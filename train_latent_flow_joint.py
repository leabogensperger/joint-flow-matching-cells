import argparse
import yaml
import os
import pathlib
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from src.vae import MicroscopyVAE
from src.flow_model import JointFlowUNet
from src.utils import setup_experiment, load_ema_model
from src.dataset import FlowCompoundDataset
from src.joint_flow_trainer import JointFlowTrainer


def set_reproducibility(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(description="Joint Latent Flow Matching over [z, c]")
    parser.add_argument("--config", type=str, default="config/latent_flow_joint.yaml")
    parser.add_argument("--drug", type=str, default=None)
    parser.add_argument("--cell_line", type=str, default=None)
    args = parser.parse_args()

    set_reproducibility(42)
    BASE_DIR = pathlib.Path(__file__).parent.resolve()

    with open(BASE_DIR / args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.drug:      cfg['training']['drug']   = args.drug
    if args.cell_line: cfg['model']['cell_line']  = args.cell_line

    t = cfg['training']
    m = cfg['model']
    drug = t.get('drug', None)

    run_dir = setup_experiment(cfg, BASE_DIR)
    ct_tag = {'log1p': 'logp1', 'sqrt': 'sqrt', 'uniform': 'uniform', 'log10': 'log10'}.get(
        t.get('conc_transform', None), '')
    suffix = f"_{m['cell_line']}_{drug or 'unknown'}" + (f"_{ct_tag}" if ct_tag else "")
    run_dir = run_dir.rename(run_dir.parent / (run_dir.name + suffix))
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[INFO] New experiment: {run_dir.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Joint Latent Flow Matching [z, c]")
    print(f"       Cell line: {m['cell_line']} | Drug: {drug}")
    print(f"       Device: {device}")

    # ── Load frozen VAE ───────────────────────────────────────────────────────
    vae = MicroscopyVAE(
        in_channels=cfg['vae']['in_channels'],
        latent_channels=cfg['vae']['latent_channels'],
        base_channels=cfg['vae'].get('base_channels', 64),
    ).to(device)

    vae_ckpt = BASE_DIR / cfg['vae']['checkpoint']
    vae, _ = load_ema_model(vae, vae_ckpt, device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"[INFO] VAE loaded")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(BASE_DIR, cfg['data'])
    target_concs = t.get('target_concentrations', None)
    include_dmso = t.get('include_dmso', True)

    train_ds = FlowCompoundDataset(
        csv_path=csv_path,
        cell_line=m['cell_line'],
        split='train',
        in_channels=cfg['vae']['in_channels'],
        transform=train_transform,
        data_root=os.path.join(BASE_DIR, cfg['data_root']),
        target_concentrations=target_concs,
        conc_transform=t.get('conc_transform', None),
        drug=drug,
        filtered_only=t.get('filtered_only', True),
        include_dmso_targets=include_dmso,
    )

    print(f"[INFO] Train: {len(train_ds):,}")

    g = torch.Generator()
    g.manual_seed(42)
    loader_args = dict(batch_size=t['batch_size'], num_workers=8, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_args)

    # ── Flow model ────────────────────────────────────────────────────────────
    flow_model = JointFlowUNet(
        dim=m['dim'],
        dim_mults=tuple(m['dim_mults']),
        channels=4,
        out_dim=4,
        resnet_block_groups=m['resnet_block_groups'],
    ).to(device)

    n_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"[INFO] JointFlowUNet: {n_params:,} parameters")

    optimizer = torch.optim.Adam(flow_model.parameters(), lr=float(t['learning_rate']))

    # ── Latent scale factor (normalizes the VAE latent to unit std) ──────────
    scale_path = run_dir / "latent_scale.pt"
    print("[INFO] Computing latent scale factor...")
    sample_mus = []
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        with torch.no_grad():
            mu, _ = vae.encode(batch["image_tgt"].to(device))
            sample_mus.append(mu.cpu())
    sample_mus = torch.cat(sample_mus, dim=0)
    latent_std = sample_mus.std().item()
    scale_factor = 1.0 / latent_std
    torch.save({"scale_factor": scale_factor}, scale_path)

    # ── VAE latent shape ──────────────────────────────────────────────────────
    sample_batch = next(iter(train_loader))
    sample_img = sample_batch["image_tgt"][:1].to(device)
    with torch.no_grad():
        z_sample, _ = vae.encode(sample_img)
    _, _, H_latent, W_latent = z_sample.shape
    vae_latent_shape = (H_latent, W_latent)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = JointFlowTrainer(
        flow_model=flow_model,
        vae=vae,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        p_uncond=t['p_uncond'],
        ema_decay=t['ema_decay'],
        scale_factor=scale_factor,
        lambda_c=t.get('lambda_c', 1.),
        vae_latent_shape=vae_latent_shape,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {'loss': [], 'z_loss': [], 'c_loss': []}
    log_interval = t.get('log_interval', 25)
    concs_drug = sorted(train_ds.df_tgt['concentration'].unique().tolist())
    if train_ds.conc_transform == 'log10':
        concs_drug = [train_ds.dmso_sentinel if c <= 0 else float(np.log10(c)) for c in concs_drug]

    for epoch in tqdm(range(1, t['epochs'] + 1), desc="Training", unit="epoch"):

        avg_loss, avg_z_loss, avg_c_loss = trainer.train_one_epoch(train_loader)
        history['loss'].append(avg_loss)
        history['z_loss'].append(avg_z_loss)
        history['c_loss'].append(avg_c_loss)

        # ── Normalized loss plot ──────────────────────────────────────────────
        epochs_range = range(1, len(history['loss']) + 1)
        z_losses = np.array(history['z_loss'])
        c_losses = np.array(history['c_loss'])
        z_norm = (z_losses - z_losses.min() + 1e-8) / (z_losses.max() - z_losses.min() + 1e-8)
        c_norm = (c_losses - c_losses.min() + 1e-8) / (c_losses.max() - c_losses.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_range, history['loss'], color='#1f77b4', label='Total Loss', linewidth=2)
        ax.plot(epochs_range, z_norm, color='#ff7f0e', label='Z Loss (normalized)', linewidth=2, alpha=0.7)
        ax.plot(epochs_range, c_norm, color='#2ca02c', label='C Loss (normalized)', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f"Joint Flow Matching — {m['cell_line']} | {drug}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(run_dir / "training_loss.png", dpi=150, bbox_inches='tight')
        plt.close()

        # ── Save best checkpoint ────────────────────────────────────────────────
        trainer.log_and_save(epoch, avg_loss)

        # ── Generate samples only every log_interval epochs ────────────────────
        if epoch % log_interval == 0:
            trainer.generate_samples(epoch, concs_drug)

        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | Z: {avg_z_loss:.6f} | C: {avg_c_loss:.6f}")


if __name__ == "__main__":
    main()
