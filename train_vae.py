import argparse
import yaml
import os
import pathlib
import numpy as np
import random
import torch
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.vae import MicroscopyVAE
from src.vae_trainer import MicroscopyVAETrainer
from src.utils import setup_experiment, plot_losses
from src.dataset import MicroscopyCellDataset


def set_reproducibility(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms(cfg, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])


def main():
    parser = argparse.ArgumentParser(description="Microscopy VAE Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/vae_microscopy.yaml",
        help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    set_reproducibility(42)

    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    config_path = BASE_DIR / args.config

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_dir = setup_experiment(cfg, BASE_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Experiment: {cfg['project_name']} | Saving to: {run_dir}")

    train_transform = get_transforms(cfg, split='train')
    val_transform   = get_transforms(cfg, split='val')

    # -------------------------------------------------------------------------
    # Dataset — trained jointly on all cell lines / drugs / concentrations
    # -------------------------------------------------------------------------
    csv_path = os.path.join(BASE_DIR, cfg['data'])
    data_root = os.path.join(BASE_DIR, cfg['data_root'])

    train_ds = MicroscopyCellDataset(
        csv_path=csv_path,
        cell_line=None,
        split='train',
        in_channels=cfg['model']['in_channels'],
        transform=train_transform,
        data_root=data_root,
        target_concentrations=None,
        drug=cfg['model'].get('drug'),
        filtered_only=cfg['training']['filtered_only']
    )
    val_ds = MicroscopyCellDataset(
        csv_path=csv_path,
        cell_line=None,
        split='val',
        in_channels=cfg['model']['in_channels'],
        transform=val_transform,
        data_root=data_root,
        target_concentrations=None,
        drug=cfg['model'].get('drug'),
        filtered_only=cfg['training']['filtered_only']
    )

    # -------------------------------------------------------------------------
    # Dataloaders
    # -------------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(42)

    loader_args = {
        "batch_size":    cfg['training']['batch_size'],
        "num_workers":   8,
        "worker_init_fn": seed_worker,
        "generator":     g,
        "pin_memory":    True
    }

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_args)

    # -------------------------------------------------------------------------
    # Model & Trainer
    # -------------------------------------------------------------------------
    model = MicroscopyVAE(
        in_channels=cfg['model']['in_channels'],
        latent_channels=cfg['model']['latent_channels'],
        base_channels=cfg['model'].get('base_channels', 64)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training']['weight_decay'])
    )

    num_epochs       = cfg['training']['epochs']
    total_steps      = len(train_loader) * num_epochs
    num_warmup_steps = int(total_steps * cfg['training']['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    trainer = MicroscopyVAETrainer(
        model, optimizer, scheduler, device,
        run_dir=run_dir,
        beta=cfg['training']['beta'],
        alpha_ssim=cfg['training'].get('alpha_ssim', 0.5),
        lam_rot=cfg['training']['lam_rot'],
        ema_decay=cfg['training']['ema_decay'],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model: MicroscopyVAE | Params: {n_params:,}")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    log_interval         = cfg['training'].get('log_interval', 1)
    train_history        = []
    val_history          = []
    val_metrics_history  = []
    sample_batch         = next(iter(val_loader))

    pbar_epoch = tqdm(range(1, num_epochs + 1), desc="[INFO] Total Progress")
    for epoch in pbar_epoch:
        avg_train_loss = trainer.train_one_epoch(train_loader)
        train_history.append(avg_train_loss)

        avg_val_loss, val_metrics = trainer.validate(val_loader)
        val_history.append(avg_val_loss)
        val_metrics_history.append(val_metrics)

        current_sample = sample_batch if (epoch % log_interval == 0) else None
        trainer.log_and_save(
            epoch, avg_train_loss, avg_val_loss, val_metrics,
            sample_batch=current_sample
        )

        plot_losses(train_history, val_history, val_metrics_history, run_dir / "training_loss.png")

        pbar_epoch.set_postfix({
            'T_loss': f"{avg_train_loss:.4f}",
            'V_loss': f"{avg_val_loss:.4f}",
            'SSIM': f"{val_metrics['ssim_loss']:.4f}"
        })


if __name__ == "__main__":
    main()
