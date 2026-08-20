"""
Test-set reconstruction quality of the microscopy VAE: PSNR, SSIM and pixel
MSE between real images and their deterministic reconstructions, plus a
saved grid of example reconstructions.
"""
import argparse
import os
import pathlib
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import torchvision.utils as vutils
from pytorch_msssim import ssim as ssim_fn

from src.vae import MicroscopyVAE
from src.dataset import MicroscopyCellDataset
from src.utils import load_ema_model
from train_vae import get_transforms


@torch.no_grad()
def evaluate(model, loader, device):
    total_mse, total_psnr, total_ssim, n = 0.0, 0.0, 0.0, 0
    example_real, example_recon = None, None

    for batch in tqdm(loader, desc="Evaluating"):
        img = batch['image'].to(device)
        recon = model.reconstruct_image_deterministic(img)

        mse = F.mse_loss(recon, img, reduction='none').mean(dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
        batch_ssim = ssim_fn(recon, img, data_range=1.0, size_average=False)

        total_mse += mse.sum().item()
        total_psnr += psnr.sum().item()
        total_ssim += batch_ssim.sum().item()
        n += img.shape[0]

        if example_real is None:
            example_real, example_recon = img[:8].cpu(), recon[:8].cpu()

    return {
        "mse": total_mse / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "n_samples": n,
    }, example_real, example_recon


def main():
    parser = argparse.ArgumentParser(description="Microscopy VAE test-set evaluation")
    parser.add_argument("--exp", type=str, required=True,
                         help="Experiment name under experiments/, e.g. microscopy_vae/<timestamp>")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data", type=str, default=None,
                         help="Override the metadata CSV baked into the checkpoint's config "
                              "(e.g. metadata/sample_metadata.csv, to sanity-check a checkpoint "
                              "against the small bundled sample instead of the full dataset).")
    parser.add_argument("--data_root", type=str, default=None,
                         help="Override data_root; required together with --data.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    exp_dir = BASE_DIR / "experiments" / args.exp
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "best_model.pth"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if args.data:
        assert args.data_root, "--data requires --data_root"
        cfg['data'], cfg['data_root'] = args.data, args.data_root

    model = MicroscopyVAE(
        in_channels=cfg['model']['in_channels'],
        latent_channels=cfg['model']['latent_channels'],
        base_channels=cfg['model'].get('base_channels', 64),
    ).to(device)
    model, _ = load_ema_model(model, checkpoint_path, device)
    model.eval()

    eval_dir = exp_dir / f"evaluation_{checkpoint_path.stem}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(cfg['data']).stem  # e.g. "sample_metadata" vs. the real dataset's CSV name

    csv_path = os.path.join(BASE_DIR, cfg['data'])
    split = args.split if args.split in pd.read_csv(csv_path)['split'].unique() else 'val'
    ds = MicroscopyCellDataset(
        csv_path=csv_path,
        cell_line=None,
        split=split,
        in_channels=cfg['model']['in_channels'],
        transform=get_transforms(cfg, split='val'),
        data_root=os.path.join(BASE_DIR, cfg['data_root']),
        filtered_only=cfg['training']['filtered_only'],
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    metrics, example_real, example_recon = evaluate(model, loader, device)

    print(f"[RESULT] data={tag} | split={split} | N={metrics['n_samples']} | "
          f"MSE={metrics['mse']:.5f} | PSNR={metrics['psnr']:.2f} | SSIM={metrics['ssim']:.4f}")

    pd.DataFrame([metrics]).to_csv(eval_dir / f"reconstruction_metrics_{tag}_{split}.csv", index=False)

    grid = vutils.make_grid(
        torch.cat([example_real, example_recon], dim=0), nrow=example_real.shape[0]
    )
    vutils.save_image(grid, eval_dir / f"reconstructions_{tag}_{split}.png")

    print(f"[SUCCESS] Results saved to {eval_dir}")


if __name__ == "__main__":
    main()
