import torch
from tqdm import tqdm
from pathlib import Path
import logging
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd


class MicroscopyVAETrainer:
    """ Trainer for the microscopy VAE: L1 + multi-scale SSIM + KL loss (Eq. 1 in the paper). """
    def __init__(self, model, optimizer, scheduler, device, run_dir, beta, alpha_ssim=0.5, lam_rot=0.0, ema_decay=0.999):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = Path(run_dir)
        self.beta = beta
        self.lam_rot = lam_rot
        self.alpha_ssim = alpha_ssim  # Weight for SSIM vs L1
        self.best_val_loss = float('inf')

        self.ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay)
        )

        # File Logging Setup
        self.logger = logging.getLogger("MicroscopyVAETrainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.run_dir / "training_vae.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(loader, desc="Microscopy VAE Training"):
            img = batch['image'].to(self.device)

            recon_img, mu, logvar = self.model(img)

            # 1. Structural loss (data_range=1.0 because images are [0, 1])
            ssim_loss = 1 - ms_ssim(recon_img, img, data_range=1.0, size_average=True)

            # 2. Pixel loss
            l1_loss = F.l1_loss(recon_img, img)

            # 3. KL divergence (mu/logvar are [B, 4, 32, 32], sum over spatial dims too)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3)))

            loss = (self.alpha_ssim * ssim_loss) + ((1 - self.alpha_ssim) * l1_loss) + (self.beta * kld_loss)

            if self.lam_rot > 0.:
                import torchvision.transforms.functional as F_tv
                angle = torch.rand(1).item() * 180 - 90
                img_rot = F_tv.rotate(img, angle)
                _, mu_rot, _ = self.model(img_rot)
                # Morphology latent should be invariant to cell orientation
                loss += self.lam_rot * F.mse_loss(mu, mu_rot)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.ema_model.update_parameters(self.model)
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.ema_model.eval()
        val_loss, val_ssim, val_kld = 0, 0, 0
        for batch in loader:
            img = batch['image'].to(self.device)
            recon_img, mu, logvar = self.model(img)

            ssim_l = 1 - ms_ssim(recon_img, img, data_range=1.0)
            l1_l = F.l1_loss(recon_img, img)
            kld_l = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3)))

            loss = (self.alpha_ssim * ssim_l) + ((1 - self.alpha_ssim) * l1_l) + (self.beta * kld_l)

            val_loss += loss.item()
            val_ssim += ssim_l.item()
            val_kld += kld_l.item()

        metrics = {'ssim_loss': val_ssim / len(loader), 'kld_loss': val_kld / len(loader)}
        return val_loss / len(loader), metrics

    def log_and_save(self, epoch, train_loss, val_loss, val_metrics, sample_batch=None):
        current_ssim = val_metrics.get('ssim_loss', 0.0)
        current_kld = val_metrics.get('kld_loss', 0.0)

        print(f"Epoch [{epoch}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SSIM: {current_ssim:.4f}")

        log_file = self.run_dir / "logs.csv"
        df = pd.DataFrame([{
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ssim_loss": current_ssim,
            "kld_loss": current_kld
        }])
        df.to_csv(log_file, mode='a', header=not log_file.exists(), index=False)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'ema_state_dict': self.ema_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'ssim_loss': current_ssim
            }
            torch.save(checkpoint_data, self.run_dir / "best_model.pth")
            print(f" -> [NEW BEST] Saved checkpoint to {self.run_dir / 'best_model.pth'}")

        if sample_batch is not None:
            self.generate_visual_report(epoch, sample_batch)

    def generate_visual_report(self, epoch, batch, n_samples=4):
        self.ema_model.eval()
        with torch.no_grad():
            real_imgs = batch['image'][:n_samples].to(self.device)
            recon_imgs, mu, _ = self.ema_model(real_imgs)

            combined = torch.cat([real_imgs, recon_imgs], dim=0)
            grid = vutils.make_grid(combined, nrow=n_samples, normalize=True)

            plt.figure(figsize=(12, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title(f"Top: Real | Bottom: Reconstruction (Epoch {epoch})")
            plt.axis('off')

            log_path = self.run_dir / "logging"
            log_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(log_path / f"vae_viz_{epoch:03d}.png")
            plt.close()
