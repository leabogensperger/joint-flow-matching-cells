import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import copy
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont


class JointFlowTrainer:
    """
    Trainer for joint flow matching over the state [z, c]: the VAE latent z
    and the scalar drug concentration c evolve with separate timesteps
    (t_z, t_c), per the dual-timestep formulation (Eq. 6-9 in the paper).
    """

    def __init__(
        self,
        flow_model,
        vae,
        optimizer,
        device,
        run_dir,
        p_uncond=0.0,
        ema_decay=0.999,
        scale_factor=1.0,
        lambda_c=1.,
        vae_latent_shape=None,
    ):
        self.flow_model = flow_model
        self.vae = vae
        self.device = device
        self.run_dir = Path(run_dir)
        self.p_uncond = p_uncond
        self.scale_factor = scale_factor
        self.vae_latent_shape = vae_latent_shape or (32, 32)
        self.lambda_c = lambda_c

        (self.run_dir / "samples").mkdir(parents=True, exist_ok=True)

        # EMA model
        self.ema_model = copy.deepcopy(flow_model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            flow_model.parameters(),
            lr=optimizer.defaults['lr']
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=5000, eta_min=1e-6)
        self.best_loss = float('inf')

        print(f"[INFO] JointFlowTrainer | VAE latent shape: {self.vae_latent_shape}")

    def train_one_epoch(self, dataloader):
        """Train and return (avg_loss, avg_z_loss, avg_c_loss)"""
        self.flow_model.train()
        self.vae.eval()

        total_loss = total_z_loss = total_c_loss = 0.0
        num_batches = 0
        H_latent, W_latent = self.vae_latent_shape

        for batch in dataloader:
            img_tgt = batch["image_tgt"].to(self.device)
            c_tgt = batch["concentration"].to(self.device)
            B = img_tgt.shape[0]

            # Encode target
            with torch.no_grad():
                z_tgt, _ = self.vae.encode(img_tgt)

            # Initial noisy states (random)
            z_0 = torch.randn(B, 4, H_latent, W_latent, device=self.device)
            c_0_scalar = torch.randn((B,), device=self.device)

            # Dual timesteps
            t_z = torch.rand(B, device=self.device)
            t_c = torch.rand(B, device=self.device)

            t_z_expand = t_z.view(B, 1, 1, 1)

            # Interpolate
            z_t = (1 - t_z_expand) * z_0 + t_z_expand * z_tgt
            c_t_scalar = (1 - t_c) * c_0_scalar + t_c * c_tgt

            # Velocity targets
            v_z_target = z_tgt - z_0
            v_c_target = c_tgt - c_0_scalar

            drop_c_mask = None
            if self.p_uncond > 0:
                drop_c_mask = torch.rand(B, device=self.device) < self.p_uncond

            v_z_pred, v_c_pred = self.flow_model(
                z_t, t_z, t_c, c_scalar=c_t_scalar, drop_c_mask=drop_c_mask
            )

            # Losses
            z_loss = F.mse_loss(v_z_pred, v_z_target)

            if drop_c_mask is not None and drop_c_mask.any():
                keep_mask = ~drop_c_mask
                if keep_mask.any():
                    c_loss = F.mse_loss(v_c_pred[keep_mask], v_c_target[keep_mask])
                else:
                    c_loss = torch.tensor(0.0, device=self.device)
            else:
                c_loss = F.mse_loss(v_c_pred, v_c_target)
            loss = z_loss + self.lambda_c * c_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self._update_ema()

            total_loss += loss.item()
            total_z_loss += z_loss.item()
            total_c_loss += c_loss.item()
            num_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        avg_z_loss = total_z_loss / max(num_batches, 1)
        avg_c_loss = total_c_loss / max(num_batches, 1)
        return avg_loss, avg_z_loss, avg_c_loss

    def _update_ema(self, decay=0.999):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.flow_model.parameters()):
                ema_p.data = decay * ema_p.data + (1 - decay) * model_p.data

    @torch.no_grad()
    def _sample_morphology_images(self, concs_target, n_samples=8, steps=50):
        """
        Sample decoded images at target concentrations from pure noise
        (dose-conditioned generation, Sec. 3.2: fixes t_c=1, integrates
        t_z: 0 -> 1 from z_0 ~ N(0, I)). Seeded for reproducibility.
        Returns: dict {c: [n_samples, 3, H, W]}
        """
        torch.manual_seed(42)

        H, W = self.vae_latent_shape
        samples_by_conc = {}

        for c_target in concs_target:
            images = []
            for _ in range(n_samples):
                z_t = torch.randn(1, 4, H, W, device=self.device)
                c_t = torch.full((1,), c_target, dtype=torch.float32, device=self.device)

                dt = 1.0 / steps
                for step in range(steps):
                    t_z = torch.full((1,), step / steps, device=self.device)
                    t_c = torch.ones(1, device=self.device)

                    v_z, _ = self.ema_model(z_t, t_z, t_c, c_scalar=c_t)
                    z_t = z_t + v_z * dt

                img = self.vae.decode(z_t / self.scale_factor).squeeze(0)
                images.append(img.cpu())

            samples_by_conc[c_target] = torch.stack(images)  # [n_samples, 3, H, W]

        return samples_by_conc

    def log_and_save(self, epoch, train_loss):
        """Save best checkpoint only."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.flow_model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
        }

        if train_loss < self.best_loss:
            self.best_loss = train_loss
            best_path = self.run_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            print(f"[SAVE] Best checkpoint @ epoch {epoch} | Loss: {train_loss:.6f}")
        else:
            print(f"[INFO] Epoch {epoch} | Loss: {train_loss:.6f} (best: {self.best_loss:.6f})")

    def generate_samples(self, epoch, concs_eval):
        """Save a morphology sample grid across concentrations (log_interval epochs only)."""
        try:
            samples_by_conc = self._sample_morphology_images(concs_eval, n_samples=8, steps=50)

            all_images = []
            for c_val in sorted(samples_by_conc.keys()):
                for img_tensor in samples_by_conc[c_val]:
                    img_pil = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_pil)
                    draw = ImageDraw.Draw(img_pil)
                    try:
                        # matplotlib bundles DejaVu Sans, giving a portable TTF across platforms
                        import matplotlib
                        font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
                        font = ImageFont.truetype(font_path, 12)
                    except Exception:
                        font = ImageFont.load_default()
                    draw.text((5, 5), f"c={c_val:.3f}", fill=(255, 255, 255), font=font)

                    img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
                    all_images.append(img_tensor)

            if all_images:
                grid = torch.stack(all_images)
                vutils.save_image(
                    grid,
                    self.run_dir / "samples" / f"epoch_{epoch:04d}_morphology.png",
                    nrow=8,
                    normalize=False
                )
                print(f"[SAMPLES] Morphology grid saved for epoch {epoch}")

        except Exception as e:
            print(f"[WARN] Sampling failed: {e}")
