import torch
from torch import nn
import torch.nn.functional as F


class ResnetBlockVAE(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return F.silu(self.block(x) + self.res_conv(x))


class MicroscopyVAE(nn.Module):
    """ High-fidelity VAE for 256x256 -> 4x32x32 Latent Space """
    def __init__(self, in_channels=3, latent_channels=4, base_channels=64):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1), # 128
            ResnetBlockVAE(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), # 64
            ResnetBlockVAE(base_channels * 2, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1), # 32
            ResnetBlockVAE(base_channels * 4, base_channels * 4),
            nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1)
        )

        # Decoder Logic corrected for PixelShuffle(2) channel scaling
        self.decoder_input = nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1)

        self.up1 = nn.Sequential(
            ResnetBlockVAE(base_channels * 4, base_channels * 8), # Prepare for shuffle
            nn.PixelShuffle(2), # 32 -> 64. Channels: (base*8)/4 = base*2
            ResnetBlockVAE(base_channels * 2, base_channels * 2)
        )

        self.up2 = nn.Sequential(
            ResnetBlockVAE(base_channels * 2, base_channels * 4), # Prepare for shuffle
            nn.PixelShuffle(2), # 64 -> 128. Channels: (base*4)/4 = base*1
            ResnetBlockVAE(base_channels, base_channels)
        )

        self.up3 = nn.Sequential(
            ResnetBlockVAE(base_channels, base_channels * 4), # Prepare for shuffle
            nn.PixelShuffle(2), # 128 -> 256. Channels: (base*4)/4 = base*1
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        moments = self.encoder(x)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        z = self.up1(z)
        z = self.up2(z)
        return self.up3(z)

    def reconstruct_image_deterministic(self, x):
        mu, _ = self.encode(x)
        recon_x = self.decode(mu)
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
