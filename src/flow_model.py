import torch
import torch.nn as nn
from einops import rearrange


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)
        h = h * (1 + scale) + shift
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        qkv = self.to_qkv(x_flat)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w) + x


class JointFlowUNet(nn.Module):
    """
    Joint flow matching UNet over the state [z, c]: predicts a velocity for the
    VAE latent z (spatial) and a velocity for the scalar drug concentration c,
    each driven by its own timestep (t_z, t_c) per the dual-timestep formulation.
    """

    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=4,
        out_dim=4,
        resnet_block_groups=8,
        time_dim=256,
    ):
        super().__init__()

        self.channels = channels
        self.out_dim = out_dim

        # ── Time embeddings (separate for t_z and t_c) ──────────────────────────
        self.time_embed_z = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.time_embed_c = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        time_emb_dim = time_dim * 2

        # ── Concentration embedding ──────────────────────────────────────────────
        # Maps scalar concentration to time_emb_dim, added to time embedding
        self.conc_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # ── Input projection ──────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(channels, dim, 3, padding=1)

        # ── Downsampling path ─────────────────────────────────────────────────
        self.downs = nn.ModuleList()
        self.down_res_blocks = nn.ModuleList()

        curr_dim = dim
        for idx, mult in enumerate(dim_mults):
            out_dim_block = dim * mult

            self.down_res_blocks.append(nn.ModuleList([
                ResnetBlock(curr_dim, out_dim_block, time_emb_dim, resnet_block_groups),
                ResnetBlock(out_dim_block, out_dim_block, time_emb_dim, resnet_block_groups),
                Attention(out_dim_block)
            ]))

            if idx != len(dim_mults) - 1:
                self.downs.append(nn.Conv2d(out_dim_block, out_dim_block, 4, 2, 1))

            curr_dim = out_dim_block

        # ── Middle ────────────────────────────────────────────────────────────
        self.mid_res1 = ResnetBlock(curr_dim, curr_dim, time_emb_dim, resnet_block_groups)
        self.mid_attn = Attention(curr_dim)
        self.mid_res2 = ResnetBlock(curr_dim, curr_dim, time_emb_dim, resnet_block_groups)

        # ── Upsampling path ───────────────────────────────────────────────────
        self.ups = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()

        # Only upsample for layers 0, 1, 2 (skip layer 3 which has no skip to concatenate)
        for idx, mult in reversed(list(enumerate(dim_mults[:-1]))):
            out_dim_block = dim * mult

            self.ups.append(nn.ConvTranspose2d(curr_dim, out_dim_block, 4, 2, 1))
            curr_dim = out_dim_block

            self.up_res_blocks.append(nn.ModuleList([
                ResnetBlock(curr_dim * 2, out_dim_block, time_emb_dim, resnet_block_groups),
                ResnetBlock(out_dim_block, out_dim_block, time_emb_dim, resnet_block_groups),
                Attention(out_dim_block)
            ]))

        # ── Output projection ─────────────────────────────────────────────────
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim, resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, out_dim, 3, padding=1)

        # Concentration velocity head (scalar output)
        bottleneck_dim = dim * dim_mults[-1]
        combined_dim = time_emb_dim + bottleneck_dim

        self.conc_velocity_head = nn.Sequential(
            nn.Linear(combined_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, 1)
        )

    def forward(self, x, t_z, t_c, c_scalar, drop_c_mask=None):
        """
        Args:
            x: [B, channels, H, W] latent state z
            t_z: [B] timestep for z (0 to 1)
            t_c: [B] timestep for c (0 to 1)
            c_scalar: [B] or [B, 1] concentration scalar
            drop_c_mask: [B] bool mask for classifier-free guidance dropout

        Returns:
            v_z: [B, out_dim, H, W] velocity prediction for the latent
            v_c: [B] velocity prediction for the concentration
        """
        # ── Time embedding ────────────────────────────────────────────────────
        t_z_emb = self.time_embed_z(t_z.unsqueeze(-1))
        t_c_emb = self.time_embed_c(t_c.unsqueeze(-1))
        time_emb = torch.cat([t_z_emb, t_c_emb], dim=1)

        # ── Concentration embedding ─────────────────────────────────────────────
        c_scalar = c_scalar.to(x.device)
        if c_scalar.dim() == 0:
            c_scalar = c_scalar.unsqueeze(0)
        if c_scalar.dim() == 1:
            c_scalar = c_scalar.unsqueeze(1)  # [B] -> [B, 1]
        c_emb = self.conc_embed(c_scalar)  # [B, time_emb_dim]

        if drop_c_mask is not None:
            keep_mask = (~drop_c_mask).float().unsqueeze(-1)  # [B, 1]
            c_emb = c_emb * keep_mask

        time_emb = time_emb + c_emb

        # ── Input projection ──────────────────────────────────────────────────
        h = self.input_conv(x)
        input_skip = h  # Save for final concatenation

        # ── Downsampling path ─────────────────────────────────────────────────
        skip_connections = []

        # Process layers 0, 1, 2 with downsampling
        for down_res_blocks, down_conv in zip(self.down_res_blocks[:-1], self.downs):
            res1, res2, attn = down_res_blocks
            h = res1(h, time_emb)
            h = res2(h, time_emb)
            h = attn(h)
            skip_connections.append(h)
            h = down_conv(h)

        # Process layer 3 (last) without downsampling
        res1, res2, attn = self.down_res_blocks[-1]
        h = res1(h, time_emb)
        h = res2(h, time_emb)
        h = attn(h)
        # Don't append this skip - it's the bottleneck

        # ── Middle ────────────────────────────────────────────────────────────
        h = self.mid_res1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, time_emb)

        # Get bottleneck features
        h_bottleneck = h.mean(dim=(2, 3))  # Global average pool [B, dim*2]

        # Concatenate with time embedding and predict c velocity with z context
        combined = torch.cat([time_emb, h_bottleneck], dim=1)
        v_c = self.conc_velocity_head(combined).squeeze(-1)  # [B]

        # ── Upsampling path ───────────────────────────────────────────────────
        for up_res_blocks, up_conv in zip(self.up_res_blocks, self.ups):
            h = up_conv(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)

            res1, res2, attn = up_res_blocks
            h = res1(h, time_emb)
            h = res2(h, time_emb)
            h = attn(h)

        # ── Output (concatenate with input_conv skip) ──────────────────────────
        h = torch.cat([h, input_skip], dim=1)
        h = self.final_res_block(h, time_emb)
        v_z = self.final_conv(h)

        return v_z, v_c
