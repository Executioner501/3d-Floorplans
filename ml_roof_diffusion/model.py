"""
model.py — Footprint-conditioned denoising diffusion for roof heightmaps.

Architecture (RoofDiffusion-style, simplified):
  · Input:  [noisy_heightmap, footprint_mask]  → 2×R×R
  · UNet predicts the noise ε; footprint conditioning by channel concat
    at every resolution (mask is downsampled alongside features).
  · Cosine β schedule; DDPM training loss = MSE(ε̂, ε);
    DDIM sampler for fast, deterministic-optional generation.

At sampling time the SAME footprint with different noise draws yields
different realistic roofs — generative diversity is native to the model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(ch, groups=8):
    g = min(groups, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


# ───────────────────────── time embedding ──────────────────────────
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / (half - 1))
        ang = t[:, None].float() * freqs[None]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1 = group_norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, cout)
        self.n2 = group_norm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(t))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        a = torch.softmax(q @ k / math.sqrt(c), dim=-1)
        return x + self.out((a @ v).transpose(1, 2).reshape(b, c, h, w))


class RoofUNet(nn.Module):
    """~9M params at base=64 — trains on a single consumer GPU."""
    def __init__(self, base=64, temb=64):
        super().__init__()
        self.temb = TimeEmbed(temb)
        td = temb * 4
        # +1 everywhere: footprint mask re-concatenated at each scale
        self.in_conv = nn.Conv2d(2, base, 3, padding=1)
        self.d1 = ResBlock(base + 1, base, td)
        self.d2 = ResBlock(base + 1, base * 2, td)
        self.d3 = ResBlock(base * 2 + 1, base * 4, td)
        self.attn = SelfAttention(base * 4)
        self.mid = ResBlock(base * 4 + 1, base * 4, td)
        self.u3 = ResBlock(base * 8 + 1, base * 2, td)
        self.u2 = ResBlock(base * 4 + 1, base, td)
        self.u1 = ResBlock(base * 2 + 1, base, td)
        self.out = nn.Sequential(group_norm(base), nn.SiLU(),
                                 nn.Conv2d(base, 1, 3, padding=1))

    @staticmethod
    def _cat_mask(x, m):
        return torch.cat([x, F.interpolate(m, size=x.shape[-2:],
                                           mode="nearest")], dim=1)

    def forward(self, x, mask, t):
        te = self.temb(t)
        h0 = self.in_conv(torch.cat([x, mask], dim=1))
        h1 = self.d1(self._cat_mask(h0, mask), te)
        h2 = self.d2(self._cat_mask(F.avg_pool2d(h1, 2), mask), te)
        h3 = self.d3(self._cat_mask(F.avg_pool2d(h2, 2), mask), te)
        h3 = self.attn(h3)
        m_ = self.mid(self._cat_mask(F.avg_pool2d(h3, 2), mask), te)
        u = F.interpolate(m_, scale_factor=2, mode="nearest")
        u = self.u3(self._cat_mask(torch.cat([u, h3], 1), mask), te)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u2(self._cat_mask(torch.cat([u, h2], 1), mask), te)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u1(self._cat_mask(torch.cat([u, h1], 1), mask), te)
        return self.out(u)


# ───────────────────────── diffusion process ───────────────────────
class RoofDiffusion:
    def __init__(self, model, timesteps=1000, device="cuda"):
        self.model, self.T, self.dev = model.to(device), timesteps, device
        s = 0.008
        t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.abar = (f / f[0]).clamp(1e-5, 1.0).to(device)   # ᾱ_0..T

    def loss(self, x0, mask):
        """x0: (B,1,R,R) normalized heightmap; mask: (B,1,R,R) in [-1,1]."""
        b = x0.shape[0]
        t = torch.randint(1, self.T + 1, (b,), device=self.dev)
        ab = self.abar[t][:, None, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        eps_hat = self.model(xt, mask, t)
        # only score pixels inside + near the footprint
        w = (mask > -0.5).float() * 0.9 + 0.1
        return (w * (eps - eps_hat) ** 2).mean()

    @torch.no_grad()
    def sample(self, mask, steps=50, eta=0.0, seed=None):
        """DDIM sampling. mask: (B,1,R,R) in [-1,1]. Different seeds →
        different roofs for the same footprint."""
        g = torch.Generator(self.dev)
        if seed is not None:
            g.manual_seed(int(seed))
        b = mask.shape[0]
        x = torch.randn(b, 1, *mask.shape[-2:], generator=g, device=self.dev)
        ts = torch.linspace(self.T, 1, steps).long().to(self.dev)
        for i, t in enumerate(ts):
            ab = self.abar[t]
            ab_prev = self.abar[ts[i + 1]] if i + 1 < steps else \
                torch.tensor(1.0, device=self.dev)
            eps = self.model(x, mask, t.repeat(b))
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            sig = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            noise = torch.randn(x.shape, generator=g, device=self.dev) \
                if i + 1 < steps else 0
            x = ab_prev.sqrt() * x0 + \
                (1 - ab_prev - sig ** 2).clamp(0).sqrt() * eps + sig * noise
        # outside the footprint the roof is undefined → zero
        return x0 * (mask > -0.5).float() + (-1) * (mask <= -0.5).float()
