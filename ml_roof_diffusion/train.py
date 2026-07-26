"""
train.py — Train footprint-conditioned roof diffusion.

Curriculum (RTX 5070 Ti class GPU, all from project root):
  1. Pregenerate synthetic data (CPU, once):
       python -m ml_roof_diffusion.dataset --pregen 20000 --out synthetic_cache --workers 8
  2. Pretrain on synthetic (60 epochs; loss plateaus early — that's normal,
     sample quality keeps improving long after):
       python -m ml_roof_diffusion.train --data synthetic --cache synthetic_cache \
              --epochs 60 --bs 128 --lr 2e-4
  3. Finetune on REAL roofs MIXED with synthetic (prevents catastrophic
     forgetting / pure-noise samples):
       python -m ml_roof_diffusion.train --data mix --cache synthetic_cache \
              --root datasets/PoznanRD --resume ckpt/roofdiff_last.pt \
              --lr 2e-5 --epochs 30 --bs 128
     (--data poznan finetunes on real roofs only.)

Checkpoints: ckpt/roofdiff_last.pt every epoch = {"model", "ema", "epoch"}.
The EMA weights are what you sample from — raw online weights look noisy.
Old plain-state-dict checkpoints still resume fine.
"""
import os
import copy
import argparse
import torch
from torch.utils.data import DataLoader

from .model import RoofUNet, RoofDiffusion
from .dataset import SyntheticRoofDataset, PoznanRDDataset, MixedRoofDataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class EMA:
    """Exponential moving average of model weights (decay 0.999).
    Diffusion samples from EMA weights are dramatically cleaner."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def load_checkpoint(path, net, ema=None):
    """Accepts both the new {'model','ema',...} dict and old raw state_dicts."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        net.load_state_dict(obj["model"])
        if ema is not None:
            ema.load_state_dict(obj.get("ema", obj["model"]))
        return obj.get("epoch", 0)
    net.load_state_dict(obj)                       # legacy format
    if ema is not None:
        ema.load_state_dict(obj)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "poznan", "mix"],
                    default="synthetic")
    ap.add_argument("--cache", default=None, help="synthetic .npz cache dir")
    ap.add_argument("--root", default=None, help="PoznanRD root")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default: 0 when data is "
                         "RAM-resident, else 4). On Windows hangs → 0.")
    ap.add_argument("--preload_workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes for one-time PoznanRD decode")
    ap.add_argument("--ema", type=float, default=0.999)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("device: cpu  !! CUDA NOT AVAILABLE — on an RTX 50-series "
              "(sm_120) install the cu128 wheel:\n"
              "   pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")

    if a.data == "synthetic":
        ds = SyntheticRoofDataset(size=a.size, cache_dir=a.cache)
    elif a.data == "poznan":
        ds = PoznanRDDataset(a.root, workers=a.preload_workers)
    else:
        assert a.cache and a.root, "--data mix needs both --cache and --root"
        ds = MixedRoofDataset(
            SyntheticRoofDataset(size=a.size, cache_dir=a.cache),
            PoznanRDDataset(a.root, workers=a.preload_workers))

    ram = getattr(ds, "ram_resident", False)
    nw = a.workers if a.workers is not None else (0 if ram else 4)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=nw,
                    pin_memory=(dev == "cuda"), drop_last=True,
                    persistent_workers=(nw > 0))
    print(f"data: {len(ds)} samples, {len(ds) // a.bs} steps/epoch, "
          f"num_workers={nw}{' (RAM-resident)' if ram else ''}")

    net = RoofUNet()
    ema = EMA(net, a.ema)
    start_ep = 0
    if a.resume:
        start_ep = load_checkpoint(a.resume, net, ema)
        print(f"resumed from {a.resume} (epoch {start_ep})")
    diff = RoofDiffusion(net, device=dev)
    ema.model.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    os.makedirs(a.out, exist_ok=True)

    for ep in range(a.epochs):
        net.train()
        tot, nb = 0.0, 0
        it = tqdm(dl, desc=f"epoch {ep + 1}/{a.epochs}", leave=False) \
            if tqdm else dl
        for step, x in enumerate(it):
            x = x.to(dev, non_blocking=True)
            hm, mask = x[:, :1], x[:, 1:]
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                loss = diff.loss(hm, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            tot += loss.item(); nb += 1
            if tqdm:
                it.set_postfix(loss=f"{tot / nb:.4f}")
            elif step % 25 == 0:
                print(f"  step {step}/{len(dl)}  loss {tot / nb:.4f}",
                      flush=True)
        print(f"epoch {ep + 1}/{a.epochs}  loss {tot / max(nb, 1):.4f}")
        ck = {"model": net.state_dict(), "ema": ema.state_dict(),
              "epoch": start_ep + ep + 1}
        torch.save(ck, os.path.join(a.out, "roofdiff_last.pt"))
        if (ep + 1) % 10 == 0:
            torch.save(ck, os.path.join(a.out, f"roofdiff_ep{ep + 1}.pt"))


if __name__ == "__main__":
    main()
