"""
MARIDA Marine Debris U-Net Training Script (FIXED)
-------------------------------------------
Model    : Lightweight U-Net (~8M params)
Dataset  : MARIDA (Sentinel-2, 11 bands, 256x256 patches, reflectance [0,1])
Loss     : Dice + Focal (handles severe class imbalance)
Device   : NVIDIA RTX 4050 Laptop GPU
"""

import sys
import time
import json
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout so special chars don't crash on Windows
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------------------------
#  CONFIGURATION
# -----------------------------------------------
class Config:
    DATA_DIR    = Path(r"c:\Users\omtil\Downloads\MARIDA")
    PATCHES_DIR = DATA_DIR / "patches"
    SPLITS_DIR  = DATA_DIR / "splits"
    CKPT_DIR    = DATA_DIR / "checkpoints"

    IN_CHANNELS  = 11
    NUM_CLASSES  = 16        # 0=unlabeled, 1-15 = MARIDA classes
    FEATURES     = [32, 64, 128, 256]   # ~8M params

    EPOCHS       = 100
    BATCH_SIZE   = 8
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    AMP          = True      # FP16 mixed precision

    DICE_WEIGHT  = 0.5
    FOCAL_WEIGHT = 0.5
    FOCAL_GAMMA  = 2.0
    IGNORE_INDEX = 0         # ignore unlabeled pixels

    T_MAX        = 100       # cosine scheduler period
    SEED         = 42

CFG = Config()
try:
    CFG.CKPT_DIR.mkdir(exist_ok=True, parents=True)
except OSError:
    pass
torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)


# -----------------------------------------------
#  DATASET
# -----------------------------------------------
# Data is already in reflectance [0.01 - 0.27].
# We use per-band mean/std computed from the actual [0,1] reflectance range.
# These are approximate MARIDA paper values (reflectance * 1, not * 10000).
BAND_MEAN = np.array([
    0.0582, 0.0745, 0.0921, 0.0850, 0.0980, 0.1580,
    0.1780, 0.1830, 0.1550, 0.0985, 0.0680
], dtype=np.float32)

BAND_STD = np.array([
    0.0200, 0.0310, 0.0360, 0.0420, 0.0410, 0.0680,
    0.0730, 0.0760, 0.0730, 0.0560, 0.0340
], dtype=np.float32)


class MARIDADataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        assert split in ("train", "val", "test")
        self.augment = augment
        self.samples = self._load_split(split)
        print(f"[{split}] {len(self.samples)} patches loaded")

    def _load_split(self, split):
        txt = CFG.SPLITS_DIR / f"{split}_X.txt"
        samples = []
        for line in txt.read_text().strip().splitlines():
            name = line.strip()
            if not name:
                continue
            parts     = name.rsplit("_", 1)
            scene_id  = "S2_" + parts[0]
            idx       = parts[1]
            scene_dir = CFG.PATCHES_DIR / scene_id
            img_path  = scene_dir / f"{scene_id}_{idx}.tif"
            lbl_path  = scene_dir / f"{scene_id}_{idx}_cl.tif"
            if img_path.exists() and lbl_path.exists():
                samples.append((img_path, lbl_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        # Per-band z-score using reflectance-scale statistics
        img = (img - BAND_MEAN[:, None, None]) / (BAND_STD[:, None, None] + 1e-8)
        return np.clip(img, -5, 5).astype(np.float32)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Load 11-band image (already in reflectance [0,1])
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)            # (11, 256, 256)

        # Load label — stored as float32 in TIF, round to nearest int
        with rasterio.open(lbl_path) as src:
            lbl = np.round(src.read(1)).astype(np.int64)  # (256, 256)

        # Guard: clip to valid class IDs
        lbl = np.clip(lbl, 0, CFG.NUM_CLASSES - 1)

        # Replace NaN/Inf in image with 0
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        img = self._normalize(img)

        if self.augment:
            img, lbl = self._augment(img, lbl)

        return torch.from_numpy(img.copy()), torch.from_numpy(lbl.copy()).long()

    def _augment(self, img, lbl):
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
            lbl = lbl[::-1, :].copy()
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()
        return img, lbl


# -----------------------------------------------
#  MODEL
# -----------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LightUNet(nn.Module):
    def __init__(self, in_channels=11, num_classes=16, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.3)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


# -----------------------------------------------
#  LOSS  (FP32 always — avoid NaN from FP16 softmax)
# -----------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=0):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.float()   # ensure FP32
        ce = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt    = torch.exp(-ce.clamp(max=80))
        focal = ((1 - pt) ** self.gamma) * ce
        mask  = targets != self.ignore_index
        if mask.sum() == 0:
            return ce.mean() * 0
        return focal[mask].mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes=16, ignore_index=0, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits, targets):
        logits = logits.float()         # ensure FP32
        probs  = F.softmax(logits, dim=1)
        mask   = targets != self.ignore_index

        total, count = 0.0, 0
        for c in range(1, self.num_classes):
            p = probs[:, c][mask]
            t = (targets[mask] == c).float()
            if t.sum() == 0:
                continue
            inter  = (p * t).sum()
            total += 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            count += 1

        return total / max(count, 1)


class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.5, focal_w=0.5, gamma=2.0,
                 num_classes=16, ignore_index=0):
        super().__init__()
        self.dice  = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.dw, self.fw = dice_w, focal_w

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + \
               self.fw * self.focal(logits, targets)


# -----------------------------------------------
#  METRICS
# -----------------------------------------------
def compute_metrics(preds, targets, num_classes, ignore_index=0):
    mask    = targets != ignore_index
    p, t    = preds[mask], targets[mask]
    if len(p) == 0:
        return 0.0, 0.0
    acc  = (p == t).float().mean().item()
    ious = []
    for c in range(1, num_classes):
        pc, tc = p == c, t == c
        if not tc.any():
            continue
        inter = (pc & tc).sum().item()
        union = (pc | tc).sum().item()
        if union > 0:
            ious.append(inter / union)
    return acc, (float(np.mean(ious)) if ious else 0.0)


# -----------------------------------------------
#  TRAIN / VALIDATE
# -----------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, all_p, all_t = 0.0, [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=CFG.AMP):
            logits = model(imgs)

        # Loss is always computed in FP32
        loss = criterion(logits.float(), lbls)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_p.append(logits.argmax(1).cpu())
        all_t.append(lbls.cpu())

    preds   = torch.cat(all_p).view(-1)
    targets = torch.cat(all_t).view(-1)
    acc, miou = compute_metrics(preds, targets, CFG.NUM_CLASSES)
    return total_loss / len(loader), acc, miou


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_p, all_t = 0.0, [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        with autocast(enabled=CFG.AMP):
            logits = model(imgs)

        loss = criterion(logits.float(), lbls)
        total_loss += loss.item()
        all_p.append(logits.argmax(1).cpu())
        all_t.append(lbls.cpu())

    preds   = torch.cat(all_p).view(-1)
    targets = torch.cat(all_t).view(-1)
    acc, miou = compute_metrics(preds, targets, CFG.NUM_CLASSES)
    return total_loss / len(loader), acc, miou


# -----------------------------------------------
#  MAIN
# -----------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu    = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("=" * 62)
    print("  MARIDA U-Net Training (Fixed)")
    print("=" * 62)
    print(f"  Device : {device} ({gpu})")
    print(f"  Epochs : {CFG.EPOCHS}  |  Batch: {CFG.BATCH_SIZE}  |  AMP: {CFG.AMP}")
    print("=" * 62)

    train_ds = MARIDADataset("train", augment=True)
    val_ds   = MARIDADataset("val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    model = LightUNet(
        in_channels=CFG.IN_CHANNELS,
        num_classes=CFG.NUM_CLASSES,
        features=CFG.FEATURES
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params/1e6:.1f}M)\n")

    criterion = CombinedLoss(
        dice_w=CFG.DICE_WEIGHT, focal_w=CFG.FOCAL_WEIGHT,
        gamma=CFG.FOCAL_GAMMA, num_classes=CFG.NUM_CLASSES,
        ignore_index=CFG.IGNORE_INDEX
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.T_MAX, eta_min=1e-6
    )
    scaler = GradScaler(enabled=CFG.AMP)

    history = {k: [] for k in
               ["train_loss","val_loss","train_acc","val_acc","train_miou","val_miou"]}
    best_miou, best_epoch = 0.0, 0

    hdr = f"{'Epoch':>6} {'T-Loss':>8} {'T-Acc':>7} {'T-mIoU':>8}  {'V-Loss':>8} {'V-Acc':>7} {'V-mIoU':>8} {'LR':>9} {'Time':>6}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    total_start = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()

        t_loss, t_acc, t_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)
        v_loss, v_acc, v_miou = validate(
            model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(
            ["train_loss","val_loss","train_acc","val_acc","train_miou","val_miou"],
            [t_loss, v_loss, t_acc, v_acc, t_miou, v_miou]
        ):
            history[k].append(v)

        star = ""
        if v_miou > best_miou:
            best_miou  = v_miou
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_miou": v_miou, "val_acc": v_acc,
            }, CFG.CKPT_DIR / "best_model.pth")
            star = " <-- best"

        print(f"{epoch:>6} {t_loss:>8.4f} {t_acc:>7.4f} {t_miou:>8.4f}  "
              f"{v_loss:>8.4f} {v_acc:>7.4f} {v_miou:>8.4f} {lr:>9.2e} "
              f"{elapsed:>5.1f}s{star}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
            }, CFG.CKPT_DIR / f"ckpt_ep{epoch:03d}.pth")

    total_time = time.time() - total_start
    print(sep)
    print(f"  Done in {total_time/60:.1f} min  |  Best val mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  Best model: {CFG.CKPT_DIR / 'best_model.pth'}")
    print(sep)

    with open(CFG.CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), CFG.CKPT_DIR / "final_model.pth")
    print("  Saved: final_model.pth  |  history.json")


if __name__ == "__main__":
    main()
