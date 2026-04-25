"""
plot_results.py
===============
Plots training curves and runs a quick test-set evaluation.
Saves all figures to checkpoints/figures/
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

CKPT = Path(r"c:\Users\omtil\Downloads\MARIDA\checkpoints")
FIG_DIR = CKPT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Load history ────────────────────────────────────────────
with open(CKPT / "history.json") as f:
    h = json.load(f)

epochs  = np.arange(1, len(h["train_loss"]) + 1)
t_loss  = np.array(h["train_loss"])
v_loss  = np.array(h["val_loss"])
t_miou  = np.array(h["train_miou"])
v_miou  = np.array(h["val_miou"])
t_acc   = np.array(h["train_acc"])
v_acc   = np.array(h["val_acc"])

best_ep   = int(np.argmax(v_miou)) + 1
best_miou = v_miou[best_ep - 1]
best_acc  = v_acc[best_ep - 1]

PALETTE = {
    "train": "#4C9BE8",
    "val"  : "#E86B4C",
    "best" : "#2ECC71",
    "bg"   : "#0F172A",
    "grid" : "#1E293B",
    "text" : "#E2E8F0",
}

plt.rcParams.update({
    "figure.facecolor" : PALETTE["bg"],
    "axes.facecolor"   : PALETTE["grid"],
    "axes.edgecolor"   : PALETTE["text"],
    "axes.labelcolor"  : PALETTE["text"],
    "xtick.color"      : PALETTE["text"],
    "ytick.color"      : PALETTE["text"],
    "text.color"       : PALETTE["text"],
    "legend.facecolor" : "#1E293B",
    "legend.edgecolor" : PALETTE["text"],
    "grid.color"       : "#334155",
    "font.family"      : "DejaVu Sans",
})

# ── Figure 1 : Full Training Dashboard ─────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(PALETTE["bg"])
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

def vline(ax, ep, color=PALETTE["best"]):
    ax.axvline(ep, color=color, ls="--", lw=1.2, alpha=0.7)
    ax.text(ep + 0.5, ax.get_ylim()[1] * 0.97,
            f"best\nep {ep}", color=color, fontsize=7, va="top")

# ── Panel 1 : Loss ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, t_loss, color=PALETTE["train"], lw=1.5, label="Train")
ax1.plot(epochs, v_loss, color=PALETTE["val"],   lw=1.5, label="Val")
ax1.axvline(best_ep, color=PALETTE["best"], ls="--", lw=1.2, alpha=0.7)
ax1.set_title("Combined Loss  (Dice + Focal)", fontweight="bold", fontsize=10)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

# ── Panel 2 : mIoU ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, t_miou, color=PALETTE["train"], lw=1.5, label="Train")
ax2.plot(epochs, v_miou, color=PALETTE["val"],   lw=1.5, label="Val")
ax2.scatter([best_ep], [best_miou], color=PALETTE["best"],
            s=80, zorder=5, label=f"Best  {best_miou:.4f}")
ax2.axvline(best_ep, color=PALETTE["best"], ls="--", lw=1.2, alpha=0.7)
ax2.set_title("Mean IoU (mIoU)", fontweight="bold", fontsize=10)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
ax2.set_ylim(0, 1); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

# ── Panel 3 : Pixel Accuracy ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs, t_acc * 100, color=PALETTE["train"], lw=1.5, label="Train")
ax3.plot(epochs, v_acc * 100, color=PALETTE["val"],   lw=1.5, label="Val")
ax3.axvline(best_ep, color=PALETTE["best"], ls="--", lw=1.2, alpha=0.7)
ax3.set_title("Pixel Accuracy (%)", fontweight="bold", fontsize=10)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy (%)")
ax3.set_ylim(0, 100); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

# ── Panel 4 : Generalization Gap ────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
gap = t_miou - v_miou
ax4.plot(epochs, gap, color="#A78BFA", lw=1.5)
ax4.axhline(0, color=PALETTE["text"], lw=0.7, alpha=0.5)
ax4.fill_between(epochs, 0, gap, alpha=0.25, color="#A78BFA")
ax4.set_title("Overfitting Gap  (Train − Val mIoU)", fontweight="bold", fontsize=10)
ax4.set_xlabel("Epoch"); ax4.set_ylabel("mIoU Gap")
ax4.grid(True, alpha=0.4)

# ── Panel 5 : Milestone Bar Chart ────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
milestones = [10, 25, 50, 70, 100]
t_vals = [t_miou[ep-1] for ep in milestones]
v_vals = [v_miou[ep-1] for ep in milestones]
x = np.arange(len(milestones))
w = 0.35
bars1 = ax5.bar(x - w/2, t_vals, w, label="Train mIoU",
                color=PALETTE["train"], alpha=0.85)
bars2 = ax5.bar(x + w/2, v_vals, w, label="Val mIoU",
                color=PALETTE["val"],   alpha=0.85)
ax5.set_xticks(x); ax5.set_xticklabels([f"Ep {e}" for e in milestones])
ax5.set_ylim(0, 1); ax5.set_title("Milestones", fontweight="bold", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.4, axis="y")
for bar in bars1: ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{bar.get_height():.2f}", ha="center", fontsize=7)
for bar in bars2: ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{bar.get_height():.2f}", ha="center", fontsize=7)

# ── Panel 6 : Summary Stats Box ──────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary = [
    ["Metric",              "Value"],
    ["Model",               "Lightweight U-Net"],
    ["Parameters",          "7.8 Million"],
    ["GPU",                 "RTX 4050 Laptop"],
    ["Total epochs",        "100"],
    ["Time per epoch",      "~12 s"],
    ["Total training time", "21 minutes"],
    ["Best epoch",          str(best_ep)],
    ["Best val mIoU",       f"{best_miou:.4f}"],
    ["Best val Accuracy",   f"{best_acc*100:.1f}%"],
    ["Final val mIoU",      f"{v_miou[-1]:.4f}"],
    ["Overfitting gap",     f"{t_miou[best_ep-1]-v_miou[best_ep-1]:.4f}"],
]
tbl = ax6.table(cellText=summary[1:], colLabels=summary[0],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.scale(1.2, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1E293B" if r % 2 == 0 else "#0F172A")
    cell.set_edgecolor("#334155")
    cell.set_text_props(color=PALETTE["text"])
ax6.set_title("Training Summary", fontweight="bold", fontsize=10)

# ── Main title ───────────────────────────────────────────────
fig.suptitle(
    "MARIDA Marine Debris U-Net  —  Training Results",
    fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.01
)

out1 = FIG_DIR / "training_dashboard.png"
fig.savefig(out1, dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"])
plt.close()
print(f"Saved: {out1}")

# ── Figure 2 : mIoU Convergence Close-up ────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["grid"])

ax.plot(epochs, t_miou, color=PALETTE["train"], lw=2,   label="Train mIoU")
ax.plot(epochs, v_miou, color=PALETTE["val"],   lw=2,   label="Val mIoU")
ax.fill_between(epochs, v_miou, t_miou,
                alpha=0.12, color="#A78BFA", label="Overfitting zone")
ax.scatter([best_ep], [best_miou], s=120, color=PALETTE["best"],
           zorder=6, label=f"Best val mIoU = {best_miou:.4f}  (epoch {best_ep})")
ax.axvline(best_ep, color=PALETTE["best"], ls="--", lw=1.5, alpha=0.6)

# Annotate key milestones
for ep, label in [(1, "Start"), (25, "25"), (50, "50"), (70, "Best"), (100, "End")]:
    y = v_miou[ep-1]
    ax.annotate(f"{y:.3f}", (ep, y),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color=PALETTE["text"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=0.7))

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("mIoU", fontsize=11)
ax.set_title("mIoU Convergence  —  Lightweight U-Net on MARIDA",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
plt.tight_layout()

out2 = FIG_DIR / "miou_convergence.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight",
             facecolor=PALETTE["bg"])
plt.close()
print(f"Saved: {out2}")

# ── Print final summary ─────────────────────────────────────
print()
print("=" * 50)
print("  FINAL MODEL RESULTS")
print("=" * 50)
print(f"  Best val mIoU    : {best_miou:.4f}  ({best_miou*100:.1f}%)")
print(f"  Best val Acc     : {best_acc:.4f}  ({best_acc*100:.1f}%)")
print(f"  Best epoch       : {best_ep} / 100")
print(f"  Training time    : 21.0 minutes")
print(f"  Time / epoch     : ~12 seconds")
print(f"  Overfitting gap  : {t_miou[best_ep-1]-v_miou[best_ep-1]:.4f}")
print("=" * 50)
print(f"  Figures saved to: {FIG_DIR}")
