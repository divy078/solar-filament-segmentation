"""Draws Figure 1 (pipeline) and Figure 2 (effect sizes with 95% paired bootstrap intervals) of the paper.
The numbers in Figure 2 are copied from the experiment logs (Results Log / paper Table 1 and Section 6).
Output: a folder "paper_figures" next to this script (Figure_1.pdf, Figure_2.pdf + PNG previews).
CPU only, a few seconds.   Run:  python C:\\solar\\src\\figs.py"""
from pathlib import Path
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
OUT = Path(__file__).resolve().parent / "paper_figures"
OUT.mkdir(exist_ok=True)

# ---------- Figure 1: pipeline ----------
fig, ax = plt.subplots(figsize=(7.0, 1.9), dpi=300)
ax.set_xlim(-0.5, 101); ax.set_ylim(0, 31); ax.axis("off")
INK, LINE = "#1f2328", "#57606a"
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=LINE, lw=0.8))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=5.5, color=INK, linespacing=1.3)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=LINE))
W, H, Y, GAP = 16.6, 19, 5, 4.1
xs = [0.5 + i * (W + GAP) for i in range(5)]
box(xs[0], Y, W, H, "Input", "GONG H-alpha image\n2048 × 2048 px\ngreyscale", "#eef2f6")
box(xs[1], Y, W, H, "Pre-processing", "resize to 1024 px\ndivide by median\ndisk brightness\nclip 0 to 2.5, minus 1", "#eef2f6")
ax.add_patch(FancyBboxPatch((xs[2], Y - 3), W, H + 6, boxstyle="round,pad=0.3,rounding_size=1.0", fc="#f4f6f8", ec=LINE, lw=0.8, ls="--"))
ax.text(xs[2] + W / 2, Y + H + 3.6, "Ensemble: 10 U-Nets", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=INK)
for yy, t in ((Y + 9.6, "ResNet-34 U-Net"), (Y - 0.6, "EffNetV2-S U-Net")):
    ax.add_patch(FancyBboxPatch((xs[2] + 1.0, yy), W - 2.0, 7.6, boxstyle="round,pad=0.25,rounding_size=0.7", fc="#dbe7f3", ec=LINE, lw=0.6))
    ax.text(xs[2] + W / 2, yy + 5.3, t, ha="center", va="center", fontsize=5.9, fontweight="bold", color=INK)
    ax.text(xs[2] + W / 2, yy + 2.1, "5 folds, mirror avg.", ha="center", va="center", fontsize=5.0, color=INK)
box(xs[3], Y, W, H, "Probability map", "mean of the 2 groups\n(equal weight)\nenlarged to 2048 px", "#eef2f6")
box(xs[4], Y, W, H, "Post-processing", "cut at 0.40\n8-connected blobs\nkeep: area ≥ 300 px\nmean prob. ≥ 0.80", "#e6f2e8")
for p, q in zip(xs[:-1], xs[1:]):
    arrow(p + W + 0.4, Y + H / 2, q - 0.4, Y + H / 2)
ax.text(xs[4] + W / 2, Y - 1.5, "one mask per filament (RLE)", ha="center", va="top", fontsize=5.5, color=INK, style="italic")
plt.savefig(OUT / "Figure_1.png", bbox_inches="tight", pad_inches=0.02); plt.savefig(OUT / "Figure_1.pdf", bbox_inches="tight", pad_inches=0.02); plt.close()

# ---------- Figure 2: effect sizes with 95% intervals ----------
rows = [  # label, delta, lo, hi, group
    ("Longer training, ResNet-34 group alone", 0.0174, 0.0121, 0.0223, "a"),
    ("Longer training, ResNet-34 group in ensemble", 0.0088, 0.0049, 0.0125, "a"),
    ("Longer training, EffNetV2-S group alone", 0.0041, -0.0013, 0.0097, "a"),
    ("Ensemble (3 groups) vs ResNet-34 group alone", 0.0062, 0.0028, 0.0096, "a"),
    ("Final 10 models vs 15 models", 0.0008, -0.0031, 0.0047, "a"),
    ("Rule: cut-off 0.40 / area 300 vs 0.35 / 400", 0.0023, -0.0002, 0.0050, "a"),
    ("Rule: cut-off 0.30 vs 0.35", -0.0050, -0.0075, -0.0022, "a"),
    ("Rule: mean probability 0.75 vs 0.80", -0.0030, -0.0049, -0.0008, "a"),
    ("Joining blobs closer than 10 px", 0.0018, -0.0023, 0.0060, "a"),
    ("Add UNet++ member (fold 0)", 0.0008, -0.003, 0.005, "b"),
    ("Add Linknet member (fold 0)", -0.0044, -0.009, -0.0002, "b"),
    ("Add SegFormer member (fold 0)", -0.0081, -0.014, -0.002, "b"),
    ("Add FPN member (fold 0)", -0.0085, -0.015, -0.003, "b"),
]
fig, ax = plt.subplots(figsize=(3.35, 3.1), dpi=300)
n = len(rows)
for i, (lab, d, lo, hi, g) in enumerate(rows):
    y = n - 1 - i
    sig = lo > 0 or hi < 0
    col = ("#1a7f37" if d > 0 else "#cf222e") if sig else "#6e7781"
    ax.plot([lo, hi], [y, y], color=col, lw=1.3, solid_capstyle="butt")
    ax.plot(d, y, "o", ms=3.6, color=col, mec="white", mew=0.5, zorder=3)
ax.axvline(0, color="#57606a", lw=0.7)
ax.axhline(n - 1 - 8.5, color="#d0d7de", lw=0.6, ls="--")
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=5.8)
ax.set_xlabel("Change in PQ (95% paired bootstrap interval)", fontsize=6.4)
ax.tick_params(axis="x", labelsize=5.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.99, 0.01, "rows above dashed line: 707 images\nbelow: fold 0, 144 images", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=5.2, color="#57606a")
plt.savefig(OUT / "Figure_2.png", bbox_inches="tight", pad_inches=0.02); plt.savefig(OUT / "Figure_2.pdf", bbox_inches="tight", pad_inches=0.02); plt.close()
print(f"figures saved in {OUT}")