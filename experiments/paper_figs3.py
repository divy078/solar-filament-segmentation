"""Final versions of the three new figures (CPU only, about 1 minute).
Re-draws the training-volume and filament-size figures from numbers.json (written by paper_figs2.py, nothing is
recomputed) and re-makes the pre-processing figure with a tighter crop and overlays on the image.
Output: C:\\solar\\results\\paper_figures3 -> Figure_2 (pre-processing), Figure_4 (training), Figure_5 (size), .pdf + .png
Run:  python C:\\solar\\src\\paper_figs3.py      then send Claude the output and the three PDFs."""
import json
import numpy as np, cv2
import pycocotools.mask as mu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import common as K
import config as C

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
SRC = C.RESULTS / "paper_figures2" / "numbers.json"
OUT = C.RESULTS / "paper_figures3"
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.loads(SRC.read_text())
BLUE, LIGHT, ORANGE, GREY = "#1f5f99", "#86b6de", "#d9822b", "#c9c9c9"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)   # 600 dpi for the image panels
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ======================= Figure 4: training volume, fold by fold =======================
T = NUM["training"]
groups = [("(a) ResNet-34", T["res_first"], T["res_long"])]
if T.get("eff_first"):
    groups.append(("(b) EfficientNetV2-S", T["eff_first"], T["eff_long"]))
fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 2.75), sharey=True, squeeze=False)
vals = [v for g in groups for v in g[1][:5] + g[2][:5]]
lo, hi = min(vals) - 0.006, max(vals) + 0.008
for ax, (title, first, long_) in zip(axes[0], groups):
    fp, fa, lp, la = first[:5], first[5], long_[:5], long_[5]
    for f in range(5):
        ax.plot([0, 1], [fp[f], lp[f]], "-o", color="#a9a9a9", lw=0.9, ms=3, zorder=2,
                label="One cross-validation fold" if f == 0 else None)
    ax.plot([0, 1], [fa, la], "-o", color="black", lw=1.8, ms=4.5, zorder=3, label="All 707 images")
    ax.text(-0.08, fa, f"{fa:.3f}", ha="right", va="center", fontsize=7)
    ax.text(1.08, la, f"{la:.3f}", ha="left", va="center", fontsize=7)
    better = sum(l > f for f, l in zip(fp, lp))
    ax.text(0.5, hi - 0.0015, f"{la - fa:+.3f} PQ, {better} of 5 folds better", ha="center", va="top", fontsize=7.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["First recipe\nbatch 8, 4,000 steps", "Long recipe\nbatch 18, 8,000 steps"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(lo, hi)
    ax.set_title(title, loc="left", fontsize=8)
    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    clean(ax)
axes[0][0].set_ylabel("PQ (rule A)")
fig.tight_layout(w_pad=2.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.985))
save(fig, "Figure_4")
print(f"Figure 4: " + "; ".join(f"{g[0]} {g[1][5]:.4f} -> {g[2][5]:.4f}" for g in groups))

# ======================= Figure 5: filaments found, by size =======================
S = NUM["size"]
bins = S["bins"]
n_all, k_all = np.array(S["model_all_n"]), np.array(S["model_all_found"])
n_mul, k_mul = np.array(S["model_multi_n"]), np.array(S["model_multi_found"])
n_hum, k_hum = np.array(S["annot_n"]), np.array(S["annot_found"])
miss = n_all - k_all
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.75), gridspec_kw={"width_ratios": [1.3, 1]})
x = np.arange(len(bins))
wd = 0.27
for s, (v, lab, col) in enumerate([(k_all / n_all, "Model, all 707 images", BLUE),
                                   (k_mul / n_mul, "Model, 296 multi-annotator images", LIGHT),
                                   (k_hum / n_hum, "Another annotator, same 296 images", ORANGE)]):
    bars = a1.bar(x + (s - 1) * wd, 100 * v, wd, label=lab, color=col, zorder=2)
    for r, val in zip(bars, v):
        a1.text(r.get_x() + r.get_width() / 2, 100 * val + 1.2, f"{100 * val:.0f}", ha="center", va="bottom", fontsize=6.5)
a1.set_xticks(x)
a1.set_xticklabels(bins, fontsize=7)
a1.set_xlabel("Filament area (pixels, 2048 × 2048 image)")
a1.set_ylabel("Filaments found (%)")
a1.set_ylim(0, 100)
a1.legend(frameon=False, fontsize=6.5, loc="upper left")
a1.set_title("(a) Share of annotated filaments found", loc="left", fontsize=8)
a1.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
a2.bar(x, k_all, 0.6, color=BLUE, label="Found", zorder=2)
a2.bar(x, miss, 0.6, bottom=k_all, color=GREY, label="Missed", zorder=2)
for b in range(len(bins)):
    a2.text(x[b], n_all[b] + n_all.max() * 0.015, f"{int(miss[b]):,} missed\n({miss[b] / miss.sum():.0%})",
            ha="center", va="bottom", fontsize=6.3, linespacing=1.1)
a2.set_xticks(x)
a2.set_xticklabels(bins, fontsize=7)
a2.set_xlabel("Filament area (pixels, 2048 × 2048 image)")
a2.set_ylabel("Annotated filaments, all 707 images")
a2.set_ylim(0, n_all.max() * 1.25)
a2.legend(frameon=False, fontsize=6.5, loc="upper left")
a2.set_title("(b) Found and missed filaments", loc="left", fontsize=8)
a2.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
for a in (a1, a2):
    clean(a)
fig.tight_layout(w_pad=2.0)
save(fig, "Figure_5")
print("Figure 5: model found " + ", ".join(f"{b} {k / n:.1%}" for b, k, n in zip(bins, k_all, n_all)))

# ======================= Figure 2: pre-processing and soft target =======================
il, GT = K.get_data()
fold = il["fold"].values
name = NUM["preprocess"]["image"]
I4 = next(i for i in range(len(il)) if str(il["stamp"].iloc[i]) in name)


def union(rles):
    return mu.decode(mu.merge(rles)) if rles else np.zeros((2048, 2048), np.uint8)


Ssum = sum(union(g).astype(np.uint8) for g in GT[I4])
n, lab, st, _ = cv2.connectedComponentsWithStats((Ssum > 0).astype(np.uint8), connectivity=8)
best = None
for k in range(1, n):
    bx, by, bw, bh, a = st[k]
    if not 2000 <= a <= 20000:
        continue
    v = Ssum[lab == k]
    score = min(np.mean((v == 1) | (v == 2)), np.mean(v == 3))
    if best is None or score > best[0]:
        best = (score, bx, by, bw, bh)
_, bx, by, bw, bh = best
s = int(np.clip(1.3 * max(bw, bh), 192, 768)) // 4 * 4
y0 = int(np.clip(by + bh / 2 - s / 2, 0, 2048 - s)) // 2 * 2
x0 = int(np.clip(bx + bw / 2 - s / 2, 0, 2048 - s)) // 2 * 2

f = int(fold[I4])
pos = list(np.flatnonzero(fold == f)).index(I4)
maps = [np.load(C.RUNS / "exps" / f"{g}_f{f}" / f"oof_fold{f}.npy", mmap_mode="r")[pos].astype(np.float32)
        for g in ("k_bs18_it8000", "k_effb18")]
prob = (0.5 * maps[0] + 0.5 * maps[1]).round().astype(np.uint8)
kept = [c[3] for c in K.extract(prob[None], 0.40)[0] if c[0] >= 300 and c[1] >= 0.80]
kept = union(kept)

path = next(p for p in C.TRAIN_IMG.iterdir() if name in p.name)
img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
x1024 = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
med = float(np.median(x1024[x1024 > 25]))
norm = np.clip(x1024 / med, 0, 2.5) - 1
soft = Ssum.astype(np.float32) / len(GT[I4])

c_in = img[y0:y0 + s, x0:x0 + s].astype(np.float32)
c_norm = norm[y0 // 2:(y0 + s) // 2, x0 // 2:(x0 + s) // 2]
c_soft = soft[y0:y0 + s, x0:x0 + s]
c_prob = cv2.resize(prob[y0 // 2:(y0 + s) // 2, x0 // 2:(x0 + s) // 2].astype(np.float32) / 255, (s, s),
                    interpolation=cv2.INTER_LINEAR)
c_kept = kept[y0:y0 + s, x0:x0 + s]
bg = cv2.resize(c_norm, (s, s), interpolation=cv2.INTER_NEAREST)
g = np.clip((bg + 1) / 2.5, 0, 1)
gray = np.dstack([g, g, g])

cols_soft = np.array([[0.35, 0.55, 0.95], [0.10, 0.75, 0.70], [1.00, 0.85, 0.10]])
over_soft = gray.copy()
for j, val in enumerate((1 / 3, 2 / 3, 1.0)):
    m = np.isclose(c_soft, val, atol=0.05)
    over_soft[m] = 0.15 * gray[m] + 0.85 * cols_soft[j]
cm = plt.get_cmap("magma")
pr = np.clip(c_prob, 0, 1)
alpha = np.clip(pr * 1.6, 0, 0.9)[..., None]
over_prob = (1 - alpha) * gray + alpha * cm(pr)[..., :3]

ext = [0, s, s, 0]
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.2))
ax[0].imshow(c_in, cmap="gray", extent=ext, interpolation="nearest")
im1 = ax[1].imshow(c_norm, cmap="gray", extent=ext, interpolation="nearest", vmin=-1, vmax=1.5)
ax[2].imshow(over_soft, extent=ext, interpolation="nearest")
ax[3].imshow(over_prob, extent=ext, interpolation="nearest")
if c_kept.any():
    ax[3].contour(np.arange(s) + 0.5, np.arange(s) + 0.5, c_kept, levels=[0.5], colors="#00e5ff", linewidths=0.7)
for a, t in zip(ax, ["(a) Input image", "(b) Normalised input", "(c) Soft training target", "(d) Model output"]):
    a.set_title(t, loc="left", fontsize=8)
    a.set_xticks([])
    a.set_yticks([])
cax = [make_axes_locatable(a).append_axes("right", size="5%", pad=0.04) for a in ax]
cax[0].set_visible(False)
cb1 = fig.colorbar(im1, cax=cax[1])
cb1.ax.tick_params(labelsize=6)
cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm([0, 1, 2, 3], 3), cmap=ListedColormap(cols_soft)),
                   cax=cax[2], ticks=[0.5, 1.5, 2.5])
cb2.ax.set_yticklabels(["1/3", "2/3", "1"], fontsize=6)
cb3 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cm), cax=cax[3])
cb3.ax.tick_params(labelsize=6)
fig.subplots_adjust(left=0.01, right=0.97, top=0.88, bottom=0.03, wspace=0.3)
save(fig, "Figure_2")
print(f"Figure 2: {name}, crop {s} x {s} px at ({x0}, {y0}) of the 2048-pixel image "
      f"({s // 2} x {s // 2} px at the network's 1024 scale); model filaments in crop: {'yes' if c_kept.any() else 'none'}")
print(f"saved in {OUT}")