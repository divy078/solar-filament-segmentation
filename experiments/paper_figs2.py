"""Three new figures for the paper (CPU only, about 10-20 min). Uses the same out-of-fold maps as paper_extras.py.
  Part 1  check: the final ensemble's pooled PQ must be 0.4367
  Part 2  share of filaments found, by filament size: model vs. another annotator      -> fig_size.pdf
  Part 3  first vs. long training recipe, fold by fold (rule A: 0.35 / 400 px / 0.80)   -> fig_training.pdf
  Part 4  pre-processing and soft target on one image                                    -> fig_preprocess.pdf
Output folder: C:\\solar\\results\\paper_figures2 (PDF + PNG of each figure and numbers.json).
Run:  python C:\\solar\\src\\paper_figs2.py
Then send Claude everything it prints, the three PDFs and numbers.json."""
import json, time
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
OUT = C.RESULTS / "paper_figures2"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
el = lambda: f"{(time.time() - t0) / 60:.1f} min"
il, GT = K.get_data()
fold = il["fold"].values
N = len(il)
pq = lambda s: float(s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2]))
NUM = {}
RULE_A, RULE_B = (0.35, 400, 0.80), (0.40, 300, 0.80)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_group(name):
    """Out-of-fold maps (N x 1024 x 1024, uint8) of one five-model group, wherever they are stored."""
    out = np.zeros((N, 1024, 1024), np.uint8)
    for f in range(5):
        pos = np.flatnonzero(fold == f)
        p1 = C.RUNS / "exps" / f"{name}_f{f}" / f"oof_fold{f}.npy"
        p2 = C.RUNS / "oof_maps" / name / f"oof_fold{f}.npy"
        p = p1 if p1.exists() else p2
        if not p.exists():
            return None
        a = np.load(p)
        assert a.shape[0] == len(pos), (str(p), a.shape, len(pos))
        out[pos] = np.asarray(a, np.float32).round().clip(0, 255).astype(np.uint8)
        del a
    return out


def average(a, b, wa=0.5):
    out = np.empty_like(a)
    for s in range(0, N, 50):
        out[s:s + 50] = (wa * a[s:s + 50].astype(np.float32) + (1 - wa) * b[s:s + 50].astype(np.float32)).round().astype(np.uint8)
    return out


def predict(maps, rule):
    thr, area, mean = rule
    return [[c[3] for c in cs if c[0] >= area and c[1] >= mean] for cs in K.extract(maps, thr)]


def counts(preds):
    cnt = np.zeros((N, 4))
    for i, pr in enumerate(preds):
        for g in GT[i]:
            cnt[i] += K.cnt(pr, g)
    return cnt


def per_fold(cnt):
    return [pq(cnt[fold == f].sum(0)) for f in range(5)], pq(cnt.sum(0))


# ---------- choose the Part 4 image first (needs only the annotations) ----------
def union(rles):
    return mu.decode(mu.merge(rles)) if rles else np.zeros((2048, 2048), np.uint8)


best = None
for i in range(N):
    if len(GT[i]) != 3 or any(not g for g in GT[i]):
        continue
    S = sum(union(g).astype(np.uint8) for g in GT[i])
    n, lab, st, _ = cv2.connectedComponentsWithStats((S > 0).astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if not 2000 <= a <= 20000:
            continue
        v = S[lab == k]
        part, full = np.mean((v == 1) | (v == 2)), np.mean(v == 3)
        score = min(part, full)
        if best is None or score > best[0]:
            best = (score, i, x, y, w, h, part, full)
if best is None:
    raise SystemExit("Part 4: no image with three annotators and a suitable filament - tell Claude")
_, I4, bx, by, bw, bh, part4, full4 = best
SZ = int(np.clip(1.5 * max(bw, bh), 384, 768)) // 4 * 4
Y4 = int(np.clip(by + bh / 2 - SZ / 2, 0, 2048 - SZ))
X4 = int(np.clip(bx + bw / 2 - SZ / 2, 0, 2048 - SZ))

# ======================= Part 1: final ensemble and check =======================
res_long = load_group("k_bs18_it8000")
eff_long = load_group("k_effb18")
final = average(res_long, eff_long)
final_map4 = final[I4].copy()
pred_final = predict(final, RULE_B)
del final
cnt_final = counts(pred_final)
S1 = cnt_final.sum(0)
print(f"=== Part 1: check ===\n  final ensemble pooled PQ {pq(S1):.4f} (should be 0.4367) | TP {int(S1[0])} FP {int(S1[1])} FN {int(S1[2])}  ({el()})", flush=True)
NUM["final"] = {"pq": pq(S1), "tp": int(S1[0]), "fp": int(S1[1]), "fn": int(S1[2])}

# ======================= Part 2: filaments found, by size =======================
BINS = [(0, 400, "< 400"), (400, 1000, "400–1,000"), (1000, 3000, "1,000–3,000"), (3000, 10 ** 9, "≥ 3,000")]


def found(g, pr):
    """For each filament in g: is it matched (IoU > 0.5) by some region in pr? With IoU > 0.5 a match is unique."""
    if not g:
        return np.zeros(0, bool)
    if not pr:
        return np.zeros(len(g), bool)
    return (np.asarray(mu.iou(pr, g, [0] * len(g))) > 0.5).any(0)


def binned(pairs):
    n, k = np.zeros(len(BINS)), np.zeros(len(BINS))
    for g, pr in pairs:
        if not g:
            continue
        a, f = np.asarray(mu.area(g), float), found(g, pr)
        for b, (lo, hi, _) in enumerate(BINS):
            m = (a >= lo) & (a < hi)
            n[b] += m.sum()
            k[b] += f[m].sum()
    return n, k


multi = [i for i in range(N) if len(GT[i]) >= 2]
n_all, k_all = binned((g, pred_final[i]) for i in range(N) for g in GT[i])
n_mul, k_mul = binned((g, pred_final[i]) for i in multi for g in GT[i])
n_hum, k_hum = binned((GT[i][x], GT[i][y]) for i in multi for x in range(len(GT[i])) for y in range(len(GT[i])) if x != y)
print("\n=== Part 2: share of annotated filaments found (IoU > 0.5), by filament area ===")
print(f"  check: filaments found by the model {int(k_all.sum())} (must equal TP {int(S1[0])})")
print(f"  {'area (px)':12s} {'model, 707 images':>24s} {'model, 296 images':>24s} {'other annotator, 296 images':>30s}")
for b, (_, _, name) in enumerate(BINS):
    print(f"  {name:12s} {k_all[b] / n_all[b]:7.1%} of {int(n_all[b]):5d}       {k_mul[b] / n_mul[b]:7.1%} of {int(n_mul[b]):5d}"
          f"       {k_hum[b] / n_hum[b]:7.1%} of {int(n_hum[b]):5d}")
miss = n_all - k_all
print(f"  missed by the model, share of all misses: " + ", ".join(f"{BINS[b][2]} {miss[b] / miss.sum():.0%}" for b in range(len(BINS))))
print(f"  ({el()})", flush=True)
NUM["size"] = {"bins": [b[2] for b in BINS], "model_all_n": n_all.tolist(), "model_all_found": k_all.tolist(),
               "model_multi_n": n_mul.tolist(), "model_multi_found": k_mul.tolist(),
               "annot_n": n_hum.tolist(), "annot_found": k_hum.tolist()}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(BINS))
wd = 0.27
series = [(k_all / n_all, "Model, all 707 images", "#1f5f99"),
          (k_mul / n_mul, "Model, 296 multi-annotator images", "#7fb2dd"),
          (k_hum / n_hum, "Another annotator, same 296 images", "#d9822b")]
for s, (v, lab, col) in enumerate(series):
    bars = a1.bar(x + (s - 1) * wd, 100 * v, wd, label=lab, color=col)
    for r, val in zip(bars, v):
        a1.text(r.get_x() + r.get_width() / 2, 100 * val + 1.2, f"{100 * val:.0f}", ha="center", va="bottom", fontsize=6.5)
a1.set_xticks(x)
a1.set_xticklabels([b[2] for b in BINS], fontsize=7)
a1.set_xlabel("Filament area (pixels, 2048 × 2048 image)")
a1.set_ylabel("Filaments found (%)")
a1.set_ylim(0, 100)
a1.legend(frameon=False, fontsize=6.5, loc="upper left")
a1.set_title("(a) Share found, by size", loc="left", fontsize=8)
a2.bar(x, k_all, 0.6, color="#1f5f99", label="Found")
a2.bar(x, miss, 0.6, bottom=k_all, color="#c9c9c9", label="Missed")
for b in range(len(BINS)):
    a2.text(x[b], n_all[b] + max(n_all) * 0.015, f"{int(miss[b])} missed", ha="center", va="bottom", fontsize=6.5)
a2.set_xticks(x)
a2.set_xticklabels([b[2] for b in BINS], fontsize=7)
a2.set_xlabel("Filament area (pixels, 2048 × 2048 image)")
a2.set_ylabel("Annotated filaments (all 707 images)")
a2.set_ylim(0, max(n_all) * 1.15)
a2.legend(frameon=False, fontsize=6.5, loc="upper right")
a2.set_title("(b) Found and missed filaments", loc="left", fontsize=8)
for a in (a1, a2):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_size")

# ======================= Part 3: first vs. long recipe, per fold =======================
print("\n=== Part 3: first vs. long training recipe, per fold, rule A (0.35 / 400 px / 0.80) ===")
res_long_pf, res_long_all = per_fold(counts(predict(res_long, RULE_A)))
del res_long
eff_long_pf, eff_long_all = per_fold(counts(predict(eff_long, RULE_A)))
del eff_long
m = load_group("res1024")
if m is None:
    raise SystemExit("Part 3: first-recipe ResNet-34 maps (res1024) not found - tell Claude")
res_first_pf, res_first_all = per_fold(counts(predict(m, RULE_A)))
del m
print(f"  ResNet-34        first {res_first_all:.4f} (paper 0.412)  long {res_long_all:.4f} (paper 0.429)")
cands = {}
a = load_group("effv2s")
if a is not None:
    cands["effv2s"] = per_fold(counts(predict(a, RULE_A)))
    b = load_group("effv2s_s2")
    if b is not None:
        cands["effv2s + effv2s_s2"] = per_fold(counts(predict(average(a, b), RULE_A)))
        del b
    del a
for k, (pf, al) in cands.items():
    print(f"  EfficientNetV2-S first, candidate '{k}': {al:.4f} (paper 0.416)")
eff_key = min(cands, key=lambda k: abs(cands[k][1] - 0.416)) if cands else None
if eff_key is None or abs(cands[eff_key][1] - 0.416) > 0.002:
    print("  !! no EfficientNetV2-S first-recipe group matches 0.416 - tell Claude; the figure shows ResNet-34 only")
    eff_key = None
else:
    eff_first_pf, eff_first_all = cands[eff_key]
    print(f"  EfficientNetV2-S first = '{eff_key}' {eff_first_all:.4f}  long {eff_long_all:.4f} (paper 0.420)")
for f in range(5):
    line = f"  fold {f}: ResNet-34 {res_first_pf[f]:.4f} -> {res_long_pf[f]:.4f} ({res_long_pf[f] - res_first_pf[f]:+.4f})"
    if eff_key:
        line += f" | EfficientNetV2-S {eff_first_pf[f]:.4f} -> {eff_long_pf[f]:.4f} ({eff_long_pf[f] - eff_first_pf[f]:+.4f})"
    print(line)
print(f"  ({el()})", flush=True)
NUM["training"] = {"res_first": res_first_pf + [res_first_all], "res_long": res_long_pf + [res_long_all],
                   "eff_long": eff_long_pf + [eff_long_all], "eff_first_key": eff_key,
                   "eff_first": (eff_first_pf + [eff_first_all]) if eff_key else None}

groups = [("(a) ResNet-34", res_first_pf, res_first_all, res_long_pf, res_long_all)]
if eff_key:
    groups.append(("(b) EfficientNetV2-S", eff_first_pf, eff_first_all, eff_long_pf, eff_long_all))
fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 2.7), sharey=True, squeeze=False)
lo = min(min(g[1] + g[3]) for g in groups) - 0.008
hi = max(max(g[1] + g[3]) for g in groups) + 0.008
for ax, (title, fp, fa, lp, la) in zip(axes[0], groups):
    for f in range(5):
        ax.plot([0, 1], [fp[f], lp[f]], "-o", color="#a6a6a6", lw=0.9, ms=3, label="One fold" if f == 0 else None)
    ax.plot([0, 1], [fa, la], "-o", color="black", lw=1.8, ms=4.5, label="All 707 images")
    ax.text(-0.07, fa, f"{fa:.3f}", ha="right", va="center", fontsize=7)
    ax.text(1.07, la, f"{la:.3f}", ha="left", va="center", fontsize=7)
    ax.text(0.5, max(fa, la) + 0.004, f"{la - fa:+.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["First recipe\nbatch 8, 4,000 steps", "Long recipe\nbatch 18, 8,000 steps"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(lo, hi)
    ax.set_title(title, loc="left", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0][0].set_ylabel("PQ (rule A)")
fig.tight_layout(w_pad=2.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98))
save(fig, "fig_training")

# ======================= Part 4: pre-processing and soft target =======================
path = next(p for p in C.TRAIN_IMG.iterdir() if il["stamp"].iloc[I4] in p.name)
img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
x1024 = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
med = float(np.median(x1024[x1024 > 25]))
norm = np.clip(x1024 / med, 0, 2.5) - 1
soft = sum(union(g).astype(np.float32) for g in GT[I4]) / len(GT[I4])
kept = union(pred_final[I4])
y0, x0, s = Y4, X4, SZ
h0, w0, hs = y0 // 2, x0 // 2, s // 2
crop_in = img[y0:y0 + s, x0:x0 + s]
crop_norm = norm[h0:h0 + hs, w0:w0 + hs]
crop_soft = soft[y0:y0 + s, x0:x0 + s]
crop_prob = final_map4[h0:h0 + hs, w0:w0 + hs].astype(np.float32) / 255
crop_kept = kept[y0:y0 + s, x0:x0 + s]
ext = [0, s, s, 0]

fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.25))
ax[0].imshow(crop_in, cmap="gray", extent=ext, interpolation="nearest")
ax[0].set_title("(a) Input image", loc="left", fontsize=8)
im1 = ax[1].imshow(crop_norm, cmap="gray", extent=ext, interpolation="nearest", vmin=-1, vmax=1.5)
ax[1].set_title("(b) Normalised network input", loc="left", fontsize=8)
cmap_s = ListedColormap(["#000000", "#3b528b", "#21918c", "#fde725"])
im2 = ax[2].imshow(crop_soft, cmap=cmap_s, norm=BoundaryNorm([-0.01, 0.2, 0.5, 0.8, 1.01], 4), extent=ext, interpolation="nearest")
ax[2].set_title("(c) Soft training target", loc="left", fontsize=8)
im3 = ax[3].imshow(crop_prob, cmap="magma", vmin=0, vmax=1, extent=ext, interpolation="bilinear")
if crop_kept.any():
    ax[3].contour(np.linspace(0, s, s), np.linspace(0, s, s), crop_kept, levels=[0.5], colors="#00e5ff", linewidths=0.8)
ax[3].set_title("(d) Model output", loc="left", fontsize=8)
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
cax = [make_axes_locatable(a).append_axes("right", size="5%", pad=0.04) for a in ax]
cax[0].set_visible(False)
cb1 = fig.colorbar(im1, cax=cax[1])
cb1.ax.tick_params(labelsize=6)
cb2 = fig.colorbar(im2, cax=cax[2], ticks=[0, 1 / 3, 2 / 3, 1])
cb2.ax.set_yticklabels(["0", "1/3", "2/3", "1"], fontsize=6)
cb3 = fig.colorbar(im3, cax=cax[3])
cb3.ax.tick_params(labelsize=6)
fig.subplots_adjust(left=0.01, right=0.97, top=0.88, bottom=0.03, wspace=0.28)
save(fig, "fig_preprocess")
print("\n=== Part 4: pre-processing figure ===")
print(f"  image {path.name} (station {il['instrument'].iloc[I4]}), crop {s} x {s} px at ({x0}, {y0}) of the 2048-pixel image")
print(f"  filament region: {part4:.0%} of its pixels marked by one or two of the three annotators, {full4:.0%} by all three")
print(f"  disk median brightness {med:.1f}; kept model filaments in the crop: {'yes' if crop_kept.any() else 'none'}")
NUM["preprocess"] = {"image": path.name, "station": str(il["instrument"].iloc[I4]), "crop": [int(x0), int(y0), int(s)],
                     "partial_share": float(part4), "full_share": float(full4), "median": med}
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
print(f"\nsaved in {OUT}: fig_size, fig_training, fig_preprocess (.pdf and .png) and numbers.json")
print(f"total time {el()}")