"""Extra analyses for the journal version (CPU only, about 20-25 min). Uses the FINAL 10-model ensemble
(ResNet-34 batch 18 / 8000 steps + EfficientNetV2-S batch 18 / 8000 steps, 0.5 each), rule 0.40 / 300 px / 0.80.
  Part 1  check: pooled PQ must be about 0.4367
  Part 2  error types of the final ensemble (same classification as diag_errors.py)
  Part 3  multi-day leakage check: PQ of validation images far in time (>= 7 days) from every training image
  Part 4  pixel Dice: model vs annotators, and annotator vs annotator
  Part 5  qualitative figure  -> C:\\solar\\results\\fig_qualitative.png
Run:  python C:\\solar\\src\\paper_extras.py      then send Claude everything it prints (and the png)."""
import time, re
from collections import Counter
from datetime import datetime
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mu
import common as K
import config as C

t0 = time.time()
el = lambda: f"{(time.time() - t0) / 60:.1f} min"
THR, AREA, MEAN = 0.40, 300, 0.80
il, GT = K.get_data()
fold = il["fold"].values
N = len(il)

# ======================= Part 1: final ensemble and check =======================
ens = np.zeros((N, 1024, 1024), np.uint8)
for f in range(5):
    pos = np.flatnonzero(fold == f)
    a = np.load(C.RUNS / "exps" / f"k_bs18_it8000_f{f}" / f"oof_fold{f}.npy").astype(np.float32)
    b = np.load(C.RUNS / "exps" / f"k_effb18_f{f}" / f"oof_fold{f}.npy").astype(np.float32)
    ens[pos] = (0.5 * a + 0.5 * b).round().astype(np.uint8)
    del a, b
comps = K.extract(ens, THR)
del ens
preds = [[c[3] for c in cs if c[0] >= AREA and c[1] >= MEAN] for cs in comps]
cnt = np.zeros((N, 4))
for i, pr in enumerate(preds):
    for g in GT[i]:
        cnt[i] += K.cnt(pr, g)
pq = lambda s: s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2])
S = cnt.sum(0)
print(f"=== Part 1: check ===\n  final ensemble pooled PQ {pq(S):.4f} (should be 0.4367) | TP {int(S[0])} FP {int(S[1])} FN {int(S[2])}  ({el()})", flush=True)


# ======================= Part 2: error types =======================
def classify(pr, g, fnc, fpc):
    """Sort every missed filament (FN) and every false blob (FP) of one annotator record into an error type."""
    if not g:
        fpc["nothing annotated there"] += len(pr)
        return
    if not pr:
        fnc["missed (best IoU < 0.1)"] += len(g)
        return
    ap = np.asarray(mu.area(pr), float)
    ag = np.asarray(mu.area(g), float)
    I = np.asarray(mu.iou(pr, g, [1] * len(g))) * ap[:, None]      # intersection in px
    iou = I / (ap[:, None] + ag[None, :] - I)
    r, c = linear_sum_assignment(-(iou * (iou > 0.5)))
    ok = iou[r, c] > 0.5
    mp, mg = set(r[ok].tolist()), set(c[ok].tolist())
    inside = I >= 0.5 * ap[:, None]
    covers = I >= 0.5 * ag[None, :]
    multi_cover = covers & (covers.sum(1, keepdims=True) >= 2)
    for j in range(len(g)):
        if j in mg:
            continue
        pcs = np.flatnonzero(inside[:, j])
        best = iou[:, j].max()
        if len(pcs) >= 2:
            fnc["split into pieces"] += 1
        elif multi_cover[:, j].any():
            fnc["merged with a neighbouring filament"] += 1
        elif best >= 0.3:
            fnc["near miss (best IoU 0.3-0.5)"] += 1
        elif best >= 0.1:
            fnc["poor overlap (best IoU 0.1-0.3)"] += 1
        else:
            fnc["missed (best IoU < 0.1)"] += 1
    for p in range(len(pr)):
        if p in mp:
            continue
        if covers[p].sum() >= 2:
            fpc["covers 2+ filaments (merge)"] += 1
        elif inside[p].any():
            fpc["piece of an annotated filament"] += 1
        elif iou[p].max() >= 0.3:
            fpc["near miss (best IoU 0.3-0.5)"] += 1
        elif I[p].max() < 0.1 * ap[p]:
            fpc["nothing annotated there"] += 1
        else:
            fpc["partial overlap"] += 1


fnc, fpc = Counter(), Counter()
for i, pr in enumerate(preds):
    for g in GT[i]:
        classify(pr, g, fnc, fpc)
print("\n=== Part 2: error types of the FINAL ensemble ===")
for title, c, tot in (("missed filaments (FN)", fnc, S[2]), ("false blobs (FP)", fpc, S[1])):
    print(f"  {title}: {int(tot)}   (classified {sum(c.values())})")
    for k, v in c.most_common():
        print(f"    {k:40s} {v:5d}  {v / max(tot, 1):6.1%}")
print(f"  ({el()})", flush=True)


# ======================= Part 3: multi-day leakage check =======================
def to_time(stamp):
    d = re.sub(r"\D", "", str(stamp))
    for n, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M"), (8, "%Y%m%d")):
        if len(d) >= n:
            return datetime.strptime(d[:n], fmt)
    raise ValueError(f"cannot read a date from stamp {stamp!r}")


T = np.array([to_time(s).timestamp() for s in il["stamp"]]) / 86400.0          # days
gap = np.zeros(N)
for f in range(5):
    v, t = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
    gap[v] = np.abs(T[v, None] - T[None, t]).min(1)
print("\n=== Part 3: time to the nearest TRAINING image, for each validation image ===")
print(f"  example stamps -> dates: {il['stamp'].iloc[0]} -> {to_time(il['stamp'].iloc[0])}, "
      f"{il['stamp'].iloc[-1]} -> {to_time(il['stamp'].iloc[-1])}")
print(f"  median gap {np.median(gap) * 24:.1f} h (earlier measurement: 44.0 h) | "
      f"share >= 2 days {np.mean(gap >= 2):.0%}, >= 7 days {np.mean(gap >= 7):.0%}, >= 30 days {np.mean(gap >= 30):.0%}")
rng = np.random.default_rng(0)
for name, m in (("gap < 2 days", gap < 2), ("gap 2-7 days", (gap >= 2) & (gap < 7)), ("gap >= 7 days", gap >= 7),
                ("all images", gap >= 0)):
    idx = np.flatnonzero(m)
    if len(idx) < 10:
        print(f"  {name:14s}: only {len(idx)} images, skipped")
        continue
    bs = pq(cnt[rng.choice(idx, (2000, len(idx)))].sum(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {name:14s}: {len(idx):3d} images, PQ {pq(cnt[idx].sum(0)):.4f} (95% CI {lo:.4f}..{hi:.4f})")
print(f"  ({el()})", flush=True)


# ======================= Part 4: pixel Dice =======================
def union(rles):
    return mu.merge(rles) if rles else None


def inter_area(a, b):
    if a is None or b is None:
        return 0.0
    return float(mu.area(mu.merge([a, b], intersect=True)))


area = lambda m: 0.0 if m is None else float(mu.area(m))
num_m = den_m = num_h = den_h = 0.0
for i, pr in enumerate(preds):
    P = union(pr)
    A = [union(g) for g in GT[i]]
    for G in A:
        num_m += 2 * inter_area(P, G)
        den_m += area(P) + area(G)
    for x in range(len(A)):
        for y in range(len(A)):
            if x != y:
                num_h += 2 * inter_area(A[x], A[y])
                den_h += area(A[x]) + area(A[y])
print("\n=== Part 4: pixel Dice (pooled over image-annotator pairs) ===")
print(f"  model vs annotators: {num_m / den_m:.3f}   annotator vs annotator (296 multi-annotator images): {num_h / den_h:.3f}")
print(f"  ({el()})", flush=True)

# ======================= Part 5: qualitative figure =======================
COL_A = [(60, 200, 60), (0, 210, 255), (230, 90, 230)]       # annotators 1-3 (BGR): green, yellow, magenta
COL_M = (50, 50, 235)                                        # model: red
small_area = lambda r: float(mu.area(r))


def best_iou(r, others):
    return float(np.max(mu.iou([r], others, [0] * len(others)))) if others else 0.0


cands = []
for i in range(N):
    if len(GT[i]) < 2 or not GT[i][0] or not preds[i]:
        continue
    a0, a1 = GT[i][0], GT[i][1]
    dis = [r for r in a0 if small_area(r) > 1500 and 0.05 < best_iou(r, a1) < 0.5]
    miss = [r for r in a0 if small_area(r) < 400 and best_iou(r, preds[i]) < 0.1]
    if dis and miss:
        t, fp, fn, s = K.cnt(preds[i], a0)
        cands.append((s / max(t + .5 * fp + .5 * fn, 1e-9), i, dis, miss))
if not cands:
    print("\n=== Part 5: no image met all conditions - tell Claude ===")
else:
    med = np.median([c[0] for c in cands])
    p_i, i, dis, miss = min(cands, key=lambda c: abs(c[0] - med))
    path = next(p for p in C.TRAIN_IMG.iterdir() if il["stamp"].iloc[i] in p.name)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    layers = [(g, COL_A[k % 3]) for k, g in enumerate(GT[i])] + [(preds[i], COL_M)]
    masks = [([mu.decode(r) for r in rl], col) for rl, col in layers]

    def panel(y0, x0, size, out=800, lw=2):
        crop = cv2.resize(img[y0:y0 + size, x0:x0 + size], (out, out),
                          interpolation=cv2.INTER_AREA if size > out else cv2.INTER_CUBIC)
        v = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for ms, col in masks:
            for m in ms:
                mm = cv2.resize(np.ascontiguousarray(m[y0:y0 + size, x0:x0 + size]), (out, out), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(v, cs, -1, col, lw)
        return v

    def window(r, size):
        x, y, w, h = mu.toBbox(r)
        cy, cx = int(y + h / 2), int(x + w / 2)
        return int(np.clip(cy - size // 2, 0, 2048 - size)), int(np.clip(cx - size // 2, 0, 2048 - size))

    dr = max(dis, key=small_area)
    sb = int(min(max(mu.toBbox(dr)[2:].max() * 1.6, 360), 900))
    yb, xb = window(dr, sb)
    mr = miss[0]
    sc = 360
    yc, xc = window(mr, sc)
    pa = panel(0, 0, 2048, lw=2)
    k = 800 / 2048
    for (y0, x0, s), lab in (((yb, xb, sb), "b"), ((yc, xc, sc), "c")):
        cv2.rectangle(pa, (int(x0 * k), int(y0 * k)), (int((x0 + s) * k), int((y0 + s) * k)), (255, 255, 255), 2)
        cv2.putText(pa, lab, (int(x0 * k) + 4, int(y0 * k) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    pb, pc = panel(yb, xb, sb, lw=3), panel(yc, xc, sc, lw=3)
    for v, lab in ((pa, "(a)"), (pb, "(b)"), (pc, "(c)")):
        cv2.rectangle(v, (0, 0), (62, 36), (0, 0, 0), -1)
        cv2.putText(v, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    gapc = np.full((800, 12, 3), 255, np.uint8)
    out = np.hstack([pa, gapc, pb, gapc, pc])
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(C.RESULTS / "fig_qualitative.png"), out)
    print(f"  figure size {out.shape[1]} x {out.shape[0]} px (journal minimum for full width: 2244 px)")
    print("\n=== Part 5: qualitative figure ===")
    print(f"  image {path.name} (station {il['instrument'].iloc[i]}, fold {fold[i]}), {len(GT[i])} annotators, "
          f"model {len(preds[i])} filaments, PQ vs annotator 1 {p_i:.3f} (median of {len(cands)} candidate images {med:.3f})")
    print(f"  (b) window {sb} px around a filament of area {small_area(dr):.0f} px; IoU with annotator 2's best match "
          f"{best_iou(dr, GT[i][1]):.2f}, with the model's best match {best_iou(dr, preds[i]):.2f}")
    print(f"  (c) window {sc} px around a small filament of area {small_area(mr):.0f} px; model's best IoU {best_iou(mr, preds[i]):.2f}")
    print(f"  colours: annotators green / yellow / magenta, model red | saved {C.RESULTS / 'fig_qualitative.png'}")
print(f"\ntotal time {el()}")