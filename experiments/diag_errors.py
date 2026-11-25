"""A16 - where do the errors come from, and does joining nearby blobs help?
Part 1  annotations: how often is ONE annotated filament drawn as several separate pieces?
Part 2  error types of the current ensemble (3 groups x 5 folds, rule thr 0.35 / area 400 / mean 0.80)
Part 3  post-processing test: blobs within r px of each other count as ONE filament (r = 0 is the current method)
CPU only, about 30-40 min.   Run:  python C:\\solar\\src\\diag_errors.py"""
import time
from collections import Counter
import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mu
import common as K

H = W = 2048
THR, AREA = 0.35, 400
R_GRID = (0, 5, 10, 20, 40)          # join setting r in px at 2048 scale (joins gaps up to about 2r)
MEAN_GRID = (0.80, 0.75)             # 0.80 = current rule, must come first
t0 = time.time()
el = lambda: f"{(time.time() - t0) / 60:.1f} min"
il, GT = K.get_data()
fold = il["fold"].values

# ======================= Part 1: annotation shapes =======================
print("=== Part 1: is one annotated filament sometimes drawn as several pieces? ===", flush=True)
rows = []
for recs in GT:
    for g in recs:
        for rle in g:
            x, y, w, h = [int(v) for v in mu.toBbox(rle)]
            y0, x0 = max(y - 45, 0), max(x - 45, 0)
            m = np.ascontiguousarray(mu.decode(rle)[y0:y + h + 45, x0:x + w + 45])
            n0 = cv2.connectedComponents(m, connectivity=8)[0] - 1
            if n0 <= 1:
                rows.append((n0,) * 5)
                continue
            d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
            rows.append((n0, *[cv2.connectedComponents((d <= r).astype(np.uint8), connectivity=8)[0] - 1
                               for r in R_GRID[1:]]))
rows = np.array(rows)
multi = rows[:, 0] >= 2
print(f"  annotated filaments (all annotators): {len(rows)}   empty masks: {(rows[:, 0] == 0).sum()}")
print(f"  drawn as ONE piece: {(rows[:, 0] == 1).mean():.1%}   as 2+ pieces: {multi.mean():.1%} ({multi.sum()})")
if multi.any():
    for j, r in enumerate(R_GRID[1:], 1):
        print(f"    of the multi-piece ones, all pieces joined at r={r:2d}: {(rows[multi, j] == 1).mean():.1%}")
print(f"  ({el()})\n", flush=True)


# ======================= helpers =======================
def extract_grouped(oof, r, thr=THR, min_area=100, max_comps=60):
    """Like K.extract, but pixels above thr that lie within r px of each other form ONE blob.
    r = 0 reproduces K.extract exactly. Blob = (area, mean_prob, peak_prob, RLE); area/mean use only real pixels."""
    out = []
    for i in range(len(oof)):
        up = cv2.resize(oof[i], (W, H), interpolation=cv2.INTER_LINEAR)
        fg = (up > int(thr * 255)).astype(np.uint8)
        if r > 0:
            near = (cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5) <= r).astype(np.uint8)
            n, lab = cv2.connectedComponents(near, connectivity=8)
            lab[fg == 0] = 0
        else:
            n, lab = cv2.connectedComponents(fg, connectivity=8)
        comps = []
        if n > 1:
            areas = np.bincount(lab.ravel(), minlength=n)[1:]
            keep = [k for k in np.argsort(-areas)[:max_comps] if areas[k] >= min_area]
            if keep:
                ids = np.array(keep) + 1
                mean = ndi.mean(up, lab, ids)
                mx = ndi.maximum(up, lab, ids)
                sls = ndi.find_objects(lab)
                for k, mn, pk in zip(ids, mean, mx):
                    sl = sls[k - 1]
                    m = np.zeros((H, W), np.uint8, order="F")
                    m[sl] = (lab[sl] == k)
                    comps.append((int(areas[k - 1]), float(mn) / 255, float(pk) / 255, mu.encode(m)))
        out.append(comps)
        if i % 200 == 0:
            print(f"    r={r}: extracted {i + 1}/{len(oof)} ({el()})", flush=True)
    return out


def score(comps, mean_min):
    """-> per-photo counts (707 x [TP, FP, FN, IoU sum]), kept predictions, per-image mean PQ."""
    cnts, preds, macro = np.zeros((len(comps), 4)), [], []
    for i, cs in enumerate(comps):
        pr = [c[3] for c in cs if c[0] >= AREA and c[1] >= mean_min]
        ms = []
        for g in GT[i]:
            t, fp, fn, s = K.cnt(pr, g)
            cnts[i] += (t, fp, fn, s)
            d = t + .5 * fp + .5 * fn
            ms.append(s / d if d > 0 else 1.0)
        preds.append(pr)
        macro.append(np.mean(ms))
    return cnts, preds, float(np.mean(macro))


def parts(c):
    s = c.sum(0)
    sq = s[3] / max(s[0], 1)
    rq = s[0] / (s[0] + .5 * s[1] + .5 * s[2])
    return sq * rq, sq, rq, s


rng = np.random.default_rng(0)
IDX = rng.integers(0, len(il), (2000, len(il)))


def boot(c, b):
    """95% interval of the pooled-PQ difference, resampling photos."""
    f = lambda s: s[:, 3] / (s[:, 0] + .5 * s[:, 1] + .5 * s[:, 2])
    d = f(c[IDX].sum(1)) - f(b[IDX].sum(1))
    return np.percentile(d, [2.5, 97.5])


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
    inside = I >= 0.5 * ap[:, None]            # blob lies mostly inside this filament
    covers = I >= 0.5 * ag[None, :]            # blob covers most of this filament
    multi_cover = covers & (covers.sum(1, keepdims=True) >= 2)
    for j in range(len(g)):
        if j in mg:
            continue
        pcs = np.flatnonzero(inside[:, j])
        best = iou[:, j].max()
        if len(pcs) >= 2:
            s = I[pcs, j].sum()
            jiou = s / (ap[pcs].sum() + ag[j] - s)
            fnc["fragmented - joining pieces would fix" if jiou > 0.5 else "fragmented - joining not enough"] += 1
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


def part2(preds, base):
    fnc, fpc = Counter(), Counter()
    for i, pr in enumerate(preds):
        for g in GT[i]:
            classify(pr, g, fnc, fpc)
    S = base.sum(0)
    for title, c, tot in (("missed filaments (FN)", fnc, S[2]), ("false blobs (FP)", fpc, S[1])):
        print(f"  {title}: {int(tot)}   (classified {sum(c.values())})")
        for k, v in c.most_common():
            print(f"    {k:42s} {v:5d}  {v / max(tot, 1):6.1%}")


# ======================= Parts 2 and 3 =======================
ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})
print(f"ensemble loaded ({el()})", flush=True)
base, table = None, []
for r in R_GRID:
    comps = extract_grouped(ens, r)
    for mm in MEAN_GRID:
        cnts, preds, macro = score(comps, mm)
        if base is None:
            base = cnts
            print(f"\ncheck: current method pooled PQ {parts(base)[0]:.4f}  (should be about 0.4247)")
            print("\n=== Part 2: error types of the current method ===")
            part2(preds, base)
            print(f"  ({el()})\n\n=== Part 3: join blobs within r px (r=0 = current) ===", flush=True)
        p, sq, rq, s = parts(cnts)
        pb = parts(base)[0]
        lo, hi = boot(cnts, base)
        nf = sum(parts(cnts[fold == f])[0] > parts(base[fold == f])[0] for f in range(5))
        table.append((r, mm, p, p - pb, lo, hi, nf))
        print(f"  r={r:2d} mean>={mm:.2f}  PQ {p:.4f} = SQ {sq:.3f} x RQ {rq:.3f}  TP {int(s[0])} FP {int(s[1])} FN {int(s[2])}"
              f"  diff {p - pb:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})  folds better {nf}/5  ({el()})", flush=True)
        K.log_result(f"A16_join_r{r}_m{int(mm * 100)}",
                     f"ensemble 3 groups x5, thr 0.35 area 400 mean {mm}, join blobs within r={r}px",
                     p, macro, [int(s[0]), int(s[1]), int(s[2]), float(s[3])])
    del comps

best = max(table, key=lambda t: t[2])
adopt = best[3] >= 0.003 and best[4] > 0 and best[6] >= 4
print(f"\nbest: r={best[0]} mean>={best[1]:.2f}  PQ {best[2]:.4f}  diff {best[3]:+.4f} "
      f"(CI {best[4]:+.4f}..{best[5]:+.4f}, folds {best[6]}/5) -> {'ADOPT joining' if adopt else 'keep current method'}")
print(f"total time {el()}")