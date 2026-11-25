"""Recounts the filament-size analysis (Figure 5) with one-to-one matching, exactly as PQ matches filaments
(CPU only, about 6 minutes). The first count matched each annotated filament independently, so one prediction
overlapping two annotated filaments that overlap each other counted twice (5,030 found against 5,029 true
positives). This version gives every prediction and every annotated filament at most one partner.
It updates the "size" block of C:\\solar\\results\\paper_figures2\\numbers.json (a copy of the old file is kept as
numbers_before_size_fix.json). Afterwards run paper_figs3.py again to redraw Figure 5.
Run:  python C:\\solar\\src\\size_fix.py"""
import json, shutil, time
import numpy as np
import pycocotools.mask as mu
from scipy.optimize import linear_sum_assignment
import common as K
import config as C

t0 = time.time()
el = lambda: f"{(time.time() - t0) / 60:.1f} min"
il, GT = K.get_data()
fold = il["fold"].values
N = len(il)
F = C.RESULTS / "paper_figures2" / "numbers.json"
NUM = json.loads(F.read_text())


def load_group(name):
    out = np.zeros((N, 1024, 1024), np.uint8)
    for f in range(5):
        pos = np.flatnonzero(fold == f)
        a = np.load(C.RUNS / "exps" / f"{name}_f{f}" / f"oof_fold{f}.npy")
        assert a.shape[0] == len(pos)
        out[pos] = np.asarray(a, np.float32).round().clip(0, 255).astype(np.uint8)
        del a
    return out


a, b = load_group("k_bs18_it8000"), load_group("k_effb18")
final = np.empty_like(a)
for s in range(0, N, 50):
    final[s:s + 50] = (0.5 * a[s:s + 50].astype(np.float32) + 0.5 * b[s:s + 50].astype(np.float32)).round().astype(np.uint8)
del a, b
pred = [[c[3] for c in cs if c[0] >= 300 and c[1] >= 0.80] for cs in K.extract(final, 0.40)]
del final
tp = sum(K.cnt(pred[i], g)[0] for i in range(N) for g in GT[i])
print(f"final ensemble extracted, TP {tp} (should be 5029)  ({el()})", flush=True)

BINS = NUM["size"]["bins"]
EDGES = [(0, 400), (400, 1000), (1000, 3000), (3000, 10 ** 9)]


def found(g, pr):
    """One-to-one matching with IoU > 0.5 (as in PQ): which filaments of g get a partner in pr."""
    out = np.zeros(len(g), bool)
    if not g or not pr:
        return out
    iou = np.asarray(mu.iou(pr, g, [0] * len(g)))
    r, c = linear_sum_assignment(-(iou * (iou > 0.5)))
    out[c[iou[r, c] > 0.5]] = True
    return out


def binned(pairs):
    n, k = np.zeros(len(EDGES)), np.zeros(len(EDGES))
    for g, pr in pairs:
        if not g:
            continue
        ar, f = np.asarray(mu.area(g), float), found(g, pr)
        for j, (lo, hi) in enumerate(EDGES):
            m = (ar >= lo) & (ar < hi)
            n[j] += m.sum()
            k[j] += f[m].sum()
    return n, k


multi = [i for i in range(N) if len(GT[i]) >= 2]
n_all, k_all = binned((g, pred[i]) for i in range(N) for g in GT[i])
n_mul, k_mul = binned((g, pred[i]) for i in multi for g in GT[i])
n_hum, k_hum = binned((GT[i][x], GT[i][y]) for i in multi for x in range(len(GT[i])) for y in range(len(GT[i])) if x != y)
print(f"check: found {int(k_all.sum())} (must equal TP {tp}), missed {int((n_all - k_all).sum())} (must be 3170)")
print(f"  {'area (px)':12s} {'model, 707 images':>22s} {'model, 296 images':>22s} {'other annotator, 296':>24s}")
for j, name in enumerate(BINS):
    print(f"  {name:12s} {k_all[j] / n_all[j]:7.1%} of {int(n_all[j]):5d}     {k_mul[j] / n_mul[j]:7.1%} of {int(n_mul[j]):5d}"
          f"     {k_hum[j] / n_hum[j]:7.1%} of {int(n_hum[j]):5d}")
miss = n_all - k_all
print("  share of all misses: " + ", ".join(f"{BINS[j]} {miss[j] / miss.sum():.1%}" for j in range(len(BINS))))
shutil.copy(F, F.with_name("numbers_before_size_fix.json"))
NUM["size"].update({"model_all_n": n_all.tolist(), "model_all_found": k_all.tolist(),
                    "model_multi_n": n_mul.tolist(), "model_multi_found": k_mul.tolist(),
                    "annot_n": n_hum.tolist(), "annot_found": k_hum.tolist(), "matching": "one-to-one"})
F.write_text(json.dumps(NUM, indent=1))
print(f"numbers.json updated  ({el()})")