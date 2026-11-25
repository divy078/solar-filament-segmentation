"""A17 ensemble swap test: does a new ResNet-34 run still help INSIDE the 3-group ensemble?
Ensemble = res1024 0.5 + effv2s 0.25 + effv2s_s2 0.25 (current), rule thr 0.35 / area 400 / mean 0.8.
The new run replaces res1024 (same weight). Scored on one fold, paired bootstrap over photos.
Usage:  set FOLD=0&& python C:\\solar\\src\\swap_test.py k_bs18_it8000_f0      (CPU, about 3-5 min)"""
import os, sys
import numpy as np
import common as K
import config as C

fold = int(os.environ.get("FOLD", "0"))
new = sys.argv[1]
il, GT = K.get_data()
va = np.flatnonzero(il.fold.values == fold)
gt = [GT[i] for i in va]
W = {"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25}


def ens(swap):
    acc = None
    for name, w in W.items():
        p = (C.RUNS / "exps" / swap / f"oof_fold{fold}.npy") if (name == "res1024" and swap) else \
            (C.RUNS / "oof_maps" / name / f"oof_fold{fold}.npy")
        a = np.load(p).astype(np.float32) * w
        assert len(a) == len(va), (p, a.shape, len(va))
        acc = a if acc is None else acc + a
    return acc.round().astype(np.uint8)


def counts(maps):
    out = np.zeros((len(maps), 4))
    for i, cs in enumerate(K.extract(maps, 0.35)):
        pr = [c[3] for c in cs if c[0] >= 400 and c[1] >= 0.8]
        for g in gt[i]:
            out[i] += K.cnt(pr, g)
    return out


pq = lambda s: s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2])
single = counts(np.load(C.RUNS / "exps" / new / f"oof_fold{fold}.npy"))
b = counts(ens(None))
n = counts(ens(new))
idx = np.random.default_rng(0).integers(0, len(va), (2000, len(va)))
d = pq(n[idx].sum(1)) - pq(b[idx].sum(1))
lo, hi = np.percentile(d, [2.5, 97.5])
gain = pq(n.sum(0)) - pq(b.sum(0))
print(f"fold {fold}, {len(va)} photos")
print(f"  {new} alone:                     {pq(single.sum(0)):.4f}")
print(f"  current ensemble (res1024 group):  {pq(b.sum(0)):.4f}")
print(f"  ensemble with {new} instead: {pq(n.sum(0)):.4f}   gain {gain:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})")
print("  ->", "WORTH IT: train the other folds" if gain >= 0.003 else "NOT worth 4.5x training cost")