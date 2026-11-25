"""A11: REPLACE the ResNet part of the ensemble with a better-recipe laptop model (fold 0) instead of adding to it."""
import time, itertools
import numpy as np, pandas as pd, cv2
import common as K
import config as C

t0 = time.time()
il, GT = K.get_data()
fold = il["fold"].values
W15 = {"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25}
pd.set_option("display.width", 200)


def per_image(comps, gt, area, mean):
    out = np.zeros((len(comps), 4))
    for i, cs in enumerate(comps):
        pred = [c[3] for c in cs if c[0] >= area and c[1] >= mean]
        for g in gt[i]:
            out[i] += K.cnt(pred, g)
    return out


def pq(a):
    return a[:, 3].sum() / (a[:, 0].sum() + .5 * a[:, 1].sum() + .5 * a[:, 2].sum())


def ci(a, b, n=500):
    rng = np.random.default_rng(0)
    d = [pq(a[x]) - pq(b[x]) for x in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return np.percentile(d, [2.5, 97.5])


ens_all = K.load_ensemble(W15)



print("=== A11: fold 0, swap the ResNet-34 half of the ensemble ===", flush=True)
pos0 = np.flatnonzero(fold == 0)
gt0 = [GT[i] for i in pos0]
res0 = K.load_ensemble({"res1024": 1.0})[pos0].astype(np.float32)
eff0 = K.load_ensemble({"effv2s": 0.5, "effv2s_s2": 0.5})[pos0].astype(np.float32)
ld = lambda n: np.load(C.RUNS / "exps" / n / "oof_fold0.npy").astype(np.float32)
base = per_image(K.extract((0.5 * res0 + 0.5 * eff0).round().astype(np.uint8), 0.35), gt0, 400, 0.8)
print(f"baseline (0.5 res + 0.5 eff): PQ {pq(base):.4f}", flush=True)
rows = []


def test(label, arr):
    a = per_image(K.extract(arr.round().astype(np.uint8), 0.35), gt0, 400, 0.8)
    lo, hi = ci(a, base)
    print(f"  {label:44s} PQ {pq(a):.4f}  delta {pq(a) - pq(base):+.4f}  95% [{lo:+.4f}, {hi:+.4f}]", flush=True)
    rows.append((label, pq(a), pq(a) - pq(base), lo, hi))


for n in ("bs18_c512", "c768_s43", "pos50", "unetpp"):
    if not (C.RUNS / "exps" / n / "oof_fold0.npy").exists():
        continue
    m = ld(n)
    test(f"{n} alone", m)
    test(f"0.5 {n} + 0.5 eff", 0.5 * m + 0.5 * eff0)
    test(f"0.25 res + 0.25 {n} + 0.5 eff", 0.25 * res0 + 0.25 * m + 0.5 * eff0)
bg = [n for n in ("bs18_c512", "c768_s43", "c1024_b4") if (C.RUNS / "exps" / n / "oof_fold0.npy").exists()]
if len(bg) > 1:
    g = np.mean([ld(n) for n in bg], axis=0)
    test(f"mean of {len(bg)} big-recipe models alone", g)
    test("0.5 big-recipe mean + 0.5 eff", 0.5 * g + 0.5 * eff0)
pd.DataFrame(rows, columns=["blend", "pq", "delta", "ci_lo", "ci_hi"]).to_csv(C.RESULTS / "A11_swap_fold0.csv", index=False)
print("saved A11_swap_fold0.csv", flush=True)