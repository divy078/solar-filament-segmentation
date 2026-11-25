"""A10: does adding a different-architecture / big-batch fold-0 model to the 15-model ensemble help?"""
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


print("=== A10: fold 0, 15-model ensemble + one laptop model (weight 0.25) ===", flush=True)
pos0 = np.flatnonzero(fold == 0)
gt0 = [GT[i] for i in pos0]
ens0 = ens_all[pos0]
base0 = per_image(K.extract(ens0, 0.35), gt0, 400, 0.8)
print(f"15-model alone, fold 0: PQ {pq(base0):.4f}", flush=True)
names = [n for n in ("bs18_c512", "c768_s43", "c1024_b4", "unetpp", "linknet", "upernet_cnx", "fpn", "segformer", "small50", "pos50")
         if (C.RUNS / "exps" / n / "oof_fold0.npy").exists()]
new = {n: np.load(C.RUNS / "exps" / n / "oof_fold0.npy") for n in names}


def test(label, arr):
    a = per_image(K.extract(arr, 0.35), gt0, 400, 0.8)
    lo, hi = ci(a, base0)
    print(f"  {label:34s} PQ {pq(a):.4f}  delta {pq(a) - pq(base0):+.4f}  95% [{lo:+.4f}, {hi:+.4f}]", flush=True)
    return (label, pq(a), pq(a) - pq(base0), lo, hi)


def mix(w_ens, group):
    m = np.mean([new[n].astype(np.float32) for n in group], axis=0)
    return (w_ens * ens0.astype(np.float32) + (1 - w_ens) * m).round().astype(np.uint8)


rows = []
for n in names:
    rows.append(test(f"+ {n} (0.25)", mix(0.75, [n])))
big = [n for n in ("bs18_c512", "c768_s43", "c1024_b4") if n in new]
if len(big) > 1:
    rows.append(test("+ big-batch/crop group (0.25)", mix(0.75, big)))
    rows.append(test("+ big-batch/crop group (0.50)", mix(0.50, big)))
alt = [n for n in ("unetpp", "linknet", "upernet_cnx") if n in new]
if len(alt) > 1:
    rows.append(test("+ other-architecture group (0.25)", mix(0.75, alt)))
    rows.append(test("+ other-architecture group (0.50)", mix(0.50, alt)))
allg = big + alt
if len(allg) > 2:
    rows.append(test("+ all of the above (0.50)", mix(0.50, allg)))
pd.DataFrame(rows, columns=["blend", "pq", "delta", "ci_lo", "ci_hi"]).to_csv(C.RESULTS / "A10_blend_fold0.csv", index=False)
print("\nsaved A10_blend_fold0.csv in C:/solar/results", flush=True)