"""Phase A2 (no GPU):
 A9  does adding a laptop-trained model to the 15-model ensemble help on fold 0?
 A1b minimum blob area below 300 (cut-offs 0.35 / 0.40)
 A8  do test-set predictions and photos look different from the training ones? (public-vs-local gap)"""
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

# ------------------------------------------------------------------ A9
print("=== A9: fold 0, 15-model ensemble + one laptop-trained model (weight 0.25) ===", flush=True)
pos0 = np.flatnonzero(fold == 0)
gt0 = [GT[i] for i in pos0]
ens0 = ens_all[pos0]
base0 = per_image(K.extract(ens0, 0.35), gt0, 400, 0.8)
print(f"15-model alone, fold 0: PQ {pq(base0):.4f}", flush=True)
names = [n for n in ("lap_control", "crop384", "r18", "effb0", "crop256", "crop768", "r101")
         if (C.RUNS / "exps" / n / "oof_fold0.npy").exists()]
new = {n: np.load(C.RUNS / "exps" / n / "oof_fold0.npy") for n in names}
rows = []
for n in names:
    arr = (0.75 * ens0.astype(np.float32) + 0.25 * new[n]).round().astype(np.uint8)
    a = per_image(K.extract(arr, 0.35), gt0, 400, 0.8)
    lo, hi = ci(a, base0)
    rows.append((f"+ {n}", pq(a), pq(a) - pq(base0), lo, hi))
    print(f"  + {n:12s} PQ {pq(a):.4f}  delta {pq(a) - pq(base0):+.4f}  95% [{lo:+.4f}, {hi:+.4f}]", flush=True)
mix = np.mean([new[n].astype(np.float32) for n in names], axis=0)
arr = (0.5 * ens0.astype(np.float32) + 0.5 * mix).round().astype(np.uint8)
a = per_image(K.extract(arr, 0.35), gt0, 400, 0.8)
lo, hi = ci(a, base0)
print(f"  + ALL {len(names)} (half weight) PQ {pq(a):.4f}  delta {pq(a) - pq(base0):+.4f}  95% [{lo:+.4f}, {hi:+.4f}]", flush=True)
pd.DataFrame(rows, columns=["blend", "pq", "delta", "ci_lo", "ci_hi"]).to_csv(C.RESULTS / "A9_blend_fold0.csv", index=False)
del new, mix, arr

# ------------------------------------------------------------------ A1b
print(f"\n=== A1b: smaller minimum areas, all 707 photos ({time.time() - t0:.0f}s) ===", flush=True)
AREAS, MEANS = (100, 150, 200, 250, 300, 400), (0.75, 0.80, 0.85)
arrs = {}
for thr in (0.35, 0.40):
    comps = K.extract(ens_all, thr, min_area=100)
    for area, mean in itertools.product(AREAS, MEANS):
        arrs[(thr, area, mean)] = per_image(comps, GT, area, mean)
    del comps
    print(f"  thr {thr} done ({time.time() - t0:.0f}s)", flush=True)
base = arrs[(0.35, 400, 0.80)]
rows = []
for (thr, area, mean), a in arrs.items():
    lo, hi = ci(a, base)
    fq = sum(pq(a[fold == f]) > pq(base[fold == f]) for f in range(5))
    rows.append(dict(thr=thr, min_area=area, min_mean=mean, pooled=pq(a), delta=pq(a) - pq(base), ci_lo=lo, ci_hi=hi, folds_better=int(fq)))
res = pd.DataFrame(rows).sort_values("pooled", ascending=False)
res.to_csv(C.RESULTS / "A1b_grid.csv", index=False)
for thr in (0.35, 0.40):
    print(f"\npooled PQ, thr {thr} (rows = min area, columns = min mean probability)")
    print(res[res.thr == thr].pivot(index="min_area", columns="min_mean", values="pooled").round(4).to_string())
print("\nTOP 8 vs baseline (thr 0.35 / area 400 / mean 0.80):")
print(res.head(8).round(4).to_string(index=False))

# ------------------------------------------------------------------ A8
print(f"\n=== A8: test predictions vs out-of-fold predictions ({time.time() - t0:.0f}s) ===", flush=True)
tp = {n: np.load(C.RUNS / "test_probs" / f"test_probs_{n}.npy", mmap_mode="r") for n in ("res", "eff", "eff2")}
n_test = tp["res"].shape[0]
print("test maps:", {k: v.shape for k, v in tp.items()})
ens_t = np.zeros((n_test, 1024, 1024), np.uint8)
for s in range(0, n_test, 30):
    sl = slice(s, s + 30)
    ens_t[sl] = (0.5 * tp["res"][sl].astype(np.float32) + 0.25 * tp["eff"][sl] + 0.25 * tp["eff2"][sl]).round().astype(np.uint8)


def blob_stats(comps):
    big = [[c for c in cs if c[0] >= 400] for cs in comps]
    kept = [[c for c in cs if c[1] >= 0.8] for cs in big]
    ka = np.array([c[0] for cs in kept for c in cs])
    bm = np.array([c[1] for cs in big for c in cs])
    return {"photos": len(comps),
            "blobs >=100px per photo": np.mean([len(cs) for cs in comps]),
            "blobs >=400px per photo": np.mean([len(b) for b in big]),
            "kept per photo (rule)": np.mean([len(k) for k in kept]),
            "share of >=400px blobs kept": sum(map(len, kept)) / max(1, sum(map(len, big))),
            "median kept area (px)": float(np.median(ka)),
            "blob mean-prob p10 / p50 / p90": " / ".join(f"{x:.2f}" for x in np.percentile(bm, [10, 50, 90]))}


s_oof = blob_stats(K.extract(ens_all, 0.35))
s_test = blob_stats(K.extract(ens_t, 0.35))
print(pd.DataFrame({"OOF (1 model per group)": s_oof, "TEST (5-model groups)": s_test}).to_string())

print("\nphoto brightness (median of pixels > 25) by instrument, train vs test:")
def med(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(np.median(g[g > 25])) if (g > 25).any() else float("nan")
rows = []
for p in sorted(C.TRAIN_IMG.iterdir()):
    rows.append(("train", p.name[14:16], med(p)))
for p in sorted(C.TEST_IMG.iterdir()):
    rows.append(("test", p.name[14:16], med(p)))
b = pd.DataFrame(rows, columns=["set", "instrument", "median_brightness"])
print(b.pivot_table(index="instrument", columns="set", values="median_brightness", aggfunc=["mean", "count"]).round(1).to_string())
print(f"\nall done in {(time.time() - t0) / 60:.1f} min. Files: A9_blend_fold0.csv, A1b_grid.csv in C:/solar/results")