"""Phase A1: threshold x min-area x min-mean-probability grid on the 15-model out-of-fold maps (no training)."""
import time, itertools
import numpy as np, pandas as pd
import common as K
import config as C

t0 = time.time()
il, GT = K.get_data()
fold = il["fold"].values
ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})

THRS = (0.30, 0.35, 0.40)
AREAS = (300, 400, 500, 700)
MEANS = (0.70, 0.75, 0.80, 0.85)
BASE = (0.35, 400, 0.80)


def per_image(comps, area, mean):
    """Per photo: [TP, FP, FN, IoU sum] summed over that photo's annotator records."""
    out = np.zeros((len(comps), 4))
    for i, cs in enumerate(comps):
        pred = [c[3] for c in cs if c[0] >= area and c[1] >= mean]
        for g in GT[i]:
            out[i] += K.cnt(pred, g)
    return out


def pq(a):
    d = a[:, 0].sum() + .5 * a[:, 1].sum() + .5 * a[:, 2].sum()
    return a[:, 3].sum() / d


arrs = {}
for thr in THRS:
    comps = K.extract(ens, thr)
    for area, mean in itertools.product(AREAS, MEANS):
        arrs[(thr, area, mean)] = per_image(comps, area, mean)
    print(f"thr {thr} done ({time.time() - t0:.0f}s)", flush=True)
    del comps

base = arrs[BASE]
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(il), len(il)) for _ in range(500)]
base_fold = [pq(base[fold == f]) for f in range(5)]
rows = []
for (thr, area, mean), a in arrs.items():
    fq = [pq(a[fold == f]) for f in range(5)]
    deltas = np.array([pq(a[b]) - pq(base[b]) for b in boots])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    rows.append(dict(thr=thr, min_area=area, min_mean=mean, pooled=pq(a), delta=pq(a) - pq(base),
                     ci_lo=lo, ci_hi=hi, folds_better=int(sum(x > y for x, y in zip(fq, base_fold))),
                     tp=int(a[:, 0].sum()), fp=int(a[:, 1].sum()), fn=int(a[:, 2].sum())))
res = pd.DataFrame(rows).sort_values("pooled", ascending=False)
C.RESULTS.mkdir(parents=True, exist_ok=True)
res.to_csv(C.RESULTS / "A1_grid.csv", index=False)

pd.set_option("display.width", 200)
print(f"\nBASELINE (thr 0.35, area 400, mean 0.80): pooled PQ {pq(base):.4f}")
print("\nTHRESHOLD alone (area 400, mean 0.80):")
print(res[(res.min_area == 400) & (res.min_mean == 0.80)].sort_values("thr").round(4).to_string(index=False))
for thr in THRS:
    print(f"\npooled PQ, thr {thr}  (rows = min area, columns = min mean probability)")
    print(res[res.thr == thr].pivot(index="min_area", columns="min_mean", values="pooled").round(4).to_string())
print("\nTOP 12 (delta and its 95% interval are vs the baseline; folds_better = out of 5):")
print(res.head(12).round(4).to_string(index=False))

best = res.iloc[0]
K.log_result("A1-base", "15-model OOF thr0.35 area400 mean0.8 (grid baseline)", pq(base), float("nan"),
             [int(base[:, 0].sum()), int(base[:, 1].sum()), int(base[:, 2].sum())])
K.log_result("A1-best", f"grid best thr{best.thr} area{int(best.min_area)} mean{best.min_mean}", best.pooled, float("nan"),
             [int(best.tp), int(best.fp), int(best.fn)], notes=f"delta {best.delta:+.4f} CI [{best.ci_lo:+.4f},{best.ci_hi:+.4f}] folds_better {int(best.folds_better)}/5")
print(f"\nfinished in {(time.time() - t0) / 60:.1f} min. Full table: C:/solar/results/A1_grid.csv")