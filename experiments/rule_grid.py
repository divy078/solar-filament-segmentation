"""A20 - re-tune the keep rule for the retrained models (CPU only, about 20 min).
The rule (cut-off / min area / min mean probability) was tuned on the OLD models. The retrained models drop many
false blobs but also lose some true ones, so a looser rule may now fit better. Ensemble D of score_eff.py:
new ResNet 0.5 + new EffNet 0.25 + old EffNet seed 43 0.25. Every setting is compared with the robust rule
(0.40 / 300 / 0.80) using a paired bootstrap over 707 photos and a count of folds that improve.
Run:  python C:\\solar\\src\\rule_grid.py"""
import time
import numpy as np
import common as K
import config as C

t0 = time.time()
el = lambda: f"{(time.time() - t0) / 60:.1f} min"
il, GT = K.get_data()
fold = il["fold"].values
W = {"k_bs18_it8000": 0.5, "k_effb18": 0.25, "effv2s_s2": 0.25}


def path(name, f):
    if name.startswith("k_"):
        return C.RUNS / "exps" / f"{name}_f{f}" / f"oof_fold{f}.npy"
    return C.RUNS / "oof_maps" / name / f"oof_fold{f}.npy"


ens = np.zeros((len(il), 1024, 1024), np.uint8)
for f in range(5):
    pos = np.flatnonzero(fold == f)
    ens[pos] = sum(np.load(path(n, f)).astype(np.float32) * w for n, w in W.items()).round().astype(np.uint8)
print(f"ensemble D loaded ({el()})", flush=True)

THRS, AREAS, MEANS = (0.30, 0.35, 0.40), (300, 400), (0.70, 0.75, 0.80)
pq = lambda s: s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2])
R = {}
for thr in THRS:
    comps = K.extract(ens, thr)
    for area in AREAS:
        for mean in MEANS:
            c = np.zeros((len(comps), 4))
            for i, cs in enumerate(comps):
                pr = [x[3] for x in cs if x[0] >= area and x[1] >= mean]
                for g in GT[i]:
                    c[i] += K.cnt(pr, g)
            R[(thr, area, mean)] = c
    print(f"  cut-off {thr:.2f} done ({el()})", flush=True)
    del comps

ref = R[(0.40, 300, 0.80)]
IDX = np.random.default_rng(0).integers(0, len(il), (2000, len(il)))
rows = []
for k, c in R.items():
    d = pq(c[IDX].sum(1)) - pq(ref[IDX].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    nf = sum(pq(c[fold == f].sum(0)) > pq(ref[fold == f].sum(0)) for f in range(5))
    s = c.sum(0)
    rows.append((pq(s), k, pq(s) - pq(ref.sum(0)), lo, hi, nf, s))
print("\ncut-off / area / mean     PQ       vs robust rule   95% interval        folds better   TP / FP / FN")
for p, k, g, lo, hi, nf, s in sorted(rows, key=lambda r: -r[0]):
    mark = "  <- robust rule (current)" if k == (0.40, 300, 0.80) else ""
    print(f"  {k[0]:.2f} / {k[1]:3d} / {k[2]:.2f}     {p:.4f}   {g:+.4f}          {lo:+.4f}..{hi:+.4f}     {nf}/5"
          f"          {int(s[0])} / {int(s[1])} / {int(s[2])}{mark}")
    K.log_result(f"A20_rule_{k[0]}_{k[1]}_{k[2]}", "ensemble D (new ResNet + new EffNet + EffNet s43), rule grid", float(p), 0.0,
                 [int(s[0]), int(s[1]), int(s[2]), float(s[3])])
best = max(rows, key=lambda r: r[0])
ok = best[1] != (0.40, 300, 0.80) and best[3] > 0 and best[5] >= 4
print(f"\n-> {'ADOPT ' + str(best[1]) if ok else 'keep the robust rule'}: best {best[1]} PQ {best[0]:.4f} "
      f"({best[2]:+.4f}, CI {best[3]:+.4f}..{best[4]:+.4f}, folds {best[5]}/5)")
print(f"total time {el()}")