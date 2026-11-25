"""A21 - is the simple 10-model ensemble as good as the submitted 20-model one? (CPU only, about 6 min)
All with the robust rule 0.40 / 300 / 0.80, on all 707 photos (paired bootstrap):
  C   new ResNet 0.5 + old EffNet s42 0.25 + old EffNet s43 0.25   (local version of the submitted final, 0.4359)
  S2  new ResNet 0.5 + new EffNet 0.5                             (10 models: the simple version)
  S1  new ResNet alone                                            (5 models)
Run:  python C:\\solar\\src\\score_simple.py"""
import time
import numpy as np
import common as K
import config as C

t0 = time.time()
il, GT = K.get_data()
fold = il["fold"].values


def path(name, f):
    if name.startswith("k_"):
        return C.RUNS / "exps" / f"{name}_f{f}" / f"oof_fold{f}.npy"
    return C.RUNS / "oof_maps" / name / f"oof_fold{f}.npy"


def load(weights):
    out = np.zeros((len(il), 1024, 1024), np.uint8)
    for f in range(5):
        pos = np.flatnonzero(fold == f)
        out[pos] = sum(np.load(path(n, f)).astype(np.float32) * w for n, w in weights.items()).round().astype(np.uint8)
    return out


def counts(maps):
    c = np.zeros((len(maps), 4))
    for i, cs in enumerate(K.extract(maps, 0.40)):
        pr = [x[3] for x in cs if x[0] >= 300 and x[1] >= 0.80]
        for g in GT[i]:
            c[i] += K.cnt(pr, g)
    return c


pq = lambda s: s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2])
R = {}
for key, w in (("C", {"k_bs18_it8000": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25}),
               ("S2", {"k_bs18_it8000": 0.5, "k_effb18": 0.5}),
               ("S1", {"k_bs18_it8000": 1.0})):
    R[key] = counts(load(w))
    s = R[key].sum(0)
    print(f"  {key:3s} PQ {pq(s):.4f}  TP {int(s[0])} FP {int(s[1])} FN {int(s[2])}  ({time.time() - t0:.0f}s)", flush=True)
IDX = np.random.default_rng(0).integers(0, len(il), (2000, len(il)))
for k in ("S2", "S1"):
    d = pq(R[k][IDX].sum(1)) - pq(R["C"][IDX].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {k} vs C: {pq(R[k].sum(0)) - pq(R['C'].sum(0)):+.4f}  (95% CI {lo:+.4f}..{hi:+.4f})")
g = pq(R["S2"].sum(0)) - pq(R["C"].sum(0))
print("->", "SWITCH to the 10-model version (same within noise)" if g > -0.003 else "KEEP the 20-model version", f"({g:+.4f})")