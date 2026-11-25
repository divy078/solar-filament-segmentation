"""A19 - full 707-photo verdict for the new EfficientNetV2-S group (batch 18, 8000 steps, Kaggle, 5 folds).
Compares, on all 707 photos with a paired bootstrap:
  A  old EfficientNetV2-S alone (effv2s, seed 42)            vs  B  new EfficientNetV2-S alone (k_effb18)
  C  current best: new ResNet 0.5 + effv2s 0.25 + effv2s_s2 0.25   (= D of score707, 0.4335)
  D  new ResNet 0.5 + new EffNet 0.25 + effv2s_s2 0.25      (new EffNet replaces the seed-42 group)
  E  new ResNet 0.5 + new EffNet 0.5                         (new EffNet replaces both old EffNet groups)
  F  best of C/D/E with the robust rule 0.40 / 300 / 0.80
CPU only, about 20 min.   Run:  python C:\\solar\\src\\score_eff.py"""
import time
import numpy as np, cv2
from scipy import ndimage as ndi
import pycocotools.mask as mu
import common as K
import config as C

H = W = 2048
t0 = time.time()
el = lambda: f"{(time.time() - t0) / 60:.1f} min"
il, GT = K.get_data()
fold = il["fold"].values
RES_NEW, EFF_NEW = "k_bs18_it8000", "k_effb18"


def path(name, f):
    if name in (RES_NEW, EFF_NEW):
        return C.RUNS / "exps" / f"{name}_f{f}" / f"oof_fold{f}.npy"
    return C.RUNS / "oof_maps" / name / f"oof_fold{f}.npy"


missing = [str(path(n, f)) for n in (RES_NEW, EFF_NEW, "effv2s", "effv2s_s2") for f in range(5) if not path(n, f).exists()]
assert not missing, "missing files:\n  " + "\n  ".join(missing)


def load(weights):
    out = np.zeros((len(il), 1024, 1024), np.uint8)
    tot = float(sum(weights.values()))
    for f in range(5):
        pos = np.flatnonzero(fold == f)
        acc = None
        for n, w in weights.items():
            a = np.load(path(n, f)).astype(np.float32) * (w / tot)
            assert len(a) == len(pos), f"{n} fold {f}: {len(a)} maps but {len(pos)} photos"
            acc = a if acc is None else acc + a
        out[pos] = acc.round().astype(np.uint8)
    return out


def extract_joined(oof, r, thr=0.35, min_area=100, max_comps=60):
    """Like K.extract, but pixels above thr within r px of each other form one blob (same as diag_errors.py)."""
    out = []
    for i in range(len(oof)):
        up = cv2.resize(oof[i], (W, H), interpolation=cv2.INTER_LINEAR)
        fg = (up > int(thr * 255)).astype(np.uint8)
        near = (cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5) <= r).astype(np.uint8)
        n, lab = cv2.connectedComponents(near, connectivity=8)
        lab[fg == 0] = 0
        comps = []
        if n > 1:
            areas = np.bincount(lab.ravel(), minlength=n)[1:]
            keep = [k for k in np.argsort(-areas)[:max_comps] if areas[k] >= min_area]
            if keep:
                ids = np.array(keep) + 1
                mean, mx, sls = ndi.mean(up, lab, ids), ndi.maximum(up, lab, ids), ndi.find_objects(lab)
                for k, mn, pk in zip(ids, mean, mx):
                    sl = sls[k - 1]
                    m = np.zeros((H, W), np.uint8, order="F")
                    m[sl] = (lab[sl] == k)
                    comps.append((int(areas[k - 1]), float(mn) / 255, float(pk) / 255, mu.encode(m)))
        out.append(comps)
    return out


def counts(comps, area=400, mean=0.80):
    c = np.zeros((len(comps), 4))
    for i, cs in enumerate(comps):
        pr = [x[3] for x in cs if x[0] >= area and x[1] >= mean]
        for g in GT[i]:
            c[i] += K.cnt(pr, g)
    return c


pq = lambda s: s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2])
IDX = np.random.default_rng(0).integers(0, len(il), (2000, len(il)))
R = {}


def run(key, label, comps, area=400, mean=0.80):
    R[key] = counts(comps, area, mean)
    s = R[key].sum(0)
    print(f"  {key}  {label:52s} PQ {pq(s):.4f}  TP {int(s[0])} FP {int(s[1])} FN {int(s[2])}  ({el()})", flush=True)
    K.log_result(f"A18_{key}", label, float(pq(s)), 0.0, [int(s[0]), int(s[1]), int(s[2]), float(s[3])])


def compare(a, b, title):
    d = pq(R[b][IDX].sum(1)) - pq(R[a][IDX].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    gain = pq(R[b].sum(0)) - pq(R[a].sum(0))
    per = [pq(R[b][fold == f].sum(0)) - pq(R[a][fold == f].sum(0)) for f in range(5)]
    print(f"  {title:44s} {gain:+.4f}  (95% CI {lo:+.4f}..{hi:+.4f})  folds better {sum(p > 0 for p in per)}/5"
          f"  per fold: {' '.join(f'{p:+.4f}' for p in per)}")


print("=== single EfficientNetV2-S group ===", flush=True)
run("A", "old EfficientNetV2-S (batch 8, 4000 steps)", K.extract(load({"effv2s": 1}), 0.35))
run("B", "new EfficientNetV2-S (batch 18, 8000 steps)", K.extract(load({EFF_NEW: 1}), 0.35))

print("\n=== 15-model ensembles (all with the new ResNet group) ===", flush=True)
W = {"C": {RES_NEW: 0.5, "effv2s": 0.25, "effv2s_s2": 0.25},
     "D": {RES_NEW: 0.5, EFF_NEW: 0.25, "effv2s_s2": 0.25},
     "E": {RES_NEW: 0.5, EFF_NEW: 0.5}}
lab = {"C": "current best (both old EffNet groups)", "D": "new EffNet replaces seed-42 group", "E": "new EffNet replaces both old groups"}
for k in "CDE":
    run(k, lab[k], K.extract(load(W[k]), 0.35))
best = max("CDE", key=lambda k: pq(R[k].sum(0)))
run("F", f"{best} with robust rule 0.40 / 300 / 0.80", K.extract(load(W[best]), 0.40), area=300)

print("\n=== verdicts (paired bootstrap over 707 photos) ===")
compare("A", "B", "new vs old EffNet, alone")
compare("C", "D", "D vs current best  <- DECISION 1")
compare("C", "E", "E vs current best  <- DECISION 2")
compare(best, "F", f"robust rule vs standard ({best})")
gain = pq(R[best].sum(0)) - pq(R["C"].sum(0))
print(f"\n-> best ensemble: {best} ({lab[best]})", "- SUBMIT it" if best != "C" and gain >= 0.003 else "- keep the current ensemble",
      f"(gain {gain:+.4f} over current best)")
print(f"total time {el()}")