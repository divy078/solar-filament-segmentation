"""Phase A3 (no GPU): what do the two probe rules score LOCALLY, and how do 'middle-ground' rules compare?
Compare the local change with the public change (public: standard 0.37, strict 0.28, loose 0.35)."""
import time
import numpy as np, pandas as pd
import common as K
import config as C

t0 = time.time()
il, GT = K.get_data()
ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})

RULES = [  # (name, cut-off, min area, min mean probability)
    ("standard (submitted so far)", .35, 400, .80),
    ("robust variant", .40, 300, .80),
    ("STRICT probe  [public 0.28]", .40, 1000, .88),
    ("LOOSE probe   [public 0.35]", .35, 150, .60),
    ("mid: cut-off .30, mean .80", .30, 300, .80),
    ("mid: cut-off .30, mean .85", .30, 300, .85),
    ("mid: cut-off .25, mean .80", .25, 300, .80),
    ("mid: cut-off .25, mean .85", .25, 300, .85),
    ("mid: cut-off .20, mean .85", .20, 300, .85),
    ("mid: cut-off .35, mean .70, area 200", .35, 200, .70),
    ("mid: cut-off .40, mean .70, area 300", .40, 300, .70),
    ("mid: cut-off .40, mean .75, area 600", .40, 600, .75),
    ("mid: cut-off .40, mean .80, area 600", .40, 600, .80),
]


def per_image(comps, area, mean):
    out = np.zeros((len(comps), 4))
    for i, cs in enumerate(comps):
        pred = [c[3] for c in cs if c[0] >= area and c[1] >= mean]
        for g in GT[i]:
            out[i] += K.cnt(pred, g)
    return out


def pq(a):
    return a[:, 3].sum() / (a[:, 0].sum() + .5 * a[:, 1].sum() + .5 * a[:, 2].sum())


rows, base = [], None
for thr in sorted({r[1] for r in RULES}, reverse=True):
    comps = K.extract(ens, thr, min_area=100)
    for name, t, area, mean in RULES:
        if t != thr:
            continue
        a = per_image(comps, area, mean)
        kept = np.mean([sum(1 for c in cs if c[0] >= area and c[1] >= mean) for cs in comps])
        if name.startswith("standard"):
            base = pq(a)
        rows.append(dict(rule=name, cutoff=t, area=area, mean=mean, local_pq=pq(a), kept_per_photo=kept,
                         recall=a[:, 0].sum() / (a[:, 0].sum() + a[:, 2].sum()),
                         precision=a[:, 0].sum() / max(1, a[:, 0].sum() + a[:, 1].sum())))
    del comps
    print(f"cut-off {thr} done ({time.time() - t0:.0f}s)", flush=True)
res = pd.DataFrame(rows)
res["delta_vs_standard"] = res["local_pq"] - base
order = [r[0] for r in RULES]
res = res.set_index("rule").loc[order].reset_index()
pd.set_option("display.width", 220)
print("\n(local = our 15-model out-of-fold maps, 707 photos; recall/precision at IoU 0.5)")
print(res.round(4).to_string(index=False))
res.to_csv(C.RESULTS / "A3_rules.csv", index=False)
print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")