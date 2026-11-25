"""A14: three things suggested by the other notebooks (all CPU, reuses saved maps)
 1. keep rule WITH a peak-probability filter (the 0.45-OOF notebook uses peak > 0.99 and min area 0)
 2. PQ by year of the photo, and PQ re-weighted to the year mix of the test set
 3. does the image polarity (bright vs dark disk skew) differ between years / train / test?"""
import itertools, json, os, re
import numpy as np, pandas as pd, cv2
import common as K
import config as C

il, GT = K.get_data()
ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})
pd.set_option("display.width", 200)


def per_image(comps, area, mean, peak):
    out = np.zeros((len(comps), 4))
    for i, cs in enumerate(comps):
        pred = [c[3] for c in cs if c[0] >= area and c[1] >= mean and c[2] >= peak]
        for g in GT[i]:
            out[i] += K.cnt(pred, g)
    return out


def pq(a):
    return a[:, 3].sum() / (a[:, 0].sum() + .5 * a[:, 1].sum() + .5 * a[:, 2].sum())


print("=== 1. keep rule with a peak filter (pooled PQ, all 707 photos, 15-model ensemble) ===", flush=True)
rows, store = [], {}
for thr in (0.35, 0.40):
    comps = K.extract(ens, thr, min_area=100)
    for area, mean, peak in itertools.product((100, 200, 300, 400), (0.0, 0.7, 0.8, 0.85), (0.0, 0.9, 0.95, 0.98, 0.99)):
        a = per_image(comps, area, mean, peak)
        rows.append(dict(thr=thr, area=area, mean=mean, peak=peak, pq=pq(a)))
        store[(thr, area, mean, peak)] = a
    print(f"  thr {thr} done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(C.RESULTS / "A14_peak_grid.csv", index=False)
base = res[(res.thr == 0.35) & (res.area == 400) & (res["mean"] == 0.8) & (res.peak == 0)].pq.iloc[0]
print(f"current rule (thr .35, area 400, mean .80, no peak filter): {base:.4f}")
for thr in (0.35, 0.40):
    for mean in (0.0, 0.8, 0.85):
        t = res[(res.thr == thr) & (res["mean"] == mean)].pivot(index="area", columns="peak", values="pq").round(4)
        print(f"\nthr {thr}, min mean {mean}: rows = min area, columns = min peak")
        print(t.to_string())
print("\nTOP 10 rules:")
print(res.sort_values("pq", ascending=False).head(10).round(4).to_string(index=False))

print("\n=== 2. PQ by year (current rule) ===")
a0 = store[(0.35, 400, 0.8, 0.0)]
il["yr"] = il.year.astype(int)
grp = {"2011-12": (2011, 2012), "2013-14": (2013, 2014), "2015-16": (2015, 2016), "2017-19": (2017, 2019), "2020-22": (2020, 2022)}
w_test = {"2011-12": 47, "2013-14": 45, "2015-16": 39, "2017-19": 25, "2020-22": 24}   # test photos per group (from A7)
tot = 0
for g, (lo, hi) in grp.items():
    m = ((il.yr >= lo) & (il.yr <= hi)).values
    print(f"  {g:8s} photos {m.sum():4d}  PQ {pq(a0[m]):.4f}   (test photos in this group: {w_test[g]})")
# year-mix reweighting: weight every train photo so the groups match the test mix
wt = np.zeros(len(il))
for g, (lo, hi) in grp.items():
    m = ((il.yr >= lo) & (il.yr <= hi)).values
    wt[m] = (w_test[g] / 180) / (m.sum() / len(il))
rw = (a0 * wt[:, None]).sum(0)
print(f"  PQ with train photos re-weighted to the test year mix: {rw[3] / (rw[0] + .5 * rw[1] + .5 * rw[2]):.4f}   (unweighted {pq(a0):.4f})")

print("\n=== 3. disk brightness skew (mean - median inside a 950 px disk) ===")
coco = json.load(open(C.TRAIN_JSON))
im = pd.DataFrame(coco["images"])
im["stamp"] = im.file_name.map(lambda s: re.search(r"(\d{14})", s).group(1))
fn = im.drop_duplicates("stamp").set_index("stamp").file_name
YY, XX = np.ogrid[:2048, :2048]
DISK = ((YY - 1023.5) ** 2 + (XX - 1023.5) ** 2) <= 950 ** 2


def skew(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    v = g[DISK].astype(np.float32)
    return float(v.mean() - np.median(v)), float(v.mean())


tr = [skew(C.TRAIN_IMG / fn[s]) for s in il.stamp]
il["skew"], il["mean"] = [t[0] for t in tr], [t[1] for t in tr]
te_files = sorted(os.listdir(C.TEST_IMG))
te = pd.DataFrame([skew(C.TEST_IMG / f) for f in te_files], columns=["skew", "mean"])
te["year"] = [int(f[:4]) for f in te_files]
print(f"  train: skew > 0 in {(il.skew > 0).sum()} of {len(il)} photos ({(il.skew > 0).mean():.1%}); test: {(te.skew > 0).sum()} of {len(te)} ({(te.skew > 0).mean():.1%})")
print("  share of photos with skew > 0, by year:")
print(pd.DataFrame({"train": il.groupby("yr").skew.apply(lambda s: (s > 0).mean()), "test": te.groupby("year").skew.apply(lambda s: (s > 0).mean())}).round(2).to_string())
m = (il.skew > 0).values
if m.any() and (~m).any():
    print(f"  PQ on skew > 0 photos: {pq(a0[m]):.4f} ({m.sum()} photos);  skew <= 0: {pq(a0[~m]):.4f} ({(~m).sum()} photos)")
print("  disk mean brightness, train by year / test by year:")
print(pd.DataFrame({"train": il.groupby("yr")["mean"].mean(), "test": te.groupby("year")["mean"].mean()}).round(1).to_string())