"""A15: image polarity / brightness by year and train vs test (the part of phaseA8 that crashed), plus PQ by polarity"""
import json, os, re
import numpy as np, pandas as pd, cv2
import common as K
import config as C

il, GT = K.get_data()
il["yr"] = il.year.astype(int)
coco = json.load(open(C.TRAIN_JSON))
im = pd.DataFrame(coco["images"])
im["stamp"] = im.file_name.map(lambda s: re.search(r"(\d{14})", s).group(1))
fn = im.drop_duplicates("stamp").set_index("stamp").file_name
YY, XX = np.ogrid[:2048, :2048]
DISK = ((YY - 1023.5) ** 2 + (XX - 1023.5) ** 2) <= 950 ** 2


def stats(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    v = g[DISK].astype(np.float32)
    return float(v.mean() - np.median(v)), float(v.mean())


tr = [stats(C.TRAIN_IMG / fn[s]) for s in il.stamp]
il["dsk"], il["dmean"] = [t[0] for t in tr], [t[1] for t in tr]
te_files = sorted(os.listdir(C.TEST_IMG))
te = pd.DataFrame([stats(C.TEST_IMG / f) for f in te_files], columns=["dsk", "dmean"])
te["yr"] = [int(f[:4]) for f in te_files]
print(f"train: skew > 0 in {(il['dsk'] > 0).sum()} of {len(il)} photos ({(il['dsk'] > 0).mean():.1%});  test: {(te['dsk'] > 0).sum()} of {len(te)} ({(te['dsk'] > 0).mean():.1%})")
print("\nshare of photos with skew > 0, by year")
print(pd.DataFrame({"train": il.groupby("yr")["dsk"].apply(lambda s: (s > 0).mean()),
                    "test": te.groupby("yr")["dsk"].apply(lambda s: (s > 0).mean())}).round(2).to_string())
print("\nmean disk brightness (0-255), by year")
print(pd.DataFrame({"train": il.groupby("yr")["dmean"].mean(), "test": te.groupby("yr")["dmean"].mean()}).round(1).to_string())
print("\nskew percentiles 5/25/50/75/95  train:", np.round(np.percentile(il["dsk"], [5, 25, 50, 75, 95]), 2),
      " test:", np.round(np.percentile(te["dsk"], [5, 25, 50, 75, 95]), 2))

ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})
comps = K.extract(ens, 0.35)
pq_i = []
for i, cs in enumerate(comps):
    pred = [c[3] for c in cs if c[0] >= 400 and c[1] >= 0.8]
    r = np.zeros(4)
    for g in GT[i]:
        r += K.cnt(pred, g)
    pq_i.append(r)
A = np.array(pq_i)
pq = lambda a: a[:, 3].sum() / (a[:, 0].sum() + .5 * a[:, 1].sum() + .5 * a[:, 2].sum())
m = (il["dsk"] > 0).values
print(f"\nPQ, skew > 0 photos: {pq(A[m]):.4f} ({m.sum()} photos);   skew <= 0 photos: {pq(A[~m]):.4f} ({(~m).sum()} photos)")
print("PQ by disk brightness quartile:")
q = pd.qcut(il["dmean"], 4, labels=False).values
for k in range(4):
    print(f"  quartile {k + 1}: PQ {pq(A[q == k]):.4f}  (disk mean {il['dmean'][q == k].min():.0f}-{il['dmean'][q == k].max():.0f})")