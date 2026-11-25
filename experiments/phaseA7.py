"""A13: do the test photos come from the same days as the training photos?
(Our cross-validation keeps whole days together. If the test photos sit right next to training photos in time,
 the test is easier than our CV and a method that uses neighbouring photos would score much higher.)"""
import os, re
import numpy as np, pandas as pd
import common as K
import config as C

il, _ = K.get_data()


def to_dt(stamp):
    d = re.sub(r"\D", "", stamp)
    fmt = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M", 14: "%Y%m%d%H%M%S"}.get(len(d))
    return pd.to_datetime(d, format=fmt, errors="coerce") if fmt else pd.NaT


te = []
for f in sorted(os.listdir(C.TEST_IMG)):
    m = K._PAT.search(f)
    if m:
        te.append(m.groups())
te = pd.DataFrame(te, columns=["stamp", "instrument"]).drop_duplicates("stamp")
print("example train stamp:", il.stamp.iloc[0], "| example test stamp:", te.stamp.iloc[0])
il["t"] = il.stamp.map(to_dt)
te["t"] = te.stamp.map(to_dt)
te["date"] = te.stamp.str[:8]
print(f"train photos {len(il)}  test photos {len(te)}")
print("train date range", il.t.min(), "->", il.t.max(), "| test date range", te.t.min(), "->", te.t.max())
print("train years:", il.year.value_counts().sort_index().to_dict())
print("test years: ", te.t.dt.year.value_counts().sort_index().to_dict())
td = set(il.date)
print(f"test photos whose calendar day also has training photos: {te.date.isin(td).sum()} of {len(te)}")
print("test instruments:", te.instrument.value_counts().to_dict())
ttr = il.t.values.astype("datetime64[m]").astype(np.int64)
gap_test = [np.abs(ttr - x).min() / 60 for x in te.t.values.astype("datetime64[m]").astype(np.int64)]
gap_cv = []
dates = il.date.values
for i, x in enumerate(ttr):
    m = dates != dates[i]
    gap_cv.append(np.abs(ttr[m] - x).min() / 60)
q = [10, 25, 50, 75, 90]
print("\nhours to the nearest TRAINING photo (percentiles 10/25/50/75/90)")
print("  test photos            :", np.round(np.percentile(gap_test, q), 1))
print("  train photo, other day :", np.round(np.percentile(gap_cv, q), 1), "  <- what our cross-validation sees")