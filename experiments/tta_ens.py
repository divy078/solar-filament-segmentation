"""A16: does the 8-view TTA gain survive averaging several models? (fold 0, laptop models)
usage: python tta_ens.py bs18_c512 bs18_s43 ...   (each needs oof_fold0.npy and oof_fold0_tta8.npy in its exps folder)"""
import sys
import numpy as np
import common as K
import config as C

names = sys.argv[1:]
il, GT = K.get_data()
va = np.flatnonzero(il.fold.values == 0)
gt = [GT[i] for i in va]


def score(maps):
    pq, mac, P = K.evaluate(K.extract(maps, 0.35), 400, 0.8, gt=gt)
    return pq


def load(n, tta):
    f = C.RUNS / "exps" / n / ("oof_fold0_tta8.npy" if tta else "oof_fold0.npy")
    return np.load(f).astype(np.float32)


for n in names:
    print(f"{n:14s} single model: 2 views {score(load(n, False).round().astype(np.uint8)):.4f}   8 views {score(load(n, True).round().astype(np.uint8)):.4f}", flush=True)
if len(names) > 1:
    a = np.mean([load(n, False) for n in names], axis=0).round().astype(np.uint8)
    b = np.mean([load(n, True) for n in names], axis=0).round().astype(np.uint8)
    print(f"average of {len(names)} models: 2 views {score(a):.4f}   8 views {score(b):.4f}   gain {score(b) - score(a):+.4f}")