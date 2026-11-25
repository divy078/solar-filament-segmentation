"""Score OOF maps trained on Kaggle with the keep rule thr 0.35 / area 400 / mean 0.8.
Usage:  set FOLD=0&& python C:\\solar\\src\\score_oof.py <EXP> [<EXP> ...]
Reads C:\\solar\\runs\\exps\\<EXP>\\oof_fold<FOLD>.npy; needs common.py and config.py in the same folder."""
import os, sys
import numpy as np
import common as K
import config as C

fold = int(os.environ.get("FOLD", "0"))
il, GT = K.get_data()
va = np.flatnonzero(il.fold.values == fold)
gt = [GT[i] for i in va]


def score(maps):
    pq, mac, P = K.evaluate(K.extract(maps, 0.35), 400, 0.8, gt=gt)
    return pq


ref = np.load(C.RUNS / "oof_maps" / "res1024" / f"oof_fold{fold}.npy")
print(f"Kaggle baseline ResNet-34 (res1024, seed 42), fold {fold}: {score(ref):.4f}", flush=True)
for n in sys.argv[1:]:
    f = C.RUNS / "exps" / n / f"oof_fold{fold}.npy"
    if not f.exists():
        print(f"{n}: file not found: {f}")
        continue
    m = np.load(f)
    assert m.shape[0] == len(va), (m.shape, len(va))
    print(f"{n:16s} fold {fold}: {score(m):.4f}", flush=True)