"""Phase A, step A0: prove the laptop reproduces our Kaggle numbers, then score the 15-model recipe."""
import time
import common as K

t0 = time.time()
il, GT = K.get_data()
print("photos:", len(il), "| per fold:", il["fold"].value_counts().sort_index().to_dict(), flush=True)


def score(exp, desc, comps, area, mean, expect=None):
    pq, mac, P = K.evaluate(comps, area, mean)
    tag = f"   (Kaggle notebook said {expect})" if expect else ""
    print(f"{desc}: pooled PQ {pq:.4f} | per-image {mac:.4f} | TP/FP/FN {P[0]}/{P[1]}/{P[2]}{tag}", flush=True)
    K.log_result(exp, desc, pq, mac, P)


# ---- check 1: ResNet-34 alone must give 0.4028 (plain) and 0.4118 (keep rule) ----
res = K.load_ensemble({"res1024": 1})
c50 = K.extract(res, 0.5)
score("A0-res-plain", "ResNet-34 thr0.5 area400", c50, 400, 0.0, 0.4028)
c35 = K.extract(res, 0.35)
score("A0-res-keep", "ResNet-34 thr0.35 area400 mean0.8", c35, 400, 0.8, 0.4118)
print(f"check 1 done ({time.time() - t0:.0f}s)\n", flush=True)

# ---- the 15-model OOF recipe: 0.5 ResNet + 0.25 EffNet seed42 + 0.25 EffNet seed43 ----
ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})
c = K.extract(ens, 0.35)
score("A0-ens15", "15-model OOF thr0.35 area400 mean0.8", c, 400, 0.8)
score("A0-ens15-m75", "15-model OOF thr0.35 area400 mean0.75", c, 400, 0.75)
print(f"all done in {time.time() - t0:.0f}s. Results appended to C:/solar/results/results_log.csv")