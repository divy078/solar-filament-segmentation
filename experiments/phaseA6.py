"""A12: where is the PQ lost, and how well do human annotators agree with each other?
 1. PQ split into SQ (mask quality of matched blobs) and RQ (detection quality), per instrument
 2. recall of annotated filaments by size
 3. human-vs-human PQ (the realistic ceiling) on the photos with 2+ annotators, compared with the model on the same photos"""
import numpy as np, pandas as pd
import pycocotools.mask as mu
import common as K
import config as C

il, GT = K.get_data()
ens = K.load_ensemble({"res1024": 0.5, "effv2s": 0.25, "effv2s_s2": 0.25})
comps = K.extract(ens, 0.35)
preds = [[c[3] for c in cs if c[0] >= 400 and c[1] >= 0.8] for cs in comps]
del ens


def pqparts(P):
    tp, fp, fn, s = P
    sq = s / tp if tp else 0.0
    rq = tp / (tp + .5 * fp + .5 * fn) if (tp + fp + fn) else 0.0
    return sq * rq, sq, rq


def tot(rows):
    return np.sum(rows, axis=0) if len(rows) else np.zeros(4)


print("=== 1. model: PQ = SQ x RQ ===")
rows = {}
for i, pr in enumerate(preds):
    for g in GT[i]:
        rows.setdefault(il.instrument[i], []).append(K.cnt(pr, g))
for k, v in list(rows.items()) + [("ALL", [r for v in rows.values() for r in v])]:
    P = tot(v)
    pq, sq, rq = pqparts(P)
    print(f"  {k:8s} records {len(v):4d}  TP {int(P[0]):5d} FP {int(P[1]):5d} FN {int(P[2]):5d}   PQ {pq:.4f} = SQ {sq:.3f} x RQ {rq:.3f}")

print("\n=== 2. recall of annotated filaments by size (2048 px scale) ===")
bins = [0, 400, 1000, 3000, 10000, 1e9]
labels = ["<400", "400-1k", "1k-3k", "3k-10k", ">10k"]
hit = {b: [0, 0] for b in labels}
for i, pr in enumerate(preds):
    for g in GT[i]:
        if not g:
            continue
        area = np.asarray(mu.area(g))
        if pr:
            m = (np.asarray(mu.iou(pr, g, [0] * len(g))) > 0.5).any(axis=0)
        else:
            m = np.zeros(len(g), bool)
        for a, ok in zip(area, m):
            b = labels[np.searchsorted(bins, a, side="right") - 1]
            hit[b][0] += int(ok); hit[b][1] += 1
for b in labels:
    print(f"  {b:8s} annotated {hit[b][1]:5d}   found {hit[b][0]:5d}   recall {hit[b][0] / max(hit[b][1], 1):.3f}")

print("\n=== 3. human vs human, photos with 2+ annotators ===")
nrec = np.array([len(g) for g in GT])
print("  annotators per photo:", dict(zip(*np.unique(nrec, return_counts=True))))
H, M = [], []
for i, recs in enumerate(GT):
    if len(recs) < 2:
        continue
    for a in range(len(recs)):
        for b in range(len(recs)):
            if a != b:
                H.append(K.cnt(recs[a], recs[b]))
        M.append(K.cnt(preds[i], recs[a]))
    # (model rows: one per record, so the two sets differ in size but both are pooled ratios)
ph, sh, rh = pqparts(tot(H))
pm, sm, rm = pqparts(tot(M))
print(f"  human predicting another human: PQ {ph:.4f} = SQ {sh:.3f} x RQ {rh:.3f}   ({len(H)} pairs)")
print(f"  model vs the same records:      PQ {pm:.4f} = SQ {sm:.3f} x RQ {rm:.3f}   ({len(M)} records)")