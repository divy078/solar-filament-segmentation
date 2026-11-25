"""Shared helpers: data + ground truth, folds, saved OOF maps, blob extraction, PQ scoring, results log."""
import json, re
from datetime import datetime
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mu
import config as C

H = W = 2048
_PAT = re.compile(r"(\d{14})([A-Za-z]{2})")
_CACHE = {}


def _ann_to_rle(seg):
    if isinstance(seg, dict):
        return mu.frPyObjects(seg, H, W) if isinstance(seg["counts"], list) else seg
    return mu.merge(mu.frPyObjects(seg, H, W))


def get_data():
    """Return (img_level, GT). img_level: one row per photo (707) incl. pinned fold.
    GT[i] = list over annotators of (list of RLE filaments)."""
    if "d" in _CACHE:
        return _CACHE["d"]
    coco = json.load(open(C.TRAIN_JSON))
    images = pd.DataFrame(coco["images"])
    anns = pd.DataFrame(coco["annotations"])
    parsed = images["file_name"].map(lambda s: _PAT.search(s).groups())
    images["stamp"] = [p[0] for p in parsed]
    images["instrument"] = [p[1] for p in parsed]
    images["year"] = images["stamp"].str[:4]
    il = images.drop_duplicates("stamp")[["stamp", "instrument", "year"]].reset_index(drop=True)
    il["date"] = il["stamp"].str[:8]

    # folds are PINNED from a file exported on Kaggle: the saved maps follow that split
    ff = C.RUNS / "scripts_logs" / "folds_photos.csv"
    assert ff.exists(), f"missing {ff} - export it from Kaggle (step 10b)"
    pinned = pd.read_csv(ff, dtype=str).drop_duplicates("stamp").set_index("stamp")["fold"]
    il["fold"] = il["stamp"].map(pinned).astype(int)

    gt_rles = {}
    for rid, g in anns.groupby(anns["image_id"].astype(str)):
        gt_rles[str(rid)] = [_ann_to_rle(s) for s in g["segmentation"]]
    ids_by_stamp = images.groupby("stamp")["id"].apply(lambda s: s.astype(str).tolist()).to_dict()
    GT = [[gt_rles[r] for r in ids_by_stamp[s]] for s in il["stamp"]]
    _CACHE["d"] = (il, GT)
    return _CACHE["d"]


def load_ensemble(weights):
    """weights = {'res1024': 0.5, 'effv2s': 0.25, ...}. Weighted mean of saved OOF probability
    maps (uint8, 0-255) -> array (707, 1024, 1024) uint8, aligned with img_level."""
    il, _ = get_data()
    out = np.zeros((len(il), 1024, 1024), np.uint8)
    tot = float(sum(weights.values()))
    for f in range(5):
        pos = np.flatnonzero(il["fold"].values == f)
        acc = None
        for name, w in weights.items():
            p = C.RUNS / "oof_maps" / name / f"oof_fold{f}.npy"
            assert p.exists(), f"missing {p}"
            a = np.load(p).astype(np.float32) * (w / tot)
            assert len(a) == len(pos), f"{name} fold {f}: {len(a)} maps but {len(pos)} photos"
            acc = a if acc is None else acc + a
        out[pos] = acc.round().astype(np.uint8)
    return out


def extract(oof, thr, min_area=100, max_comps=60):
    """Per photo: upsample to 2048, cut off at thr, 8-connected blobs.
    Each blob = (area, mean_prob, peak_prob, RLE)."""
    out = []
    for i in range(len(oof)):
        up = cv2.resize(oof[i], (W, H), interpolation=cv2.INTER_LINEAR)
        n, lab, st, _ = cv2.connectedComponentsWithStats((up > int(thr * 255)).astype(np.uint8), connectivity=8)
        comps = []
        if n > 1:
            areas = st[1:, cv2.CC_STAT_AREA]
            keep = [k for k in np.argsort(-areas)[:max_comps] if areas[k] >= min_area]
            if keep:
                ids = np.array(keep) + 1
                mean = ndi.mean(up, lab, ids)
                mx = ndi.maximum(up, lab, ids)
                for k, mn, pk in zip(ids, mean, mx):
                    x, y, w, h = st[k, :4]
                    m = np.zeros((H, W), np.uint8, order="F")
                    m[y:y + h, x:x + w] = (lab[y:y + h, x:x + w] == k)
                    comps.append((int(st[k, 4]), float(mn) / 255, float(pk) / 255, mu.encode(m)))
        out.append(comps)
    return out


def cnt(pred, gt):
    """One-to-one matching at IoU > 0.5 -> (tp, fp, fn, sum of matched IoU)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt), 0.0
    iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)))
    r, c = linear_sum_assignment(-(iou * (iou > 0.5)))
    ok = iou[r, c] > 0.5
    t = int(ok.sum())
    return t, len(pred) - t, len(gt) - t, float(iou[r, c][ok].sum())


def evaluate(comps, min_area, min_mean, min_peak=0, gt=None):
    """Keep blobs with area>=min_area, mean prob>=min_mean, peak>=min_peak, then score.
    Returns (pooled PQ, per-image mean PQ, [TP, FP, FN, IoU sum])."""
    _, GT = get_data()
    gt = GT if gt is None else gt
    P = [0, 0, 0, 0.0]
    macro = []
    for i, cs in enumerate(comps):
        pred = [c[3] for c in cs if c[0] >= min_area and c[1] >= min_mean and c[2] >= min_peak]
        ms = []
        for g in gt[i]:
            t, fp, fn, s = cnt(pred, g)
            P[0] += t; P[1] += fp; P[2] += fn; P[3] += s
            d = t + .5 * fp + .5 * fn
            ms.append(s / d if d > 0 else 1.0)
        macro.append(np.mean(ms))
    d = P[0] + .5 * P[1] + .5 * P[2]
    return P[3] / d, float(np.mean(macro)), P


def log_result(exp, desc, pq, macro, P, notes=""):
    """Append one row to results/results_log.csv (the table for the paper and the teacher)."""
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    f = C.RESULTS / "results_log.csv"
    row = pd.DataFrame([dict(time=datetime.now().strftime("%Y-%m-%d %H:%M"), exp=exp, description=desc,
                             pooled_pq=round(pq, 4), per_image_pq=round(macro, 4),
                             tp=P[0], fp=P[1], fn=P[2], notes=notes)])
    row.to_csv(f, mode="a", header=not f.exists(), index=False)