"""DEMO for the teacher - what the Solar Filament project does, in 3 steps (CPU only, about 2 minutes).
  1. the data: photos, expert drawings, the 5 day-grouped folds
  2. the score: old vs new models on fold 0 (144 photos the models never saw during training)
  3. a picture: one photo with the experts' outlines (green) next to the model's outlines (red)
Run:  python C:\\solar\\src\\demo.py"""
import os, time
import numpy as np, cv2
import pycocotools.mask as mu
import common as K
import config as C

t0 = time.time()
el = lambda: f"{time.time() - t0:.0f}s"
il, GT = K.get_data()

print("=" * 72)
print("STEP 1 - THE DATA")
n_rec = sum(len(g) for g in GT)
n_fil = sum(len(r) for g in GT for r in g)
print(f"  {len(il)} photos of the Sun, 2048 x 2048 pixels, from {il['instrument'].nunique()} telescopes")
print(f"  {n_rec} expert drawings with {n_fil} outlined filaments ({sum(len(g) >= 2 for g in GT)} photos drawn by 2-3 experts)")
print(f"  5 folds grouped by calendar day (photos per fold: {', '.join(map(str, np.bincount(il['fold'].values)))})")
print("  -> every photo is scored by a model that never saw it, or any photo from the same day")

print("=" * 72)
print("STEP 2 - THE SCORE on fold 0 (144 hidden photos)   metric: Panoptic Quality (PQ), 0 to 1")
va = np.flatnonzero(il["fold"].values == 0)
gt = [GT[i] for i in va]
maps = {
    "old_res": np.load(C.RUNS / "oof_maps" / "res1024" / "oof_fold0.npy"),
    "new_res": np.load(C.RUNS / "exps" / "k_bs18_it8000_f0" / "oof_fold0.npy"),
}
eff = 0.25 * np.load(C.RUNS / "oof_maps" / "effv2s" / "oof_fold0.npy").astype(np.float32) + \
      0.25 * np.load(C.RUNS / "oof_maps" / "effv2s_s2" / "oof_fold0.npy").astype(np.float32)
maps["old_ens"] = (0.5 * maps["old_res"].astype(np.float32) + eff).round().astype(np.uint8)
maps["new_ens"] = (0.5 * maps["new_res"].astype(np.float32) + eff).round().astype(np.uint8)
del eff
labels = {"old_res": "ResNet-34 U-Net, old training (batch 8, 4000 steps)",
          "new_res": "ResNet-34 U-Net, new training (batch 18, 8000 steps)",
          "old_ens": "Ensemble ResNet + 2 EfficientNets, old ResNet",
          "new_ens": "Ensemble ResNet + 2 EfficientNets, new ResNet"}
score = {}
for k in maps:
    pq, _, P = K.evaluate(K.extract(maps[k], 0.35), 400, 0.80, gt=gt)
    score[k] = pq
    print(f"  {labels[k]:54s} PQ {pq:.4f}   found {P[0]}, false {P[1]}, missed {P[2]}  ({el()})", flush=True)
print(f"  gain from longer training: single model {score['new_res'] - score['old_res']:+.4f}, "
      f"ensemble {score['new_ens'] - score['old_ens']:+.4f}")
print("  for reference: two human experts agree with each other at PQ 0.340")

print("=" * 72)
print("STEP 3 - A PICTURE (saved and opened)")
j = int(np.argmin([abs(len(GT[i][0]) - 8) for i in va]))          # a typical photo: about 8 filaments
i = va[j]
stamp = il["stamp"][i]
path = next(p for p in C.TRAIN_IMG.iterdir() if stamp in p.name)
img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
cs = K.extract(maps["new_ens"][j:j + 1], 0.35)[0]
pred = [c[3] for c in cs if c[0] >= 400 and c[1] >= 0.80]
tp, fp, fn, iou_sum = K.cnt(pred, GT[i][0])                       # compared with the expert shown
pq1 = iou_sum / max(tp + .5 * fp + .5 * fn, 1e-9)
S = 768


def panel(rles, colour, title):
    v = cv2.cvtColor(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    for r in rles:
        m = cv2.resize(mu.decode(r).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cnts, -1, colour, 2)
    cv2.rectangle(v, (0, 0), (S, 40), (0, 0, 0), -1)
    cv2.putText(v, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return v


out = np.hstack([panel([], (0, 0, 0), "Photo " + path.stem),
                 panel(GT[i][0], (0, 220, 0), f"Expert: {len(GT[i][0])} filaments"),
                 panel(pred, (0, 0, 255), f"Model: {len(pred)} filaments, PQ {pq1:.2f}")])
C.RESULTS.mkdir(parents=True, exist_ok=True)
png = C.RESULTS / "demo_panel.png"
cv2.imwrite(str(png), out)
print(f"  photo {path.name}: expert drew {len(GT[i][0])}, model found {len(pred)} (matched {tp}), PQ on this photo {pq1:.3f}")
print(f"  saved {png}")
try:
    os.startfile(str(png))                                          # opens the picture on Windows
except Exception:
    pass
print(f"done in {el()}")
