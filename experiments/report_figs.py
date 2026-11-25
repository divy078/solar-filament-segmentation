"""Numbers and example figure for the technical report (CPU only, about 4 min).
Final 10-model ensemble (new ResNet-34 0.5 + new EfficientNetV2-S 0.5), robust rule 0.40 / 300 / 0.80, all 707 photos:
  - pooled PQ, SQ, RQ with a 95% bootstrap interval, PQ per fold, per station, recall by filament size
  - fig_examples.png: three held-out photos, expert outlines (green) and model outlines (red)
Run:  python C:\\solar\\src\\report_figs.py      (writes C:\\solar\\results\\fig_examples.png)"""
import numpy as np, cv2
import pycocotools.mask as mu
import common as K
import config as C

il, GT = K.get_data()
fold = il["fold"].values
ens = np.zeros((len(il), 1024, 1024), np.uint8)
for f in range(5):
    pos = np.flatnonzero(fold == f)
    a = np.load(C.RUNS / "exps" / f"k_bs18_it8000_f{f}" / f"oof_fold{f}.npy").astype(np.float32)
    b = np.load(C.RUNS / "exps" / f"k_effb18_f{f}" / f"oof_fold{f}.npy").astype(np.float32)
    ens[pos] = (0.5 * a + 0.5 * b).round().astype(np.uint8)
comps = K.extract(ens, 0.40)
preds = [[c[3] for c in cs if c[0] >= 300 and c[1] >= 0.80] for cs in comps]

cnt = np.zeros((len(il), 4))
per_photo = []
for i, pr in enumerate(preds):
    for g in GT[i]:
        cnt[i] += K.cnt(pr, g)
    t, fp, fn, s = K.cnt(pr, GT[i][0])
    per_photo.append(s / max(t + .5 * fp + .5 * fn, 1e-9))
per_photo = np.array(per_photo)
pq = lambda s: s[..., 3] / (s[..., 0] + .5 * s[..., 1] + .5 * s[..., 2])
S = cnt.sum(0)
sq, rq = S[3] / S[0], S[0] / (S[0] + .5 * S[1] + .5 * S[2])
idx = np.random.default_rng(0).integers(0, len(il), (2000, len(il)))
boot = pq(cnt[idx].sum(1))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"FINAL 10-model ensemble, 707 photos: PQ {pq(S):.4f} (95% CI {lo:.4f}..{hi:.4f}) = SQ {sq:.3f} x RQ {rq:.3f}"
      f" | TP {int(S[0])} FP {int(S[1])} FN {int(S[2])}")
fpq = [pq(cnt[fold == f].sum(0)) for f in range(5)]
print("PQ per fold:", " ".join(f"{p:.4f}" for p in fpq), f"| mean {np.mean(fpq):.4f} sd {np.std(fpq, ddof=1):.4f}")
for st in sorted(il["instrument"].unique()):
    m = il["instrument"].values == st
    print(f"  station {st}: {m.sum():3d} photos, PQ {pq(cnt[m].sum(0)):.4f}")

bins, labels = [0, 400, 1000, 3000, 10000, 1e9], ["<400", "400-1k", "1k-3k", "3k-10k", ">10k"]
hit = {b: [0, 0] for b in labels}
for i, pr in enumerate(preds):
    for g in GT[i]:
        if not g:
            continue
        found = (np.asarray(mu.iou(pr, g, [0] * len(g))) > 0.5).any(0) if pr else np.zeros(len(g), bool)
        for a, ok in zip(np.asarray(mu.area(g)), found):
            b = labels[np.searchsorted(bins, a, side="right") - 1]
            hit[b][0] += int(ok); hit[b][1] += 1
print("recall by filament size (px at 2048):", " | ".join(f"{b}: {hit[b][0] / max(hit[b][1], 1):.2f} (n={hit[b][1]})" for b in labels))

# three typical held-out photos from different stations: per-photo PQ closest to the median
order = np.argsort(np.abs(per_photo - np.median(per_photo)))
chosen, seen = [], set()
for i in order:
    st = il["instrument"][i]
    if st not in seen and len(GT[i][0]) >= 4:
        chosen.append(i); seen.add(st)
    if len(chosen) == 3:
        break
for i in order:                                   # fallback: fewer than 3 stations available
    if len(chosen) == 3:
        break
    if i not in chosen and len(GT[i][0]) >= 1:
        chosen.append(i)
S_PX = 700
panels = []
for i in chosen:
    path = next(p for p in C.TRAIN_IMG.iterdir() if il["stamp"][i] in p.name)
    img = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (S_PX, S_PX), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for rles, col in ((GT[i][0], (40, 200, 40)), (preds[i], (40, 40, 230))):
        for r in rles:
            m = cv2.resize(mu.decode(r).astype(np.uint8), (S_PX, S_PX), interpolation=cv2.INTER_NEAREST)
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, c, -1, col, 2)
    cv2.rectangle(v, (0, 0), (S_PX, 34), (0, 0, 0), -1)
    cv2.putText(v, f"{il['instrument'][i]} {il['stamp'][i][:8]}  PQ {per_photo[i]:.2f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    panels.append(v)
    print(f"example: {path.name} station {il['instrument'][i]} fold {fold[i]} | expert {len(GT[i][0])} filaments,"
          f" model {len(preds[i])}, PQ vs this expert {per_photo[i]:.3f}")
gap = np.full((S_PX, 8, 3), 255, np.uint8)
out = np.hstack([x for p in panels for x in (p, gap)][:-1])
C.RESULTS.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(C.RESULTS / "fig_examples.png"), out)
print("saved", C.RESULTS / "fig_examples.png", f"| median per-photo PQ {np.median(per_photo):.3f}")