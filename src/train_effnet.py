"""Training script for the EfficientNetV2-S U-Net group (segmentation-models-pytorch) (one process trains the requested folds).
Used on Kaggle (2 x NVIDIA T4) with the competition data attached. Settings come from environment variables.
Final models in weights/ were trained with:
    EXP=effv2s_b18 ENCODER=tu-tf_efficientnetv2_s RES=1024 BATCH=18 ITERS=8000 SEED=42 FOLDS=<f> python train_effnet.py
Outputs go to /kaggle/working/exps/<EXP>/: unet_fold<f>.pt (weights) and oof_fold<f>.npy (out-of-fold maps).
The 5 folds are grouped by calendar day and stratified by station (StratifiedGroupKFold, seed 42);
the resulting split is stored in data/folds_photos.csv."""
# One experiment = one process, configured by environment variables. Trains the requested folds and saves
# checkpoints + out-of-fold probabilities to /kaggle/working/exps/<EXP>/.
import os, glob, json, re, time, random
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pycocotools import mask as mu
from sklearn.model_selection import StratifiedGroupKFold

E = os.environ.get
EXP, RES = E("EXP", "baseline"), int(E("RES", "1024")); FOLDS = [int(x) for x in E("FOLDS", "0").split(",")]
ITERS, SEED, ENCODER = int(E("ITERS", "4000")), int(E("SEED", "42")), E("ENCODER", "resnet34")
LOSS, TARGET, ALPHA, BETA = E("LOSS", "bcedice"), E("TARGET", "soft"), float(E("ALPHA", "0.3")), float(E("BETA", "0.7"))
CROP, LR, WD, POS_CROP_P, FOLD_SEED, H = 512, 3e-4, 1e-4, 0.7, 42, 2048; BATCH = int(E('BATCH', '8'))
WORK = f"{E('OUT', '/kaggle/working/exps')}/{EXP}"; os.makedirs(WORK, exist_ok=True)
dev = torch.device("cuda:0")
print(f"EXP={EXP} RES={RES} folds={FOLDS} iters={ITERS} seed={SEED} encoder={ENCODER} loss={LOSS} target={TARGET} batch={BATCH}", flush=True)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---- data + folds (identical recipe to the main notebook) ----
ann_path = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)[0]
train_imgs = glob.glob("/kaggle/input/**/train_images/*.jp*g", recursive=True)
pat = re.compile(r"(\d{14})([A-Za-z]{2})"); parse = lambda n: pat.search(n).groups()
path_by_key = {"".join(parse(os.path.basename(p))): p for p in train_imgs}
coco = json.load(open(ann_path)); images = pd.DataFrame(coco["images"]); anns = pd.DataFrame(coco["annotations"])
images["id"] = images["id"].astype(str)
images[["stamp", "instrument"]] = images["file_name"].apply(lambda s: pd.Series(parse(s)))
images["key"] = images["stamp"] + images["instrument"]
img_level = images.drop_duplicates("stamp")[["stamp", "instrument", "key"]].reset_index(drop=True)
img_level["date"] = img_level["stamp"].str[:8]; img_level["fold"] = -1
for f, (_, vi) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(img_level, y=img_level["instrument"], groups=img_level["date"])):
    img_level.loc[vi, "fold"] = f
stamp_to_ids = images.groupby("stamp")["id"].apply(list).to_dict()
def ann_to_rle(a):
    seg = a["segmentation"]
    if isinstance(seg, dict): return mu.frPyObjects(seg, H, H) if isinstance(seg["counts"], list) else seg
    return mu.merge(mu.frPyObjects(seg, H, H))
gt_rles = {str(k): [ann_to_rle(a) for _, a in g.iterrows()] for k, g in anns.groupby(anns["image_id"].astype(str))}

def norm_img(g):
    med = max(float(np.median(g[g > 25])) if (g > 25).any() else 1.0, 1.0)
    return (np.clip(g / med, 0, 2.5) - 1.0).astype(np.float32)
def load_small(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return g if RES == 2048 else cv2.resize(g, (RES, RES), interpolation=cv2.INTER_AREA)
t0 = time.time(); X = np.zeros((len(img_level), RES, RES), np.float16); Y = np.zeros((len(img_level), RES, RES), np.uint8)
for i, r in img_level.iterrows():
    X[i] = norm_img(load_small(path_by_key[r.key])); acc = np.zeros((H, H), np.float32)
    for rid in stamp_to_ids[r.stamp]: acc += mu.decode(mu.merge(gt_rles[rid]))
    acc /= len(stamp_to_ids[r.stamp])
    Y[i] = ((acc if RES == 2048 else cv2.resize(acc, (RES, RES), interpolation=cv2.INTER_AREA)) * 255).round().astype(np.uint8)
if TARGET == "majority": Y = np.where(Y >= 128, 255, 0).astype(np.uint8)     # keep pixels most annotators marked
elif TARGET == "union": Y = np.where(Y > 0, 255, 0).astype(np.uint8)         # any annotator marked
print(f"cached in {time.time()-t0:.0f}s", flush=True)

class CropDS(torch.utils.data.Dataset):
    def __init__(self, idx, n): self.idx, self.n = np.array(idx), n
    def __len__(self): return self.n
    def __getitem__(self, k):
        i = self.idx[np.random.randint(len(self.idx))]; x, y = X[i], Y[i]
        if np.random.rand() < POS_CROP_P:
            pos = np.flatnonzero(y.ravel() > 127)
            if len(pos):
                p = pos[np.random.randint(len(pos))]; cy, cx = divmod(p, RES)
                y0 = int(np.clip(cy - CROP // 2 + np.random.randint(-128, 129), 0, RES - CROP)); x0 = int(np.clip(cx - CROP // 2 + np.random.randint(-128, 129), 0, RES - CROP))
            else: y0, x0 = np.random.randint(0, RES - CROP + 1, 2)
        else: y0, x0 = np.random.randint(0, RES - CROP + 1, 2)
        xc = x[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32); yc = y[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
        if np.random.rand() < .5: xc, yc = xc[:, ::-1], yc[:, ::-1]
        if np.random.rand() < .5: xc, yc = xc[::-1], yc[::-1]
        k90 = np.random.randint(4); xc, yc = np.rot90(xc, k90), np.rot90(yc, k90)
        xc = xc * np.random.uniform(.85, 1.15) + np.random.uniform(-.05, .05)
        return torch.from_numpy(np.ascontiguousarray(xc))[None], torch.from_numpy(np.ascontiguousarray(yc))[None]

def conv_bn(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
import segmentation_models_pytorch as smp
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=3, classes=1)  # raises if weights unavailable
        self.pretrained = True
    def forward(self, x): return self.m(x.repeat(1, 3, 1, 1))

def loss_fn(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y); p = torch.sigmoid(logit)
    if LOSS == "tversky":       # ALPHA weighs false positives, BETA weighs misses (BETA > ALPHA favours recall)
        tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
        return 0.5 * bce + 0.5 * (1 - (tp + 1) / (tp + ALPHA * fp + BETA * fn + 1))
    inter = (p * y).sum(); return 0.5 * bce + 0.5 * (1 - (2 * inter + 1) / (p.sum() + y.sum() + 1))

@torch.no_grad()
def predict(model, arr_list):
    model.eval(); out = []
    for a in arr_list:
        x = torch.from_numpy(np.asarray(a, np.float32))[None, None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            p = 0.5 * (torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(-1)).float()).flip(-1))
        if torch.isnan(p).any(): p = 0.5 * (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1))  # fp16 safety
        out.append((p[0, 0].cpu().numpy() * 255).round().astype(np.uint8))
    return out

for f in FOLDS:
    ckpt, oof_path = f"{WORK}/unet_fold{f}.pt", f"{WORK}/oof_fold{f}.npy"
    if os.path.exists(ckpt) and os.path.exists(oof_path): print(f"fold {f}: already done, skipping", flush=True); continue
    va = img_level.index[img_level.fold == f].to_numpy(); tr = img_level.index[img_level.fold != f].to_numpy()
    seed_all(SEED + f); model = UNet().to(dev)
    dl = torch.utils.data.DataLoader(CropDS(tr, ITERS * BATCH), batch_size=BATCH, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=ITERS, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler(); t0 = time.time(); run = []
    for it, (xb, yb) in enumerate(dl, 1):
        model.train(); xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16): lg = model(xb)
        loss = loss_fn(lg.float(), yb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); run.append(loss.item())
        if it % 500 == 0 or it == 1: print(f"fold {f} it {it}/{ITERS} loss {np.mean(run[-500:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), ckpt); np.save(oof_path, np.stack(predict(model, [X[i] for i in va])))
    print(f"fold {f} done in {(time.time()-t0)/60:.1f} min", flush=True)
    del model, opt; torch.cuda.empty_cache()
print("ALL DONE", flush=True)