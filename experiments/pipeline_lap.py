"""Laptop training script. One experiment = one process, configured by environment variables.
Same recipe as the Kaggle pipeline (U-Net, soft multi-annotator target, BCE+Dice, AdamW one-cycle, fp16),
but: paths from config.py, folds pinned from file, data cached on disk, options for crop/batch/encoder,
and the result is scored and appended to results/results_log.csv automatically."""
import os, re, json, time, queue, threading
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import segmentation_models_pytorch as smp
from pycocotools import mask as mu
import config as C
import common as K

E = os.environ.get
EXP, RES = E("EXP", "lap_control"), int(E("RES", "1024"))
FOLDS = [int(x) for x in E("FOLDS", "0").split(",")]
ITERS, SEED, ENCODER = int(E("ITERS", "4000")), int(E("SEED", "42")), E("ENCODER", "resnet34")
LOSS, TARGET = E("LOSS", "bcedice"), E("TARGET", "soft")
ALPHA, BETA = float(E("ALPHA", "0.3")), float(E("BETA", "0.7"))
CROP, BATCH, LR = int(E("CROP", "512")), int(E("BATCH", "8")), float(E("LR", "3e-4"))
WD, POS_CROP_P, H = 1e-4, float(E("POS_CROP_P", "0.7")), 2048
SMOKE = E("SMOKE", "0") == "1"
ARCH = E("ARCH", "")   # "" = our own U-Net; else an smp class name: UnetPlusPlus, FPN, DeepLabV3Plus, MAnet
OUT = C.RUNS / "exps" / EXP
OUT.mkdir(parents=True, exist_ok=True)
CACHE = C.RUNS.parent / "cache"
PAT = re.compile(r"(\d{14})([A-Za-z]{2})")
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"EXP={EXP} RES={RES} folds={FOLDS} iters={ITERS} seed={SEED} encoder={ENCODER} crop={CROP} batch={BATCH} "
      f"loss={LOSS} target={TARGET} lr={LR} smoke={SMOKE}", flush=True)
json.dump(dict(EXP=EXP, RES=RES, FOLDS=FOLDS, ITERS=ITERS, SEED=SEED, ENCODER=ENCODER, LOSS=LOSS, TARGET=TARGET,
               CROP=CROP, BATCH=BATCH, LR=LR, ARCH=ARCH, WD=WD, POS_CROP_P=POS_CROP_P, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0)), open(OUT / "config.json", "w"), indent=1)


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data (cached on disk)
def norm_img(g):
    med = max(float(np.median(g[g > 25])) if (g > 25).any() else 1.0, 1.0)
    return (np.clip(g / med, 0, 2.5) - 1.0).astype(np.float32)


def load_arrays():
    il, GT = K.get_data()
    fx, fy = CACHE / f"X_{RES}.npy", CACHE / f"Y_{RES}.npy"
    if fx.exists() and fy.exists():
        print("loading cached arrays", flush=True)
        return il, GT, np.load(fx), np.load(fy)
    CACHE.mkdir(parents=True, exist_ok=True)
    path_by_key = {}
    for p in C.TRAIN_IMG.iterdir():
        m = PAT.search(p.name)
        if m:
            path_by_key[m.group(1) + m.group(2)] = p
    t0 = time.time()
    X = np.zeros((len(il), RES, RES), np.float16)
    Y = np.zeros((len(il), RES, RES), np.uint8)
    for i, r in il.iterrows():
        g = cv2.imread(str(path_by_key[r.stamp + r.instrument]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if RES != H:
            g = cv2.resize(g, (RES, RES), interpolation=cv2.INTER_AREA)
        X[i] = norm_img(g)
        acc = np.zeros((H, H), np.float32)
        for rec in GT[i]:
            acc += mu.decode(mu.merge(rec))
        acc /= len(GT[i])
        if RES != H:
            acc = cv2.resize(acc, (RES, RES), interpolation=cv2.INTER_AREA)
        Y[i] = (acc * 255).round().astype(np.uint8)
        if i % 100 == 0:
            print(f"  caching {i}/{len(il)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(fx, X); np.save(fy, Y)
    print(f"cache built in {time.time() - t0:.0f}s", flush=True)
    return il, GT, X, Y


il, GT, X, Y = load_arrays()
if TARGET == "majority":
    Y = np.where(Y >= 128, 255, 0).astype(np.uint8)      # pixels most annotators marked
elif TARGET == "union":
    Y = np.where(Y > 0, 255, 0).astype(np.uint8)         # any annotator marked


class Sampler:
    """Random training windows: POS_CROP_P of them centred near a random filament pixel, else anywhere."""
    def __init__(self, tr, seed):
        self.tr = np.array(tr)
        self.rng = np.random.default_rng(seed)
        self.pos = {int(i): np.flatnonzero(Y[i].ravel() > 127) for i in self.tr}

    def one(self):
        r = self.rng
        i = int(self.tr[r.integers(len(self.tr))])
        hi = RES - CROP
        if r.random() < POS_CROP_P and len(self.pos[i]):
            p = int(self.pos[i][r.integers(len(self.pos[i]))])
            cy, cx = divmod(p, RES)
            y0 = int(np.clip(cy - CROP // 2 + r.integers(-128, 129), 0, hi))
            x0 = int(np.clip(cx - CROP // 2 + r.integers(-128, 129), 0, hi))
        else:
            y0, x0 = int(r.integers(0, hi + 1)), int(r.integers(0, hi + 1))
        xc = X[i, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
        yc = Y[i, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
        if r.random() < .5: xc, yc = xc[:, ::-1], yc[:, ::-1]
        if r.random() < .5: xc, yc = xc[::-1], yc[::-1]
        k = int(r.integers(4)); xc, yc = np.rot90(xc, k), np.rot90(yc, k)
        xc = xc * r.uniform(.85, 1.15) + r.uniform(-.05, .05)
        return np.ascontiguousarray(xc, dtype=np.float32), np.ascontiguousarray(yc, dtype=np.float32)


def batches(sampler, n):
    """Cut windows in a background thread so the GPU never waits for the CPU."""
    q = queue.Queue(maxsize=8)

    def work():
        try:
            for _ in range(n):
                xs, ys = zip(*[sampler.one() for _ in range(BATCH)])
                q.put((torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))[:, None]))
            q.put(None)
        except Exception as e:
            q.put(e)
    threading.Thread(target=work, daemon=True).start()
    while True:
        b = q.get()
        if b is None:
            return
        if isinstance(b, Exception):
            raise b
        yield b


# ---------------------------------------------------------------- models
def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


RES_CFG = {"resnet18": (torchvision.models.resnet18, "ResNet18_Weights", (64, 64, 128, 256, 512)),
           "resnet34": (torchvision.models.resnet34, "ResNet34_Weights", (64, 64, 128, 256, 512)),
           "resnet50": (torchvision.models.resnet50, "ResNet50_Weights", (64, 256, 512, 1024, 2048)),
           "resnet101": (torchvision.models.resnet101, "ResNet101_Weights", (64, 256, 512, 1024, 2048))}


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        ctor, wts, (c0, c1, c2, c3, c4) = RES_CFG[ENCODER]
        enc = ctor(weights=getattr(torchvision.models, wts).IMAGENET1K_V1)     # raises if the download fails
        self.l0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu); self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.d4 = nn.Sequential(conv_bn(c4 + c3, 256), conv_bn(256, 256))
        self.d3 = nn.Sequential(conv_bn(256 + c2, 128), conv_bn(128, 128))
        self.d2 = nn.Sequential(conv_bn(128 + c1, 64), conv_bn(64, 64))
        self.d1 = nn.Sequential(conv_bn(64 + c0, 64), conv_bn(64, 64))
        self.d0 = nn.Sequential(conv_bn(64, 32), conv_bn(32, 32)); self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x0 = self.l0(x); x1 = self.l1(self.pool(x0)); x2 = self.l2(x1); x3 = self.l3(x2); x4 = self.l4(x3)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        d = self.d4(up(x4, x3)); d = self.d3(up(d, x2)); d = self.d2(up(d, x1)); d = self.d1(up(d, x0))
        return self.head(self.d0(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)))


class SmpUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = getattr(smp, ARCH or "Unet")(ENCODER, encoder_weights="imagenet", in_channels=3, classes=1)

    def forward(self, x):
        return self.m(x.repeat(1, 3, 1, 1))


def build_model():
    return (ResUNet() if (ENCODER in RES_CFG and not ARCH) else SmpUNet()).to(dev)


def loss_fn(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    if LOSS == "tversky":
        tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
        return 0.5 * bce + 0.5 * (1 - (tp + 1) / (tp + ALPHA * fp + BETA * fn + 1))
    inter = (p * y).sum()
    return 0.5 * bce + 0.5 * (1 - (2 * inter + 1) / (p.sum() + y.sum() + 1))


@torch.no_grad()
def predict(model, arrs):
    model.eval()
    out = []
    for a in arrs:
        x = torch.from_numpy(np.asarray(a, np.float32))[None, None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            p = 0.5 * (torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(-1)).float()).flip(-1))
        if not torch.isfinite(p).all():                      # rare fp16 overflow: redo this photo in fp32
            p = 0.5 * (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1))
        out.append((p[0, 0].cpu().numpy() * 255).round().astype(np.uint8))
    return out


# ---------------------------------------------------------------- score + log
def score_and_log(f, va, oof, minutes):
    gt = [GT[i] for i in va]
    comps50 = K.extract(oof, 0.5)
    plain = K.evaluate(comps50, 400, 0.0, gt=gt)
    comps35 = K.extract(oof, 0.35)
    keep = K.evaluate(comps35, 400, 0.8, gt=gt)
    ref_file = C.RUNS / "oof_maps" / "res1024" / f"oof_fold{f}.npy"
    ref_pq = float("nan")
    if ref_file.exists():
        ref = np.load(ref_file)[:len(va)]
        ref_pq = K.evaluate(K.extract(ref, 0.35), 400, 0.8, gt=gt)[0]
    desc = f"{ARCH or 'unet'} {ENCODER} crop{CROP} batch{BATCH} it{ITERS} seed{SEED} {LOSS}/{TARGET} fold{f}; keep rule thr0.35/a400/m0.8"
    note = f"plain thr0.5 {plain[0]:.4f}; Kaggle res1024 fold{f} {ref_pq:.4f}; delta {keep[0] - ref_pq:+.4f}; {minutes:.1f} min"
    if SMOKE:
        note = "SMOKE " + note
    K.log_result(f"{EXP}-f{f}", desc, keep[0], keep[1], keep[2], notes=note)
    print(f"RESULT {EXP} fold {f}: keep-rule PQ {keep[0]:.4f} (per-image {keep[1]:.4f}) | plain {plain[0]:.4f} | "
          f"Kaggle ResNet-34 baseline on this fold {ref_pq:.4f} | delta {keep[0] - ref_pq:+.4f} | {minutes:.1f} min", flush=True)


for f in FOLDS:
    ckpt, oof_path = OUT / f"unet_fold{f}.pt", OUT / f"oof_fold{f}.npy"
    if ckpt.exists() and oof_path.exists() and not SMOKE:
        print(f"fold {f}: already done, skipping", flush=True)
        continue
    va = il.index[il.fold == f].to_numpy()
    tr = il.index[il.fold != f].to_numpy()
    if SMOKE:
        va = va[:10]
    seed_all(SEED + f)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=ITERS, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    sampler = Sampler(tr, SEED + f)
    t0, run = time.time(), []
    every = 5 if SMOKE else 500
    for it, (xb, yb) in enumerate(batches(sampler, ITERS), 1):
        model.train()
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(xb)
        loss = loss_fn(lg.float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run.append(loss.item())
        if it % every == 0 or it == 1:
            print(f"fold {f} it {it}/{ITERS} loss {np.mean(run[-every:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    minutes = (time.time() - t0) / 60
    oof = np.stack(predict(model, [X[i] for i in va]))
    if not SMOKE:
        torch.save(model.state_dict(), ckpt)
        np.save(oof_path, oof)
    score_and_log(f, va, oof, minutes)
    del model, opt
    torch.cuda.empty_cache()
print("ALL DONE", flush=True)