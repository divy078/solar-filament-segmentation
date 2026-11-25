"""Predict the 180 test photos with the 5 retrained models of one group (batch 18, 8000 steps, trained on Kaggle). Laptop GPU, about 2-5 min.
Model, normalisation, 1024 px size and flip averaging are copied exactly from the Kaggle pipeline.py / pipeline_smp.py.
  python C:\\solar\\src\\predict_test.py        ResNet-34 group        -> runs\\test_probs\\test_probs_res_b18.npy
  python C:\\solar\\src\\predict_test.py eff    EfficientNetV2-S group -> runs\\test_probs\\test_probs_eff_b18.npy
(180 x 1024 x 1024 uint8, same format as the older test_probs_*.npy files)"""
import sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import config as C

RES = 1024
dev = torch.device("cuda")
GROUP = sys.argv[1] if len(sys.argv) > 1 else "res"
assert GROUP in ("res", "eff"), "argument must be res or eff"
NEW = {"res": "k_bs18_it8000", "eff": "k_effb18"}[GROUP]
OUT = C.RUNS / "test_probs" / f"test_probs_{GROUP}_b18.npy"
OLD = C.RUNS / "test_probs" / f"test_probs_{GROUP}.npy"


def conv_bn(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


CH = (64, 64, 128, 256, 512)


class UNet(nn.Module):                      # identical to the Kaggle pipeline.py (ResNet-34); weights come from the checkpoint
    def __init__(self):
        super().__init__()
        enc = torchvision.models.resnet34(weights=None)
        self.l0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu); self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        c0, c1, c2, c3, c4 = CH
        self.d4 = nn.Sequential(conv_bn(c4 + c3, 256), conv_bn(256, 256)); self.d3 = nn.Sequential(conv_bn(256 + c2, 128), conv_bn(128, 128))
        self.d2 = nn.Sequential(conv_bn(128 + c1, 64), conv_bn(64, 64)); self.d1 = nn.Sequential(conv_bn(64 + c0, 64), conv_bn(64, 64))
        self.d0 = nn.Sequential(conv_bn(64, 32), conv_bn(32, 32)); self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1); x0 = self.l0(x); x1 = self.l1(self.pool(x0)); x2 = self.l2(x1); x3 = self.l3(x2); x4 = self.l4(x3)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        d = self.d4(up(x4, x3)); d = self.d3(up(d, x2)); d = self.d2(up(d, x1)); d = self.d1(up(d, x0))
        return self.head(self.d0(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)))


class EffUNet(nn.Module):                   # identical to the Kaggle pipeline_smp.py; weights come from the checkpoint
    def __init__(self):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.m = smp.Unet("tu-tf_efficientnetv2_s", encoder_weights=None, in_channels=3, classes=1)

    def forward(self, x): return self.m(x.repeat(1, 3, 1, 1))


def norm_img(g):
    med = max(float(np.median(g[g > 25])) if (g > 25).any() else 1.0, 1.0)
    return (np.clip(g / med, 0, 2.5) - 1.0).astype(np.float32)


def load_small(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return cv2.resize(g, (RES, RES), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def predict(model, arrs):
    model.eval(); out, nan_fixed = [], 0
    for a in arrs:
        x = torch.from_numpy(np.asarray(a, np.float32))[None, None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            p = 0.5 * (torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(-1)).float()).flip(-1))
        if torch.isnan(p).any():           # fp16 safety net: redo this photo in full precision
            p = 0.5 * (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)); nan_fixed += 1
        out.append((p[0, 0].cpu().numpy() * 255).round().astype(np.uint8))
    return out, nan_fixed


t0 = time.time()
names = [p for p in sorted(C.TEST_IMG.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg")]   # same order as make_sub.py
assert len(names) == 180, f"expected 180 test photos, found {len(names)}"
ckpts = [C.RUNS / "exps" / f"{NEW}_f{f}" / f"unet_fold{f}.pt" for f in range(5)]
missing = [str(c) for c in ckpts if not c.exists()]
assert not missing, "missing checkpoints:\n  " + "\n  ".join(missing)
X = [norm_img(load_small(p)).astype(np.float16) for p in names]      # stored as float16, as in training
print(f"loaded {len(X)} test photos ({time.time() - t0:.0f}s) | GPU: {torch.cuda.get_device_name(0)}", flush=True)

probs = np.zeros((len(X), RES, RES), np.float32)
for f, ck in enumerate(ckpts):
    m = (UNet() if GROUP == "res" else EffUNet()).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev))
    p, nf = predict(m, X)
    probs += np.stack(p).astype(np.float32) / 5
    print(f"  fold {f} model done ({time.time() - t0:.0f}s){f', {nf} photos redone in fp32' if nf else ''}", flush=True)
    del m; torch.cuda.empty_cache()
out = probs.round().astype(np.uint8)
np.save(OUT, out)
print(f"saved {OUT}  shape {out.shape}")

# sanity check against the same group's old test maps: should be similar, not identical
old = np.load(OLD, mmap_mode="r")
a, b = out[::6].astype(np.float32).ravel(), old[::6].astype(np.float32).ravel()
corr = float(np.corrcoef(a, b)[0, 1])
fa, fb = (a > 89).mean() * 100, (b > 89).mean() * 100
print(f"check vs old {GROUP} test maps: correlation {corr:.3f} (expect about 0.85-0.97) | "
      f"pixels above 0.35: new {fa:.2f}% old {fb:.2f}% (expect similar)")
print("OK" if corr > 0.8 and 0.5 < fa / max(fb, 1e-9) < 2 else "CHECK: maps look different from the old ones - tell Claude")