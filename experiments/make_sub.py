"""Make a Kaggle submission CSV from the saved test probability maps (no GPU).
   python make_sub.py NAME KIND THR AREA MEAN      e.g.  python make_sub.py ens20_strict ens20 0.40 1000 0.88
       KIND = ens20     3 groups incl. 1536 px (the submitted recipe, public 0.37)
              ens15     ResNet + 2 EfficientNet seeds
              ens20b18  ens20 with the new ResNet-34 group (batch 18, 8000 steps; needs test_probs_res_b18.npy)
              ens15b18  ens15 with the new ResNet-34 group
              ens20b18_e1  ens20b18 with the new EfficientNet group in place of the seed-42 EfficientNet group
              ens20b18_e2  ens20b18 with the new EfficientNet group in place of both old EfficientNet groups
              ens10b18  new ResNet-34 group + new EfficientNetV2-S group, equal weight (10 models)
       THR = cut-off, AREA = min blob area (2048 scale), MEAN = min mean probability
   python make_sub.py check                        rebuilds ens20 thr0.35/area400/mean0.8 and compares with the
                                                   Kaggle-made sub_ens20_m80.csv (proves the laptop pipeline works)"""
import sys
import numpy as np, pandas as pd
import common as K
import config as C

KINDS = {"ens20": ("res", "eff", "eff2", True), "ens15": ("res", "eff", "eff2", False),
         "ens20b18": ("res_b18", "eff", "eff2", True), "ens15b18": ("res_b18", "eff", "eff2", False),
         "ens20b18_e1": ("res_b18", "eff_b18", "eff2", True), "ens20b18_e2": ("res_b18", "eff_b18", "eff_b18", True),
         "ens10b18": ("res_b18", "eff_b18", "eff_b18", False)}


def blend(kind):
    if kind not in KINDS:
        raise SystemExit("KIND must be one of: " + ", ".join(KINDS))
    rn, ea, eb, with1536 = KINDS[kind]
    mm = {n: np.load(C.RUNS / "test_probs" / f"test_probs_{n}.npy", mmap_mode="r") for n in {rn, ea, eb, "r1536"}}
    n = mm[rn].shape[0]
    out = np.zeros((n, 1024, 1024), np.uint8)
    for s in range(0, n, 30):
        sl = slice(s, s + 30)
        res, eff, eff2 = (mm[k][sl].astype(np.float32) for k in (rn, ea, eb))
        if with1536:
            p = (res + 0.5 * (eff + eff2) + mm["r1536"][sl].astype(np.float32)) / 3
        else:
            p = 0.5 * (res + 0.5 * (eff + eff2))
        out[sl] = p.round().astype(np.uint8)
    return out


def make(name, kind, thr, area, mean):
    names = [p.stem for p in sorted(C.TEST_IMG.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg")]
    probs = blend(kind)
    assert len(names) == len(probs), (len(names), len(probs))
    comps = K.extract(probs, thr, min_area=100)
    rows, zero = [], 0
    for stem, cs in zip(names, comps):
        keep = [c for c in cs if c[0] >= area and c[1] >= mean]
        if not keep and cs:
            keep = [max(cs, key=lambda c: c[1])]          # never leave a photo empty (same as the Kaggle version)
        zero += (len(keep) == 0)
        for j, c in enumerate(keep, 1):
            rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": c[3]["counts"].decode("ascii")})
    sub = pd.DataFrame(rows)
    C.SUBS.mkdir(parents=True, exist_ok=True)
    path = C.SUBS / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"{path}: {len(sub)} rows, {len(sub) / len(names):.2f} filaments/photo, {zero} photos with none "
          f"| {kind} thr {thr} area {area} mean {mean}")
    return sub


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sub = make("check_ens20_m80", "ens20", 0.35, 400, 0.80)
        old = pd.read_csv(C.RUNS / "scripts_logs" / "sub_ens20_m80.csv")
        ph = lambda d: d.filament_id.str.rsplit("_", n=1).str[0].value_counts()
        a, b = ph(sub), ph(old)
        same = (a.reindex(b.index).fillna(0) == b).mean()
        shared = len(set(sub.segmentation_rle) & set(old.segmentation_rle)) / len(old)
        print(f"Kaggle-made file: {len(old)} rows | rebuilt: {len(sub)} rows")
        print(f"photos with the same blob count: {same:.1%} | identical outlines: {shared:.1%}")
        print("PASS" if abs(len(sub) - len(old)) / len(old) < 0.01 and same > 0.9 else "CHECK: numbers differ, tell Claude")
    elif len(sys.argv) == 6:
        make(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    else:
        print(__doc__)