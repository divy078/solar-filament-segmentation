from pathlib import Path
import json
import cv2
import numpy as np

# ---- paths (change only this block when moving to another machine) ----
DATA = Path(r"C:\Users\Divyansh\Downloads\MAGFiLO_1.0_Kaggle_2026")
TRAIN_IMG = DATA / "train" / "train_images"
TEST_IMG = DATA / "test" / "test_images"
TRAIN_JSON = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RUNS = Path(r"C:\solar\runs")
RESULTS = Path(r"C:\solar\results")
SUBS = Path(r"C:\solar\subs")
# ------------------------------------------------------------------------

if __name__ == "__main__":
    for name, p in [("train images", TRAIN_IMG), ("test images", TEST_IMG),
                    ("train json", TRAIN_JSON), ("runs", RUNS)]:
        print(f"{name:13s} exists={p.exists()}  {p}")

    train_files = sorted(TRAIN_IMG.iterdir())
    test_files = sorted(TEST_IMG.iterdir())
    print("train photos:", len(train_files), "| test photos:", len(test_files))
    print("first train file:", train_files[0].name)
    print("first test file:", test_files[0].name)

    js = json.load(open(TRAIN_JSON))
    print("json keys:", list(js.keys()))
    print("images:", len(js["images"]), "| annotations:", len(js["annotations"]))
    print("first image entry:", js["images"][0])
    a = js["annotations"][0]
    print("first annotation keys:", list(a.keys()))

    img = cv2.imread(str(train_files[0]), cv2.IMREAD_UNCHANGED)
    print("photo shape/dtype:", img.shape, img.dtype,
          "| min", img.min(), "median", int(np.median(img)), "max", img.max())