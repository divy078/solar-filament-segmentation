"""Run a list of experiments one after another (fold 0 screens). Usage:
    python queue_lap.py smoke    -> 2 tiny test runs (checks everything works, ~5 min)
    python queue_lap.py          -> the real queue (skips experiments that already finished)"""
import os, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).parent
EXPS = Path(r"C:\solar\runs\exps")
LOGS = EXPS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

JOBS = [
    dict(EXP="lap_control"),                                          # same recipe as Kaggle: does the laptop reproduce it?
    dict(EXP="crop384", CROP="384"),                                  # window size (teacher Q1): smaller windows
    dict(EXP="r18", ENCODER="resnet18"),                              # ResNet depth down (Q3)
    dict(EXP="effb0", ENCODER="tu-tf_efficientnetv2_b0"),             # EfficientNet family, smaller (Q4)
    dict(EXP="crop256", CROP="256"),                                  # even smaller windows
    dict(EXP="crop768", CROP="768"),                                  # larger windows
    dict(EXP="r101", ENCODER="resnet101"),                            # ResNet depth up
]
SMOKE_JOBS = [dict(EXP="smoke_r34", SMOKE="1", ITERS="25"),
              dict(EXP="smoke_eff", SMOKE="1", ITERS="25", ENCODER="tu-tf_efficientnetv2_s")]

CLEAN = ("SMOKE", "ITERS", "EXP", "ENCODER", "CROP", "BATCH", "LR", "SEED", "LOSS", "TARGET", "RES", "FOLDS", "POS_CROP_P")
smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
for j in (SMOKE_JOBS if smoke else JOBS):
    exp = j["EXP"]
    if not smoke and (EXPS / exp / "oof_fold0.npy").exists():
        print(f"{exp}: already finished, skipping", flush=True)
        continue
    env = {k: v for k, v in os.environ.items() if k not in CLEAN}
    env.update({"FOLDS": "0", "PYTHONUNBUFFERED": "1", **j})
    t = time.time()
    print(f"[{time.strftime('%H:%M')}] starting {exp} ...", flush=True)
    with open(LOGS / f"{exp}.log", "w") as lf:
        rc = subprocess.call([sys.executable, str(SRC / "pipeline_lap.py")], env=env, stdout=lf, stderr=subprocess.STDOUT)
    lines = open(LOGS / f"{exp}.log", errors="replace").read().splitlines()
    res = [l for l in lines if l.startswith("RESULT")]
    print(f"[{time.strftime('%H:%M')}] {exp}: exit code {rc}, {(time.time() - t) / 60:.1f} min", flush=True)
    print("   " + (res[-1] if res else "NO RESULT. last log lines: " + " | ".join(lines[-4:])), flush=True)
print("QUEUE DONE", flush=True)