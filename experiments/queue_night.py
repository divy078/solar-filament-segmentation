"""Overnight queue of fold-0 experiments (same machinery as queue_lap.py). Run:  python queue_night.py
Ordered by importance: if you stop early, the most useful runs are already done. Finished runs are skipped."""
import os, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).parent
EXPS = Path(r"C:\solar\runs\exps")
LOGS = EXPS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

JOBS = [
    dict(EXP="ctrl_s43", SEED="43"),                                   # noise floor: same recipe, another seed
    dict(EXP="ctrl_s44", SEED="44"),                                   # noise floor, second repeat
    dict(EXP="bs18_c512", BATCH="18"),                                 # 512 windows, same pixels per step as crop768
    dict(EXP="c768_s43", CROP="768", SEED="43"),                       # does the crop768 gain repeat?
    dict(EXP="c1024_b4", CROP="1024", BATCH="4"),                      # whole photo at once
    dict(EXP="fpn", ARCH="FPN", ENCODER="resnet34"),                   # other architectures (teacher Q5)
    dict(EXP="dlv3p", ARCH="DeepLabV3Plus", ENCODER="resnet34"),
    dict(EXP="unetpp", ARCH="UnetPlusPlus", ENCODER="resnet34"),       # heaviest, last
]
CLEAN = ("SMOKE", "ITERS", "EXP", "ENCODER", "CROP", "BATCH", "LR", "SEED", "LOSS", "TARGET", "RES", "FOLDS", "POS_CROP_P", "ARCH")
for j in JOBS:
    exp = j["EXP"]
    if (EXPS / exp / "oof_fold0.npy").exists():
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