"""Queue 2: the 'small improvements' screens (fold 0, laptop). Waits until the night queue has finished, then runs.
   python queue2.py smoke      (one tiny test run that uses ALL the new options, ~5 min)
   python queue2.py            (start it any time; it waits for queue_night.py to end)
   python queue2.py now        (do not wait)"""
import os, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).parent
EXPS = Path(r"C:\solar\runs\exps")
LOGS = EXPS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

JOBS = [
    dict(EXP="disk_norm", NORM="disk"),            # B1: remove limb darkening per photo
    dict(EXP="ema", EMA="1"),                      # B9: average weights over the last ~1000 steps
    dict(EXP="aug_strong", AUG="strong"),          # more varied training windows: any rotation + zoom
    dict(EXP="small50", SMALLP="0.5"),             # B8: aim half the windows at small filaments
    dict(EXP="pos90", POS_CROP_P="0.9"),           # B3: 90% of windows centred on filaments (was 70%)
    dict(EXP="pos50", POS_CROP_P="0.5"),           # B3: 50%
]
SMOKE_JOBS = [dict(EXP="smoke_v2", SMOKE="1", ITERS="25", NORM="disk", EMA="1", AUG="strong", SMALLP="0.5")]
CLEAN = ("SMOKE", "ITERS", "EXP", "ENCODER", "CROP", "BATCH", "LR", "SEED", "LOSS", "TARGET", "RES", "FOLDS",
         "POS_CROP_P", "ARCH", "NORM", "AUG", "EMA", "EMA_DECAY", "SMALLP")


def night_running():
    cmd = ("(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and "
           "$_.CommandLine -like '*queue_night.py*'} | Measure-Object).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out) > 0
    except Exception as e:
        print("could not check the night queue (", e, ") - starting without waiting", flush=True)
        return False


smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
if not smoke and not (len(sys.argv) > 1 and sys.argv[1] == "now"):
    while night_running():
        print(f"[{time.strftime('%H:%M')}] night queue still running, waiting ...", flush=True)
        time.sleep(300)

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
        rc = subprocess.call([sys.executable, str(SRC / "pipeline_lap2.py")], env=env, stdout=lf, stderr=subprocess.STDOUT)
    lines = open(LOGS / f"{exp}.log", errors="replace").read().splitlines()
    res = [l for l in lines if l.startswith("RESULT")]
    print(f"[{time.strftime('%H:%M')}] {exp}: exit code {rc}, {(time.time() - t) / 60:.1f} min", flush=True)
    print("   " + (res[-1] if res else "NO RESULT. last log lines: " + " | ".join(lines[-4:])), flush=True)
print("QUEUE 2 DONE", flush=True)