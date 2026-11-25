"""Queue 3: other architectures for ensemble diversity (fold 0, laptop). Waits for queue_night.py AND queue2.py to finish.
   python queue3.py            (start it any time)
   python queue3.py now        (do not wait)"""
import os, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).parent
EXPS = Path(r"C:\solar\runs\exps")
LOGS = EXPS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

JOBS = [
    dict(EXP="segformer", ARCH="Segformer", ENCODER="mit_b2"),                 # transformer encoder (MiT-B2)
    dict(EXP="upernet_cnx", ARCH="UPerNet", ENCODER="tu-convnext_tiny"),       # ConvNeXt-Tiny encoder
    dict(EXP="manet", ARCH="MAnet", ENCODER="resnet34"),                       # attention decoder
    dict(EXP="linknet", ARCH="Linknet", ENCODER="resnet34"),
]
CLEAN = ("SMOKE", "ITERS", "EXP", "ENCODER", "CROP", "BATCH", "LR", "SEED", "LOSS", "TARGET", "RES", "FOLDS",
         "POS_CROP_P", "ARCH", "NORM", "AUG", "EMA", "EMA_DECAY", "SMALLP")


def night_running():
    cmd = ("(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and "
           "($_.CommandLine -like '*queue_night.py*' -or $_.CommandLine -like '*queue2.py*')} | Measure-Object).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out) > 0
    except Exception as e:
        print("could not check the night queue (", e, ") - starting without waiting", flush=True)
        return False


smoke = False
if not smoke and not (len(sys.argv) > 1 and sys.argv[1] == "now"):
    while night_running():
        print(f"[{time.strftime('%H:%M')}] earlier queues still running, waiting ...", flush=True)
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
print("QUEUE 3 DONE", flush=True)