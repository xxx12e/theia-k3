"""
THEIA paper — overnight experiment orchestrator.

Runs 5 new experiments sequentially to strengthen the Transformer baseline
reliability claim:

  Matched protocol (extends n=5 → n=8):
    seeds 31415, 27182, 14142  ×  tf8l_5seed recipe
    (AdamW lr=1e-3, cosine, batch 4096, no warmup)

  Tuned protocol (extends n=1 → n=3):
    seeds 123, 256  ×  tf8l_tuned_seed42 recipe
    (AdamW lr=1e-4, β2=0.98, 5-epoch warmup, cosine, batch 2048, grad clip 1.0)

All runs log stdout to overnight_logs/ and write JSON results to
overnight_results/. Morning aggregation: run overnight_report.py.

================================================================
IMPORTANT: BEFORE STARTING
================================================================
You MUST verify the CLI interface of your existing scripts below.
Open each script and confirm:
  1. It accepts --seed N as an argument, OR
  2. It has a SEED = ... line at the top that you can sed-replace

If neither is true, scroll to `run_matched()` and `run_tuned()` and
adapt the cmd[] list to match your actual invocation pattern.

The assumed scripts are:
    MATCHED_SCRIPT = 'tf8l_5seed.py'
    TUNED_SCRIPT   = 'tf8l_tuned_seed42.py'

If your paths differ, edit the constants below.
================================================================
"""
# NOTE: the recipe described above misstates the callee's actual configuration
# (tf8l_5seed.py runs lr=5e-4, batch=2048); see paper App. H, Correction.

import subprocess
import sys
import os
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

# --- configure: paths to the training scripts ---
REPO_ROOT = Path(".")  # adjust if running from elsewhere
MATCHED_SCRIPT = REPO_ROOT / "tf8l_5seed.py"
TUNED_SCRIPT   = REPO_ROOT / "tf8l_tuned_seed42.py"
PYTHON = sys.executable  # use the same Python you're launching this with

MATCHED_SEEDS = [31415, 27182, 14142]   # 3 new seeds, extends n=5 → n=8
TUNED_SEEDS   = [123, 256]              # 2 new seeds, extends n=1 → n=3

# Output dirs (created automatically)
LOG_DIR    = Path("overnight_logs")
RESULT_DIR = Path("overnight_results")


def run_one(label: str, cmd: list, env_extra: dict = None):
    """Run one experiment, stream stdout to log file, record wall-clock."""
    LOG_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] START {label}")
    print(f"  cmd: {' '.join(str(x) for x in cmd)}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# {label}\n# cmd: {' '.join(str(x) for x in cmd)}\n# started: {datetime.now()}\n\n")
        lf.flush()
        try:
            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
            rc = result.returncode
        except FileNotFoundError as e:
            rc = -1
            lf.write(f"\n\n[ERROR] FileNotFoundError: {e}\n")

    dur = time.time() - start
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] END   {label}  [{status}]  {dur/60:.1f} min")

    # Persist a tiny meta file so overnight_report.py can find everything
    meta = {
        "label": label,
        "cmd": [str(x) for x in cmd],
        "returncode": rc,
        "duration_min": round(dur / 60, 2),
        "log": str(log_path),
        "ended": datetime.now().isoformat(timespec="seconds"),
    }
    (RESULT_DIR / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    return rc == 0


def run_matched():
    """3 new Transformer seeds under matched protocol."""
    if not MATCHED_SCRIPT.exists():
        print(f"[SKIP matched] {MATCHED_SCRIPT} not found")
        return
    for seed in MATCHED_SEEDS:
        label = f"matched_seed{seed}"
        # tf8l_5seed.py uses --output-root (not --output-dir), and it
        # creates seed_N/ subdirectory inside output-root automatically.
        cmd = [PYTHON, str(MATCHED_SCRIPT), "--seed", str(seed),
               "--output-root", str(RESULT_DIR)]
        run_one(label, cmd)


def run_tuned():
    """2 new Transformer seeds under tuned (Transformer-standard) protocol.

    tf8l_tuned_seed42.py has no CLI args — SEED, OUT_DIR, and filenames are
    hardcoded.  We do in-place sed-replace: swap SEED and output paths, run,
    then restore the original file.  Ugly but safe (try/finally guarantees
    restore even on crash).
    """
    if not TUNED_SCRIPT.exists():
        print(f"[SKIP tuned] {TUNED_SCRIPT} not found")
        return

    import re
    original_text = TUNED_SCRIPT.read_text(encoding="utf-8")

    for seed in TUNED_SEEDS:
        label = f"tuned_seed{seed}"
        out_dir = str(RESULT_DIR / label).replace("\\", "/")

        # Patch the script text for this seed
        patched = original_text
        # 1. Replace SEED = 42 with SEED = <new>
        patched = re.sub(r'SEED\s*=\s*42\b', f'SEED = {seed}', patched)
        # 2. Replace OUT_DIR path
        patched = re.sub(
            r'OUT_DIR\s*=\s*"[^"]*"',
            f'OUT_DIR = "{out_dir}"',
            patched,
        )
        # 3. Replace seed42 in filenames → seed<N>
        patched = patched.replace("seed42_result.json", f"seed{seed}_result.json")
        patched = patched.replace("seed42_log.txt", f"seed{seed}_log.txt")
        patched = patched.replace("seed42_result", f"seed{seed}_result")

        TUNED_SCRIPT.write_text(patched, encoding="utf-8")
        try:
            cmd = [PYTHON, str(TUNED_SCRIPT)]
            run_one(label, cmd)
        finally:
            # Always restore the original, even if the run crashes
            TUNED_SCRIPT.write_text(original_text, encoding="utf-8")


def main():
    print(f"THEIA overnight runner — started {datetime.now()}")
    print(f"Python: {PYTHON}")
    print(f"Matched script: {MATCHED_SCRIPT}  exists={MATCHED_SCRIPT.exists()}")
    print(f"Tuned script:   {TUNED_SCRIPT}  exists={TUNED_SCRIPT.exists()}")
    print(f"Matched seeds:  {MATCHED_SEEDS}")
    print(f"Tuned seeds:    {TUNED_SEEDS}")
    print()
    print("Estimated total wall-clock:")
    print("  3 matched × ~45 min  +  2 tuned × ~45 min  =  ~3.75 hours")
    print()

    overall_start = time.time()
    run_matched()
    run_tuned()
    overall_dur = (time.time() - overall_start) / 60

    print(f"\n{'='*60}")
    print(f"ALL DONE — total {overall_dur:.1f} min")
    print(f"Logs in:    {LOG_DIR.resolve()}")
    print(f"Results in: {RESULT_DIR.resolve()}")
    print(f"Now run: python overnight_report.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
