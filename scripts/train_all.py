"""Train all Phase 2 ML models in sequence (Task 7).

Runs the five train scripts one after another as subprocesses with the same
Python interpreter (``sys.executable``), each with ``cwd`` set to the repo
root. Stops at the first failing script with a clear error message.

Usage (from the repo root)::

    python scripts/train_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPTS = [
    "src/models/train_forecast.py",
    "src/models/train_failure.py",
    "src/models/train_anomaly.py",
    "src/models/train_cost.py",
    "src/models/train_ore_quality.py",
]


def run_script(script: str) -> float:
    """Run one train script; return elapsed seconds (raises on failure)."""
    print(f"=== {script} ===")
    start = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-15:]) or "(no stderr captured)"
        raise RuntimeError(f"FAILED at {script} (exit {proc.returncode}):\n{tail}")
    out = (proc.stdout or "").strip()
    if out:
        print(out)
    print(f"OK ({elapsed:.1f}s)\n")
    return elapsed


def main() -> None:
    print(f"train_all: running {len(TRAIN_SCRIPTS)} train scripts (cwd={REPO_ROOT})\n")
    timings: dict[str, float] = {}
    try:
        for script in TRAIN_SCRIPTS:
            timings[script] = run_script(script)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    total = sum(timings.values())
    print("=== train_all: SUCCESS — all 5 models trained ===")
    for script, elapsed in timings.items():
        print(f"  {script}: {elapsed:.1f}s")
    print(f"  total: {total:.1f}s")


if __name__ == "__main__":
    main()
