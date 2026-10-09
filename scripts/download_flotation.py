"""Download the real flotation-plant dataset from Kaggle (optional).

The dashboard works fully without this file (it falls back to synthetic ore
quality data). When the CSV is present in data/raw/, the ore quality module
automatically uses the real dataset.

Requires a Kaggle API token:
  1. Go to https://www.kaggle.com/settings
  2. Click "Create New Token" -> saves kaggle.json to your machine
  3. Move it to ~/.kaggle/kaggle.json (Linux/macOS) or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)

Usage:
    python scripts/download_flotation.py
    python scripts/download_flotation.py --help
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "edumagalhaes/quality-prediction-in-a-mining-process"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "raw"

KAGGLE_SETUP_GUIDE = """\
Kaggle API credentials not found.

To get them:
  1. Log in at https://www.kaggle.com/settings
  2. Click "Create New Token" (downloads kaggle.json)
  3. Place it at:
       Linux/macOS : ~/.kaggle/kaggle.json
       Windows     : %USERPROFILE%\\.kaggle\\kaggle.json
  4. Re-run: python scripts/download_flotation.py

The dashboard runs fine without this file (synthetic fallback).
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the 'Quality Prediction in a Mining Process' dataset from Kaggle."
    )
    parser.add_argument("--dataset", default=DATASET, help="Kaggle dataset slug.")
    parser.add_argument("--dest", default=str(DEST_DIR), help="Destination directory.")
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        print("The 'kaggle' CLI is not installed. Install it with: pip install kaggle")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(dest), "--unzip"]
    print(f"Downloading {args.dataset} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        print("Download failed.")
        if "401" in stderr or "Unauthorized" in stderr or "kaggle.json" in stderr:
            print(KAGGLE_SETUP_GUIDE)
        else:
            print(stderr)
        return result.returncode

    csvs = sorted(dest.glob("*.csv"))
    if not csvs:
        print("Download finished but no CSV found in", dest)
        return 1
    print("Done. Files:")
    for csv in csvs:
        print(f"  {csv} ({csv.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
