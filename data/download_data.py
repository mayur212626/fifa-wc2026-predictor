"""
download_data.py — fetch the raw match data so the repo is reproducible.

Pulls the Mart Jurisoo international results files straight from GitHub into
data/raw/. Run this once after cloning the project:

    python data/download_data.py

Data source: https://github.com/martj42/international_results  (master branch)
Credit: Mart Jurisoo. See data/README.md for attribution/license notes.
"""

from pathlib import Path
import sys
import requests

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]

# data/raw relative to this script's location
RAW_DIR = Path(__file__).resolve().parent / "raw"


def download(filename: str) -> None:
    url = f"{BASE}/{filename}"
    dest = RAW_DIR / filename
    print(f"Downloading {filename} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"FAILED ({e})")
        return
    dest.write_bytes(resp.content)
    kb = len(resp.content) / 1024
    print(f"ok  ({kb:,.0f} KB)  ->  {dest}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving into: {RAW_DIR}\n")
    for f in FILES:
        download(f)
    # confirm the critical file landed
    if (RAW_DIR / "results.csv").exists():
        print("\nDone. results.csv is ready.")
        return 0
    print("\nWARNING: results.csv missing — check your network/proxy.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
