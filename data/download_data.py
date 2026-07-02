"""
download_data.py
=================
Downloads the raw Instacart Market Basket Analysis data from Kaggle.
Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars set) -- in Colab, upload kaggle.json and run the setup
cell shown in the README.

Instacart is a COMPETITION, not a plain dataset, so it uses
`kaggle competitions download`, and you must first accept the competition
rules at https://www.kaggle.com/c/instacart-market-basket-analysis/rules
(one click, no submission required) or the download will 403.

The competition also ships several of its CSVs as nested zips
(e.g. order_products__prior.csv.zip), so extraction recurses one level.

Usage:
    python data/download_data.py
    python data/download_data.py --competition instacart-market-basket-analysis
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"


def _extract_all(directory: Path) -> None:
    for zip_path in list(directory.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(directory)
        zip_path.unlink()
    remaining = list(directory.glob("*.zip"))
    if remaining:
        _extract_all(directory)


def download(competition: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(RAW_DIR)],
        check=True,
    )
    _extract_all(RAW_DIR)
    expected = {"orders.csv", "products.csv", "order_products__prior.csv", "order_products__train.csv"}
    found = {p.name for p in RAW_DIR.glob("*.csv")}
    missing = expected - found
    if missing:
        print(f"WARNING: expected files not found after extraction: {missing}")
    print(f"Downloaded and extracted to {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition",
        default="instacart-market-basket-analysis",
        help="Kaggle competition slug (must accept rules on Kaggle's site first)",
    )
    args = parser.parse_args()
    download(args.competition)
