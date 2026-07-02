"""
download_data.py
=================

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
