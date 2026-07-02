"""
download_data.py
=================

"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

EXPECTED_FILES = {"orders.csv", "products.csv", "order_products__prior.csv", "order_products__train.csv"}


def _check_expected(directory: Path) -> None:
    found = {p.name for p in directory.glob("*.csv")}
    missing = EXPECTED_FILES - found
    if missing:
        print(f"WARNING: expected files not found after download: {missing}")


def download_via_kagglehub(dataset: str = "psparks/instacart-market-basket-analysis") -> None:
    import kagglehub  # pip install kagglehub

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(dataset))
    print(f"kagglehub cached the dataset at {cache_path}")

    # Some mirrors nest CSVs one level deep -- copy everything found, flat,
    # into data/raw/ so build_interactions.py always has one known location.
    for csv_path in cache_path.rglob("*.csv"):
        shutil.copy2(csv_path, RAW_DIR / csv_path.name)

    _check_expected(RAW_DIR)
    print(f"Copied CSVs to {RAW_DIR}")


def _extract_all(directory: Path) -> None:
    for zip_path in list(directory.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(directory)
        zip_path.unlink()
    if list(directory.glob("*.zip")):
        _extract_all(directory)


def download_via_competition(competition: str = "instacart-market-basket-analysis") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(RAW_DIR)],
        check=True,
    )
    _extract_all(RAW_DIR)
    _check_expected(RAW_DIR)
    print(f"Downloaded and extracted to {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["kagglehub", "competition"], default="kagglehub")
    parser.add_argument(
        "--slug",
        default=None,
        help="Override the dataset slug (kagglehub) or competition slug (competition)",
    )
    args = parser.parse_args()

    if args.method == "kagglehub":
        download_via_kagglehub(args.slug or "psparks/instacart-market-basket-analysis")
    else:
        download_via_competition(args.slug or "instacart-market-basket-analysis")
