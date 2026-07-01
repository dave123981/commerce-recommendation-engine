"""
download_data.py
=================
Downloads the raw dataset from Kaggle. Requires a Kaggle API token at
~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY env vars set) --
in Colab, upload kaggle.json and run the setup cell shown in the README.

Usage:
    python data/download_data.py --dataset olistbr/brazilian-ecommerce
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"


def download(dataset: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(RAW_DIR)],
        check=True,
    )
    for zip_path in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()
    print(f"Downloaded and extracted to {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="olistbr/brazilian-ecommerce",
        help="Kaggle dataset slug, e.g. 'olistbr/brazilian-ecommerce'",
    )
    args = parser.parse_args()
    download(args.dataset)
