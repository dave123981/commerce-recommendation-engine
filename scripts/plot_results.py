"""
plot_results.py
================
Generates the version-comparison bar chart from results.csv.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-csv path/to/results.csv --output path/to/chart.png

Expects results.csv with columns:
    version, precision@10, recall@10, map@10, ndcg@10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path(__file__).parent.parent / "results.csv"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "version_comparison.png"

# consistent color-per-version across chart regenerations
VERSION_COLORS = {
    "v1_popularity": "#94a3b8",
    "v2_association": "#f87171",
    "v3_collaborative": "#34d399",
    "v4_matrix_factorization": "#60a5fa",
    "v5_neural": "#a78bfa",
}
VERSION_LABELS = {
    "v1_popularity": "V1 Popularity",
    "v2_association": "V2 Association",
    "v3_collaborative": "V3 Collaborative",
    "v4_matrix_factorization": "V4 Matrix Fact.",
    "v5_neural": "V5 Neural",
}
METRIC_LABELS = {
    "precision@10": "Precision@10",
    "recall@10": "Recall@10",
    "map@10": "MAP@10",
    "ndcg@10": "NDCG@10",
}


def plot_results(results_csv: Path, output_path: Path) -> None:
    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found. Run each version's fit + evaluate_model() cell "
            "and log a row to results.csv first (see README quickstart)."
        )

    df = pd.read_csv(results_csv)
    metrics = [c for c in df.columns if c != "version"]

    versions = df["version"].tolist()
    labels = [VERSION_LABELS.get(v, v) for v in versions]
    colors = [VERSION_COLORS.get(v, "#999999") for v in versions]

    fig, ax = plt.subplots(figsize=(11, 6))
    n_versions = len(versions)
    n_metrics = len(metrics)
    bar_width = 0.8 / max(n_versions, 1)
    x = np.arange(n_metrics)

    for i, (version, label, color) in enumerate(zip(versions, labels, colors)):
        values = df.loc[df["version"] == version, metrics].values.flatten()
        offset = (i - (n_versions - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=label, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_ylabel("Score")
    ax.set_title("Recommender Version Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, df[metrics].values.max() * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_results(args.results_csv, args.output)