"""
metrics.py
==========
One evaluation harness, used identically for every version so the
version-to-version comparison table is actually apples-to-apples.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_based_split(interactions: pd.DataFrame, test_frac: float = 0.2):
    """Hold out each user's most recent interactions as test data.
    """
    interactions = interactions.sort_values("timestamp")
    train_rows, test_rows = [], []

    for _, group in interactions.groupby("user_id"):
        n_test = max(1, int(len(group) * test_frac)) if len(group) > 1 else 0
        if n_test == 0:
            train_rows.append(group)
        else:
            train_rows.append(group.iloc[:-n_test])
            test_rows.append(group.iloc[-n_test:])

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=interactions.columns)
    return train, test


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits, score = 0, 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 1) for i, item in enumerate(top_k, start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, test_interactions: pd.DataFrame, k: int = 10) -> dict:
    """Run the four core ranking metrics for a fitted model against test data.

    """
    precisions, recalls, maps, ndcgs = [], [], [], []

    for user_id, group in test_interactions.groupby("user_id"):
        relevant = set(group["item_id"])
        recs = model.recommend(user_id=user_id, n=k, exclude_seen=True)
        recommended_ids = [r["item_id"] for r in recs]

        precisions.append(precision_at_k(recommended_ids, relevant, k))
        recalls.append(recall_at_k(recommended_ids, relevant, k))
        maps.append(average_precision_at_k(recommended_ids, relevant, k))
        ndcgs.append(ndcg_at_k(recommended_ids, relevant, k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
