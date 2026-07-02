"""
v1_popularity.py
================
Version 1: "Most purchased products"

Simplest possible baseline. Every version after this one is judged against
it
Includes a recency-decay option so it's not a completely static baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    version = "v1_popularity"

    def __init__(self, half_life_days: float | None = None, **kwargs):
        """
        half_life_days: if set, applies exponential recency decay so recent
        purchases count more than old ones. If None, plain purchase counts.
        """
        super().__init__(**kwargs)
        self.half_life_days = half_life_days
        self._user_items: dict[int, set] = {}
        self._scores: pd.Series | None = None  # item_id -> score, sorted desc

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        df = interactions.copy()

        if self.half_life_days:
            now = pd.to_datetime(df["timestamp"]).max()
            age_days = (now - pd.to_datetime(df["timestamp"])).dt.total_seconds() / 86400
            weight = np.power(0.5, age_days / self.half_life_days)
        else:
            weight = 1.0

        df["_weight"] = weight
        scores = df.groupby("item_id")["_weight"].sum().sort_values(ascending=False)

        self._scores = scores
        self._popularity_fallback = scores  # V1 IS the fallback
        self._user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def _known_users(self) -> set:
        return set(self._user_items.keys())

    def _seen_items(self, user_id) -> set:
        return self._user_items.get(user_id, set())

    def _recommend_for_known_user(self, user_id, n: int, exclude_items: set) -> list[dict]:
        # Popularity is global -> "known user" path is identical to fallback,
        # just filtered by that user's own seen items.
        results = []
        for item, score in self._scores.items():
            if item in exclude_items:
                continue
            results.append({"item_id": item, "score": float(score)})
            if len(results) >= n:
                break
        return results
