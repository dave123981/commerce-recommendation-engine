"""
base.py
=======
Defines the single API contract that every recommender version (V1-V5) must
implement. Nothing outside this file should ever need to know which version
is running underneath — that's the whole point of the exercise.

Interactions DataFrame contract (input to `fit`):
    user_id     : hashable (int/str)
    item_id     : hashable (int/str)
    order_id    : hashable, groups items purchased together in one basket
                  (required for V2 association rules, optional otherwise)
    timestamp   : pd.Timestamp or int, used for time-based train/test splits
    quantity    : optional numeric, defaults to 1 if absent

Recommend output contract:
    list[dict] like [{"item_id": ..., "score": float}, ...]
    - sorted descending by score
    - length <= n
    - MUST NOT raise on unseen user_id -> fall back to popularity
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class all recommender versions inherit from."""

    #: bump this in subclasses if you change a version's internal format
    version: str = "base"

    def __init__(self, **kwargs):
        self.is_fitted: bool = False
        self._popularity_fallback: pd.Series | None = None  # item_id -> count

    # ------------------------------------------------------------------ #
    # Required methods every subclass must implement
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRecommender":
        """Train the model. Must set self.is_fitted = True and return self."""
        raise NotImplementedError

    @abstractmethod
    def _recommend_for_known_user(self, user_id, n: int, exclude_items: set) -> list[dict]:
        """Version-specific recommendation logic for a user seen during fit().
        Subclasses implement only this; `recommend()` below handles the
        cold-start fallback and the shared contract so you don't repeat it
        in every version.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared logic — do not override in subclasses
    # ------------------------------------------------------------------ #
    def recommend(self, user_id, n: int = 10, exclude_seen: bool = True) -> list[dict]:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .recommend().")

        exclude_items = self._seen_items(user_id) if exclude_seen else set()

        if user_id in self._known_users():
            recs = self._recommend_for_known_user(user_id, n, exclude_items)
        else:
            recs = []  # unknown user -> straight to fallback below

        if len(recs) < n:
            recs = self._fill_with_popularity(recs, n, exclude_items | {r["item_id"] for r in recs})

        return recs[:n]

    def _fill_with_popularity(self, recs: list[dict], n: int, exclude_items: set) -> list[dict]:
        """Cold-start / sparse-result fallback shared by all versions."""
        if self._popularity_fallback is None:
            return recs
        needed = n - len(recs)
        top_up = [
            {"item_id": item, "score": float(score)}
            for item, score in self._popularity_fallback.items()
            if item not in exclude_items
        ][:needed]
        return recs + top_up

    def _known_users(self) -> set:
        raise NotImplementedError  # set by subclass during fit()

    def _seen_items(self, user_id) -> set:
        raise NotImplementedError  # set by subclass during fit()

    # ------------------------------------------------------------------ #
    # Persistence — identical across versions
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls) and not isinstance(obj, BaseRecommender):
            raise TypeError(f"{path} does not contain a BaseRecommender subclass")
        return obj
