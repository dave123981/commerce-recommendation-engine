"""
v3_collaborative.py
====================
Version 3: "Users like you also bought..."

Item-based k-NN collaborative filtering using cosine similarity over a
sparse user-item matrix. Item-based (not user-based) because it's more
stable at e-commerce scale — item similarity changes slowly, user
similarity churns constantly as people buy things.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    version = "v3_collaborative"

    def __init__(self, n_neighbors: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self._user_to_idx: dict = {}
        self._item_to_idx: dict = {}
        self._idx_to_item: dict = {}
        self._user_item: sp.csr_matrix | None = None
        self._knn: NearestNeighbors | None = None
        self._user_items: dict[int, set] = {}

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeRecommender":
        df = interactions.copy()
        qty = df["quantity"] if "quantity" in df.columns else pd.Series(1, index=df.index)
        df["_weight"] = qty

        users = df["user_id"].unique()
        items = df["item_id"].unique()
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {it: i for i, it in enumerate(items)}
        self._idx_to_item = {i: it for it, i in self._item_to_idx.items()}

        rows = df["user_id"].map(self._user_to_idx)
        cols = df["item_id"].map(self._item_to_idx)
        self._user_item = sp.csr_matrix(
            (df["_weight"], (rows, cols)), shape=(len(users), len(items))
        )

        # item-item similarity via k-NN on the transposed (item x user) matrix
        item_user = self._user_item.T.tocsr()
        self._knn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, item_user.shape[0]), metric="cosine"
        )
        self._knn.fit(item_user)

        self._popularity_fallback = df.groupby("item_id")["_weight"].sum().sort_values(ascending=False)
        self._user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def _known_users(self) -> set:
        return set(self._user_items.keys())

    def _seen_items(self, user_id) -> set:
        return self._user_items.get(user_id, set())

    def _recommend_for_known_user(self, user_id, n: int, exclude_items: set) -> list[dict]:
        seen_items = self._user_items.get(user_id, set())
        scores: dict = {}

        for item in seen_items:
            item_idx = self._item_to_idx.get(item)
            if item_idx is None:
                continue
            distances, neighbor_idxs = self._knn.kneighbors(
                self._user_item.T[item_idx], n_neighbors=min(self.n_neighbors + 1, self._user_item.shape[1])
            )
            for dist, n_idx in zip(distances[0], neighbor_idxs[0]):
                neighbor_item = self._idx_to_item[n_idx]
                if neighbor_item in exclude_items or neighbor_item == item:
                    continue
                similarity = 1 - dist
                scores[neighbor_item] = scores.get(neighbor_item, 0.0) + similarity

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"item_id": item, "score": float(score)} for item, score in ranked]
