"""
v4_matrix_factorization.py
============================
Version 4: Matrix Factorization (latent factors)

Uses scikit-learn's TruncatedSVD on the sparse user-item matrix. This is the
"pure sklearn" MF option so the project stays consistent with its stated
stack; swap in `implicit.als.AlternatingLeastSquares` if you want a proper
implicit-feedback ALS model later (recommended as a stretch goal — mention
it in the README as future work).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .base import BaseRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    version = "v4_matrix_factorization"

    def __init__(self, n_factors: int = 50, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_factors = n_factors
        self.random_state = random_state
        self._user_to_idx: dict = {}
        self._item_to_idx: dict = {}
        self._idx_to_item: dict = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._user_items: dict[int, set] = {}

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationRecommender":
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
        user_item = sp.csr_matrix((df["_weight"], (rows, cols)), shape=(len(users), len(items)))

        n_components = min(self.n_factors, min(user_item.shape) - 1)
        svd = TruncatedSVD(n_components=max(n_components, 2), random_state=self.random_state)
        self._user_factors = svd.fit_transform(user_item)          # (n_users, k)
        self._item_factors = svd.components_.T                    # (n_items, k)

        self._popularity_fallback = df.groupby("item_id")["_weight"].sum().sort_values(ascending=False)
        self._user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def _known_users(self) -> set:
        return set(self._user_items.keys())

    def _seen_items(self, user_id) -> set:
        return self._user_items.get(user_id, set())

    def _recommend_for_known_user(self, user_id, n: int, exclude_items: set) -> list[dict]:
        user_idx = self._user_to_idx[user_id]
        user_vec = self._user_factors[user_idx]                    # (k,)
        scores = self._item_factors @ user_vec                     # (n_items,)

        ranked_idx = np.argsort(-scores)
        results = []
        for idx in ranked_idx:
            item = self._idx_to_item[idx]
            if item in exclude_items:
                continue
            results.append({"item_id": item, "score": float(scores[idx])})
            if len(results) >= n:
                break
        return results
