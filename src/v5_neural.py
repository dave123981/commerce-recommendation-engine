"""
v5_neural.py
============
Version 5: Neural Recommendation (Neural Collaborative Filtering) using TensorFlow.

Embedding layers for user_id and item_id, concatenated and passed through an
MLP to predict an implicit-feedback score. Trained with negative sampling
since we only observe positive (purchased) interactions.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

from .base import BaseRecommender


class NeuralRecommender(BaseRecommender):
    version = "v5_neural"

    def __init__(
        self,
        embedding_dim: int = 32,
        epochs: int = 5,
        batch_size: int = 512,
        negative_ratio: int = 4,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.random_state = random_state

        self._user_to_idx: dict = {}
        self._item_to_idx: dict = {}
        self._idx_to_item: dict = {}
        self._all_item_idxs: np.ndarray | None = None
        self._model: Model | None = None
        self._user_items: dict[int, set] = {}

    # ------Definfing the model build parameters------ #
    def _build_model(self, n_users: int, n_items: int) -> Model:
        user_input = layers.Input(shape=(1,), name="user")
        item_input = layers.Input(shape=(1,), name="item")

        user_emb = layers.Embedding(n_users, self.embedding_dim, name="user_embedding")(user_input)
        item_emb = layers.Embedding(n_items, self.embedding_dim, name="item_embedding")(item_input)

        x = layers.Concatenate()([layers.Flatten()(user_emb), layers.Flatten()(item_emb)])
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        output = layers.Dense(1, activation="sigmoid", name="interaction_prob")(x)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _sample_negatives(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    
        n_items = len(self._item_to_idx)
        positive_keys = (df["user_idx"].astype(np.int64) * n_items + df["item_idx"].astype(np.int64)).values

        n_negatives = len(df) * self.negative_ratio
        oversample = 2  # generate extra candidates since some will collide with positives
        candidate_users = rng.integers(0, len(self._user_to_idx), size=n_negatives * oversample)
        candidate_items = rng.integers(0, len(self._item_to_idx), size=n_negatives * oversample)
        candidate_keys = candidate_users.astype(np.int64) * n_items + candidate_items.astype(np.int64)

        is_positive = pd.Series(candidate_keys).isin(positive_keys).values  # vectorized hash lookup
        valid_users = candidate_users[~is_positive][:n_negatives]
        valid_items = candidate_items[~is_positive][:n_negatives]

        return pd.DataFrame({"user_idx": valid_users, "item_idx": valid_items})

    # ---------Fitting the model-------------- #
    def fit(self, interactions: pd.DataFrame) -> "NeuralRecommender":
        tf.random.set_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        df = interactions.copy()
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {it: i for i, it in enumerate(items)}
        self._idx_to_item = {i: it for it, i in self._item_to_idx.items()}
        self._all_item_idxs = np.arange(len(items))

        df["user_idx"] = df["user_id"].map(self._user_to_idx)
        df["item_idx"] = df["item_id"].map(self._item_to_idx)

        positives = df[["user_idx", "item_idx"]].drop_duplicates().assign(label=1)
        negatives = self._sample_negatives(positives, rng).assign(label=0)
        train = pd.concat([positives, negatives]).sample(frac=1, random_state=self.random_state)

        self._model = self._build_model(len(users), len(items))
        self._model.fit(
            [train["user_idx"].values, train["item_idx"].values],
            train["label"].values,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_split=0.1,
        )

        self._popularity_fallback = df["item_id"].value_counts()
        self._user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def _known_users(self) -> set:
        return set(self._user_items.keys())

    def _seen_items(self, user_id) -> set:
        return self._user_items.get(user_id, set())

    def _recommend_for_known_user(self, user_id, n: int, exclude_items: set) -> list[dict]:
        user_idx = self._user_to_idx[user_id]
        candidate_item_idxs = self._all_item_idxs
        user_batch = np.full_like(candidate_item_idxs, user_idx)
        scores = self._model(
            [user_batch, candidate_item_idxs], training=False
        ).numpy().flatten()

        ranked_positions = np.argsort(-scores)
        results = []
        for pos in ranked_positions:
            item = self._idx_to_item[candidate_item_idxs[pos]]
            if item in exclude_items:
                continue
            results.append({"item_id": item, "score": float(scores[pos])})
            if len(results) >= n:
                break
        return results
