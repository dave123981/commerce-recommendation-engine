"""
v2_association.py
==================
Version 2: "People who bought X also bought Y"

Uses basket-level (order_id) association rule mining via mlxtend's FP-Growth,
which scales much better than classic Apriori on real transaction data.

Computes pairwise association rules (X -> Y) directly from a sparse
item-item co-occurrence matrix, using only scipy.sparse (a scikit-learn
dependency)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import BaseRecommender


class AssociationRecommender(BaseRecommender):
    version = "v2_association"

    def __init__(
        self,
        min_support: float = 0.0,
        min_confidence: float = 0.1,
        min_lift: float = 1.0,
        min_cooccurrence: int = 10,
        max_items: int | None = None,
        top_k_rules_per_item: int = 50,
        **kwargs,
    ):


# min_support: fraction-of-all-baskets threshold. 
# min_cooccurrence: minimum RAW COUNT of baskets containing both items. 
# min_lift / min_confidence: relevance filters, applied only to pairs that already cleared min_cooccurrence.


        super().__init__(**kwargs)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.min_cooccurrence = min_cooccurrence
        self.max_items = max_items  # optional safety cap on catalog size
        self.top_k_rules_per_item = top_k_rules_per_item  # bounds memory of the rules lookup
        self._rules_map: dict = {}
        self._user_items: dict[int, set] = {}

    def fit(self, interactions: pd.DataFrame) -> "AssociationRecommender":
        if "order_id" not in interactions.columns:
            raise ValueError("V2 requires an 'order_id' column to group baskets.")

        item_freq_full = interactions["item_id"].value_counts()

        df = interactions
        if self.max_items:
            top_items = set(item_freq_full.head(self.max_items).index)
            df = interactions[interactions["item_id"].isin(top_items)]

        # dedupe (order_id, item_id): an item shouldn't count twice toward
        # co-occurrence just because it had a quantity/reorder flag row
        pairs = df[["order_id", "item_id"]].drop_duplicates()

        orders = pairs["order_id"].unique()
        items = pairs["item_id"].unique()
        order_to_idx = {o: i for i, o in enumerate(orders)}
        item_to_idx = {it: i for i, it in enumerate(items)}
        idx_to_item = {i: it for it, i in item_to_idx.items()}

        rows = pairs["order_id"].map(order_to_idx).values
        cols = pairs["item_id"].map(item_to_idx).values
        data = np.ones(len(pairs), dtype=np.int32)
        basket_item = sp.csr_matrix((data, (rows, cols)), shape=(len(orders), len(items)))

        n_baskets = basket_item.shape[0]
        item_basket_counts = np.asarray(basket_item.sum(axis=0)).flatten()  # freq per item

        cooccurrence = (basket_item.T @ basket_item).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()

        coo = cooccurrence.tocoo()
        support = coo.data / n_baskets
        antecedent_freq = item_basket_counts[coo.row]
        consequent_freq = item_basket_counts[coo.col]
        confidence = coo.data / antecedent_freq
        lift = (coo.data * n_baskets) / (antecedent_freq * consequent_freq)

        mask = (
            (coo.data >= self.min_cooccurrence)
            & (support >= self.min_support)
            & (confidence >= self.min_confidence)
            & (lift >= self.min_lift)
        )
        if not mask.any():
            raise ValueError(
                "No rules passed the min_cooccurrence/min_support/min_confidence/min_lift thresholds — "
                "try lowering min_cooccurrence or min_confidence. "
                f"Got {n_baskets:,} baskets, {len(items):,} items considered."
            )

        rules_map: dict = {}
        for r, c, lft in zip(coo.row[mask], coo.col[mask], lift[mask]):
            antecedent_item = idx_to_item[r]
            consequent_item = idx_to_item[c]
            rules_map.setdefault(antecedent_item, []).append((consequent_item, float(lft)))

        for antecedent in rules_map:
            rules_map[antecedent] = sorted(
                rules_map[antecedent], key=lambda x: x[1], reverse=True
            )[: self.top_k_rules_per_item]

        self._rules_map = rules_map
        # popularity fallback uses the FULL catalog even if max_items capped mining,
        # so cold-start users still get sensible recs beyond the mined item set
        self._popularity_fallback = item_freq_full
        self._user_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def _known_users(self) -> set:
        return set(self._user_items.keys())

    def _seen_items(self, user_id) -> set:
        return self._user_items.get(user_id, set())

    def _recommend_for_known_user(self, user_id, n: int, exclude_items: set) -> list[dict]:
        seen = self._user_items.get(user_id, set())
        candidates: dict = {}
        for item in seen:
            for consequent, lift in self._rules_map.get(item, []):
                if consequent in exclude_items:
                    continue
                candidates[consequent] = max(candidates.get(consequent, 0.0), lift)

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"item_id": item, "score": score} for item, score in ranked]

    def related_items(self, item_id, n: int = 10) -> list[dict]:
        """Bonus method (outside the base contract): 'people who bought X also
        bought' lookup for a single product, useful for product-page widgets.
        """
        rules = sorted(self._rules_map.get(item_id, []), key=lambda x: x[1], reverse=True)[:n]
        return [{"item_id": item, "score": score} for item, score in rules]       