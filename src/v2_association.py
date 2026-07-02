"""
v2_association.py
==================
Version 2: "People who bought X also bought Y"

Uses basket-level (order_id) association rule mining via mlxtend's FP-Growth,
which scales much better than classic Apriori on real transaction data.

"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from .base import BaseRecommender


class AssociationRecommender(BaseRecommender):
    version = "v2_association"

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._rules: pd.DataFrame | None = None      # antecedent -> [(item, lift)]
        self._rules_map: dict = {}
        self._user_items: dict[int, set] = {}

    def fit(self, interactions: pd.DataFrame) -> "AssociationRecommender":
        if "order_id" not in interactions.columns:
            raise ValueError("V2 requires an 'order_id' column to group baskets.")

        baskets = interactions.groupby("order_id")["item_id"].apply(list).tolist()

        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = fpgrowth(basket_df, min_support=self.min_support, use_colnames=True)
        if frequent_itemsets.empty:
            raise ValueError(
                "No frequent itemsets found — lower min_support. "
                f"Got {len(baskets)} baskets, {len(te.columns_)} unique items."
            )

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        rules = rules[rules["antecedents"].apply(len) == 1]  # keep simple X -> Y rules
        rules["antecedent_item"] = rules["antecedents"].apply(lambda s: next(iter(s)))
        rules["consequent_item"] = rules["consequents"].apply(lambda s: next(iter(s)))
        rules = rules.sort_values("lift", ascending=False)

        self._rules_map = {}
        for _, row in rules.iterrows():
            self._rules_map.setdefault(row["antecedent_item"], []).append(
                (row["consequent_item"], float(row["lift"]))
            )

        item_freq = interactions["item_id"].value_counts()
        self._popularity_fallback = item_freq
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
