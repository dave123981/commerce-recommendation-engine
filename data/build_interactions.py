"""
build_interactions.py
======================


Usage:
    python data/build_interactions.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
OUT_PATH = Path(__file__).parent / "interactions.csv"

# Arbitrary anchor date -- only relative spacing/order matters downstream
REFERENCE_DATE = pd.Timestamp("2020-01-01")


def build_interactions(
    min_orders_per_user: int = 5,
    min_purchases_per_item: int = 10,
) -> pd.DataFrame:
    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        dtype={
            "order_id": "int32",
            "user_id": "int32",
            "order_number": "int16",
            "order_dow": "int8",
            "order_hour_of_day": "int8",
        },
    )
    orders = orders[orders["eval_set"] != "test"].copy()

    prior = pd.read_csv(RAW_DIR / "order_products__prior.csv", dtype={"order_id": "int32", "product_id": "int32"})
    train = pd.read_csv(RAW_DIR / "order_products__train.csv", dtype={"order_id": "int32", "product_id": "int32"})
    order_products = pd.concat([prior, train], ignore_index=True)[["order_id", "product_id"]]

    # --- reconstruct a per-user relative timestamp -----------------------
    orders = orders.sort_values(["user_id", "order_number"])
    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0)
    orders["_cum_days"] = orders.groupby("user_id")["days_since_prior_order"].cumsum()
    orders["timestamp"] = REFERENCE_DATE + pd.to_timedelta(orders["_cum_days"], unit="D")

    # --- join orders -> products -> build interactions --------------------
    interactions = order_products.merge(
        orders[["order_id", "user_id", "timestamp"]], on="order_id", how="inner"
    )
    interactions = interactions.rename(columns={"product_id": "item_id"})
    interactions["quantity"] = 1
    interactions = interactions[["user_id", "item_id", "order_id", "timestamp", "quantity"]]

    # --- filter sparse users/items (see README EDA step) -----------------
    orders_per_user = interactions.groupby("user_id")["order_id"].nunique()
    keep_users = orders_per_user[orders_per_user >= min_orders_per_user].index
    interactions = interactions[interactions["user_id"].isin(keep_users)]

    purchases_per_item = interactions.groupby("item_id")["order_id"].nunique()
    keep_items = purchases_per_item[purchases_per_item >= min_purchases_per_item].index
    interactions = interactions[interactions["item_id"].isin(keep_items)]

    return interactions.reset_index(drop=True)


def add_product_names(interactions: pd.DataFrame) -> pd.DataFrame:
    """Optional: attach product_name for readable qualitative checks
    (e.g. eyeballing V2's association rules). Not part of the core schema.
    """
    products = pd.read_csv(RAW_DIR / "products.csv", dtype={"product_id": "int32"})
    return interactions.merge(
        products[["product_id", "product_name"]], left_on="item_id", right_on="product_id", how="left"
    ).drop(columns="product_id")


if __name__ == "__main__":
    interactions = build_interactions()
    print(f"Built {len(interactions):,} interactions | "
          f"{interactions['user_id'].nunique():,} users | "
          f"{interactions['item_id'].nunique():,} items | "
          f"{interactions['order_id'].nunique():,} orders")
    interactions.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
