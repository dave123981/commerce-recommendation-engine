# Product Recommendation System — V1 to V5

A progressive series of product recommenders, each swapping in a more
sophisticated technique while implementing the **exact same API contract**
(`fit`, `recommend`, `evaluate`, `save`, `load`). The shared contract is
enforced by [`tests/test_contract.py`](tests/test_contract.py), which runs
identical assertions against every version.

| Version | Technique | Library |
|---|---|---|
| V1 | Popularity — most purchased products | pandas |
| V2 | Association rules — "people who bought X also bought Y" | mlxtend (FP-Growth) |
| V3 | Collaborative filtering — "users like you bought..." | scikit-learn (item-based k-NN) |
| V4 | Matrix factorization — latent factors | scikit-learn (TruncatedSVD) |
| V5 | Neural recommendation | TensorFlow / Keras (NCF) |

## Results

*(fill in after running `evaluate_model` for each version on the same
time-based test split — this table is the whole point of the project)*

| Version | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|---|---|---|---|---|
| V1 Popularity | | | | |
| V2 Association | | | | |
| V3 Collaborative | | | | |
| V4 Matrix Factorization | | | | |
| V5 Neural | | | | |

## Repo structure

```
recsys-project/
├── src/
│   ├── base.py                    # BaseRecommender — the shared contract
│   ├── metrics.py                 # evaluate_model(), time_based_split()
│   ├── v1_popularity.py
│   ├── v2_association.py
│   ├── v3_collaborative.py
│   ├── v4_matrix_factorization.py
│   └── v5_neural.py
├── tests/
│   └── test_contract.py           # same test suite, run against every version
├── data/
│   ├── download_data.py           # Kaggle competition download helper
│   ├── build_interactions.py      # raw Instacart CSVs -> interactions.csv
│   └── raw/                       # gitignored
├── models/                        # saved artifacts, gitignored
├── notebooks/                     # one Colab notebook per version
├── requirements.txt
└── README.md
```

## The API contract

Every version implements:

```python
model = SomeRecommender(**hyperparams)
model.fit(interactions_df)                    # same input schema for all versions
recs = model.recommend(user_id, n=10)          # -> [{"item_id": ..., "score": float}, ...]
metrics = evaluate_model(model, test_df, k=10) # -> {"precision@10": ..., "recall@10": ..., ...}
model.save("models/v1.pkl")
loaded = SomeRecommender.load("models/v1.pkl")
```

**Interactions DataFrame schema** (input to every `fit()`):

| column | type | notes |
|---|---|---|
| `user_id` | int/str | required |
| `item_id` | int/str | required |
| `order_id` | int/str | required for V2 (groups items into baskets) |
| `timestamp` | datetime | required for time-based splitting |
| `quantity` | numeric | optional, defaults to 1 |

Cold start is handled once, in `BaseRecommender`: any `user_id` unseen
during `fit()` (or a known user with too few candidate recommendations)
automatically falls back to the popularity ranking, so V5 never crashes
on a brand-new user.

## Quickstart (local)

```bash
git clone https://github.com/<you>/recsys-project.git
cd recsys-project
pip install -r requirements.txt
pytest tests/test_contract.py -v
```

```python
from src.v1_popularity import PopularityRecommender
from src.metrics import evaluate_model, time_based_split
import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["timestamp"])
train, test = time_based_split(interactions, test_frac=0.2)

model = PopularityRecommender().fit(train)
print(model.recommend(user_id=train["user_id"].iloc[0], n=10))
print(evaluate_model(model, test, k=10))
```

## Running in Google Colab

Each notebook in `notebooks/` starts the same way:

```python
# 1. Kaggle credentials
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Clone your own repo so the notebook trains against versioned source
!git clone https://github.com/<you>/recsys-project.git
%cd recsys-project
!pip install -r requirements.txt -q

# 3. Download + build the interactions table (accept competition rules on
#    kaggle.com first, one click, no submission needed)
!python data/download_data.py
!python data/build_interactions.py

# 4. Train + evaluate (identical evaluate_model call in every notebook)
from src.v1_popularity import PopularityRecommender
from src.metrics import evaluate_model, time_based_split
...
```

Save trained artifacts to Google Drive (`from google.colab import drive`)
if they're too large for GitHub; commit only the `results.csv` row each
notebook appends so the results table above stays reproducible.

## Dataset: Instacart Market Basket Analysis

Using [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis)
instead of Olist — Instacart has genuine repeat-purchase behavior (that's
literally what its `reordered` flag captures), which V3/V4/V5 need in order
to learn anything beyond "recommend whatever's popular." Olist is mostly
one-time buyers, which caps collaborative filtering and matrix
factorization at not-much-better-than-V1.

**It's a competition, not a plain dataset** — accept the rules once at
https://www.kaggle.com/c/instacart-market-basket-analysis/rules (no
submission needed), then:

```bash
python data/download_data.py                    # kaggle competitions download
python data/build_interactions.py                # raw CSVs -> interactions.csv
```

`build_interactions.py` handles three Instacart-specific quirks:

1. **No real calendar dates** — only `days_since_prior_order` per user.
   The script reconstructs a per-user *relative* timestamp by cumulatively
   summing that field from an arbitrary anchor date. Absolute dates are
   synthetic, but each user's true purchase order and spacing is preserved,
   which is all `time_based_split` needs.
2. **`eval_set == 'test'` orders have no products attached** (that's the
   original competition's held-out prediction target) — dropped.
3. **Sparse-user/item filtering** — defaults to users with >=5 orders and
   items with >=10 total purchases, tunable via `build_interactions()`
   kwargs. Tighten this if V2/V3 are slow; Instacart has ~200K users and
   ~50K products at full scale.

Output lands at `data/interactions.csv` in the shared schema every version
expects. Run `build_interactions.py`'s `add_product_names()` helper if you
want readable product names for eyeballing V2's association rules.

## Future work

- Swap V4's `TruncatedSVD` for `implicit.als.AlternatingLeastSquares`
  (proper implicit-feedback ALS)
- Two-tower architecture for V5 with side features (category, price)
- Hybrid model combining V2's association rules with V5's embeddings
