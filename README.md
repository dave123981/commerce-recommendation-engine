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
│   └── test_contract.py           
├── data/
│   ├── download_data.py           
│   └── raw/                       
├── models/                        
├── notebooks/                     
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
during `fit()`.
