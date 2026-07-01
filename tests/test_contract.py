"""
test_contract.py
=================
The single most important file in this repo. It runs the SAME suite of
assertions against every version (V1-V5), proving the "same API contract"
claim in the README rather than just asserting it in prose.

Run with:  pytest tests/test_contract.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v1_popularity import PopularityRecommender
from src.v2_association import AssociationRecommender
from src.v3_collaborative import CollaborativeRecommender
from src.v4_matrix_factorization import MatrixFactorizationRecommender

# V5 needs TensorFlow — skip cleanly if it isn't installed in this environment
try:
    from src.v5_neural import NeuralRecommender
    HAS_TF = True
except ImportError:
    HAS_TF = False


@pytest.fixture
def sample_interactions():
    """Small synthetic dataset with an obvious repeated pattern (items 1 & 2
    always co-occur) so association rules / CF have something to find.
    """
    rng = np.random.default_rng(42)
    rows = []
    order_id = 0
    for user in range(1, 21):
        for _ in range(rng.integers(3, 8)):
            order_id += 1
            basket = [1, 2] if rng.random() < 0.5 else list(rng.choice(range(1, 16), size=3, replace=False))
            for item in basket:
                rows.append(
                    {
                        "user_id": user,
                        "item_id": int(item),
                        "order_id": order_id,
                        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=order_id),
                        "quantity": 1,
                    }
                )
    return pd.DataFrame(rows)


def _versions():
    versions = [
        (PopularityRecommender, {}),
        (AssociationRecommender, {"min_support": 0.02, "min_confidence": 0.1}),
        (CollaborativeRecommender, {"n_neighbors": 5}),
        (MatrixFactorizationRecommender, {"n_factors": 5}),
    ]
    if HAS_TF:
        versions.append((NeuralRecommender, {"epochs": 1, "embedding_dim": 8}))
    return versions


@pytest.mark.parametrize("RecClass,kwargs", _versions())
def test_fit_returns_self_and_sets_fitted_flag(RecClass, kwargs, sample_interactions):
    model = RecClass(**kwargs)
    result = model.fit(sample_interactions)
    assert result is model
    assert model.is_fitted is True


@pytest.mark.parametrize("RecClass,kwargs", _versions())
def test_recommend_output_shape(RecClass, kwargs, sample_interactions):
    model = RecClass(**kwargs).fit(sample_interactions)
    recs = model.recommend(user_id=1, n=5)

    assert isinstance(recs, list)
    assert len(recs) <= 5
    for r in recs:
        assert set(r.keys()) == {"item_id", "score"}
        assert isinstance(r["score"], float)


@pytest.mark.parametrize("RecClass,kwargs", _versions())
def test_recommend_handles_unknown_user_gracefully(RecClass, kwargs, sample_interactions):
    """Cold start: an unseen user_id must never raise, and should fall back
    to the popularity ranking baked into every version via BaseRecommender.
    """
    model = RecClass(**kwargs).fit(sample_interactions)
    recs = model.recommend(user_id=999999, n=5)
    assert isinstance(recs, list)
    assert len(recs) <= 5


@pytest.mark.parametrize("RecClass,kwargs", _versions())
def test_exclude_seen_removes_purchased_items(RecClass, kwargs, sample_interactions):
    model = RecClass(**kwargs).fit(sample_interactions)
    user_id = sample_interactions["user_id"].iloc[0]
    seen = set(sample_interactions.loc[sample_interactions["user_id"] == user_id, "item_id"])

    recs = model.recommend(user_id=user_id, n=20, exclude_seen=True)
    recommended_ids = {r["item_id"] for r in recs}
    assert recommended_ids.isdisjoint(seen)


@pytest.mark.parametrize("RecClass,kwargs", _versions())
def test_recommend_before_fit_raises(RecClass, kwargs):
    model = RecClass(**kwargs)
    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, n=5)


@pytest.mark.parametrize("RecClass,kwargs", _versions())
def test_save_and_load_roundtrip(RecClass, kwargs, sample_interactions, tmp_path):
    model = RecClass(**kwargs).fit(sample_interactions)
    save_path = tmp_path / f"{model.version}.pkl"
    model.save(save_path)

    loaded = RecClass.load(save_path)
    assert loaded.is_fitted is True

    original_recs = model.recommend(user_id=1, n=5)
    loaded_recs = loaded.recommend(user_id=1, n=5)
    assert [r["item_id"] for r in original_recs] == [r["item_id"] for r in loaded_recs]
