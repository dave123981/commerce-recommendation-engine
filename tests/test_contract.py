import pytest
from src.v1_popularity import PopularityRecommender
from src.v2_association import AssociationRecommender
from src.v3_collaborative import CollaborativeRecommender
from src.v4_matrix_factorization import MatrixFactorizationRecommender
from src.v5_neural import NeuralRecommender

ALL_VERSIONS = [PopularityRecommender, AssociationRecommender, CollaborativeRecommender, MatrixFactorizationRecommender, NeuralRecommender]

@pytest.mark.parametrize("RecClass", ALL_VERSIONS)
def test_recommend_shape(RecClass, sample_interactions):
    model = RecClass().fit(sample_interactions)
    recs = model.recommend(user_id=123, n=5)
    assert len(recs) <= 5
    assert all("item_id" in r and "score" in r for r in recs)