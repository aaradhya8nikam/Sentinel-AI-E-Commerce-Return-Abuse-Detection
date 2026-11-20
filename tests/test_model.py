from generate_data import generate_dataset
from src.features import build_features

def test_feature_matrix_is_numeric_and_finite():
    df = generate_dataset(300, seed=11)
    x = build_features(df)
    assert x.shape[0] == 300
    assert x.select_dtypes(exclude="number").empty
    assert x.isna().sum().sum() == 0
