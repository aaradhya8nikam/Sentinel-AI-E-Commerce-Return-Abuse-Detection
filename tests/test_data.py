from generate_data import generate_dataset

def test_dataset_shape_and_labels():
    df = generate_dataset(500, seed=7)
    assert len(df) == 500
    assert {"transaction_id", "fraud_type", "is_return_abuse"} <= set(df.columns)
    assert set(df["is_return_abuse"].unique()) <= {0, 1}
    assert df["is_return_abuse"].nunique() == 2
