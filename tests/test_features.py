import pandas as pd
from src.features import build_features, FEATURE_COLUMNS

def test_feature_builder_has_stable_schema():
    row = pd.DataFrame([{
        "customer_age": 25, "account_age_days": 30, "is_first_order": 1,
        "order_value_inr": 10000, "items_count": 2, "returns_past_month": 2,
        "orders_past_month": 3, "shipping_mismatch": 1, "days_to_return": 2,
        "item_category": "fashion", "payment_method": "upi",
        "refund_to_original_payment": 1, "previous_return_rate": 0.4
    }])
    x = build_features(row)
    assert list(x.columns) == FEATURE_COLUMNS
    assert x.loc[0, "value_per_item"] == 5000
    assert x.loc[0, "quick_return_flag"] == 1
