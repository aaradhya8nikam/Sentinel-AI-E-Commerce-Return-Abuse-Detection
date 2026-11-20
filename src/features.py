"""Shared feature engineering used by both training and inference."""
import pandas as pd

TARGET = "is_return_abuse"
ID_COLUMNS = ["transaction_id", "customer_id", "fraud_type"]
NUMERIC_FEATURES = [
    "customer_age", "account_age_days", "is_first_order", "order_value_inr",
    "items_count", "returns_past_month", "orders_past_month",
    "shipping_mismatch", "days_to_return", "refund_to_original_payment",
    "previous_return_rate", "value_per_item", "return_velocity",
    "return_frequency", "quick_return_flag", "high_value_first_order",
    "account_is_new", "return_order_ratio",
]
CATEGORICAL_FEATURES = ["item_category", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + [
    f"{col}_{value}" for col, values in {
        "item_category": ["beauty", "electronics", "fashion", "home", "other"],
        "payment_method": ["card", "cod", "upi", "wallet"],
    }.items() for value in values
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready numeric features in a stable, fixed column order."""
    required = [
        "customer_age", "account_age_days", "is_first_order", "order_value_inr",
        "items_count", "returns_past_month", "orders_past_month",
        "shipping_mismatch", "days_to_return", "item_category",
        "payment_method", "refund_to_original_payment", "previous_return_rate",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.copy()
    x["value_per_item"] = x["order_value_inr"] / x["items_count"].clip(lower=1)
    x["return_velocity"] = x["returns_past_month"] / (x["account_age_days"] / 30 + 1)
    x["return_frequency"] = x["returns_past_month"] / x["orders_past_month"].clip(lower=1)
    x["quick_return_flag"] = (x["days_to_return"] <= 3).astype(int)
    x["high_value_first_order"] = (
        (x["is_first_order"] == 1) & (x["order_value_inr"] >= 18000)
    ).astype(int)
    x["account_is_new"] = (x["account_age_days"] <= 45).astype(int)
    x["return_order_ratio"] = (
        x["returns_past_month"] / x["orders_past_month"].clip(lower=1)
    )
    encoded = pd.get_dummies(
        x[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        dtype=int,
    )
    # Reindex ensures a stable schema even if an inference batch lacks a category.
    return encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0).astype(float)
