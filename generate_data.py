
"""Generate reproducible synthetic e-commerce return-abuse transactions."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SEED = 42


def generate_dataset(n: int = 10_000, seed: int = SEED) -> pd.DataFrame:
    if n < 100:
        raise ValueError("n must be at least 100")

    rng = np.random.default_rng(seed)

    # Customer and transaction characteristics
    account_age_days = np.clip(
        rng.gamma(3.2, 170, n).astype(int), 1, 3000
    )

    is_first_order = (
        rng.random(n)
        < np.clip(0.55 - account_age_days / 6000, 0.08, 0.55)
    ).astype(int)

    order_value_inr = np.round(
        np.clip(rng.lognormal(8.25, 0.8, n), 300, 75000), 2
    )

    items_count = rng.integers(1, 9, n)

    returns_past_month = np.minimum(
        rng.poisson(0.75, n), 9
    )

    orders_past_month = np.maximum(
        1, rng.poisson(3.2, n) + 1
    )

    shipping_mismatch = (
        rng.random(n) < 0.075
    ).astype(int)

    days_to_return = np.clip(
        rng.gamma(2.4, 5.0, n).astype(int), 0, 45
    )

    category = rng.choice(
        ["fashion", "electronics", "home", "beauty", "other"],
        n,
        p=[0.34, 0.18, 0.19, 0.13, 0.16]
    )

    payment_method = rng.choice(
        ["card", "upi", "wallet", "cod"],
        n,
        p=[0.34, 0.36, 0.18, 0.12]
    )

    refund_to_original_payment = (
        rng.random(n) < 0.91
    ).astype(int)

    prior_return_rate = np.clip(
        (returns_past_month + rng.poisson(1.0, n))
        / (orders_past_month + rng.integers(1, 8, n)),
        0, 1
    )

    # Latent behavioral risk signals
    wardrobing_signal = (
        (category == "fashion") * 1.5
        + (days_to_return <= 3) * 1.3
        + (order_value_inr > 12000) * 0.8
        + (returns_past_month >= 2) * 1.0
    )

    fee_avoidance_signal = (
        shipping_mismatch * 2.0
        + (returns_past_month >= 3) * 1.3
        + (orders_past_month >= 5) * 0.6
        + (refund_to_original_payment == 0) * 0.8
    )

    burner_signal = (
        (account_age_days < 45) * 1.8
        + is_first_order * 1.0
        + (order_value_inr > 18000) * 0.8
        + shipping_mismatch * 0.7
    )

    # Select the strongest behavioral pattern
    scenario_scores = np.column_stack([
        wardrobing_signal,
        fee_avoidance_signal,
        burner_signal
    ])

    scenario_index = scenario_scores.argmax(axis=1)

    # Create a risk probability using stronger behavioral signals.
    # Noise remains to prevent perfectly deterministic labels.
    strongest_signal = scenario_scores.max(axis=1)

    logit = (
        -5.0
        + 1.15 * strongest_signal
        + 1.2 * (prior_return_rate > 0.45)
        + rng.normal(0, 0.45, n)
    )

    probability = 1 / (1 + np.exp(-logit))

    labels = (
        rng.random(n) < probability
    ).astype(int)

    scenario_names = np.array([
        "wardrobing",
        "fee_avoidance",
        "burner_account"
    ])

    fraud_type = np.where(
        labels == 1,
        scenario_names[scenario_index],
        "legitimate"
    )

    df = pd.DataFrame({
        "transaction_id": [
            f"TXN{i:06d}" for i in range(1, n + 1)
        ],
        "customer_id": [
            f"CUST{x:05d}"
            for x in rng.integers(1, max(2, n // 2), n)
        ],
        "customer_age": rng.integers(18, 76, n),
        "account_age_days": account_age_days,
        "is_first_order": is_first_order,
        "order_value_inr": order_value_inr,
        "items_count": items_count,
        "returns_past_month": returns_past_month,
        "orders_past_month": orders_past_month,
        "shipping_mismatch": shipping_mismatch,
        "days_to_return": days_to_return,
        "item_category": category,
        "payment_method": payment_method,
        "refund_to_original_payment": refund_to_original_payment,
        "previous_return_rate": np.round(prior_return_rate, 4),
        "fraud_type": fraud_type,
        "is_return_abuse": labels,
    })

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/transactions.csv")
    )

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(args.rows, args.seed)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df):,} rows to {args.output}")
    print("Class distribution:")
    print(
        df["is_return_abuse"]
        .value_counts(normalize=True)
        .rename("fraction")
    )


if __name__ == "__main__":
    main()
