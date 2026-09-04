import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000 # Increased for better learning

data = {
    'order_id': range(1, n + 1),
    'customer_age': np.random.randint(18, 75, n),
    'order_value_inr': np.random.uniform(500, 50000, n),
    'items_count': np.random.randint(1, 10, n),
    'is_first_order': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'shipping_mismatch': np.random.choice([0, 1], n, p=[0.9, 0.1]),
    'returns_past_month': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
}

df = pd.DataFrame(data)

def determine_abuse(row):
    # Complex logic for AI to find
    score = 0
    if row['shipping_mismatch'] == 1: score += 2
    if row['returns_past_month'] >= 3: score += 3
    if row['is_first_order'] == 1 and row['order_value_inr'] > 20000: score += 2
    
    # 70% chance of abuse if score is high
    if score >= 4:
        return np.random.choice([1, 0], p=[0.75, 0.25])
    return np.random.choice([1, 0], p=[0.02, 0.98])

df['is_return_abuse'] = df.apply(determine_abuse, axis=1)
df.to_csv('transactions.csv', index=False)
print("Dataset created: 10,000 transactions.")