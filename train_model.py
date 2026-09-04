import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Load data
try:
    df = pd.read_csv('transactions.csv')
except FileNotFoundError:
    print("Error: transactions.csv not found. Run generate_data.py first!")
    exit()

# Feature Engineering
def add_features(df):
    df['val_per_item'] = df['order_value_inr'] / df['items_count']
    df['return_velocity'] = df['returns_past_month'] / (df['items_count'] + 1)
    df['risk_multiplier'] = df['shipping_mismatch'] * df['returns_past_month']
    return df

df = add_features(df)
X = df.drop(columns=['order_id', 'is_return_abuse'])
y = df['is_return_abuse']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train XGBoost 
print(f"Training on {len(X_train)} samples...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=15, # Increased weight to help with rare fraud
    random_state=42
)
model.fit(X_train, y_train)

# 3. Save the Model and the Column Names (Crucial for Dashboard)
joblib.dump(model, 'return_abuse_model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl') 
print("✅ Model and features.pkl saved successfully.")

# 4. Evaluation
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred))

# 5. SHAP (Wrapped in try-except to prevent crashing)
print("\nAttempting to generate SHAP plot...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary.png')
    print("✅ SHAP plot saved as 'shap_summary.png'")
except Exception as e:
    print(f"⚠️ SHAP plotting skipped: {e}")