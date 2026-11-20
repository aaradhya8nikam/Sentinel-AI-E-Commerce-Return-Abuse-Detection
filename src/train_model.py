"""Train, evaluate and persist the Sentinel AI XGBoost pipeline."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from src.features import build_features, TARGET
from src.cost_optimizer import optimize_threshold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

def train(data_path=DATA_PATH, artifact_dir=ARTIFACT_DIR, report_dir=REPORT_DIR):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    if TARGET not in df:
        raise ValueError(f"Dataset must contain {TARGET}")
    y = df[TARGET].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    X = build_features(df)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best, threshold_table = optimize_threshold(y_val, val_prob)
    threshold = best["threshold"]

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "threshold_selected_on_validation": float(threshold),
        "test_confusion_matrix_labels_tn_fp_fn_tp": confusion_matrix(
            y_test, test_pred, labels=[0, 1]
        ).ravel().tolist(),
        "test_rows": int(len(y_test)),
        "validation_cost_inr": float(best["total_cost_inr"]),
    }

    joblib.dump(model, artifact_dir / "return_abuse_model.pkl")
    joblib.dump(list(X.columns), artifact_dir / "features.pkl")
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(artifact_dir / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "false_positive_cost_inr": 500,
                   "missed_fraud_cost_inr": 4500}, f, indent=2)
    threshold_table.to_csv(report_dir / "threshold_cost_analysis.csv", index=False)

    # SHAP summary on a bounded test sample to keep training time manageable.
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved model artifacts to {artifact_dir}")
    return metrics

if __name__ == "__main__":
    train()
