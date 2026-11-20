"""Cost-sensitive threshold selection and business impact calculations."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FALSE_POSITIVE_COST = 500.0
MISSED_FRAUD_COST = 4500.0

def evaluate_threshold(y_true, probabilities, threshold,
                       false_positive_cost=FALSE_POSITIVE_COST,
                       missed_fraud_cost=MISSED_FRAUD_COST):
    predictions = (np.asarray(probabilities) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    total_cost = fp * false_positive_cost + fn * missed_fraud_cost
    return {
        "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp), "total_cost_inr": float(total_cost),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }

def optimize_threshold(y_true, probabilities,
                       false_positive_cost=FALSE_POSITIVE_COST,
                       missed_fraud_cost=MISSED_FRAUD_COST):
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = [
        evaluate_threshold(y_true, probabilities, t,
                           false_positive_cost, missed_fraud_cost)
        for t in thresholds
    ]
    results = pd.DataFrame(rows).sort_values(
        ["total_cost_inr", "threshold"], ascending=[True, False]
    )
    return results.iloc[0].to_dict(), results
