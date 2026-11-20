"""Local SHAP explanations and structured audit records."""
import numpy as np
import shap

def explain_one(model, feature_frame, transaction_id, probability, threshold):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(feature_frame)
    values = np.asarray(explanation.values)
    if values.ndim == 3:  # Some multiclass SHAP versions return class dimension.
        values = values[:, :, 1]
    contributions = values[0]
    ranked = sorted(
        zip(feature_frame.columns, contributions),
        key=lambda pair: pair[1], reverse=True
    )
    positive = [
        {"feature": name, "shap_value": float(value)}
        for name, value in ranked if value > 0
    ][:5]
    negative = [
        {"feature": name, "shap_value": float(value)}
        for name, value in sorted(
            zip(feature_frame.columns, contributions), key=lambda pair: pair[1]
        ) if value < 0
    ][:5]
    return {
        "transaction_id": str(transaction_id),
        "risk_score": float(probability),
        "threshold": float(threshold),
        "decision": "HIGH_RISK" if probability >= threshold else "LOW_RISK",
        "top_risk_factors": positive,
        "risk_reducing_factors": negative,
        "explanation_note": (
            "SHAP values explain feature contributions to the model output; "
            "they are not proof of fraud or causal evidence."
        ),
    }, explanation
