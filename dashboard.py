import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.features import build_features
from src.explain import explain_one
from src.cost_optimizer import evaluate_threshold, FALSE_POSITIVE_COST, MISSED_FRAUD_COST

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ROOT / "data" / "transactions.csv"

st.set_page_config(page_title="Sentinel AI | Return Abuse Monitor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load(ART / "return_abuse_model.pkl")
    features = joblib.load(ART / "features.pkl")
    with open(ART / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(ART / "threshold_config.json", encoding="utf-8") as f:
        config = json.load(f)
    return model, features, metrics, config

st.title("Sentinel AI")
st.caption("Explainable e-commerce return-abuse risk monitoring")
if not (ART / "return_abuse_model.pkl").exists():
    st.error("Model artifacts not found. Run: python generate_data.py && python -m src.train_model")
    st.stop()

model, feature_names, metrics, config = load_assets()
default_threshold = float(config["threshold"])
threshold = st.sidebar.slider("Risk threshold", 0.05, 0.95, default_threshold, 0.01)
fp_cost = st.sidebar.number_input("False-positive cost (₹)", min_value=0, value=500, step=100)
fn_cost = st.sidebar.number_input("Missed-fraud cost (₹)", min_value=0, value=4500, step=250)

if DATA.exists():
    base_df = pd.read_csv(DATA)
else:
    base_df = pd.DataFrame()

uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
if uploaded is not None:
    base_df = pd.read_csv(uploaded)

if base_df.empty:
    st.warning("No transaction data loaded.")
    st.stop()

required_target = "is_return_abuse"
raw_features = build_features(base_df).reindex(columns=feature_names, fill_value=0)
probabilities = model.predict_proba(raw_features)[:, 1]
base_df = base_df.copy()
base_df["risk_score"] = probabilities
base_df["decision"] = np.where(probabilities >= threshold, "FLAGGED", "CLEAR")

flagged = base_df[base_df["decision"] == "FLAGGED"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(base_df):,}")
c2.metric("Flagged", f"{len(flagged):,}", f"{len(flagged)/len(base_df):.1%}")
c3.metric("Average risk", f"{base_df['risk_score'].mean():.1%}")
c4.metric("Model test precision", f"{metrics['precision']:.1%}")
st.caption("Model test metrics are calculated from the held-out test set, not the current dashboard upload.")

overview, explorer, explain_tab, impact, model_tab = st.tabs(
    ["Overview", "Transaction Explorer", "SHAP Explainability", "Business Impact", "Model Evaluation"]
)
with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Risk score distribution")
        st.bar_chart(base_df["risk_score"].round(1).value_counts().sort_index())
    with right:
        st.subheader("Decision breakdown")
        st.bar_chart(base_df["decision"].value_counts())
    st.subheader("Highest-risk transactions")
    display_cols = [c for c in ["transaction_id", "order_value_inr", "returns_past_month",
                    "shipping_mismatch", "risk_score", "decision"] if c in base_df.columns]
    st.dataframe(base_df.sort_values("risk_score", ascending=False)[display_cols].head(20),
                 use_container_width=True)

with explorer:
    st.subheader("Search and filter transactions")
    min_risk, max_risk = st.slider("Risk score range", 0.0, 1.0, (0.0, 1.0))
    view = base_df[base_df["risk_score"].between(min_risk, max_risk)]
    only_flagged = st.checkbox("Show flagged only")
    if only_flagged:
        view = view[view["decision"] == "FLAGGED"]
    st.dataframe(view, use_container_width=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False),
                       "sentinel_transaction_report.csv", "text/csv")

with explain_tab:
    st.subheader("Transaction-level explanation")
    ids = base_df["transaction_id"].astype(str).tolist()
    selected_id = st.selectbox("Choose transaction", ids)
    idx = base_df.index[base_df["transaction_id"].astype(str) == selected_id][0]
    position = base_df.index.get_loc(idx)
    row_features = raw_features.iloc[[position]]
    probability = float(probabilities[position])
    audit, shap_explanation = explain_one(model, row_features, selected_id, probability, threshold)
    st.metric("Abuse probability", f"{probability:.1%}",
              "FLAGGED" if probability >= threshold else "CLEAR")
    st.write("**Top risk-increasing factors**")
    st.json(audit["top_risk_factors"])
    st.write("**Risk-reducing factors**")
    st.json(audit["risk_reducing_factors"])
    st.caption(audit["explanation_note"])
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_explanation[0], show=False, max_display=12)
    st.pyplot(fig, clear_figure=True)
    st.download_button("Download audit JSON", json.dumps(audit, indent=2),
                       f"{selected_id}_audit.json", "application/json")

with impact:
    st.subheader("Cost-sensitive decision analysis")
    if required_target not in base_df.columns:
        st.info("Upload a labelled dataset with is_return_abuse to calculate realized FP/FN cost.")
    else:
        result = evaluate_threshold(base_df[required_target], probabilities, threshold,
                                    fp_cost, fn_cost)
        a, b, c = st.columns(3)
        a.metric("False positives", result["fp"])
        b.metric("Missed fraud", result["fn"])
        c.metric("Estimated total cost", f"₹{result['total_cost_inr']:,.0f}")
        st.write("Cost = FP × false-positive cost + FN × missed-fraud cost.")
        st.dataframe(pd.DataFrame([result]), use_container_width=True)
    st.caption("For unlabelled live transactions, realized FP/FN cost cannot be known until outcomes are reviewed.")

with model_tab:
    st.subheader("Held-out test-set metrics")
    cols = st.columns(3)
    for col, (name, key) in zip(cols, [
        ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")
    ]):
        col.metric(name, f"{metrics[key]:.1%}")
    st.json(metrics)
    shap_image = ROOT / "reports" / "shap_summary.png"
    if shap_image.exists():
        st.image(str(shap_image), caption="Global SHAP feature importance")
