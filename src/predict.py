"""Reusable inference functions."""
from pathlib import Path
import joblib
import pandas as pd
from src.features import build_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"

def load_model(artifact_dir=ARTIFACT_DIR):
    model = joblib.load(Path(artifact_dir) / "return_abuse_model.pkl")
    feature_names = joblib.load(Path(artifact_dir) / "features.pkl")
    return model, feature_names

def predict_transactions(raw_df: pd.DataFrame, model=None, feature_names=None):
    if model is None or feature_names is None:
        model, feature_names = load_model()
    X = build_features(raw_df).reindex(columns=feature_names, fill_value=0)
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities
