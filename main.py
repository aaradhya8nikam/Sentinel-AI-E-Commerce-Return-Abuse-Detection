import joblib
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# 1. Define the Input Schema with validation
class TransactionData(BaseModel):
    customer_age: int = Field(..., ge=18, le=100)
    order_value_inr: float = Field(..., gt=0)
    items_count: int = Field(..., ge=1)
    is_first_order: int = Field(..., ge=0, le=1)
    shipping_mismatch: int = Field(..., ge=0, le=1)
    returns_past_month: int = Field(..., ge=0)

# 2. Initialize FastAPI
app = FastAPI(title="Razorpay Hackathon: Risk Analyzer API")

# Global variables for model and explainer
model = None
explainer = None
feature_names = [
    'customer_age', 'order_value_inr', 'items_count', 
    'is_first_order', 'shipping_mismatch', 'returns_past_month',
    'value_per_item', 'is_high_risk_age' # Add these!
]

@app.on_event("startup")
def load_resources():
    """Load model and SHAP explainer on startup to keep inference fast."""
    global model, explainer
    try:
        model = joblib.load('return_abuse_model.pkl')
        # We use a TreeExplainer for Random Forest performance
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Error loading model resources: {e}")

@app.post("/analyze-risk")
async def analyze_risk(data: TransactionData):
    """
    Analyzes transaction data to predict return abuse risk.
    Returns:
    - risk_probability: 0.0 to 1.0
    - audit_trail: Top 2 contributing factors for the score
    """
    try:
        # Convert Pydantic model to DataFrame for the model
        input_dict = data.dict()
        df_input = pd.DataFrame([input_dict])

        # Get risk probability
        # [0] for class 0 (safe), [1] for class 1 (abuse)
        probs = model.predict_proba(df_input)[0]
        risk_score = float(probs[1])

        # Generate Audit Trail (Local Feature Importance)
        # We calculate SHAP values for this specific instance
        shap_values = explainer.shap_values(df_input)
        
        # shap_values[1] contains values for the 'Abuse' class
        # We zip feature names with their local impact values
        feature_impacts = dict(zip(feature_names, shap_values[1][0]))
        
        # Sort factors by impact (reasons why the score is high)
        # We take features with the highest positive influence on the 'Abuse' flag
        sorted_reasons = sorted(feature_impacts.items(), key=lambda x: x[1], reverse=True)
        top_reasons = [reason[0].replace('_', ' ').title() for reason in sorted_reasons[:2]]

        return {
            "status": "success",
            "risk_probability": round(risk_score, 4),
            "is_flagged": risk_score > 0.5,
            "audit_trail": top_reasons
        }

    except Exception as e:
        # Razorpay Hackathon Rule: Graceful failure handling
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}. Please check input format."
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)