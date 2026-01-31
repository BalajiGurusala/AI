import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from feast import FeatureStore
import tensorflow as tf
from tensorflow import keras

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_lending_data, DROP_COLS

app = FastAPI(title="Lending Club Credit Scoring API")

# Load artifacts
MODEL_KERAS_PATH = os.path.join("models", "best_model.keras")
MODEL_PKL_PATH = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("data", "minmax_scaler.pkl")
FEATURE_REPO_PATH = "feature_repo"
DATA_PATH = os.path.join("data", "loans.csv")

model = None
model_type = None # "keras" or "sklearn"
scaler = None
store = None
loans_df = None

class LoanRequest(BaseModel):
    borrower_id: int
    loan_amnt: float = Field(..., gt=0, le=40000, description="Requested loan amount (max 40k matches training data)")
    term: str = "36 months"
    purpose: str = "debt_consolidation"

@app.on_event("startup")
def load_artifacts():
    global model, model_type, scaler, store, loans_df
    try:
        # Load Best Model (Detect type)
        if os.path.exists(MODEL_KERAS_PATH):
            model = keras.models.load_model(MODEL_KERAS_PATH)
            model_type = "keras"
            print(f"Keras model loaded from {MODEL_KERAS_PATH}")
        elif os.path.exists(MODEL_PKL_PATH):
            model = joblib.load(MODEL_PKL_PATH)
            model_type = "sklearn"
            print(f"Sklearn/XGBoost model loaded from {MODEL_PKL_PATH}")
        else:
            print("Warning: No model found in models/ directory")

        # Load Scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")
        else:
            print(f"Warning: Scaler not found at {SCALER_PATH}")

        # Initialize Feature Store
        if os.path.exists(FEATURE_REPO_PATH):
            store = FeatureStore(repo_path=FEATURE_REPO_PATH)
            print(f"Feature Store connected at {FEATURE_REPO_PATH}")
        else:
            print(f"Warning: Feature Repo not found at {FEATURE_REPO_PATH}")
            
        # Load loans.csv for lookup
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            if 'id' in df.columns:
                df = df.drop_duplicates(subset=['id'])
                df.set_index('id', inplace=True)
                loans_df = df
                print(f"Loaded loans.csv with {len(df)} records")
            else:
                print("Warning: loans.csv does not have 'id' column")
        else:
            print(f"Warning: Data file not found at {DATA_PATH}")
            
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.post("/predict")
def predict_default(request: LoanRequest):
    if not model or not scaler or not store:
        raise HTTPException(status_code=503, detail="Service not ready (artifacts missing)")

    try:
        # 1. Fetch from CSV
        csv_data = {}
        if loans_df is not None:
            if request.borrower_id in loans_df.index:
                csv_data = loans_df.loc[request.borrower_id].to_dict()
                csv_data.pop('loan_status', None)
                csv_data.pop('member_id', None) 
            else:
                print(f"Warning: Borrower {request.borrower_id} not found in loans.csv")

        # 2. Fetch from Feast
        feature_vector = store.get_online_features(
            features=[
                "lending_features:annual_inc",
                "lending_features:dti",
                "lending_features:revol_bal",
                "lending_features:revol_util",
                "lending_features:open_acc",
                "lending_features:total_acc",
                "lending_features:mort_acc",
                "lending_features:pub_rec",
                "lending_features:pub_rec_bankruptcies",
                "lending_features:earliest_cr_year",
                "lending_features:home_ownership"
            ],
            entity_rows=[{"borrower_id": request.borrower_id}]
        ).to_dict()
        
        feast_data = {k: v[0] for k, v in feature_vector.items() if k != "borrower_id" and v is not None and len(v) > 0 and v[0] is not None}

        # 3. Combine Data
        combined_data = csv_data.copy()
        combined_data.update(feast_data)
        request_data = request.dict(by_alias=True, exclude_unset=True)
        combined_data.update(request_data)
        
        # 4. Prepare DataFrame
        full_df = pd.DataFrame([combined_data])
        
        # 5. Preprocess
        processed_df = preprocess_lending_data(full_df)
        
        # 6. Align columns with Scaler
        processed_df = processed_df.drop(columns=[c for c in DROP_COLS if c in processed_df.columns], errors='ignore')
        processed_df = processed_df.select_dtypes(exclude=['object']) 
        
        if hasattr(scaler, "feature_names_in_"):
            expected_features = scaler.feature_names_in_
        else:
            raise HTTPException(status_code=500, detail="Scaler configuration error")

        for col in expected_features:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        X = processed_df[expected_features]
        X = X.fillna(0)
        
        # 7. Scale
        X_scaled = scaler.transform(X)
        
        # 8. Predict
        if model_type == "keras":
            prob = model.predict(X_scaled, verbose=0)[0][0]
        else:
            # sklearn models (LogReg, RF, XGB)
            prob = model.predict_proba(X_scaled)[0][1]
        
        return {
            "borrower_id": request.borrower_id,
            "default_probability": float(prob),
            "is_approved": bool(prob < 0.20), 
            "risk_score": int((1 - prob) * 850),
            "input_data_summary": {
                "loan_amnt": combined_data.get("loan_amnt"),
                "term": combined_data.get("term"),
                "grade": combined_data.get("grade"),
                "int_rate": combined_data.get("int_rate")
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))