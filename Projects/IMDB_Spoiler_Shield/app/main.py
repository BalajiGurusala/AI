from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys
import uvicorn
import boto3
from feast import FeatureStore

# THE FIX: Add the project root to path so 'preprocessing' can be found
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/src")

try:
    from preprocessing import clean_text
    print("✅ Successfully imported cleaning logic.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"DEBUG: sys.path is {sys.path}")
    raise

app = FastAPI(title="IMDB Spoiler Shield API", description="Production Serving Layer")

# Global variables
model = None
fs = None

class ReviewRequest(BaseModel):
    movie_id: str
    review_text: str

class PredictionResponse(BaseModel):
    is_spoiler: bool
    probability: float
    message: str
    enriched_features: dict

@app.on_event("startup")
def startup_event():
    global model, fs
    local_model_path = os.getenv("MODEL_PATH", "/opt/airflow/models/model.joblib")
    
    try:
        model = joblib.load(local_model_path)
        print(f"Model loaded from {local_model_path}.")
    except Exception as e:
        print(f"Model load error: {e}")

    try:
        fs = FeatureStore(repo_path="/opt/airflow/feature_repo")
        print("Feast Feature Store initialized.")
    except Exception as e:
        print(f"Feast init error: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "feast_loaded": fs is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    enriched_features = {}
    if fs:
        try:
            feature_vector = fs.get_online_features(
                features=[
                    "movie_stats:duration_minutes",
                    "movie_stats:rating",
                    "movie_stats:genre",
                    "movie_stats:avg_review_length",
                    "movie_stats:avg_sentiment_score",
                ],
                entity_rows=[{"movie_id": request.movie_id}]
            ).to_dict()
            enriched_features = {k: v[0] for k, v in feature_vector.items() if k != "movie_id"}
        except Exception as e:
            print(f"Feast retrieval error: {e}")

    try:
        cleaned_text = clean_text(request.review_text)
        prediction = model.predict([cleaned_text])[0]
        probability = model.predict_proba([cleaned_text])[0][1]

        return PredictionResponse(
            is_spoiler=bool(prediction),
            probability=float(probability),
            message="Spoiler detected" if prediction else "Safe to read",
            enriched_features=enriched_features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)