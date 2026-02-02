from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys
import uvicorn
import boto3
from feast import FeatureStore

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import clean_text

app = FastAPI(title="IMDB Spoiler Shield API", description="API enriched by Feast Feature Store")

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
    
    # Load Model
    bucket = os.getenv("S3_BUCKET")
    local_model_path = os.getenv("MODEL_PATH", "../models/model.joblib")
    
    if bucket:
        print(f"Downloading model from S3 bucket: {bucket}...")
        try:
            s3 = boto3.client('s3')
            # Ensure local dir exists
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            s3.download_file(bucket, "models/model.joblib", local_model_path)
            print("Model downloaded from S3.")
        except Exception as e:
            print(f"Failed to download model from S3: {e}")

    try:
        model = joblib.load(local_model_path)
        print(f"Model loaded from {local_model_path}.")
    except Exception as e:
        print(f"Model load error: {e}")

    # Load Feast Feature Store
    # In Docker, feature_repo is at /app/feature_repo
    try:
        fs = FeatureStore(repo_path="/app/feature_repo")
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

    # 1. Fetch enriched features from Feast Online Store
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
            
            # Extract values
            enriched_features = {k: v[0] for k, v in feature_vector.items() if k != "movie_id"}
        except Exception as e:
            print(f"Feast retrieval error: {e}")

    # 2. Preprocess and Predict
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
