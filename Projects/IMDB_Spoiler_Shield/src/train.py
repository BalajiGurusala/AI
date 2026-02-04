import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
import joblib
import boto3
import json

def train_model(data_dir="data/processed"):
    bucket = os.getenv("S3_BUCKET")
    
    if bucket:
        print(f"Loading data from S3 bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
    else:
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

    print(f"Reading train data from {train_path}...")
    # Use chunksize to read, but for baseline we just sample a chunk to avoid OOM
    # 50k rows is statistically sufficient for a baseline comparison
    train_df = pd.read_csv(train_path, nrows=10000)
    test_df = pd.read_csv(test_path, nrows=2000)

    print(f"Training Baseline on sampled data: {len(train_df)} rows")

    # Fill NaN just in case
    train_df['clean_review'] = train_df['clean_review'].fillna('')
    test_df['clean_review'] = test_df['clean_review'].fillna('')

    X_train = train_df['clean_review']
    y_train = train_df['label']
    X_test = test_df['clean_review']
    y_test = test_df['label']

    # Step 1: Compute class weights based on training data
    print("Computing class weights...")
    classes = np.unique(y_train)
    if len(classes) > 1:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    else:
        class_weight_dict = None
        print("Warning: Single class detected in sample. Skipping class weights.")

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("IMDB_Spoiler_Shield")

    with mlflow.start_run():
        print("Training model with class weights...")
        
        # Pipeline: TF-IDF + Logistic Regression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000, class_weight=class_weight_dict))
        ])

        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")

        metrics = {
            "model_type": "TF-IDF + Logistic Regression (Baseline)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        # Log metrics and params
        if class_weight_dict:
            # Convert integer keys to strings for MLflow
            str_key_weights = {str(k): v for k, v in class_weight_dict.items()}
            mlflow.log_params(str_key_weights)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Save model artifact locally
        if not os.path.exists("models"):
            os.makedirs("models")
        
        local_model_path = os.path.join("models", "model.joblib")
        joblib.dump(pipeline, local_model_path)
        print(f"Model saved locally to {local_model_path}")

        # Save metrics locally
        local_metrics_path = os.path.join("models", "metrics.json")
        with open(local_metrics_path, "w") as f:
            json.dump(metrics, f)
        
        # Upload to S3
        if bucket:
            s3 = boto3.client('s3')
            
            s3_model_path = "models/model.joblib"
            print(f"Uploading model to s3://{bucket}/{s3_model_path}...")
            s3.upload_file(local_model_path, bucket, s3_model_path)
            
            s3_metrics_path = "models/metrics.json"
            print(f"Uploading metrics to s3://{bucket}/{s3_metrics_path}...")
            s3.upload_file(local_metrics_path, bucket, s3_metrics_path)

if __name__ == "__main__":
    train_model()