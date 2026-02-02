import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Fill NaN just in case
    train_df['clean_review'] = train_df['clean_review'].fillna('')
    test_df['clean_review'] = test_df['clean_review'].fillna('')

    X_train = train_df['clean_review']
    y_train = train_df['label']
    X_test = test_df['clean_review']
    y_test = test_df['label']

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("IMDB_Spoiler_Shield")

    with mlflow.start_run():
        print("Training model...")
        
        # Pipeline: TF-IDF + Logistic Regression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")

        metrics = {
            "model_type": "TF-IDF + Logistic Regression",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Save model artifact locally
        if not os.path.exists("models"):
            os.makedirs("models")
        
        local_model_path = "models/model.joblib"
        joblib.dump(pipeline, local_model_path)
        print(f"Model saved locally to {local_model_path}")

        # Save metrics locally
        local_metrics_path = "models/metrics.json"
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
