import pandas as pd
import joblib
import os
import mlflow
import boto3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping

def evaluate_model(data_dir="data/processed", model_path="models/model.joblib"):
    bucket = os.getenv("S3_BUCKET")
    
    if bucket:
        print(f"Loading data from S3 bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
        
        # Download model from S3 to local for loading
        s3 = boto3.client('s3')
        local_model_path = "models/model.joblib"
        # Ensure dir
        if not os.path.exists("models"):
            os.makedirs("models")
            
        print(f"Downloading model from s3://{bucket}/models/model.joblib...")
        s3.download_file(bucket, "models/model.joblib", local_model_path)
        model_path = local_model_path
    else:
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

    print("Loading data for evaluation...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df['clean_review'] = train_df['clean_review'].fillna('')
    test_df['clean_review'] = test_df['clean_review'].fillna('')

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Predictions for Evidently
    train_df['prediction'] = model.predict(train_df['clean_review'])
    test_df['prediction'] = model.predict(test_df['clean_review'])

    # Evidently Report
    column_mapping = ColumnMapping()
    column_mapping.target = 'label'
    column_mapping.prediction = 'prediction'

    # Create report
    report = Report(metrics=[
        DataDriftPreset(), 
        ClassificationPreset()
    ])

    print("Running Evidently Report...")
    report.run(reference_data=train_df[['label', 'prediction']], 
               current_data=test_df[['label', 'prediction']],
               column_mapping=column_mapping)

    report_path = "data/evidently_report.html"
    report.save_html(report_path)
    print(f"Evidently report saved locally to {report_path}")
    
    if bucket:
        s3_report_path = "reports/evidently_report.html"
        print(f"Uploading report to s3://{bucket}/{s3_report_path}...")
        s3 = boto3.client('s3')
        s3.upload_file(report_path, bucket, s3_report_path)

    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    # In a real DAG, we'd pass the run_id to log artifact to the same run.
    # For now, this is standalone.

if __name__ == "__main__":
    evaluate_model()