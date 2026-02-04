import pandas as pd
import joblib
import os
import mlflow
import boto3

# Fix imports for Evidently 0.7.x in this env
try:
    from evidently.report import Report
except ImportError:
    from evidently import Report

try:
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
except ImportError:
    from evidently.presets import DataDriftPreset, ClassificationPreset

try:
    from evidently import ColumnMapping
except ImportError:
    try:
        from evidently.pipeline.column_mapping import ColumnMapping
    except ImportError:
        # Fallback or simple dict if ColumnMapping class is gone
        ColumnMapping = None

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

    # Create report
    report = Report(metrics=[
        DataDriftPreset(), 
        # ClassificationPreset requires explicit config in this version, skipping for now
    ])

    print("Running Evidently Report...")
    
    # Rename 'label' to 'target' for Evidently auto-discovery
    eval_train = train_df[['label', 'prediction']].rename(columns={'label': 'target'})
    eval_test = test_df[['label', 'prediction']].rename(columns={'label': 'target'})
    
    # Run without column_mapping argument to avoid TypeError in some versions
    report.run(reference_data=eval_train, 
               current_data=eval_test)

    # Debugging: Inspect what Report actually is in this environment
    print(f"Report Type: {type(report)}")
    print(f"Report Attributes: {dir(report)}")

    report_path = "data/evidently_report.html"
    
    # Robust saving for different Evidently versions
    try:
        if hasattr(report, 'save_html'):
            report.save_html(report_path)
        elif hasattr(report, 'get_html'):
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report.get_html())
        elif hasattr(report, 'save'):
             report.save(report_path.replace('.html', '.json'))
        else:
             print("WARNING: Could not save report. No known save method found.")
    except Exception as e:
        print(f"Failed to save report: {e}")
            
    print(f"Evidently report processing complete.")
    
    if bucket and os.path.exists(report_path):
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