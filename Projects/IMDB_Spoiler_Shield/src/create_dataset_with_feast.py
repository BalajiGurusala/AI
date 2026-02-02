import pandas as pd
import os
from datetime import datetime
from feast import FeatureStore
import boto3

def create_dataset():
    # 1. Setup paths
    bucket = os.getenv("S3_BUCKET")
    data_dir = "data/processed"
    if bucket:
        print(f"Loading data from S3 bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
    else:
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

    # 2. Load entity dataframe (Reviews)
    print("Loading base review data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 3. Add timestamps for Feast
    # In a real scenario, this should be the actual review time. 
    # For now, we'll assume reviews are recent enough or map them if available.
    # Since preprocessing dropped dates, we'll use current time for training snapshot.
    # Feast requires 'event_timestamp' to find features valid AS OF that time.
    train_df['event_timestamp'] = pd.to_datetime(datetime.now())
    test_df['event_timestamp'] = pd.to_datetime(datetime.now())

    # 4. Initialize Feature Store
    # Assuming the repo path is local relative to this script execution
    repo_path = "feature_repo" 
    fs = FeatureStore(repo_path=repo_path)

    # 5. Define features to fetch
    features = [
        "movie_stats:duration_minutes",
        "movie_stats:rating",
        "movie_stats:avg_review_length",
        "movie_stats:avg_sentiment_score",
        # "movie_stats:genre" # Genre is categorical/string, usually needs encoding. Excluding for simple XGBoost/NN start.
    ]

    print("Fetching historical features from Feast...")
    
    # Train set
    training_df = fs.get_historical_features(
        entity_df=train_df,
        features=features
    ).to_df()

    # Test set
    testing_df = fs.get_historical_features(
        entity_df=test_df,
        features=features
    ).to_df()

    print(f"Enriched Train Shape: {training_df.shape}")
    print(f"Enriched Test Shape: {testing_df.shape}")

    # 6. Save enriched datasets
    if bucket:
        train_out = f"s3://{bucket}/processed/train_with_features.csv"
        test_out = f"s3://{bucket}/processed/test_with_features.csv"
        print(f"Saving to S3: {train_out}")
        training_df.to_csv(train_out, index=False)
        testing_df.to_csv(test_out, index=False)
    else:
        train_out = os.path.join(data_dir, "train_with_features.csv")
        test_out = os.path.join(data_dir, "test_with_features.csv")
        print(f"Saving locally: {train_out}")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        training_df.to_csv(train_out, index=False)
        testing_df.to_csv(test_out, index=False)

if __name__ == "__main__":
    create_dataset()
