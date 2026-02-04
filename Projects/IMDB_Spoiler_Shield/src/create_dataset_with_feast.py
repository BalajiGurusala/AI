import pandas as pd
import os
from datetime import datetime
from feast import FeatureStore
import boto3

def process_in_chunks(fs, entity_df_path, output_path, features, chunksize=50000):
    """
    Reads entity_df in chunks, fetches features, and appends to output CSV.
    This manages memory usage for large datasets.
    """
    first_chunk = True
    
    # Iterate over chunks to avoid OOM
    for chunk_df in pd.read_csv(entity_df_path, chunksize=chunksize):
        # Feast requires event_timestamp for point-in-time correctness
        chunk_df['event_timestamp'] = pd.to_datetime(datetime.now())
        
        print(f"Fetching historical features from Feast for chunk of size {len(chunk_df)}...")
        # Historical retrieval performs the join between the entity (movie_id) and the features
        enriched_df = fs.get_historical_features(
            entity_df=chunk_df,
            features=features
        ).to_df()
        
        # Write to local file. 'w' for first chunk, 'a' (append) for subsequent
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        enriched_df.to_csv(output_path, mode=mode, header=header, index=False)
        
        first_chunk = False
        print(f"Appended enriched chunk to {output_path}")

def create_dataset():
    # 1. Setup paths
    bucket = os.getenv("S3_BUCKET")
    data_dir = "data/processed"
    os.makedirs(data_dir, exist_ok=True)
    
    train_out_local = os.path.join(data_dir, "train_with_features.csv")
    test_out_local = os.path.join(data_dir, "test_with_features.csv")

    if bucket:
        print(f"S3 integration active. Bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
    else:
        print("Local mode active. Using data/processed.")
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

    # 2. Initialize Feast Store
    # Path is relative to project root
    repo_path = "feature_repo" 
    fs = FeatureStore(repo_path=repo_path)

    # 3. Define features to join
    features = [
        "movie_stats:duration_minutes",
        "movie_stats:rating",
        "movie_stats:avg_review_length",
        "movie_stats:avg_sentiment_score",
    ]

    # 4. Process Train and Test datasets
    print(f"Enriching Train Data: {train_path}")
    process_in_chunks(fs, train_path, train_out_local, features)
    
    print(f"Enriching Test Data: {test_path}")
    process_in_chunks(fs, test_path, test_out_local, features)

    # 5. Sync to Cloud
    if bucket:
        s3 = boto3.client('s3')
        print(f"Uploading enriched datasets to s3://{bucket}/processed/...")
        try:
            s3.upload_file(train_out_local, bucket, "processed/train_with_features.csv")
            s3.upload_file(test_out_local, bucket, "processed/test_with_features.csv")
            print("S3 Sync Complete.")
        except Exception as e:
            print(f"S3 Upload failed: {e}")

if __name__ == "__main__":
    create_dataset()