import pandas as pd
import os
from datetime import datetime
import subprocess
import json
import boto3
from src.data_loader import load_data

def convert_duration_to_minutes(duration_str):
    if pd.isna(duration_str) or not isinstance(duration_str, str):
        return 0.0
    hours = 0
    minutes = 0
    parts = duration_str.split()
    for part in parts:
        if 'h' in part:
            try: hours = int(part.replace('h', ''))
            except: pass
        elif 'min' in part:
            try: minutes = int(part.replace('min', ''))
            except: pass
    return float(hours * 60 + minutes)

def generate_features(data_dir="data", output_dir="data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bucket = os.getenv("S3_BUCKET")
    reviews_path = os.path.join(data_dir, "IMDB_reviews.json")
    details_path = os.path.join(data_dir, "IMDB_movie_details.json")

    print("Loading data for feature engineering...")
    
    # Download if not present
    if not os.path.exists(reviews_path) or not os.path.exists(details_path):
        load_data(data_dir)

    # Load Details
    with open(details_path, 'r') as file:
        details_data = [json.loads(line) for line in file]
    df_details = pd.DataFrame(details_data)

    # Load Reviews to calculate aggregated stats
    with open(reviews_path, 'r') as file:
        reviews_data = [json.loads(line) for line in file]
    df_reviews = pd.DataFrame(reviews_data)

    print("Aggregating review features...")
    # Calculate review length
    df_reviews['review_length'] = df_reviews['review_text'].apply(lambda x: len(str(x)))
    # Dummy sentiment score based on rating (normalized 0 to 1)
    df_reviews['sentiment_score'] = pd.to_numeric(df_reviews['rating'], errors='coerce').fillna(5.0) / 10.0

    # Group by movie_id
    movie_reviews_agg = df_reviews.groupby('movie_id').agg({
        'review_length': 'mean',
        'sentiment_score': 'mean'
    }).reset_index().rename(columns={
        'review_length': 'avg_review_length',
        'sentiment_score': 'avg_sentiment_score'
    })

    print("Merging features...")
    df_details = df_details.merge(movie_reviews_agg, on='movie_id', how='left')
    df_details['avg_review_length'] = df_details['avg_review_length'].fillna(0.0)
    df_details['avg_sentiment_score'] = df_details['avg_sentiment_score'].fillna(0.5)

    # Engineering core features
    df_details['duration_minutes'] = df_details['duration'].apply(convert_duration_to_minutes)
    df_details['rating'] = pd.to_numeric(df_details['rating'], errors='coerce').fillna(0.0)
    df_details['genre'] = df_details['genre'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")

    # Feast timestamps
    df_details['event_timestamp'] = pd.to_datetime("2020-01-01")
    df_details['created_timestamp'] = pd.to_datetime(datetime.now())

    feature_df = df_details[['movie_id', 'duration_minutes', 'rating', 'genre', 'avg_review_length', 'avg_sentiment_score', 'event_timestamp', 'created_timestamp']]
    
    if bucket:
        parquet_path = f"s3://{bucket}/processed/movie_features.parquet"
        print(f"Saving features to S3: {parquet_path}")
        feature_df.to_parquet(parquet_path, index=False)
    else:
        parquet_path = os.path.join(output_dir, "movie_features.parquet")
        print(f"Saving features locally: {parquet_path}")
        feature_df.to_parquet(parquet_path, index=False)

    # Feast Ops
    repo_path = "feature_repo" # Relative to project root
    print("Applying Feast registry...")
    subprocess.run(["feast", "apply"], cwd=repo_path, check=True)

    print("Materializing features to Online Store...")
    subprocess.run(["feast", "materialize-incremental", datetime.now().isoformat()], cwd=repo_path, check=True)

if __name__ == "__main__":
    generate_features()
