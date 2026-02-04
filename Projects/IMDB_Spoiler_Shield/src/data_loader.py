import os
import json
import pandas as pd
import boto3
import shutil
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print(f"Upload Successful: {local_file} to s3://{bucket}/{s3_file}")
        return True
    except FileNotFoundError:
        print(f"The file was not found: {local_file}")
        return False
    except NoCredentialsError:
        print("Credentials not available for AWS S3")
        return False
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return False

def download_data(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    reviews_file = "IMDB_reviews.json"
    details_file = "IMDB_movie_details.json"
    
    reviews_file_path = os.path.join(data_dir, reviews_file)
    details_file_path = os.path.join(data_dir, details_file)

    # Check S3 first if configured
    bucket = os.getenv("S3_BUCKET")
    s3_loaded = False
    
    if bucket:
        print(f"Checking S3 bucket {bucket} for data...")
        s3 = boto3.client('s3')
        try:
            # Only download if local doesn't exist
            if not os.path.exists(reviews_file_path):
                s3.download_file(bucket, f"raw/{reviews_file}", reviews_file_path)
            if not os.path.exists(details_file_path):
                s3.download_file(bucket, f"raw/{details_file}", details_file_path)
            print("Loaded data from S3.")
            s3_loaded = True
        except Exception as e:
            print(f"S3 interaction error: {e}. Checking local /data directory.")

    # Local Path Check (Primary for local runs)
    if not os.path.exists(reviews_file_path):
        root_reviews_path = os.path.join("/data", reviews_file)
        if os.path.exists(root_reviews_path):
            print(f"Copying {reviews_file} from /data...")
            shutil.copy(root_reviews_path, reviews_file_path)
        else:
            print(f"Warning: {reviews_file} not found in {data_dir} or /data")

    if not os.path.exists(details_file_path):
        root_details_path = os.path.join("/data", details_file)
        if os.path.exists(root_details_path):
            print(f"Copying {details_file} from /data...")
            shutil.copy(root_details_path, details_file_path)
        else:
            print(f"Warning: {details_file} not found in {data_dir} or /data")

    # If bucket is set but files were loaded from local /data, sync them to S3
    if bucket and not s3_loaded and os.path.exists(reviews_file_path):
        print("Syncing local raw data to S3...")
        upload_to_s3(reviews_file_path, bucket, f"raw/{reviews_file}")
        upload_to_s3(details_file_path, bucket, f"raw/{details_file}")

    return reviews_file_path, details_file_path

def load_data(data_dir="data"):
    reviews_path, details_path = download_data(data_dir)

    if not os.path.exists(reviews_path) or not os.path.exists(details_path):
        raise FileNotFoundError("Raw data files missing. Ensure they are in /data or S3.")

    print("Loading reviews...")
    # Use chunking if possible, but for now we load full
    # For OOM prevention, we will just ensure we don't return the big DF
    with open(reviews_path, 'r') as file:
        reviews_data = [json.loads(line) for line in file]
    df_reviews = pd.DataFrame(reviews_data)

    print("Loading details...")
    with open(details_path, 'r') as file:
        details_data = [json.loads(line) for line in file]
    df_details = pd.DataFrame(details_data)

    # Merge
    print("Merging datasets...")
    df_both = pd.merge(df_reviews, df_details, on='movie_id')
    
    # Save merged data to disk instead of returning
    merged_path = os.path.join(data_dir, "merged_raw.csv")
    print(f"Saving merged data to {merged_path}...")
    df_both.to_csv(merged_path, index=False)
    
    # Return path string (Tiny memory footprint)
    return merged_path

if __name__ == "__main__":
    load_data()