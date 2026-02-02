import os
import gdown
import json
import pandas as pd
import boto3
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

    details_file_id = "1wkPlzRwOKSj3tjvX3wOnrSJj8iFPumqS"
    reviews_file_id = "1hvzjvoGCwGMYyQ0tYj_FBnV-Ett0_ZTh"

    reviews_file_path = os.path.join(data_dir, "IMDB_reviews.json")
    details_file_path = os.path.join(data_dir, "IMDB_movie_details.json")

    # Check S3 first if configured
    bucket = os.getenv("S3_BUCKET")
    s3_downloaded = False
    if bucket:
        print(f"Checking S3 bucket {bucket} for data...")
        s3 = boto3.client('s3')
        try:
            s3.download_file(bucket, "raw/IMDB_reviews.json", reviews_file_path)
            s3.download_file(bucket, "raw/IMDB_movie_details.json", details_file_path)
            print("Downloaded data from S3.")
            s3_downloaded = True
        except Exception as e:
            print(f"Could not download from S3: {e}. Falling back to GDown.")

    if not s3_downloaded:
        if not os.path.exists(reviews_file_path):
            print(f"Downloading reviews to {reviews_file_path} from Drive...")
            gdown.download(id=reviews_file_id, output=reviews_file_path, quiet=False)

        if not os.path.exists(details_file_path):
            print(f"Downloading details to {details_file_path} from Drive...")
            gdown.download(id=details_file_id, output=details_file_path, quiet=False)
        
        # Upload to S3 for future use
        if bucket:
            print("Uploading raw data to S3...")
            upload_to_s3(reviews_file_path, bucket, "raw/IMDB_reviews.json")
            upload_to_s3(details_file_path, bucket, "raw/IMDB_movie_details.json")

    return reviews_file_path, details_file_path

def load_data(data_dir="data"):
    reviews_path, details_path = download_data(data_dir)

    print("Loading reviews...")
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
    return df_both

if __name__ == "__main__":
    load_data()