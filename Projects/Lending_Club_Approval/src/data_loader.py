import pandas as pd
import boto3
from io import BytesIO
import os

DEFAULT_BUCKET = "ik-lending-club-bucket"
DEFAULT_FILE_KEY = "lending_club_loans.csv"

def load_data(bucket_name: str = DEFAULT_BUCKET, file_key: str = DEFAULT_FILE_KEY) -> pd.DataFrame:
    """
    Loads the Lending Club dataset from an S3 bucket or local path.
    
    Args:
        bucket_name: Name of the S3 bucket.
        file_key: Key (path) to the file in the bucket.
        
    Returns:
        pd.DataFrame: The raw dataframe.
    """
    # Check if we are running locally and have a local copy (faster for dev)
    local_path = os.path.join("data", file_key)
    if os.path.exists(local_path):
        print(f"Loading data from local file: {local_path}")
        return pd.read_csv(local_path)

    print(f"Downloading data from S3: s3://{bucket_name}/{file_key}")
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
        return df
    except Exception as e:
        print(f"Error loading from S3: {e}")
        raise e

if __name__ == "__main__":
    # Simple test to verify it runs
    try:
        df = load_data()
        print(f"Successfully loaded data with shape: {df.shape}")
    except:
        print("Could not load data. Ensure file exists in 'data/' folder or S3 credentials are set.")