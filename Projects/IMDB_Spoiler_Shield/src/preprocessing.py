import pandas as pd
import numpy as np
import re
import nltk
import ssl
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from data_loader import load_data
import os

# Global variable for lazy loading
STOP_WORDS = None

def ensure_stopwords():
    global STOP_WORDS
    if STOP_WORDS is None:
        try:
            nltk.data.find('corpora/stopwords.zip')
        except LookupError:
            print("Downloading NLTK stopwords...")
            # SSL Fix for Docker/Corporate networks
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))
        # Custom stop words from the notebook
        custom_stop_words = [
            '@', "'", '.', '"', '/', '!', ',', "'ve", "...", "n't", '$', "'s", 
            '"', "''", '..', '&', '*', ';', '”', '``', ':', '#', '!', '-', 
            '?', '%', "'d", "'m", '+', '++'
        ]
        stop_words.update(custom_stop_words)
        STOP_WORDS = stop_words

def clean_text(text):
    # Ensure stopwords are loaded
    ensure_stopwords()
    
    # Basic cleaning
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    
    # Replace non-alphabetic characters with space to avoid merging words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) 
    
    # Tokenize (split by whitespace)
    words = text.split()
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in STOP_WORDS]
    
    return " ".join(filtered_words)

def preprocess_data(data_dir="data", output_dir="data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load_data now returns a PATH, not a dataframe
    merged_path = load_data(data_dir)
    print(f"Loading merged data from {merged_path}...")
    df = pd.read_csv(merged_path)
    
    # Simple sampling for demonstration if dataset is huge.
    # In production/EC2 with S3, we might want to use the full dataset.
    # Uncomment to sample:
    # df = df.sample(n=5000, random_state=42)

    print("Cleaning text and removing stopwords...")
    df['clean_review'] = df['review_text'].apply(clean_text)
    
    # Label encoding
    df['label'] = df['is_spoiler'].astype(int)

    # Split
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        # Save to S3
        print(f"Saving processed data to S3 bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    else:
        # Save locally
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")
    
    return train_path, test_path

if __name__ == "__main__":
    preprocess_data()