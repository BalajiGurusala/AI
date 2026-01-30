import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_lending_data

def main():
    # Load data (it should find data/loans.csv based on how load_data is written)
    # The file is actually data/loans.csv, but data_loader defaults to lending_club_loans.csv
    # I will specify the file key explicitly.
    try:
        df = load_data(file_key="loans.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to direct read if data_loader fails
        df = pd.read_csv("data/loans.csv")
        
    print(f"Original shape: {df.shape}")
    
    # Preprocess
    df_cleaned = preprocess_lending_data(df)
    
    # Ensure Feast requirements
    if 'id' not in df_cleaned.columns:
         # If for some reason id was missing from raw data, create one
         df_cleaned['id'] = range(len(df_cleaned))
    
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Columns: {df_cleaned.columns.tolist()[:10]}...")
    
    # Save to Parquet
    output_path = "data/lending_club_cleaned.parquet"
    df_cleaned.to_parquet(output_path, index=False)
    print(f"Successfully saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()
