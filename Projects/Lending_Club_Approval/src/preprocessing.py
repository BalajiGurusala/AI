import pandas as pd
import numpy as np

def preprocess_lending_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the lending club dataset by handling missing values,
    cleaning string columns, mapping categorical values to numeric,
    and performing one-hot encoding.
    
    Refactored from Balaji_Gurusala_LendingClubLoanApprovalML.ipynb.
    
    Args:
        df (pd.DataFrame): Raw lending club dataframe.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for splitting and modeling.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # 1. Missing Value Treatment
    # For revol_util and emp_length, fill missing values with the mode
    if 'revol_util' in df.columns:
        revol_util_mode = df['revol_util'].mode()[0]
        df['revol_util'] = df['revol_util'].fillna(revol_util_mode)
        
    if 'emp_length' in df.columns:
        emp_length_mode = df['emp_length'].mode()[0]
        df['emp_length'] = df['emp_length'].fillna(emp_length_mode)

    # 2. String Cleaning & Numeric Conversion
    
    # Remove 'months' suffix and convert to int
    if 'term' in df.columns:
        df['term'] = df['term'].str.replace('months', '').astype(int)
        
    # Remove '%' symbol and convert to float
    if 'int_rate' in df.columns:
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
        
    if 'revol_util' in df.columns:
        # Check if it's already float (from previous steps or input)
        if df['revol_util'].dtype == 'object':
            df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)
        
    # Handle earliest_cr_line to get earliest_cr_year
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_year'] = pd.to_datetime(df['earliest_cr_line']).dt.year
    else:
        # Placeholder for earliest_cr_year if missing
        df['earliest_cr_year'] = 2000 

    # Clean emp_length and convert to numeric using mapping
    if 'emp_length' in df.columns and df['emp_length'].dtype == 'object':
        df['emp_length'] = df['emp_length'].str.replace(r'[a-zA-Z]', '', regex=True)
        df['emp_length'] = df['emp_length'].str.replace(' ', '')
        emp_len_map = {
            '<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10+': 10
        }
        df['emp_length'] = df['emp_length'].map(emp_len_map).fillna(0).astype(int)

    # 3. Categorical Mapping
    
    # Map grade
    if 'grade' in df.columns:
        grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df['grade'] = df['grade'].map(grade_map)
        
    # Map sub_grade
    if 'sub_grade' in df.columns:
        sub_grade_map = {
            'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4,
            'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9,
            'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14,
            'D1': 15, 'D2': 16, 'D3': 17, 'D4': 18, 'D5': 19,
            'E1': 20, 'E2': 21, 'E3': 22, 'E4': 23, 'E5': 24,
            'F1': 25, 'F2': 26, 'F3': 27, 'F4': 28, 'F5': 29,
            'G1': 30, 'G2': 31, 'G3': 32, 'G4': 33, 'G5': 34
        }
        df['sub_grade'] = df['sub_grade'].map(sub_grade_map)
        
    # Map loan_status (Target)
    if 'loan_status' in df.columns:
        loan_status_map = {
            'Fully Paid': 0,
            'Charged Off': 1,
            'Default': 1
        }
        df['loan_status'] = df['loan_status'].map(loan_status_map)

    # 4. One-Hot Encoding (KEEP home_ownership for Feast)
    if 'home_ownership' in df.columns:
        df['home_ownership_orig'] = df['home_ownership'] # Keep original
    
    ohe_columns = ['home_ownership', 'verification_status', 'purpose']
    # Filter only columns that actually exist in the dataframe
    ohe_columns = [col for col in ohe_columns if col in df.columns]
    if ohe_columns:
        df = pd.get_dummies(data=df, columns=ohe_columns, drop_first=True)

    if 'home_ownership_orig' in df.columns:
        df['home_ownership'] = df['home_ownership_orig']
        df = df.drop(columns=['home_ownership_orig'])

    # 5. Add event_timestamp for Feast
    if 'event_timestamp' not in df.columns:
        df['event_timestamp'] = pd.to_datetime('now')

    # Rename id to borrower_id or create it
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'borrower_id'})
    elif 'borrower_id' not in df.columns:
        df['borrower_id'] = range(len(df))

    # Ensure other columns exist for the schema
    for col in ['mort_acc', 'pub_rec_bankruptcies', 'pub_rec']:
        if col not in df.columns:
            df[col] = 0.0

    # Ensure integer type for earliest_cr_year
    df['earliest_cr_year'] = df['earliest_cr_year'].astype(np.int64)

    # 6. Drop Irrelevant Columns
    cols_to_drop = ['member_id', 'installment', 'last_pymnt_amnt']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    return df

# Columns to drop after preprocessing but before scaling/training
# These are IDs, targets, or string columns retained for Feast/Analysis but not for the ML model
DROP_COLS = [
    'loan_status', 
    'borrower_id', 
    'event_timestamp', 
    'id', 
    'member_id', 
    'home_ownership', # String retained for Feast, dropped for model (OHE exists)
    'emp_title', 
    'issue_d', 
    'title', 
    'zip_code', 
    'addr_state', 
    'earliest_cr_line', 
    'application_type', 
    'initial_list_status', 
    'last_pymnt_d', 
    'next_pymnt_d', 
    'last_credit_pull_d'
]
