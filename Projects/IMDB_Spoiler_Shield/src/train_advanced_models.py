import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import os
import joblib
import json
import boto3

# Device config with MPS support
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

def load_enriched_data(data_dir="data/processed"):
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        print(f"Loading data from S3 bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train_with_features.csv"
        test_path = f"s3://{bucket}/processed/test_with_features.csv"
    else:
        train_path = os.path.join(data_dir, "train_with_features.csv")
        test_path = os.path.join(data_dir, "test_with_features.csv")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Enriched data not found. Please run src/create_dataset_with_feast.py first.")
        raise

    # Fill NaN
    train_df['clean_review'] = train_df['clean_review'].fillna('')
    test_df['clean_review'] = test_df['clean_review'].fillna('')
    
    # Fill Numerical NaNs (if any from Feast join)
    feat_cols = ["duration_minutes", "rating", "avg_review_length", "avg_sentiment_score"]
    for col in feat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(train_df[col].mean()) # Fill test with train mean
    
    return train_df, test_df

def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    print(f"Generating embeddings using {model_name}...")
    model = SentenceTransformer(model_name, device=str(device))
    embeddings = model.encode(texts.tolist(), show_progress_bar=True, batch_size=32)
    return embeddings

# Updated NN to handle combined input
class CombinedNN(nn.Module):
    def __init__(self, input_dim):
        super(CombinedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_neural_network(X_train, y_train, X_test, y_test, class_weights):
    print("Training Neural Network...")
    input_dim = X_train.shape[1]
    model = CombinedNN(input_dim).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Weighted BCEWithLogitsLoss
    pos_weight = torch.tensor([class_weights[1]/class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Loop
    epochs = 10
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds_cls = (torch.sigmoid(logits) > 0.5).float()
        acc = accuracy_score(y_test, preds_cls.cpu().numpy())
        f1 = f1_score(y_test, preds_cls.cpu().numpy())
    
    return model, acc, f1

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("IMDB_Spoiler_Shield_Advanced_Feast")
    
    # 1. Load Data (Enriched with Feast Features)
    train_df, test_df = load_enriched_data()
    
    # Use full dataset
    # train_df = train_df.sample(frac=0.1, random_state=42)
    # test_df = test_df.sample(frac=0.1, random_state=42)
    
    y_train = train_df['label']
    y_test = test_df['label']
    
    # 2. Process Features
    # A. Text Embeddings
    print("Processing Text...")
    X_train_emb = get_embeddings(train_df['clean_review'].astype(str))
    X_test_emb = get_embeddings(test_df['clean_review'].astype(str))
    
    # B. Numerical Features from Feast
    print("Processing Numerical Features...")
    feat_cols = ["duration_minutes", "rating", "avg_review_length", "avg_sentiment_score"]
    
    # Check if cols exist (in case Feast join failed silently or partial data)
    valid_cols = [c for c in feat_cols if c in train_df.columns]
    
    if valid_cols:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(train_df[valid_cols])
        X_test_num = scaler.transform(test_df[valid_cols])
        
        # Concatenate
        X_train_combined = np.hstack((X_train_emb, X_train_num))
        X_test_combined = np.hstack((X_test_emb, X_test_num))
        print(f"Combined Feature Shape: {X_train_combined.shape}")
    else:
        print("Warning: Feast features not found in columns. Using only text embeddings.")
        X_train_combined = X_train_emb
        X_test_combined = X_test_emb
    
    # Compute class weights
    classes = np.unique(y_train)
    if len(classes) > 1:
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {0: weights[0], 1: weights[1]}
    else:
        weights = np.array([1.0, 1.0])
        class_weight_dict = {0: 1.0, 1: 1.0}
    
    bucket = os.getenv("S3_BUCKET")
    if not os.path.exists("models"):
        os.makedirs("models")

    # 3. XGBoost (Combined Features)
    with mlflow.start_run(run_name="XGBoost_Feast"):
        print("\nTraining XGBoost (Feast Enriched)...")
        xgb_clf = xgb.XGBClassifier(
            scale_pos_weight=weights[1]/weights[0],
            use_label_encoder=False,
            eval_metric='logloss',
            device="cuda" if device.type == "cuda" else "cpu"
        )
        xgb_clf.fit(X_train_combined, y_train)
        y_pred = xgb_clf.predict(X_test_combined)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"XGBoost - Acc: {acc:.4f}, F1: {f1:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.xgboost.log_model(xgb_clf, "xgboost_model")
        
        joblib.dump(xgb_clf, "models/xgboost_feast_model.joblib")
        if bucket:
            boto3.client('s3').upload_file("models/xgboost_feast_model.joblib", bucket, "models/xgboost_model.joblib")

    # 4. Neural Network (Combined Features)
    with mlflow.start_run(run_name="NN_Feast"):
        nn_model, nn_acc, nn_f1 = train_neural_network(X_train_combined, y_train, X_test_combined, y_test, weights)
        print(f"Neural Net - Acc: {nn_acc:.4f}, F1: {nn_f1:.4f}")
        
        mlflow.log_metric("accuracy", nn_acc)
        mlflow.log_metric("f1", nn_f1)
        mlflow.pytorch.log_model(nn_model, "nn_model")
        
        torch.save(nn_model.state_dict(), "models/nn_feast_model.pt")
        if bucket:
            boto3.client('s3').upload_file("models/nn_feast_model.pt", bucket, "models/nn_model.pt")

    # 5. Logistic Regression with SBERT + Feast (Comparison)
    with mlflow.start_run(run_name="LogReg_SBERT_Feast"):
        print("\nTraining LogReg (SBERT + Feast)...")
        lr_clf = LogisticRegression(class_weight=class_weight_dict, max_iter=1000)
        lr_clf.fit(X_train_combined, y_train)
        y_pred = lr_clf.predict(X_test_combined)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"LogReg (SBERT + Feast) - Acc: {acc:.4f}, F1: {f1:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        
        joblib.dump(lr_clf, "models/logreg_sbert_feast.joblib")
        if bucket:
            boto3.client('s3').upload_file("models/logreg_sbert_feast.joblib", bucket, "models/logreg_sbert.joblib")

if __name__ == "__main__":
    main()