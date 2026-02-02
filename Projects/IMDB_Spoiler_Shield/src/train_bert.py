import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os
import mlflow
import boto3
import json
import tempfile

import ray
from ray import train, tune
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader

# --- Data Loading (S3 Aware) ---
def load_data_dfs(data_dir="data/processed"):
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        print(f"Loading data from S3 bucket: {bucket}")
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
    else:
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return dummy data for testing if files don't exist
        return pd.DataFrame({'clean_review': ['test'], 'label': [0]}), pd.DataFrame({'clean_review': ['test'], 'label': [0]})

    # 30% subsample for BERT training as per notebook strategy
    train_df = train_df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    
    # Fill NaN
    train_df['clean_review'] = train_df['clean_review'].fillna('')
    test_df['clean_review'] = test_df['clean_review'].fillna('')
    
    return train_df, test_df

# --- Dataset Class ---
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Training Function (Per Worker) ---
def train_func(config):
    # Hyperparameters from config
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 2e-5)
    epochs = config.get("epochs", 2)
    data_dir = config.get("data_dir", "data/processed")

    # Load Data (Each worker loads data - efficient for small datasets, 
    # for huge datasets Ray Data is preferred but sticking to pandas for simplicity here)
    train_df, test_df = load_data_dfs(data_dir)
    
    X_train = train_df['clean_review'].values
    y_train = train_df['label'].values
    X_test = test_df['clean_review'].values
    y_test = test_df['label'].values

    # Compute Class Weights (on CPU numpy before converting to tensor)
    classes = np.unique(y_train)
    # Handle edge case with single class in subset
    if len(classes) > 1:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    else:
        class_weights = np.array([1.0, 1.0])
    
    # Model & Tokenizer
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Prepare Model for Distributed Training (Ray handles DDP/FSDP wrapping)
    model = prepare_model(model)
    
    # Datasets
    train_ds = ReviewDataset(X_train, y_train, tokenizer)
    test_ds = ReviewDataset(X_test, y_test, tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    
    # Prepare DataLoaders (Ray shards data across workers)
    train_loader = prepare_data_loader(train_loader)
    test_loader = prepare_data_loader(test_loader)

    # Optimizer & Loss
    # Get device from model (Ray places it correctly)
    device = train.torch.get_device()
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        val_steps = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Report metrics to Ray Train/Tune
        metrics = {
            "loss": avg_val_loss,
            "accuracy": val_acc,
            "f1_score": val_f1,
            "epoch": epoch
        }
        
        # Save Checkpoint
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # We must access the underlying model for saving if wrapped
            if hasattr(model, "module"):
                torch.save(model.module.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)

def run_training():
    # 1. Detect Environment and Config Resources
    use_gpu = False
    resources_per_worker = {"CPU": 1}
    num_workers = 1
    
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU(s).")
        use_gpu = True
        num_gpus = torch.cuda.device_count()
        # Scale to all available GPUs
        num_workers = num_gpus
        resources_per_worker = {"GPU": 1, "CPU": 2} # Assume 2 CPUs per GPU
        print(f"Launching {num_workers} workers with GPU support.")
        
    elif torch.backends.mps.is_available():
        print("Apple Metal (MPS) detected. Using Local Execution with MPS.")
        # Ray Train distributed backend doesn't fully support MPS via NCCL/Gloo efficiently yet.
        # We set 1 worker and will force the device inside the training function if needed,
        # but usually 'use_gpu=True' in Ray implies CUDA/ROCm.
        # For MPS, we often treat it as a single local worker without 'use_gpu=True' in Ray ScalingConfig
        # but force the device to MPS in PyTorch.
        use_gpu = False 
        num_workers = 1
        # We will manually handle device placement for MPS if Ray doesn't assign it.
    else:
        print("No GPU detected. Using CPU.")
        num_workers = 1

    # 2. Define Scaling Config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker
    )

    # 3. Setup TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "batch_size": 16, 
            "epochs": 2, 
            "lr": 2e-5,
            "data_dir": "data/processed"
        },
        scaling_config=scaling_config,
    )

    # 4. Run Training
    print("Starting Ray Training...")
    result = trainer.fit()
    
    print(f"Training completed.\nMetrics: {result.metrics}")
    print(f"Checkpoint saved at: {result.checkpoint}")

    # 5. MLflow Logging (Log final best result)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("IMDB_Spoiler_Shield_BERT_Ray")
    
    with mlflow.start_run():
        mlflow.log_metrics({
            "accuracy": result.metrics.get("accuracy", 0),
            "f1_score": result.metrics.get("f1_score", 0),
            "loss": result.metrics.get("loss", 0)
        })
        
        # Handle Artifacts (S3)
        # Note: Ray saves checkpoints locally. We can upload the best one.
        if result.checkpoint:
            with result.checkpoint.as_directory() as checkpoint_dir:
                model_path = os.path.join(checkpoint_dir, "model.pt")
                
                # Save locally to project models/
                if not os.path.exists("models"):
                    os.makedirs("models")
                import shutil
                shutil.copy(model_path, "models/bert_ray_model.pt")
                print("Model copied to models/bert_ray_model.pt")

                bucket = os.getenv("S3_BUCKET")
                if bucket:
                    s3 = boto3.client('s3')
                    s3.upload_file("models/bert_ray_model.pt", bucket, "models/bert_model.pt")
                    print("Model uploaded to S3.")

if __name__ == "__main__":
    # If running on Mac with MPS, set explicit var to help PyTorch find it if Ray doesn't
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
    run_training()