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

# --- Data Loading ---
def load_data_dfs(data_dir="data/processed"):
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        train_path = f"s3://{bucket}/processed/train.csv"
        test_path = f"s3://{bucket}/processed/test.csv"
    else:
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except:
        return pd.DataFrame({'clean_review': ['test'], 'label': [0]}), pd.DataFrame({'clean_review': ['test'], 'label': [0]})

    # Use full dataset
    # train_df = train_df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    
    train_df['clean_review'] = train_df['clean_review'].fillna('')
    test_df['clean_review'] = test_df['clean_review'].fillna('')
    return train_df, test_df

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.texts[idx]), add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

# --- Training Function ---
def train_func(config):
    model_name = config.get("model_name")
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 2e-5)
    epochs = config.get("epochs", 1)

    train_df, test_df = load_data_dfs()
    
    classes = np.unique(train_df['label'])
    class_weights = compute_class_weight('balanced', classes=classes, y=train_df['label']) if len(classes) > 1 else np.array([1.0, 1.0])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = prepare_model(model)
    
    train_loader = prepare_data_loader(DataLoader(ReviewDataset(train_df['clean_review'].values, train_df['label'].values, tokenizer), batch_size=batch_size, shuffle=True))
    test_loader = prepare_data_loader(DataLoader(ReviewDataset(test_df['clean_review'].values, test_df['label'].values, tokenizer), batch_size=batch_size*2, shuffle=False))

    device = train.torch.get_device()
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss = loss_fn(outputs.logits, batch['labels'].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), os.path.join(temp_dir, "model.pt"))
            train.report({"accuracy": acc, "f1_score": f1, "epoch": epoch}, checkpoint=Checkpoint.from_directory(temp_dir))

def run_training():
    models_to_train = ["distilbert-base-uncased", "bert-base-uncased"]
    
    use_gpu = torch.cuda.is_available()
    num_workers = torch.cuda.device_count() if use_gpu else 1
    
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    for model_name in models_to_train:
        print(f"\n>>> Starting Ray Training for: {model_name}")
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={"model_name": model_name, "batch_size": 16, "epochs": 1, "lr": 2e-5},
            scaling_config=scaling_config,
        )
        result = trainer.fit()
        
        # MLflow & S3
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("IMDB_Spoiler_Shield_BERT_Ray")
        
        with mlflow.start_run(run_name=f"Ray_{model_name}"):
            mlflow.log_params({"model_name": model_name})
            mlflow.log_metrics({"accuracy": result.metrics["accuracy"], "f1_score": result.metrics["f1_score"]})
            
            if result.checkpoint:
                with result.checkpoint.as_directory() as checkpoint_dir:
                    local_path = f"models/{model_name.replace('-', '_')}.pt"
                    os.makedirs("models", exist_ok=True)
                    import shutil
                    shutil.copy(os.path.join(checkpoint_dir, "model.pt"), local_path)
                    
                    bucket = os.getenv("S3_BUCKET")
                    if bucket:
                        boto3.client('s3').upload_file(local_path, bucket, local_path)
                        print(f"Uploaded {model_name} to S3.")

if __name__ == "__main__":
    run_training()
