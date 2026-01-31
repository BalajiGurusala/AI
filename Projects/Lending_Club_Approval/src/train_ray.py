import os
import sys
import joblib
import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.tensorflow
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback

# Import project modules
from src.data_loader import load_data, DEFAULT_BUCKET
from src.preprocessing import preprocess_lending_data, DROP_COLS

def upload_to_s3(local_path, s3_key):
    """
    Utility to upload a file to S3 if requested or in cloud mode.
    """
    upload_requested = os.getenv("UPLOAD_TO_S3", "false").lower() == "true"
    is_cloud = sys.platform == "linux" 
    
    if upload_requested or is_cloud:
        try:
            print(f"Attempting S3 upload for {local_path}...")
            s3 = boto3.client('s3')
            s3.upload_file(local_path, DEFAULT_BUCKET, s3_key)
            print(f"Successfully uploaded to s3://{DEFAULT_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"Warning: S3 upload failed for {local_path}: {e}")
    else:
        print(f"Skipping S3 upload for {local_path} (Local Mode). Set UPLOAD_TO_S3=true to enable.")

def prepare_data(scaler_path="data/minmax_scaler.pkl"):
    """
    Loads, preprocesses, and scales the data. Saves the scaler for inference.
    """
    try:
        df = load_data(file_key="loans.csv")
    except:
        df = pd.read_csv("data/loans.csv")
        
    df = preprocess_lending_data(df)
    
    target_col = 'loan_status'
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    X = X.select_dtypes(exclude=['object'])
    y = df[target_col]
    
    X = X.fillna(0)
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure directory exists
    if not os.path.exists(os.path.dirname(scaler_path)):
        os.makedirs(os.path.dirname(scaler_path))
        
    joblib.dump(scaler, scaler_path)
    
    # Upload scaler
    upload_to_s3(scaler_path, "artifacts/minmax_scaler.pkl")
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def get_xgboost_config(config):
    params = {
        "n_estimators": config.get("n_estimators", 100),
        "max_depth": config.get("max_depth", 6),
        "learning_rate": config.get("learning_rate", 0.1),
        "subsample": config.get("subsample", 1.0),
        "use_label_encoder": False,
        "eval_metric": 'logloss',
        "random_state": 60
    }
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and sys.platform != "darwin":
         params["device"] = "cuda"
         params["tree_method"] = "hist"
    return params

def train_lending_model(config, data=None):
    X_train, y_train, X_test, y_test = data
    model_type = config.get("model_type", "keras")

    if model_type == "logistic_regression":
        mlflow.sklearn.autolog()
        model = LogisticRegression(
            C=config.get("C", 1.0), 
            max_iter=1000,
            class_weight=config.get("class_weight", None),
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        train.report({"accuracy": acc, "loss": loss, "precision": precision, "recall": recall, "f1": f1})

    elif model_type == "decision_tree":
        mlflow.sklearn.autolog()
        model = DecisionTreeClassifier(
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", None),
            criterion=config.get("criterion", "gini"),
            random_state=60
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        train.report({"accuracy": acc, "loss": loss, "precision": precision, "recall": recall, "f1": f1})

    elif model_type == "random_forest":
        mlflow.sklearn.autolog()
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", "sqrt"),
            bootstrap=config.get("bootstrap", True),
            random_state=60
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        train.report({"accuracy": acc, "loss": loss, "precision": precision, "recall": recall, "f1": f1})

    elif model_type == "xgboost":
        mlflow.xgboost.autolog()
        xgb_params = get_xgboost_config(config)
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        train.report({"accuracy": acc, "loss": loss, "precision": precision, "recall": recall, "f1": f1})

    elif model_type == "keras":
        mlflow.tensorflow.autolog()
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(config.get("units_1", 64), activation="relu"),
            layers.Dropout(config.get("dropout_1", 0.2)),
            layers.Dense(config.get("units_2", 32), activation="relu"),
            layers.Dropout(config.get("dropout_2", 0.2)),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.get("lr", 0.001)),
            loss="binary_crossentropy", 
            metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        model.fit(
            X_train, y_train, epochs=config.get("epochs", 10), 
            batch_size=64, validation_data=(X_test, y_test), verbose=0,
            callbacks=[keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l: train.report({
                    "loss": l["val_loss"], 
                    "accuracy": l["val_accuracy"],
                    "precision": l.get("val_precision", 0),
                    "recall": l.get("val_recall", 0)
                })
            )]
        )

def save_best_model(config, data, output_dir="models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    X_train, y_train, X_test, y_test = data
    model_type = config.get("model_type", "keras")
    print(f"Retraining best model: {model_type}...")
    
    saved_path = ""
    
    if model_type == "logistic_regression":
        model = LogisticRegression(
            C=config.get("C", 1.0), 
            max_iter=1000,
            class_weight=config.get("class_weight", None),
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        saved_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(model, saved_path)

    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", None),
            criterion=config.get("criterion", "gini"),
            random_state=60
        )
        model.fit(X_train, y_train)
        saved_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(model, saved_path)

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", "sqrt"),
            bootstrap=config.get("bootstrap", True),
            random_state=60
        )
        model.fit(X_train, y_train)
        saved_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(model, saved_path)

    elif model_type == "xgboost":
        xgb_params = get_xgboost_config(config)
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        saved_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(model, saved_path)

    elif model_type == "keras":
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(config.get("units_1", 64), activation="relu"),
            layers.Dropout(config.get("dropout_1", 0.2)),
            layers.Dense(config.get("units_2", 32), activation="relu"),
            layers.Dropout(config.get("dropout_2", 0.2)),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.get("lr", 0.001)),
            loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.fit(
            X_train, y_train, epochs=config.get("epochs", 10), 
            batch_size=64, verbose=1
        )
        saved_path = os.path.join(output_dir, "best_model.keras")
        model.save(saved_path)
    
    print(f"Best model saved locally to {saved_path}")
    
    # Upload to S3 (uses basename, e.g. artifacts/best_model.keras)
    upload_to_s3(saved_path, f"artifacts/{os.path.basename(saved_path)}")

def get_tune_resources():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # On macOS, Ray doesn't manage Metal GPUs as 'gpu' resource. TF uses it implicitly.
            # Requesting gpu=1 will cause the scheduler to hang waiting for a resource that Ray doesn't see.
            if sys.platform == "darwin":
                return {"cpu": 4}
            return {"cpu": 4, "gpu": 1}
    except:
        pass
    return {"cpu": 4}

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    data_tuple = (X_train, y_train, X_test, y_test)

    # Comprehensive Search Space (RESTORED)
    search_space = {
        "model_type": tune.choice(["logistic_regression", "decision_tree", "random_forest", "xgboost", "keras"]),
        
        # Logistic Regression
        "C": tune.loguniform(1e-4, 100),
        "class_weight": tune.choice([None, "balanced"]),
        
        # Tree-based Shared
        "max_depth": tune.choice([None, 10, 20, 30, 40, 50]),
        "min_samples_split": tune.choice([2, 5, 10, 20]),
        "min_samples_leaf": tune.choice([1, 2, 4, 5, 10]),
        
        # Decision Tree & RF Specific
        "criterion": tune.choice(["gini", "entropy"]),
        "max_features": tune.choice([None, "sqrt", "log2"]),
        "bootstrap": tune.choice([True, False]),
        
        # Ensemble Specific
        "n_estimators": tune.choice([50, 100, 200, 300]),
        "learning_rate": tune.loguniform(1e-3, 0.3), # XGBoost
        "subsample": tune.uniform(0.5, 1.0), # XGBoost
        
        # Keras Specific
        "units_1": tune.choice([32, 64, 128, 256]),
        "dropout_1": tune.uniform(0.1, 0.5),
        "units_2": tune.choice([16, 32, 64]),
        "dropout_2": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": 10
    }

    if not ray.is_initialized():
        ray.init(num_cpus=14, ignore_reinit_error=True)

    tune_resources = get_tune_resources()
    print(f"Tune Resources: {tune_resources}")

    analysis = tune.run(
        tune.with_parameters(train_lending_model, data=data_tuple),
        config=search_space,
        num_samples=5, 
        scheduler=ASHAScheduler(metric="loss", mode="min"),
        resources_per_trial=tune_resources,
        callbacks=[MLflowLoggerCallback(experiment_name="Lending_Club_Cloud_Ready", save_artifact=False)]
    )
    
    best_config = analysis.get_best_config(metric="loss", mode="min")
    print("Best result: ", best_config)
    save_best_model(best_config, data_tuple)
