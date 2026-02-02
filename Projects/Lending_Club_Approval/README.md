# Lending Club Credit Scoring MLOps Pipeline

A production-grade MLOps system for real-time credit risk assessment. This project demonstrates an end-to-end machine learning lifecycle, from feature engineering and distributed training to orchestration and model serving.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![Ray](https://img.shields.io/badge/Training-Ray%20Tune-orange)
![Feast](https://img.shields.io/badge/Feature%20Store-Feast-purple)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-teal)
![Airflow](https://img.shields.io/badge/Orchestration-Airflow-red)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## Overview Video

https://drive.google.com/file/d/1RAaVF-X9JN8CT3bUdmTxCHuDdqvQRTfA/view?usp=sharing

## 📖 Project Overview

This system predicts the probability of loan default for Lending Club applicants. Unlike simple notebook experiments, this project implements a **robust, scalable MLOps architecture** designed for real-world deployment. It solves key challenges such as training-serving skew, model versioning, and automated retraining.

**Key Capabilities:**
*   **Real-time Inference**: Sub-second credit scoring using an HTTP API.
*   **Feature Consistency**: Uses **Feast** to serve the exact same feature definitions for training and inference.
*   **Distributed Tuning**: Leverages **Ray Tune** to optimize hyperparameters across multiple model architectures (XGBoost, Keras, Random Forest) in parallel.
*   **Experiment Tracking**: Full lineage of metrics, parameters, and artifacts via **MLflow**.
*   **Automated Pipelines**: **Apache Airflow** DAGs orchestrate data materialization and model retraining schedules.
*   **Cloud Native**: Dockerized services ready for deployment on Kubernetes or EC2.

---

## 🏗️ System Architecture

```text
+------------------+        +----------------------+        +-------------------------+
|   Data Source    | -----> | Feast Feature Store  | -----> |  Online Store (SQLite)  |
|   (loans.csv)    |        |   (Offline Store)    |        | (Low Latency Retrieval) |
+--------+---------+        +----------------------+        +-----------+-------------+
         |                                                              ^
         | Batch Load                                                   |
         v                                                              | Fetch Features
+--------+---------+                                         +----------+-----------+
|  Training Engine |                                         |    FastAPI Serving   |
|    (Ray Tune)    | <-------------------------------------- |      (Inference)     |
+--------+---------+          (Triggered by Airflow)         +----------+-----------+
         |                                                              ^
         | Log Metrics                                                  | Load Model
         v                                                              |
+--------+---------+        +----------------------+        +-----------+-----------+
| MLflow Tracking  |        |   Model Artifacts    | -----> |   Best Model.keras    |
| (Exp Management) |        |     (S3 / Local)     |        |   MinMax Scaler.pkl   |
+------------------+        +----------------------+        +-----------------------+
```

The system is composed of four main decoupled services orchestrated by Docker Compose:

1.  **Feature Store (Feast)**:
    *   Ingests data from `data/lending_club_cleaned.parquet`.
    *   Serves low-latency features via an online store (SQLite/Redis) to the API.
    *   Ensures point-in-time correctness for historical data.

2.  **Training Engine (Ray + MLflow)**:
    *   `src/train_ray.py`: A script that performs distributed hyperparameter tuning.
    *   Optimizes models: Logistic Regression, Decision Trees, Random Forests, XGBoost, and Deep Learning (Keras).
    *   Logs best models and scalers to MLflow and local/S3 artifacts.

3.  **Model Serving (FastAPI)**:
    *   `app/main.py`: A REST API that loads the best model and scaler.
    *   Fetches real-time features from Feast based on `borrower_id`.
    *   Merges features with live request data to produce a risk score.

4.  **Orchestrator (Airflow)**:
    *   `dags/lending_club_training_dag.py`: Manages the lifecycle.
    *   **Task 1**: Sync Feature Store (`feast materialize`).
    *   **Task 2**: Trigger Model Training (`python src/train_ray.py`).

---

## 📂 Project Structure

```
├── app/
│   └── main.py              # FastAPI application for inference
├── config/                  # Configuration files
├── dags/
│   └── lending_club_training_dag.py  # Airflow pipeline definition
├── data/
│   ├── loans.csv            # Raw dataset
│   ├── lending_club_cleaned.parquet # Processed data for Feature Store
│   └── online_store.db      # Feast online store (SQLite)
├── docker/                  # Docker-specific resources
├── feature_repo/            # Feast Feature Store definitions
│   ├── feature_store.yaml
│   └── definitions.py
├── mlruns/                  # MLflow tracking data (local)
├── models/                  # Saved model artifacts (best_model.keras)
├── src/
│   ├── data_loader.py       # Data ingestion utilities
│   ├── preprocessing.py     # Shared data transformation logic
│   └── train_ray.py         # Main training and tuning script
├── requirements.txt         # Python dependencies
├── docker-compose.yaml      # Orchestration for local stack
└── Dockerfile               # Image definition for API/Worker
```

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Python 3.11+ (for local dev)

### Option 1: Run Full Stack with Docker (Recommended)
This spins up Airflow, MLflow, Postgres, and the FastAPI service in containers.

1.  **Build and Start Services**:
    ```bash
    docker-compose up --build
    ```
    *Wait for a few minutes for Airflow to initialize and install dependencies.*

2.  **Access Interfaces**:
    *   **FastAPI Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
    *   **Airflow UI**: [http://localhost:8080](http://localhost:8080) (User: `admin`, Pass: `admin`)
    *   **MLflow UI**: [http://localhost:5000](http://localhost:5000)

3.  **Trigger Training**:
    *   Go to Airflow UI -> Enable `lending_club_training_pipeline` -> Trigger DAG.
    *   Watch the training progress in MLflow.

### Option 2: Local Development
Run components individually on your machine.

1.  **Setup Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Initialize Feature Store**:
    ```bash
    cd feature_repo
    feast apply
    feast materialize-incremental $(date +%Y-%m-%d)
    cd ..
    ```

3.  **Run Training**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python src/train_ray.py
    ```

4.  **Start API**:
    ```bash
    uvicorn app.main:app --port 8000
    ```

---

## 🧪 Testing the API

Once the API is running (Port 8000), you can score a loan application.

**Example Request (High Risk Borrower):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "borrower_id": 1071795,
           "loan_amnt": 5600.0,
           "term": "60 months",
           "purpose": "small_business"
         }'
```

**Expected Response:**
```json
{
  "borrower_id": 1071795,
  "default_probability": 0.4009,
  "is_approved": false,
  "risk_score": 509,
  "input_data_summary": { ... }
}
```

---

## 🛠️ Design Decisions

*   **Feast vs. Direct Loading**: Feast is used for serving to guarantee that the `dti` or `annual_inc` used at inference time is consistent with the definitions used during training, preventing **training-serving skew**.
*   **Ray Tune**: We chose Ray over simple grid search because it supports parallel distributed execution and advanced schedulers (ASHA), allowing us to efficiently search a large hyperparameter space across multiple model types.
*   **Hybrid Storage**: Artifacts are stored locally for development but the code is built to seamlessly switch to **S3** (`UPLOAD_TO_S3=true`) for cloud deployments.

## 🤝 Contributing
1.  Fork the repo
2.  Create a feature branch
3.  Submit a Pull Request

---
*Created by [Balaji Gurusala] - 2026
