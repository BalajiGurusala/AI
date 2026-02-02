# IMDB Spoiler Shield 🛡️🎬

A production-grade MLOps system to detect spoilers in movie reviews. This project demonstrates a complete end-to-end machine learning lifecycle, from data ingestion and feature engineering to distributed training and real-time serving, built on a robust cloud-native stack.

## 🏗️ Architecture

The system is designed to be modular, scalable, and cloud-agnostic (optimized for AWS EC2 & S3).

```mermaid
graph TD
    Data[Raw Data (S3/Drive)] --> Ingest(Data Ingestion)
    Ingest --> Preprocess(Text Preprocessing)
    Ingest --> FeatureEng(Feature Engineering)
    
    subgraph "Feature Store (Feast)"
        FeatureEng --> Offline[Offline Store (S3/Parquet)]
        FeatureEng --> Online[Online Store (Redis)]
    end
    
    subgraph "Training Pipeline (Airflow)"
        Preprocess --> Baseline[Train Baseline (LogReg)]
        Preprocess --> BERT[Train BERT (Ray/Distributed)]
        
        Offline --> CreateDS(Create Feast Dataset)
        Preprocess --> CreateDS
        CreateDS --> Advanced[Train Advanced (XGBoost/NN)]
        
        Baseline --> Eval(Evaluation / Drift Detection)
        BERT --> Compare(Model Comparison)
        Advanced --> Compare
    end
    
    subgraph "Artifact Store"
        Compare --> S3Artifacts[S3 Bucket (Models/Metrics)]
        Eval --> S3Reports[S3 Bucket (Evidently Reports)]
    end
    
    subgraph "Serving (FastAPI)"
        S3Artifacts --> LoadModel(Load Best Model)
        Online --> Enrich(Enrich Features)
        UserRequest --> Enrich --> Predict --> Response
    end
```

### 🚀 Key Components

*   **Orchestration:** **Apache Airflow** manages the DAG execution, ensuring dependencies between ingestion, feature engineering, and model training are respected.
*   **Feature Store:** **Feast** manages features (`duration`, `rating`, `sentiment`) consistency.
    *   *Offline:* Parquet files on S3 for training point-in-time joins.
    *   *Online:* Redis for low-latency retrieval during inference.
*   **Experiment Tracking:** **MLflow** logs parameters, metrics, and artifacts for every training run.
*   **Distributed Training:** **Ray Train & Ray Tune** enable scaling BERT fine-tuning across multiple GPUs (EC2 A100s) or local accelerators (Mac Metal/MPS).
*   **Monitoring:** **Evidently AI** detects data drift and model performance degradation.
*   **Storage:** **AWS S3** acts as the central data lake and artifact repository.
*   **Serving:** **FastAPI** provides a high-performance REST API for real-time predictions.

---

## 🧠 Models

We employ a multi-tier modeling strategy to balance speed and accuracy:

1.  **Baseline Model (Logistic Regression):**
    *   **Input:** TF-IDF vectors.
    *   **Optimization:** Class weights applied to handle the 75/25 imbalance.
    *   **Use Case:** Quick benchmarks and interpretability.

2.  **Advanced NLP (DistilBERT):**
    *   **Input:** Raw text (Tokenized).
    *   **Training:** Fine-tuned `distilbert-base-uncased` using **PyTorch** and **Ray**.
    *   **Strategy:** Trained on a stratified 30% subsample to optimize compute/performance trade-off. Supports FSDP/DDP for multi-GPU setups.

3.  **Hybrid Advanced Models (XGBoost & Neural Networks):**
    *   **Input:** Hybrid features combining **Sentence Embeddings** (SBERT `all-MiniLM-L6-v2`) + **Feast Features** (Movie metadata).
    *   **Training:** Leverages `scale_pos_weight` for imbalance handling.
    *   **Use Case:** Captures both semantic meaning and metadata context (e.g., long reviews on short movies are suspicious).

---

## 🛠️ Setup & Installation

### Prerequisites
*   Docker & Docker Compose
*   Python 3.10+
*   AWS Account (S3 Bucket access)

### 1. Environment Configuration
Create a `.env` file in the root directory:
```bash
AIRFLOW_UID=502
S3_BUCKET=your-s3-bucket-name
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

### 2. Local Setup (Mac/Linux)
 The system automatically detects Apple Silicon (MPS) for acceleration.

```bash
# Install dependencies
pip install -r requirements.txt

# Start Infrastructure
docker-compose up -d --build
```

### 3. EC2 Setup (GPU/NVIDIA)
For training on A100s or T4s:
1.  Launch an instance with the Deep Learning AMI.
2.  Clone the repo.
3.  Ensure `nvidia-docker` is installed.
4.  Run `docker-compose up -d`.
5.  *Note:* The `src/train_bert.py` script automatically detects CUDA and scales workers via Ray.

---

## 🏃‍♂️ Usage

### Running the Pipeline (Airflow)
1.  Access Airflow UI at `http://localhost:8080`.
2.  Trigger the `imdb_spoiler_pipeline` DAG.
3.  Watch the tasks execute:
    *   `create_feast_dataset`: Generates training data from the Feature Store.
    *   `train_bert_model`: Launches distributed training.
    *   `compare_models`: Selects the winner.

### Manual Training
You can run specific training scripts inside the environment:

```bash
# Train BERT (Auto-detects GPU/MPS)
python src/train_bert.py

# Train XGBoost/NN with Feast Features
python src/create_dataset_with_feast.py
python src/train_advanced_models.py
```

### Inference API
The API loads the best model from S3 on startup.

```bash
# Start API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Test Prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"movie_id": "tt0111161", "review_text": "Bruce Willis was a ghost the whole time!"}'
```

---

## 📂 Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── app/                  # FastAPI Serving Layer
│   └── main.py
├── dags/                 # Airflow DAGs
│   └── imdb_pipeline.py
├── data/                 # Local data storage (synced with S3)
├── docker/               # Dockerfiles
├── feature_repo/         # Feast Feature Definitions
│   ├── feature_store.yaml
│   └── definitions.py
├── notebooks/            # Exploratory Analysis
├── src/                  # Core Source Code
│   ├── compare_models.py
│   ├── create_dataset_with_feast.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_advanced_models.py  # XGBoost/NN
│   └── train_bert.py             # Ray/Distributed BERT
├── docker-compose.yml
├── requirements.txt
└── README.md
```