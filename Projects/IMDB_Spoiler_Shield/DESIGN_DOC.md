# IMDB Spoiler Shield: System Design Document

**Date:** February 2026  
**Author:** Balaji Gurusala  
**Status:** Production Ready

---

## 1. Executive Summary

The **IMDB Spoiler Shield** is an end-to-end MLOps system designed to detect spoiler content in movie reviews. Unlike simple notebook experiments, this project implements a production-grade lifecycle including automated orchestration, distributed training, feature stores, drift monitoring, and real-time serving.

The system is built to be **cloud-agnostic** (running seamlessly on local Mac Silicon and AWS EC2 NVIDIA clusters) and **modular** (allowing components to be swapped without breaking the pipeline).

---

## 2. System Architecture

The architecture follows the **Lakehouse** pattern where S3 acts as the central source of truth, and compute is decoupled from storage.

### High-Level Data Flow
1.  **Ingestion:** Raw JSON data is pulled from S3/Local source.
2.  **Feature Store:** Metadata features are computed and stored in **Feast**.
3.  **Training:** Three parallel model pipelines (Baseline, Hybrid, Deep Learning) train on the data.
4.  **Evaluation:** Models are compared, and the best artifact is promoted to S3.
5.  **Serving:** FastAPI loads the artifact and queries Feast for real-time inference.

---

## 3. Component Deep Dive

### 3.1. Orchestration (Apache Airflow)
*   **Role:** The "Traffic Controller" of the system.
*   **Design Choice:** We used `LocalExecutor` within Docker for simplicity and reliability.
*   **Key DAG:** `imdb_pipeline.py` defines the dependency graph:
    *   Ensures features are materialized *before* training data is generated.
    *   Ensures data is processed *before* distributed training starts.
    *   Aggregates results in a final comparison task.

### 3.2. Feature Store (Feast)
*   **Role:** Solves **Training-Serving Skew** by ensuring the features used to train the model are exactly the same as those used during prediction.
*   **Implementation:**
    *   **Offline Store (S3/Parquet):** Used for generating point-in-time correct training datasets (`src/create_dataset_with_feast.py`).
    *   **Online Store (Redis):** Used by the API (`app/main.py`) to fetch movie metadata (duration, rating) in milliseconds to enrich incoming requests.
*   **Why?** Without Feast, the API would need to compute "average sentiment" on the fly, which is too slow.

### 3.3. The Model Factory (Three-Tier Strategy)
We implemented a tiered approach to balance cost vs. accuracy:

| Tier | Model | Technology | Rationale |
| :--- | :--- | :--- | :--- |
| **Baseline** | Logistic Regression | Scikit-Learn + TF-IDF | Fast, interpretable baseline. Establishes the "floor" for performance. |
| **Hybrid** | XGBoost / Neural Net | SBERT + Feast Metadata | Combines semantic meaning (Embeddings) with structured context (Movie Duration, Rating). |
| **SOTA** | DistilBERT / BERT | PyTorch + Ray | State-of-the-Art NLP. Captures deep contextual nuance (e.g., sarcasm, plot twists). |

### 3.4. Distributed Training (Ray)
*   **Role:** Scaling compute.
*   **Problem:** BERT training is memory-intensive and slow on CPUs.
*   **Solution:** We integrated **Ray Train**.
    *   **On Mac:** Auto-detects MPS (Metal) for local acceleration.
    *   **On EC2:** Auto-detects CUDA and scales across all available NVIDIA GPUs using `DataParallelTrainer`.
*   **Key Script:** `src/train_bert.py` handles the hardware abstraction.

### 3.5. Experiment Tracking (MLflow)
*   **Role:** The "Log Book" of the project.
*   **Implementation:** Every training run logs:
    *   **Parameters:** Learning rates, batch sizes, class weights.
    *   **Metrics:** Accuracy, F1-Score, Recall (Crucial for spoilers!).
    *   **Artifacts:** The actual serialized model (`.joblib`, `.pt`).
*   **Storage:** Metadata in Postgres, Artifacts in S3.

### 3.6. Monitoring & Observability (Evidently AI)
*   **Role:** Detecting Model Decay.
*   **Implementation:**
    *   **Drift Detection:** `src/evaluate.py` compares Training vs. Test distributions.
    *   **Simulation:** `src/simulate_drift.py` artificially generates "bad data" scenarios (e.g., spam attacks, null values) to prove the monitoring stack works before production issues occur.
    *   **Dashboard:** A persistent UI to visualize these reports.

### 3.7. Inference API (FastAPI)
*   **Role:** Real-time prediction.
*   **Design:** Stateless microservice.
    *   **Startup:** Downloads the best model from S3.
    *   **Request:** Receives `review_text` + `movie_id`.
    *   **Enrichment:** Queries Redis (Feast) for movie stats.
    *   **Response:** Returns `is_spoiler` probability.

---

## 4. Infrastructure & DevOps

### 4.1. Containerization (Docker)
*   **Philosophy:** "Write Once, Run Anywhere."
*   **Services:** We separated concerns into distinct containers:
    *   `airflow-scheduler/webserver`: Workflow management.
    *   `mlflow-server`: Tracking.
    *   `fastapi-app`: Prediction.
    *   `redis/postgres`: State storage.
*   **Production Override:** `docker-compose.prod.yml` injects NVIDIA runtime configurations only when needed (EC2), keeping local Mac development clean.

### 4.2. Infrastructure as Code
*   **`deploy_stack.sh`**: A bash script that acts as an "Installer."
    *   Detects User IDs (UIDs) to fix permission issues.
    *   Detects GPUs to apply production configs.
    *   Configures `.env` with AWS secrets.

### 4.3. CI/CD (GitHub Actions)
*   **Pipeline:** Triggered on Push/PR.
*   **Checks:**
    1.  **Linting (Flake8):** Enforces code style.
    2.  **Build Validation:** Ensures `Dockerfile` builds successfully on Linux (preventing "It works on my machine" issues).

---

## 5. Key Learnings & Decisions

1.  **Class Imbalance:** Spoilers are rare (~25%). We addressed this using **Class Weights** (`compute_class_weight='balanced'`) in all models rather than blindly oversampling, preserving the natural data distribution.
2.  **Memory Management:** Loading 500k reviews caused OOM kills. We implemented **Chunking** in data loading and **Sampling** for resource-heavy BERT training to ensure stability.
3.  **Security:** NLTK and HuggingFace downloads faced SSL issues in Docker. We implemented explicit **SSL Context Patching** to ensure the pipeline runs in restricted network environments.

---

## 6. Future Roadmap

1.  **Model Registry:** Automate the promotion of the "Best" model to Production (currently we just compare metrics).
2.  **Kubernetes (K8s):** Migrate from Docker Compose to Helm Charts for auto-scaling the API.
3.  **LLM Integration:** Replace DistilBERT with a Large Language Model (Llama-3 or GPT-4) via prompt engineering for zero-shot classification.
