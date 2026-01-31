# Portfolio Video Demo Script: Lending Club MLOps

**Goal:** Demonstrate a production-ready, end-to-end MLOps pipeline for credit risk scoring, featuring automated retraining, feature stores, and cloud deployment.

---

## 1. Introduction (0:00 - 0:15)
*   "Hi, I'm Balaji. This is an end-to-end **MLOps Pipeline** built to automate credit risk assessment for Lending Club data."
*   "The goal was to move beyond simple notebooks and build a **production-grade system** that handles data consistency, distributed training, and automated deployment."
*   "I'm running this live on an **AWS EC2 g4dn.xlarge** instance, leveraging NVIDIA T4 GPUs."

## 2. Architecture & Tech Stack (0:15 - 0:30)
*   **Show:** The Architecture Diagram (ASCII or image) or `docker-compose.yaml`.
*   "The stack is fully containerized using **Docker Compose**:
    *   **Feast**: Manages the Feature Store for consistent training and serving data.
    *   **Airflow**: Orchestrates the entire ETL and training workflow.
    *   **Ray Tune**: Handles distributed hyperparameter optimization across CPU/GPU.
    *   **MLflow**: Tracks experiments and manages model artifacts.
    *   **FastAPI**: Serves the final model for real-time predictions."

## 3. The Pipeline in Action (0:30 - 0:50)
*   **Show:** **Airflow UI** (DAGs).
*   "Here in Airflow, we have the `lending_club_training_pipeline`."
*   "First, it syncs the **Feature Store** to ensure fresh data."
*   "Then, it triggers the **Model Tuning** task. This spins up a Ray cluster to train multiple model architectures in parallel—XGBoost, Logistic Regression, and Keras Neural Networks."
*   **Show:** **MLflow UI**.
*   "As you can see in MLflow, we are tracking multiple trials in real-time. We log accuracy, loss, and hyperparameters for every run."

## 4. Artifacts & Infrastructure (0:50 - 1:10)
*   **Show:** **AWS S3 Console** (or mention S3 integration).
*   "Once the best model is identified, it is automatically versioned and uploaded to an **S3 Bucket** for durability."
*   **Show:** Terminal (nvidia-smi or Ray logs).
*   "The infrastructure is optimized for scale. We are utilizing the **NVIDIA T4 GPU**, and Ray handles the resource allocation dynamically."

## 5. Inference & API (1:10 - 1:30)
*   **Show:** **FastAPI Docs** (Swagger UI).
*   "Finally, the trained model is hot-swapped into our **FastAPI** service."
*   "Let's test it with a live request. I'll send a sample borrower profile..."
*   *(Click Execute)* -> "And we get an immediate approval probability score."
*   "This completes the cycle from raw data to production API."

## 6. Closing (1:30)
*   "Thank you for watching. The full code and deployment scripts are available on my GitHub."
