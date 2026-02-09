# System Architecture: ShopTalk MLOps

## 1. High-Level Data Flow

### A. Training & Ingestion Pipeline (Kaggle + Airflow)
* **Input:** Amazon Berkeley Objects (ABO) Dataset (Images + Metadata CSV).
* **Image Processing (Kaggle GPU):**
    * Raw Images -> **BLIP/CLIP Model** -> Generated Captions.
    * *Output:* Enriched CSV (Product Description + Image Captions).
* **Embedding Fine-Tuning (Kaggle GPU):**
    * Enriched Text -> **Triplet Loss Training** -> Fine-Tuned Embedding Model adapter.
    * *Artifacts:* Saved to **MLflow Registry** (`.pt` files).
* **Vector Indexing (Airflow DAG):**
    * Download Model -> Generate Embeddings -> Upsert to **ChromaDB**.

### B. Inference Loop (MacBook Local / AWS EC2)
1.  **Voice Input:** User Microphone -> `Audio Data`.
2.  **Transcribe:** `Audio` -> **Whisper (Base)** -> `Query Text`.
3.  **Retrieve:** `Query Text` -> **Hybrid Search** (Keyword + Semantic) -> `Top-K Documents`.
4.  **Generate:** `Top-K` + `Prompt` -> **GPT-4o/Llama3** -> `Natural Response`.
5.  **Speak:** `Response` -> **ElevenLabs/gTTS** -> `Audio Output`.

### C. Monitoring & Feedback (MLOps)
* **Evidently AI:** Asynchronously checks `Query Embeddings` for drift vs. `Training Data`.
* **Prometheus:** Scrapes API latency metrics from FastAPI.

## 2. Component Diagram
* **Frontend:** Streamlit (User Interface).
* **Backend:** FastAPI (Orchestrates Whisper, Chroma, and LLM).
* **Orchestrator:** Apache Airflow (Dockerized).
* **Storage:**
    * **Vector DB:** ChromaDB (Persisted in `data/chroma`).
    * **Model Registry:** MLflow.

## 3. Deployment Strategy
* **Dev:** `docker-compose.yml` spins up Airflow, Chroma, MLflow, and App locally.
* **Prod:** AWS EC2 (g4dn.xlarge).
    * Model weights pulled from S3/MLflow.
    * Docker containers orchestrated via simple shell script.