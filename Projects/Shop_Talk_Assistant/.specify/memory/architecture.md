# System Architecture: ShopTalk MLOps

## 1. High-Level Data Flow

### A. Training & Ingestion Pipeline (Kaggle + Airflow)
* **Input:** Amazon Berkeley Objects (ABO) Dataset (Images + **Metadata JSON Lines, gzipped**). Downloaded from `s3://amazon-berkeley-objects/` (public, unsigned). Nested `[{language_tag, value}]` fields require English extraction/flattening. **Note: ABO has no price field**; synthetic prices assigned during ingestion if price filtering is needed.
* **Image Processing (Kaggle GPU):**
    * Raw Images -> **BLIP/CLIP Model** -> Generated Captions.
    * *Output:* Enriched dataset (Product Description + Image Captions).
* **Embedding Fine-Tuning (Kaggle GPU):**
    * Enriched Text -> **Triplet Loss Training** -> Fine-Tuned Embedding Model adapter.
    * *Artifacts:* Saved to **MLflow Registry** (`.pt` files).
* **Vector Indexing (Airflow DAG):**
    * Download Model -> Generate Embeddings -> Upsert to **ChromaDB**.

### B. Inference Loop (MacBook Local / AWS EC2)
1.  **User Input:**
    * **Text Mode (Primary):** Direct string input.
    * **Voice Mode (Secondary):** Audio -> **Whisper** -> Text String.
2.  **Unified Processor:**
    * `Text String` (from either source) -> **Hybrid Search** (Keyword + Semantic).
    * *Result:* `Top-K Documents` (Context).
3.  **Generation:**
    * `Context` + `User Query` -> **LLM (GPT-4o)** -> `Natural Response`.
4.  **Output:**
    * **Display:** `Natural Response` (Text).
    * **Playback (Optional):** `Natural Response` -> **TTS (ElevenLabs/gTTS)** -> Audio.

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