# Project Requirements: ShopTalk

## Clarifications
### Session 2025-02-08
- Q: How should the app handle users and sessions (user identity & access)? → A: Multi-user, anonymous (no login); each browser/session has isolated chat history.
- Q: Where should per-session chat history be stored? → A: In-memory only (server or Streamlit session state); lost on tab close or refresh.
- Q: When speech-to-text fails (no speech, noise, timeout, API error), what should happen? → A: Show a short error message in the chat (e.g., "Couldn't hear that—try again") and allow retry.
- Q: What is the expected usage pattern / concurrent load for the Streamlit/FastAPI app? → A: Single user or demo only; no target for concurrent users or QPS.
- Q: Should the app retain or store user voice recordings or full chat transcripts? → A: Retain for evaluation only: store anonymized transcripts for the 50 qualitative responses only.
- Q: When hybrid search returns no products, what should the user see or hear? → A: Show a short message in chat (e.g. "No products match that. Try different keywords or filters."); optional TTS readback.
- Q: When RAG retrieval or LLM/TTS fails (e.g. timeout, API error), what should the user see or hear? → A: Show a short error message in the chat (e.g. "Something went wrong. Please try again."); allow retry; no TTS for the error.
- Q: While listening, searching, or generating, should the UI show explicit status feedback? → A: Show simple status messages (e.g. "Listening…", "Searching…", "Generating…") in the chat area or near the mic.

## 1. Core Features
### Core Features
**Multi-Modal Chat Interface (Primary):**
    * Standard text-based chat (like ChatGPT).
    * Users can type queries ("red shirt under $50").
    * System returns text responses + product cards.
**Voice Interaction (Secondary/Optional):**
    * "Mic" button populates the text input field (Speech-to-Text).
    * Optional "Read Aloud" button for responses (Text-to-Speech).
    * *Constraint:* The app must be fully functional without a microphone.
* **User model:** Multi-user, anonymous (no login). Each browser/session is isolated.
* **Hybrid Search:** Must support semantic search ("red shirt") AND keyword filtering ("under $50").
* **RAG Pipeline:** Retrieve product metadata + Generate natural language response.
* **Empty search (zero results):** When hybrid search returns no products, show a short message in chat (e.g. "No products match that. Try different keywords or filters."); TTS readback of that message is optional.
* **RAG/LLM/TTS failure (retrieval down, LLM timeout, TTS error):** Show a short error message in the chat (e.g. "Something went wrong. Please try again."); allow retry; do not play TTS for the error.
* **Voice Interaction:**
    * User clicks "Mic" -> Speak query.
    * App transcribes -> Searches -> Generates Answer.
    * App plays audio response back.
    * **STT failure (no speech, noise, timeout, API error):** Show a short error message in the chat (e.g., "Couldn't hear that—try again") and allow the user to tap Mic again to retry.
* **Chat History:** Context-aware within a session (e.g., "Show me the blue one" refers to previous search in that session). Stored in-memory only (e.g., Streamlit session state); lost on tab close or refresh. No persistence layer for chat.

## 2. Performance Goals
* **Voice-to-Text latency:** < 2 seconds.
* **RAG Retrieval latency:** < 1 second.
* **Total Round Trip:** < 5 seconds (Voice in -> Audio out).
* **Load:** Single user or demo only; no target for concurrent users or QPS.

## 3. Data Strategy
* **Source:** Amazon Berkeley Objects (ABO) Dataset (**JSON Lines, gzipped** + Images). Downloaded from `s3://amazon-berkeley-objects/` (public bucket, unsigned access).
  * **Format note (from EDA):** ABO metadata is `.json.gz` (JSON Lines), **not CSV**. Fields use nested `[{language_tag, value}]` arrays requiring English extraction/flattening.
  * **Price note (from EDA):** ABO **does not contain a price field**. For `price_max` filter support, synthetic prices must be assigned during ingestion (e.g., by category) or the filter must be marked optional/demo-only.
  * **Images:** Referenced by `main_image_id` (not URL); thumbnails in `abo-images-small.tar`.
* **Processing:**
    * **Images:** Captioning via BLIP/CLIP (running on Kaggle).
    * **Text:** Flatten nested ABO fields → chunk and embed via HuggingFace models.
* **Storage:**
    * **Dev:** ChromaDB (Local).
    * **Prod:** AWS OpenSearch or ChromaDB on EC2.
* **User data / retention:** Do not persist voice recordings or general chat transcripts. Anonymized transcripts may be stored only for the 50 qualitative evaluation responses (for scoring Helpfulness and Naturalness).

## 4. User Interface (Streamlit)
* **Components:**
    * Sidebar: Filters (Price, Category) + "Record" button.
    * Main Area: Chat Interface (User bubble vs. AI bubble).
    * Product Cards: Display Image, Title, Price, and "Add to Cart" (mock) button.
* **Loading / status feedback:** Show simple status messages (e.g. "Listening…", "Searching…", "Generating…") in the chat area or near the mic while the app is processing.

## 5. Research & Academic Requirements (Mandatory for Grading)
* **Model Comparison:**
    * Benchmark retrieval precision of **Base Embedding Model** vs. **Fine-Tuned Model**.
    * Compare latency/throughput of **ChromaDB** vs. **Milvus** (or FAISS).
* **Fine-Tuning Strategy:**
    * Use **Triplet Loss** to train the embedding model on the ABO dataset.
    * Apply **LoRA/QLORA** to optimize training on Kaggle GPUs.
* **Evaluation Metrics:**
    * **Quantitative:** Report P95 and P99 latency for the full pipeline.
    * **Retrieval:** Measure "Precision@K" (Top-5 results relevance).
    * **Qualitative:** Manually score 50 query responses for "Helpfulness" and "Naturalness."
* **Technical Constraints:**
    * **Model Loading:** Models must load *once* at startup (Global state), NEVER during inference.
    * **Consistency:** Inference API must use the exact same data transformers as the training pipeline.

## 6. MLOps & Infrastructure Requirements
* **Pipeline Orchestration (Apache Airflow):**
    * Must automate the "Ingestion Pipeline" (Load Data -> Chunk -> Embed -> Store).
    * Schedule: Daily updates to the Vector DB to reflect new product data.
* **Experiment Tracking (MLflow):**
    * Track every experiment (e.g., "Run 1: Chunk Size 500", "Run 2: Chunk Size 1000").
    * Log metrics: Retrieval Precision, Latency, and Generation Faithfulness.
    * Model Registry: Version control for the Fine-Tuned Embedding Model.
* **Hyperparameter Optimization (Ray Tune):**
    * Automate the search for optimal parameters (Top-K retrieval count, Chunk size, Learning rate).
    * Execution: Must be capable of running on Kaggle GPUs or AWS.
* **Monitoring & Observability (Evidently AI + Grafana):**
    * **Data Drift:** Detect if user queries shift away from training data distribution.
    * **Performance Monitoring:** Real-time dashboard for System Latency (P95/P99) and Error Rates.
    * **Alerting:** Trigger alerts if Model Faithfulness drops below a threshold.
* **Containerization & Deployment:**
    * **Docker:** All services (Airflow, FastAPI, Streamlit, Chroma) must be containerized in a single `docker-compose.yml` for local dev.
    * **CI/CD:** GitHub Actions to build and push Docker images to AWS ECR on push to `main`.