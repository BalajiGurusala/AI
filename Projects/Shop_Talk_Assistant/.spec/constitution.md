# Project Constitution: ShopTalk AI Assistant

## 1. Project & Environment
* **Project Name:** ShopTalk - AI Shopping Assistant
* **Dataset:** Amazon Berkeley Objects (ABO) Dataset
* **Development (App Logic):** Local MacBook (Apple Silicon M-Series)
* **Training (Fine-Tuning):** Kaggle Kernels (NVIDIA T4/P100 GPUs)
    * *Why:* Leverage free 30-40hrs/week of CUDA compute for LoRA/QLORA.
* **Production:** AWS EC2 (Instance: `g4dn.xlarge` or `g5.xlarge`)

## 2. Application Tech Stack (The "App")
* **Orchestration:** LangChain
* **Vector DB:** * *Development:* ChromaDB (Local persistence)
    * *Production:* AWS OpenSearch (or Scalable ChromaDB on EC2)
    * *Comparison:* Must benchmark retrieval speed vs. FAISS or Milvus.
* **Embeddings:** * *Base:* HuggingFace `all-MiniLM-L6-v2` (Fast/Local) or `OpenAI Embeddings`.
    * *Fine-Tuned:* Custom version trained on ABO dataset (via Kaggle).
* **Image Processing:** BLIP or CLIP (Run on Kaggle for batch captioning; Store captions in Vector DB).
* **LLM:** Comparative Study:
    * *Proprietary:* GPT-4o (Benchmark).
    * *Open Source:* Llama 3 (via Ollama or Groq).
* **Voice STT:** OpenAI Whisper (Local `base` model).
* **Voice TTS:** ElevenLabs (High fidelity) or gTTS (Fallback).
* **Frontend:** Streamlit.
* **API:** FastAPI.

## 3. MLOps Infrastructure (The "Platform")
* **Pipeline Orchestration:** Apache Airflow.
* **Experiment Tracking:** MLflow (Hosted on Databricks Community or Local).
    * *Critical Task:* Log training runs from Kaggle to MLflow.
* **Hyperparameter Tuning:** Ray Tune (Run on Kaggle).
* **Monitoring:** Evidently AI + Prometheus/Grafana.

## 4. Coding Standards
* **Validation:** Use `pydantic` models for all data exchange (Inputs/Outputs).
* **Hybrid Workflow:** * *Heavy Compute:* Perform Fine-tuning & Bulk Image Captioning on Kaggle. 
    * *Artifacts:* Download trained models (`.pt` or `.bin`) from Kaggle -> `models/` folder locally.
* **Typing:** Python `typing` is mandatory.
* **Secrets:** Use `.env` files. (On Kaggle, use "Kaggle Secrets" add-on).
* **Hardware Awareness:** Code must detect `mps` (Mac) vs `cuda` (Kaggle/AWS) dynamically.
* **Documentation:** Google-style docstrings for all modules and functions.