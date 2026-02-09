# Quickstart: ShopTalk

**Feature**: ShopTalk AI Assistant | **Date**: 2025-02-08

Minimal steps to run the app locally (after implementation).

---

## Prerequisites

- Python 3.11+
- (Optional) Docker and Docker Compose for full stack (Airflow, ChromaDB, MLflow)
- ABO dataset or a sample CSV + pre-built ChromaDB index for dev

---

## 1. Repository and Environment

```bash
git clone <repo-url>
cd Shop_Talk_Assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):

- `OPENAI_API_KEY` (if using OpenAI embeddings/LLM)
- `ELEVENLABS_API_KEY` (if using ElevenLabs TTS)
- Optional: `LLM_PROVIDER=openai|ollama|groq`, `EMBEDDING_MODEL`, etc.

---

## 2. Data and Vector Store (Dev)

**Option A – Pre-built index**

- Place ChromaDB data in `data/chroma/` or set `CHROMA_PATH` in `.env`.

**Option B – Ingest from ABO**

- Run ingestion pipeline (or Airflow DAG) to load ABO CSV + optional image captions, chunk, embed, and upsert to ChromaDB. See `pipelines/` or backend docs.

---

## 3. Models (Local)

- **Whisper**: Downloaded automatically (e.g. `base`) on first use, or place in `models/`.
- **Embedding**: HuggingFace `all-MiniLM-L6-v2` loads from hub; fine-tuned adapter from Kaggle/MLflow goes in `models/` and is loaded at startup.
- **LLM**: GPT-4o via API or Llama 3 via Ollama/Groq; configured via `.env`.

---

## 4. Run Backend

```bash
cd backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

---

## 5. Run Frontend

```bash
cd frontend
streamlit run src/app.py
```

- Open the URL shown (e.g. `http://localhost:8501`).
- Use sidebar for filters and "Record" for voice; chat shows status ("Listening…", "Searching…", "Generating…") and assistant replies with product cards.

---

## 6. Docker Compose (All Services)

From repo root:

```bash
docker-compose up -d
```

Brings up FastAPI, Streamlit, ChromaDB, and optionally Airflow/MLflow per `docker-compose.yml`. Check service URLs in the compose file.

---

## 7. Verify

- **Health**: `curl http://localhost:8000/health` → `{"status":"ok", ...}`.
- **Voice**: Click Record, say e.g. "red shirt under 50 dollars"; expect transcript, reply, and product cards (or "No products match…" / error message per spec).
- **Errors**: STT failure → "Couldn't hear that—try again"; pipeline failure → "Something went wrong. Please try again."

---

## References

- [requirements.md](requirements.md) – Features and clarifications
- [architecture.md](architecture.md) – Data flow and components
- [data-model.md](data-model.md) – Entities and DTOs
- [contracts/api-openapi.yaml](contracts/api-openapi.yaml) – API contract
- [test_strategy.md](test_strategy.md) – How to run tests and evaluations
