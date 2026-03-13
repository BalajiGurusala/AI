# Backend (`backend/`)

FastAPI service for search, RAG chat, and voice endpoints.

## What It Owns

- API app bootstrap: `backend/src/api/main.py`
- Routes: `backend/src/api/routes.py`
- Schemas: `backend/src/models/schemas.py`
- Services:
  - embeddings/index loader: `backend/src/services/embeddings.py`
  - RAG orchestration + LLM routing: `backend/src/services/rag.py`
  - voice STT/TTS: `backend/src/services/voice.py`
- Settings/env: `backend/src/config.py`

## API Endpoints

- `GET /health`
- `POST /api/v1/search`
- `POST /api/v1/chat`
- `POST /api/v1/voice/query`

## Startup Lifecycle

At app startup (`lifespan`):
1. Load product data and indexes from `DATA_DIR`.
2. Load embedding models (SentenceTransformer + CLIP).
3. Initialize LLM manager (Ollama -> OpenAI -> Groq fallback).
4. Load voice service (Whisper + gTTS).

## Key Environment Variables

- `DATA_DIR` (default `/app/data`)
- `TEXT_MODEL_ID` (default `all-MiniLM-L6-v2`)
- `CLIP_MODEL_ID` (default `openai/clip-vit-base-patch32`)
- `FINETUNED_MODEL_PATH` (optional, preferred ST model path)
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `OPENAI_API_KEY`, `GROQ_API_KEY`
- `CORS_ORIGINS`

## Fine-Tuned Model Behavior

SentenceTransformer source selection:
1. `FINETUNED_MODEL_PATH` (if set and exists)
2. `DATA_DIR/models/finetuned-shoptalk-emb` (if exists)
3. fallback base model id

Text index selection:
1. `finetuned_text_index.npy` if present
2. fallback `rag_text_index.npy`

## Local Run

From project root:

- Install deps: `pip install -r backend/requirements.txt`
- Run API: `uvicorn backend.src.api.main:app --host 0.0.0.0 --port 8000`

Check:
- `curl http://localhost:8000/health`
