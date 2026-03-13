# App (`app/`)

Streamlit chat UI for ShopTalk.

Main entrypoint: `app/streamlit_app.py`

## Modes

### 1) Thin-Client Mode (recommended for deployment)

When `BACKEND_URL` is set and reachable:
- Streamlit calls backend APIs (`/chat`, `/search`, `/voice/query`)
- Backend handles retrieval, LLM generation, and voice processing

### 2) Standalone Mode (local fallback)

When backend is not reachable:
- Streamlit loads artifacts directly from `data/`
- Uses local models for retrieval
- Optional local LLM calls (Ollama/OpenAI/Groq)

## Features

- Chat-driven shopping assistant
- Product card rendering with optional images
- Metadata filters (`price_max`, `category`, `top_k`)
- Voice input/output (Whisper + gTTS in standalone mode)
- Session context for multi-turn responses

## Fine-Tuned Model Support (Standalone)

SentenceTransformer load order:
1. `FINETUNED_MODEL_PATH` env (if path exists)
2. `data/models/finetuned-shoptalk-emb`
3. fallback base model id from config

Text index load order:
1. `finetuned_text_index.npy`
2. fallback `rag_text_index.npy`

## Local Run

From project root:

- Install deps: `pip install -r app/requirements.txt`
- Run app: `streamlit run app/streamlit_app.py`

Optional env:
- `BACKEND_URL=http://localhost:8000`
- `OPENAI_API_KEY=...`
- `GROQ_API_KEY=...`
- `FINETUNED_MODEL_PATH=data/models/finetuned-shoptalk-emb`
