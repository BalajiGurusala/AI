# ShopTalk AI Assistant — Code Structure Video Walkthrough

> A scene-by-scene script for recording a video demonstration of the codebase, explaining each folder's purpose, key files, and how they connect.

**Estimated duration:** ~18–22 minutes

---

## Pre-Recording Setup

1. Open the project in your IDE (VS Code / Cursor)
2. Collapse all folders in the explorer so you start clean
3. Have a terminal ready at the project root
4. (Optional) Have the running EC2 app open in a browser tab for quick references

**Recommended IDE settings for recording:**
- Font size: 16–18px for readability
- Theme: Light or dark with good contrast
- Close all unrelated tabs
- Hide activity bar icons you won't use

---

## Scene 1: Project Overview & Root Files (2 min)

**Action:** Show the top-level directory tree in the terminal.

```bash
ls -la
```

**Then expand the explorer sidebar to show all top-level folders.**

### Talking Points

> "Let me walk you through the ShopTalk AI Assistant codebase. This is an end-to-end ML application — from data exploration and model training in Jupyter notebooks, all the way to a deployed, containerized service on AWS EC2."
>
> "At the top level, you can see the project is organized into clear concerns:"
>
> - **`notebooks/`** — the offline ML pipeline. This is where data prep, embedding training, and evaluation happen. Think of this as the *artifact production* layer.
> - **`src/`** — shared core logic, specifically the hybrid search engine. This is imported by both notebooks and the production services, keeping one canonical implementation.
> - **`backend/`** — the FastAPI backend that serves the API, loads models, and orchestrates the RAG pipeline.
> - **`app/`** — the Streamlit chat UI. It can run in two modes: standalone (loads models itself) or thin-client (calls the backend API).
> - **`frontend/`** — Docker packaging for the Streamlit app.
> - **`deploy/`** — operational scripts for EC2 setup and deployment.
> - **`data/`** — runtime artifacts exported from notebooks. Not committed to git — populated before deployment.
> - **`specs/`** and **`.spec/`** — requirement documents, architecture specs, and the data model that guided implementation.
> - **`backend/tests/`** — 170 tests across unit, contract, and integration layers.

**Show these root files briefly:**

> "At the root level, we also have:"
>
> - **`docker-compose.yml`** — the main orchestration file that ties Ollama, Backend, and Frontend together
> - **`docker-compose.gpu.yml`** / **`docker-compose.cpu.yml`** / **`docker-compose.local.yml`** — environment-specific overrides for GPU, CPU-only, and local corporate proxy scenarios
> - **`README.md`** — project overview with Mermaid architecture diagrams
> - **`INSTALL.md`** — installation instructions for local macOS and EC2
> - **`.env.example`** — environment variable template for API keys and configuration

---

## Scene 2: Notebooks — The ML Pipeline (4 min)

**Action:** Open `notebooks/` folder. Open each notebook file header briefly.

### Talking Points

> "The `notebooks/` folder is where all the ML work lives. There are five notebooks, run in order, and each one produces artifacts that feed the next."

**Open `notebooks/README.md` briefly** to show the execution order table.

#### NB01: EDA (`01-shoptalk-eda.ipynb`)

> "Notebook 1 is exploratory data analysis on the Amazon Berkeley Objects dataset — about 147,000 product listings with images, metadata, and multilingual fields. It examines data quality, category distributions, and identifies which fields are useful for retrieval."

#### NB02: Image Captioning (`02-image-captioning.ipynb`)

> "Notebook 2 generates image captions using BLIP, a vision-language model. These captions are appended to product descriptions, creating richer text representations for hybrid search. This is the multimodal enrichment step."

#### NB03: RAG Prototype (`03-rag-prototype.ipynb`)

**Show the key exports:**

> "Notebook 3 is the core retrieval system. It builds the hybrid search prototype:"
>
> - Encodes all products with SentenceTransformer (`all-MiniLM-L6-v2`) for text embeddings
> - Encodes product images with CLIP for image embeddings
> - Exports: `rag_products.pkl`, `rag_text_index.npy`, `rag_image_index.npy`, `rag_config.json`
>
> "These four files are the **minimum required artifacts** for the serving stack to function."

#### NB04: LLM Integration (`04-llm-integration.ipynb`)

> "Notebook 4 integrates the retrieval system with LLM response generation. It connects to OpenAI GPT-4o-mini, evaluates response quality using LLM-as-judge metrics — faithfulness, relevance, helpfulness, naturalness — and compares different provider configurations."
>
> "Exports: evaluation CSVs, config JSONs, and standardized evaluation queries."

#### NB05: Fine-Tuning (`05-fine-tuning.ipynb`)

> "Notebook 5 fine-tunes the embedding model using triplet loss — anchor, positive, negative training. It compares base vs full fine-tuned vs LoRA experiments, achieving a 79% improvement in retrieval quality."
>
> "Key exports: `finetuned_text_index.npy` and the `models/finetuned-shoptalk-emb/` directory containing the fine-tuned SentenceTransformer. The serving stack automatically prefers these over the base model."

**Transition:**

> "Everything from notebooks flows into the `data/` directory, which the serving stack reads at startup. Let me show you that contract."

---

## Scene 3: Data Directory — The Artifact Contract (1.5 min)

**Action:** Open `data/` folder in explorer. Open `data/README.md`.

```bash
ls -la data/
```

### Talking Points

> "The `data/` directory is the bridge between training and serving. It's not committed to git — it's populated by copying notebook outputs."

**Show the directory listing:**

> "You can see the required files:"
> - `rag_products.pkl` — the product catalog (9,190 products)
> - `rag_text_index.npy` / `finetuned_text_index.npy` — embedding vectors, 384 dimensions
> - `rag_image_index.npy` — CLIP image embeddings, 512 dimensions
> - `rag_config.json` — model IDs, alpha parameter, and settings
>
> "And the recommended fine-tuned model directory under `data/models/finetuned-shoptalk-emb/`."
>
> "The serving stack has a resolution order: it prefers the fine-tuned index and model, falls back to the base model if unavailable. This means you can deploy with just the base artifacts, and upgrade to fine-tuned artifacts without any code changes."

---

## Scene 4: Shared Core — `src/search.py` (3 min)

**Action:** Open `src/search.py`. This is the most important shared file.

### Talking Points

> "The `src/` directory contains one critical module: `search.py`. This is the **single source of truth** for all search logic — used by notebooks, the Streamlit app in standalone mode, and the backend service."

**Scroll through the constants section (lines 1–60):**

> "At the top, you see the tuned hyperparameters: `ALPHA_DEFAULT` controls the text-vs-image fusion weight, and we have several reranking penalties — head noun miss, gender miss, color miss, type miss, and qualifier miss. These are multiplicative penalties applied in stage 2."

**Show the `COLOR_FAMILIES` dictionary:**

> "Color awareness is built into reranking. The system groups colors into families — for example, 'navy', 'cobalt', 'azure' all belong to the 'blue' family. When a user asks for 'red shoes', products with colors in a different family get penalized."

**Scroll to the `hybrid_search` function:**

> "The `hybrid_search` function is the main entry point. It takes a query, the product DataFrame, text and image indexes, encoder functions, and filter parameters."
>
> "The pipeline is two stages:"
> 1. **Vector retrieval** — dot-product similarity against text and image indexes, fused with configurable alpha weight
> 2. **Stage-2 reranking** — lexical overlap scoring, head-noun matching, gender/color/type/qualifier penalties
>
> "This approach gives us semantic understanding from embeddings plus precision from heuristic reranking."

**Briefly show `apply_rerank` function:**

> "The reranking function is where relevance precision lives. For example, if you search 'sports shoes for men' and a result is 'Leather Dress Shoe', it gets penalized by the type-miss penalty because 'dress' doesn't match 'sports', and by the qualifier-miss penalty because 'sports' doesn't appear in the title."

**Transition:**

> "This shared module is why notebooks and production services always produce identical search results — no train-serve skew."

---

## Scene 5: Backend — FastAPI Service (4 min)

**Action:** Open `backend/` folder, expand the directory tree.

```
backend/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── config.py          ← Pydantic settings
│   ├── api/
│   │   ├── main.py        ← App + lifespan
│   │   └── routes.py      ← API endpoints
│   ├── models/
│   │   └── schemas.py     ← Pydantic request/response models
│   └── services/
│       ├── embeddings.py  ← Model loading + encoding
│       ├── rag.py         ← RAG pipeline + LLM routing
│       └── voice.py       ← Whisper STT + gTTS TTS
└── tests/                 ← 170 tests
```

### 5a: Configuration (`config.py`)

**Open `backend/src/config.py`:**

> "Configuration is driven by Pydantic `BaseSettings`. Every setting has a sensible default but can be overridden by environment variables or `.env` file. Data directory, model IDs, LLM provider URLs, API keys, CORS origins — all configurable without code changes."

### 5b: Application Entry Point (`main.py`)

**Open `backend/src/api/main.py`:**

> "The FastAPI application uses a lifespan context manager — models load once at startup, never during requests. This follows the architecture spec: 'Models load once at startup, global state.'"
>
> "At startup:"
> 1. Embedding service loads the product catalog, both embedding indexes, and both encoder models (SentenceTransformer + CLIP)
> 2. LLM manager initializes with fallback: Ollama → OpenAI → Groq
> 3. Voice service loads Whisper for speech-to-text and gTTS for text-to-speech
>
> "CORS middleware is configured for the Streamlit frontend. Product images are served as static files if available."

### 5c: API Routes (`routes.py`)

**Open `backend/src/api/routes.py`:**

> "There are four endpoints:"
>
> - **`GET /health`** — reports component status: embedding loaded, LLM available, STT/TTS available, product count
> - **`POST /api/v1/search`** — pure search: query in, ranked products out
> - **`POST /api/v1/chat`** — the main RAG endpoint: query in, search results + LLM-generated natural language recommendation out
> - **`POST /api/v1/voice/query`** — voice pipeline: audio file in, Whisper transcription → RAG search → TTS audio out
>
> "Notice how the voice endpoint reuses the same RAG pipeline as the chat endpoint — it just adds STT on the front and TTS on the back."

### 5d: Schemas (`schemas.py`)

**Open `backend/src/models/schemas.py`:**

> "All data exchange uses Pydantic models — this is mandated by the project constitution. You can see `Product` with 12 fields matching the ABO dataset, `ChatRequest`/`ChatResponse` for the RAG endpoint, `SearchRequest`/`SearchResponse` for pure search, and `VoiceQueryResponse` for the voice pipeline."
>
> "Notice the `top_k` field has validation — `ge=1, le=20` — so invalid requests are rejected at the schema level, not deep in the search code."

### 5e: Services Layer (briefly)

**Open `backend/src/services/embeddings.py` briefly:**

> "The embedding service is a singleton that holds the loaded models, the product DataFrame, and both indexes. It exposes `encode_text` and `encode_clip` methods. CLIP loading is wrapped in try/except — if it fails due to network issues, the system falls back to text-only search."

**Open `backend/src/services/rag.py` briefly:**

> "The RAG service ties search and LLM together. It calls `hybrid_search` from the shared `src/search.py`, formats the top results as context, and passes them to the active LLM. The `LLMManager` class handles multi-provider fallback — it tries Ollama first, then OpenAI, then Groq."

### 5f: Dockerfile

**Open `backend/Dockerfile`:**

> "The backend Dockerfile is based on Python 3.11 slim. It installs system dependencies like ffmpeg for Whisper, then Python dependencies. There's a best-effort model pre-download step at build time — it tries to cache SentenceTransformer and CLIP into the image layer. If this fails (e.g., behind a proxy), it's non-fatal — runtime loading handles it."

---

## Scene 6: App — Streamlit UI (2.5 min)

**Action:** Open `app/` folder.

### Talking Points

**Open `app/streamlit_app.py`:**

> "The Streamlit app is a single-file application — about 1,100 lines — that provides the chat interface."
>
> "It operates in two modes:"
> - **Thin-client mode**: When `BACKEND_URL` is set and the backend is reachable, Streamlit acts as a pure UI — all search, RAG, and voice processing happens on the backend via HTTP
> - **Standalone mode**: When no backend is available, the app loads models directly — SentenceTransformer, CLIP, and LLM — and runs the full pipeline locally
>
> "This dual-mode design means you can develop and demo locally without Docker, and deploy to production with the full microservice stack."

**Scroll to show `load_search_models` function:**

> "Model loading uses Streamlit's `@st.cache_resource` decorator — models load once and are cached across sessions. Both ST and CLIP are loaded eagerly at startup."

**Scroll to show the sidebar section:**

> "The sidebar provides: LLM model selection (Ollama/OpenAI/Groq), category filter, results slider, catalog stats, voice input toggle, and text-to-speech toggle."

**Scroll to show the chat input and voice processing:**

> "Voice input uses a two-step mechanism to avoid Streamlit widget lifecycle issues: Step 1 captures raw audio and immediately reruns, Step 2 handles the heavy transcription on the next execution. This prevents UI freezes."

**Scroll to show `render_product_card`:**

> "Product cards display images, titles, categories, brands, and colors. Images fall back to public AWS S3 URLs if local files aren't available — this is important because the full image dataset is 40 GB."

---

## Scene 7: Frontend & Docker Packaging (1.5 min)

**Action:** Open `frontend/Dockerfile`.

### Talking Points

> "The `frontend/` folder is purely Docker packaging — no application code lives here. The actual UI code is in `app/`. This separation keeps container concerns (base image, system packages, startup command) separate from application logic."

**Open `frontend/Dockerfile`:**

> "It copies both `app/` and `src/` into the container, because in standalone mode the Streamlit app imports the shared search module. Streamlit runs headless on port 8501 with usage stats disabled."

**Open `docker-compose.yml`:**

> "The Docker Compose file orchestrates three services:"
>
> 1. **Ollama** — local LLM server with automatic model pull on first start
> 2. **Backend** — FastAPI service, depends on Ollama being healthy
> 3. **Frontend** — Streamlit UI, depends on backend being healthy
>
> "The dependency chain ensures services start in the right order. The backend has a 60-second start period because model loading takes 30–60 seconds."
>
> "Environment-specific overrides are handled by separate compose files:"
> - `docker-compose.gpu.yml` — adds NVIDIA GPU reservation for Ollama
> - `docker-compose.cpu.yml` — no-op override for CPU-only hosts
> - `docker-compose.local.yml` — mounts the host's HuggingFace cache to work behind corporate proxies

---

## Scene 8: Deploy Scripts — EC2 Operations (1.5 min)

**Action:** Open `deploy/` folder.

### Talking Points

**Open `deploy/setup-ec2.sh`:**

> "The setup script is a one-time bootstrap for Ubuntu 22.04 EC2 instances. It installs Docker with the Compose plugin, configures NVIDIA container runtime if a GPU is detected, creates the data directory, and adds the user to the Docker group."

**Open `deploy/deploy.sh`:**

> "The deploy script handles day-2 operations. It supports four commands:"
> - **`deploy`** — validates data artifacts exist, builds images, starts services, waits for backend health
> - **`restart`** — restarts without rebuilding
> - **`logs`** — tails compose logs
> - **`stop`** — stops and removes containers
>
> "It validates that `rag_text_index.npy` exists before deploying — this catches the common 'forgot to upload data' error early."

---

## Scene 9: Tests — 170 Tests, Three Layers (2.5 min)

**Action:** Open `backend/tests/` folder.

### Talking Points

**Open `backend/tests/README.md` and show the test pyramid:**

> "The test suite has 170 tests across three layers, all running in under 1 second with no external dependencies — no ML models, no API keys, no running services."

**Show the three layers:**

> "**Unit tests (128 tests)** cover six modules:"
> - `test_search.py` — tokenization, overlap scoring, head-noun extraction, gender/color detection, reranking penalties, full `hybrid_search` entry point
> - `test_schemas.py` — all 9 Pydantic models validated
> - `test_data_processing.py` — text cleaning edge cases (emojis, unicode, HTML entities)
> - `test_voice_utils.py` — voice service lifecycle, mocked Whisper transcription
> - `test_config.py` — settings defaults and env overrides
> - `test_spec_compliance.py` — cross-references 5 spec documents to verify the code matches the architecture and requirements

**Open `backend/tests/conftest.py` briefly:**

> "All tests run against synthetic fixtures — 20 fake products spanning 10 categories with realistic titles, brands, colors, and random embedding vectors. This means tests are deterministic, fast, and portable."

> "**Contract tests (31 tests)** validate API schema shapes against the OpenAPI spec for all four endpoints."
>
> "**Integration tests (11 tests)** run the full search + RAG pipeline end-to-end with synthetic data, including category and price filtering, zero-results edge cases, and LLM context formatting."

**Run the tests live:**

```bash
PYTHONPATH=. python -m pytest backend/tests/ -v --tb=short
```

> "170 tests, all green, under 1 second."

---

## Scene 10: Specs & Architecture Docs (1.5 min)

**Action:** Open `.spec/` folder.

### Talking Points

> "The project was built spec-first. The `.spec/` folder contains the governing documents:"
>
> - **`requirements.md`** — functional and non-functional requirements
> - **`architecture.md`** — system architecture, component diagram, data flow
> - **`data-model.md`** — entity definitions (Product, ChatMessage, Filters)
> - **`constitution.md`** — coding standards: 'Use pydantic for all data exchange', 'Models load once at startup', '.env for secrets'
> - **`test_strategy.md`** — testing approach mapped to the codebase
>
> "The `test_spec_compliance.py` tests actually parse these spec documents and verify that the code follows the stated architecture. For example, it checks that FastAPI is used, that Streamlit exists, that pydantic models have the required fields."

**Open `specs/001-shoptalk-mvp-setup/plan.md` briefly:**

> "The implementation plan was written before coding started — it captures all technical decisions: Python 3.11, FastAPI + Streamlit, hybrid search with ChromaDB, Whisper STT, voice as secondary, and target platform (MacBook dev, Kaggle training, EC2 prod)."

---

## Scene 11: End-to-End Data Flow Recap (1.5 min)

**Action:** Open `README.md` and show the Mermaid architecture diagram. Or draw on a whiteboard/slide.

### Talking Points

> "Let me tie it all together with the end-to-end data flow."
>
> "**Training time** (Kaggle notebooks):"
> 1. Raw ABO data → EDA → Image captioning → Enriched products
> 2. Enriched products → Embedding generation → Text + image indexes
> 3. Text queries → RAG evaluation → LLM integration tuning
> 4. Triplet training → Fine-tuned SentenceTransformer model
> 5. All artifacts exported to `data/`
>
> "**Serving time** (Docker on EC2):"
> 1. Backend loads artifacts from `data/` at startup
> 2. User types a query in Streamlit →  Streamlit calls `POST /api/v1/chat`
> 3. Backend encodes query with SentenceTransformer + CLIP
> 4. `src/search.py` runs hybrid search: vector dot-product → score fusion → reranking
> 5. Top-K products formatted as context → sent to LLM (Ollama/OpenAI/Groq)
> 6. LLM generates natural language recommendation
> 7. Response + product cards returned to Streamlit UI
>
> "For voice queries, the flow adds Whisper STT at the front and gTTS at the back, but the core search and RAG pipeline is identical."
>
> "The key architectural principle is **no train-serve skew**: `src/search.py` is the single source of truth used in notebooks during evaluation AND in production during serving. Same code, same behavior."

---

## Scene 12: Wrap-Up (0.5 min)

**Action:** Show the project root one more time.

### Talking Points

> "To summarize the ShopTalk codebase:"
>
> | Layer | Folder | Key Responsibility |
> |-------|--------|----|
> | ML Pipeline | `notebooks/` | Data prep, captioning, search prototyping, LLM eval, fine-tuning |
> | Shared Logic | `src/` | Hybrid search engine — single source of truth |
> | API Service | `backend/` | FastAPI: model loading, RAG orchestration, voice processing |
> | UI | `app/` | Streamlit chat interface with dual mode (standalone / thin-client) |
> | Packaging | `frontend/`, `backend/Dockerfile` | Docker images |
> | Deployment | `deploy/`, `docker-compose*.yml` | EC2 setup, compose orchestration |
> | Artifacts | `data/` | Bridge between training and serving |
> | Testing | `backend/tests/` | 170 tests: unit, contract, integration — no external deps |
> | Specification | `.spec/`, `specs/` | Architecture, requirements, data model, coding standards |
>
> "The design follows a clean separation: notebooks produce artifacts, the shared search module ensures consistency, and the serving stack (FastAPI + Streamlit + Docker) consumes those artifacts in production."
>
> "Thank you for watching."

---

## Quick Reference: File-to-Purpose Index

Use this as a cheat sheet during recording if you need to quickly find what a file does.

| File | Purpose |
|------|---------|
| `src/search.py` | Hybrid search + reranking (THE core engine) |
| `backend/src/api/main.py` | FastAPI app, lifespan model loading |
| `backend/src/api/routes.py` | 4 API endpoints (health, search, chat, voice) |
| `backend/src/config.py` | Pydantic settings from env/.env |
| `backend/src/models/schemas.py` | 9 Pydantic request/response models |
| `backend/src/services/embeddings.py` | ST + CLIP model loading, encoding |
| `backend/src/services/rag.py` | RAG pipeline, LLM routing (Ollama/OpenAI/Groq) |
| `backend/src/services/voice.py` | Whisper STT + gTTS TTS |
| `app/streamlit_app.py` | Streamlit UI (standalone + thin-client modes) |
| `backend/Dockerfile` | Backend container with model pre-caching |
| `frontend/Dockerfile` | Frontend container for Streamlit |
| `docker-compose.yml` | 3-service orchestration (Ollama + Backend + Frontend) |
| `docker-compose.gpu.yml` | NVIDIA GPU override for Ollama |
| `docker-compose.local.yml` | HF cache mount for corporate proxy |
| `deploy/setup-ec2.sh` | One-time EC2 bootstrap |
| `deploy/deploy.sh` | Build, deploy, restart, stop, logs |
| `backend/tests/conftest.py` | Synthetic fixtures (20 products, mock encoders) |
| `data/rag_config.json` | Model IDs, alpha, serving config |
| `.spec/constitution.md` | Coding standards (pydantic, typing, env) |
| `.spec/architecture.md` | System architecture, data flow |

---

## Recording Tips for Code Walkthrough

1. **Use split screen**: IDE on the left (70%), terminal on the right (30%)
2. **Zoom into code**: Use `Cmd+=` to increase font size when showing specific functions
3. **Use the minimap**: Scroll the minimap to give viewers a sense of file size and structure
4. **Collapse/expand folders**: Start each scene by expanding the relevant folder, collapse when done
5. **Use terminal for structure**: `ls` and `tree` commands help show directory layouts
6. **Keep pace moderate**: Pause 2–3 seconds when opening a new file to let viewers orient
7. **Highlight key lines**: Click on important lines to move the cursor highlight there
8. **Run tests live**: Running `pytest` during the test scene adds credibility — viewers see real green checkmarks
