# Implementation Plan: ShopTalk AI Assistant

**Branch**: `main` (or `001-shoptalk-mvp-setup`) | **Date**: 2025-02-08 | **Spec**: [requirements.md](.specify/memory/requirements.md)  
**Input**: Feature specification from `.specify/memory/requirements.md`

## Summary

ShopTalk is an AI shopping assistant with **text-based chat as primary** and **voice as secondary/optional**; the app must be fully functional without a microphone. Hybrid (semantic + keyword) product search and RAG over the Amazon Berkeley Objects (ABO) dataset. Technical approach: Streamlit frontend, FastAPI backend, LangChain orchestration, ChromaDB/OpenSearch for vectors; optional Whisper STT and ElevenLabs/gTTS TTS; fine-tuning and image captioning on Kaggle; MLOps via Airflow, MLflow, Ray Tune, Evidently AI, and Grafana.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: FastAPI, Streamlit, LangChain, ChromaDB, OpenAI Whisper, ElevenLabs/gTTS, HuggingFace (all-MiniLM-L6-v2, BLIP/CLIP), pydantic  
**Storage**: ChromaDB (dev local); AWS OpenSearch or ChromaDB on EC2 (prod); MLflow Model Registry; no persistent chat DB  
**Testing**: pytest (unit, integration, contract); Evidently AI + Prometheus/Grafana for monitoring  
**Target Platform**: Local MacBook (Apple Silicon) for dev; Kaggle for training; AWS EC2 (g4dn.xlarge) for prod  
**Project Type**: web (backend FastAPI + frontend Streamlit)  
**Performance Goals**: Voice-to-Text < 2s, RAG retrieval < 1s, total round trip (voice in → audio out) < 5s; P95/P99 reported  
**Constraints**: Models load once at startup (global state); inference uses same transformers as training; single-user/demo load  
**Scale/Scope**: Single user or demo; no concurrency target; 50-query qualitative evaluation set

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Constitution Rule | Status | Notes |
|-------------------|--------|--------|
| Pydantic for all data exchange | PASS | Plan uses pydantic for API and internal DTOs |
| Python typing mandatory | PASS | Enforced in coding standards |
| .env for secrets; Kaggle Secrets on Kaggle | PASS | Assumed in deployment |
| Hardware: detect mps (Mac) vs cuda (Kaggle/AWS) | PASS | Documented in constitution |
| Google-style docstrings | PASS | In scope for implementation |
| LangChain orchestration | PASS | In constitution |
| ChromaDB dev / OpenSearch or Chroma EC2 prod | PASS | In plan |
| Fine-tuning & bulk captioning on Kaggle; artifacts to local `models/` | PASS | In plan |
| Streamlit frontend, FastAPI API | PASS | In plan |
| Airflow, MLflow, Ray Tune, Evidently AI, Grafana | PASS | In plan |

No violations. Complexity tracking table omitted.

## Project Structure

### Documentation (this feature)

```text
.specify/memory/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 API contracts
├── requirements.md      # Feature spec
├── architecture.md      # High-level architecture
└── test_strategy.md     # Test & evaluation plan
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/          # Pydantic models, domain entities
│   ├── services/        # RAG, STT, TTS, hybrid search
│   └── api/             # FastAPI routes
└── tests/
    ├── contract/
    ├── integration/
    └── unit/

frontend/
├── src/
│   ├── components/      # Streamlit UI components
│   ├── pages/           # App pages/flows
│   └── services/        # API client, session state
└── tests/

data/                    # ChromaDB persistence (dev), ABO data
models/                  # Downloaded fine-tuned models from Kaggle
pipelines/               # Airflow DAGs, ingestion scripts (optional for MVP)
```

## Execution Plan

### Phase 0: Research & Prototyping (Notebooks)
**Goal**: Validate data and RAG logic in a sandbox before writing production code.
- [ ] T000a Create `notebooks/` directory and requirements.
- [ ] T000b [Kaggle] ShopTalk EDA (Price distribution, Image URL check).
- [ ] T000c [Kaggle] Image Captioning pipeline (BLIP/CLIP) -> `enriched_products.csv`.
- [ ] T000d [Local] `01_rag_prototype.ipynb`: Load data, ChromaDB, test "Text Search".
- [ ] T000e [Local] `02_voice_test.ipynb`: Record audio -> Whisper -> Print Text.

### Phase 1: Setup (Shared Infrastructure)
**Goal**: Initialize project skeleton.
- [ ] T001 Create directory structure (backend, frontend, data).
- [ ] T002 Create .env and config.
- [ ] T003 Initialize Python projects (pyproject.toml / requirements.txt).
- [ ] T004 Setup Docker / Docker Compose skeleton.

### Phase 2: Foundational (Blocking Prerequisites)
**Goal**: Core classes that don't depend on features.
- [ ] T008 Implement Pydantic Models (Product, ChatRequest, ChatResponse).
- [ ] T009 Implement VectorStore service (ChromaDB wrapper).
- [ ] T010 Implement Embedding service (Singleton loader).
- [ ] T011 Create FastAPI skeleton + Health Check.

### Phase 3: User Story 1 – Text Query + RAG (MVP)
**Goal**: Primary Text API (No Voice yet).
- [ ] T027 Implement `POST /api/v1/chat` (Text -> Search -> RAG -> Text Response).
- [ ] T028 Implement `POST /api/v1/search` (Debug endpoint).
- [ ] T029 Wire API routes.

### Phase 4: User Story 2 – Streamlit UI & Voice
**Goal**: The Frontend Interface.
- [ ] T031 Implement Streamlit Chat UI.
- [ ] T032 Add "Record" button (Audio Capture).
- [ ] T033 Implement Backend Voice Adapter (`POST /api/v1/voice` -> STT -> Chat API).

### Phase 5: Production & Polish
**Goal**: Deployment ready.
- [ ] T040 Finalize Docker containers.
- [ ] T041 Write Quickstart guide.

**Structure Decision**: Web application (backend + frontend) chosen per constitution (Streamlit + FastAPI). Backend holds orchestration, vector store, STT/TTS/LLM; frontend is Streamlit with session-scoped chat state.

## Requirements Coverage (Problem Statement → Plan/Tasks)

*Traceability from [problem_statement_shop_assist.pdf](problem_statement/problem_statement_shop_assist.pdf) to this plan and [tasks.md](tasks.md).*

### Core Problem & Deliverables

| Problem Statement Requirement | Plan / Tasks Coverage |
|-------------------------------|------------------------|
| Smart assistant: query → relevant products + natural language response | Phase 3 (US1): RAG pipeline, POST /api/v1/chat, POST /api/v1/search |
| User input understanding (complex queries, synonyms) | Hybrid search (semantic + metadata filters); RAG + LLM |
| RAG for retrieval (accuracy + speed) | RAG &lt; 1s goal; ChromaDB/OpenSearch; LangChain |
| NLG for natural language recommendations | RAG generation service (T025); LLM post-retrieval |
| Dataset: ABO, product description/keywords/images | Phase 0 T000b/c; Phase 5 US3 ingestion |
| Data preprocessing (category, brand, description, image link, tags) | T041–T042: data_processing.py, ingest_abo script |
| Embedding generation (pretrained/finetuned) | T010 embeddings; optional fine-tuning (research/Kaggle) |
| Vector DB (Chroma/Milvus) | ChromaDB dev; OpenSearch/Chroma prod (plan); T009 |
| Image captioning → append to descriptions | T000c Kaggle pipeline → enriched_products.csv |
| LLMs for generation post-retrieval | T025 RAG service; LangChain |
| RAG pipeline as REST endpoint | POST /api/v1/chat, POST /api/v1/search, POST /api/v1/voice/query |
| UI: Streamlit/Gradio, text input, result + product identifier | Phase 4 (US2): T031–T040; product cards with product_ids |
| (Optional) Fine-tuned models, LORA/QLORA, triplet loss | Plan: fine-tuning on Kaggle; Phase 7 / requirements §5 |
| (Optional) Voice input | Phase 3 T027b voice/query; Phase 4 Mic + Read Aloud |
| (Optional) Follow-up / conversational history | T039 session chat history for follow-ups; T032 chat state |

### Submission Guidelines

| Requirement | Coverage |
|-------------|----------|
| Colab/Kaggle notebooks + models | Phase 0 notebooks; Kaggle for EDA + captioning; models in `models/` |
| Preprocessed data + documentation | US3 T041–T043; quickstart.md |
| AWS code for RAG and inference + documentation | Backend deployable to AWS EC2; T045 README, T048 Docker |
| Video recording of working bot | **Task**: T051 (Phase 6) – record and document demo video |
| requirements.txt, README, instructions | T004–T005; T045 README; quickstart |
| System architecture document | architecture.md in .specify/memory; **Task**: T052 – ensure architecture doc is complete and linked |
| Document results/experiences | test_strategy.md; T047 eval export; T102 Golden Set |
| Docker and deployment steps | T048 docker-compose; quickstart |

### Model Evaluation & Testing (Problem Statement)

| Requirement | Coverage |
|-------------|----------|
| Precision testing for retrieval (test dataset: query + positive products) | test_strategy.md Precision@5; T047/T102 50-query Gold Standard |
| Qualitative testing for generative results | test_strategy RAGAS/TruLens; 50-query Helpfulness/Naturalness |
| P95 and P99 latency | Plan performance goals; test_strategy latency benchmarking |
| Plan for further testing/improvement | test_strategy.md §2–3; Phase 7 T102 |

### Evaluation Criteria (Weights) → Plan

| Criterion (Weight) | Coverage |
|--------------------|----------|
| EDA and Data Preparation (15%) | T000b ShopTalk EDA; T041–T042 preprocessing |
| Experimentation with Models (25%) | Phase 0 notebooks; fine-tuning/Kaggle; requirements §5; Phase 7 |
| Deployment (25%) | Model load once (T029); REST API; T048 Docker; T045/T052 docs |
| E2E testing (15%) | Contract/integration tests; test_strategy; T047/T102 |
| UI/UX (5%) | US2 Streamlit; conversational history; product cards; status messages |
| Solution Documentation (15%) | README, quickstart, architecture, test_strategy; T045, T049, T052 |

### Gaps Addressed

- **Product URL**: Problem statement FAQ says “textual results with URL if available”. API Product schema has `image_url`; add optional `product_url` in contract if dataset provides links (task note in Phase 6).
- **Video demo**: Added as T051 in tasks.md.
- **System architecture document**: Explicit task T052 to complete and link architecture.md.

## Complexity Tracking

No constitution violations. Table left empty.
