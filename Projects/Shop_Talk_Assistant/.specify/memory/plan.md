# Implementation Plan: ShopTalk AI Assistant

**Branch**: `feature/001-shoptalk-mvp-setup` | **Date**: 2025-02-08 | **Spec**: [requirements.md](.specify/memory/requirements.md)  
**Input**: Feature specification from `.specify/memory/requirements.md`

## Summary

ShopTalk is an AI shopping assistant with voice interaction, hybrid (semantic + keyword) product search, and RAG over the Amazon Berkeley Objects (ABO) dataset. Primary technical approach: Streamlit frontend, FastAPI backend, LangChain orchestration, ChromaDB/OpenSearch for vectors, Whisper STT, ElevenLabs/gTTS TTS; fine-tuning and image captioning on Kaggle; MLOps via Airflow, MLflow, Ray Tune, Evidently AI, and Grafana.

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

**Structure Decision**: Web application (backend + frontend) chosen per constitution (Streamlit + FastAPI). Backend holds orchestration, vector store, STT/TTS/LLM; frontend is Streamlit with session-scoped chat state.

## Complexity Tracking

No constitution violations. Table left empty.
