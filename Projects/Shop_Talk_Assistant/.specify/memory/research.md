# Research: ShopTalk Implementation Decisions

**Feature**: ShopTalk AI Assistant | **Date**: 2025-02-08

All Technical Context items are resolved from constitution and requirements. This document records rationale and alternatives for key choices.

---

## 1. Embedding Model (Base vs Fine-Tuned)

**Decision**: Base: HuggingFace `all-MiniLM-L6-v2`. Fine-tuned: custom adapter on ABO via Triplet Loss (LoRA/QLORA) on Kaggle; artifacts in MLflow and local `models/`.

**Rationale**: MiniLM is fast and runs locally; fine-tuning on ABO improves retrieval for product semantics. Triplet Loss + LoRA keeps Kaggle training feasible.

**Alternatives considered**: OpenAI Embeddings (better quality, cost and latency); sentence-transformers larger models (slower on CPU/Mac).

---

## 2. Vector Database

**Decision**: Dev: ChromaDB (local persistence in `data/chroma`). Prod: AWS OpenSearch or ChromaDB on EC2. Benchmark retrieval vs Milvus/FAISS for reporting.

**Rationale**: ChromaDB is simple for dev and supports hybrid search patterns; OpenSearch fits AWS production. Benchmark satisfies academic comparison requirement.

**Alternatives considered**: FAISS (in-memory, no persistence); Milvus (more ops overhead for single-user MVP).

---

## 3. LLM for RAG Generation

**Decision**: Comparative study: GPT-4o (benchmark) and Llama 3 (Ollama or Groq). Implementation supports both via config.

**Rationale**: Grading requires comparing proprietary vs open-source; single-user/demo allows optional local Ollama.

**Alternatives considered**: Llama-only (no benchmark); Claude (not in constitution).

---

## 4. Speech-to-Text (STT)

**Decision**: OpenAI Whisper local `base` model.

**Rationale**: Constitution specifies Whisper base; runs on MacBook; <2s latency target achievable.

**Alternatives considered**: Whisper large (slower); cloud STT APIs (cost, latency).

---

## 5. Text-to-Speech (TTS)

**Decision**: ElevenLabs (primary), gTTS (fallback).

**Rationale**: Constitution specifies both; high fidelity for demo; fallback for reliability.

**Alternatives considered**: TTS-only (e.g., gTTS only) rejected for quality; Piper/Coqui (not in constitution).

---

## 6. Multi-Modal Interface (Text vs Voice)

**Decision**: Text-based chat is primary; voice is secondary/optional. The app must be fully functional without a microphone. Mic button populates the text input (STT); optional "Read Aloud" for TTS.

**Rationale**: Requirements state standard text chat (like ChatGPT) with typed queries as primary; voice adds accessibility and demo value but is not required for core flow.

**Alternatives considered**: Voice-first (rejected per spec); voice-only (rejected – text required).

---

## 7. Hybrid Search (Semantic + Keyword)

**Decision**: Implement hybrid as: (1) semantic search via vector similarity (ChromaDB/OpenSearch); (2) keyword filters (e.g., price, category) applied in application or via metadata filter. Combine into single ranked result set.

**Rationale**: Requirements mandate both "red shirt" (semantic) and "under $50" (keyword); vector DB supports metadata filtering; LangChain retriever can wrap combined logic.

**Alternatives considered**: Two-step only (no fusion) rejected; pure keyword rejected for semantic requirement.

---

## 8. Session and Chat State

**Decision**: In-memory only (Streamlit session state / FastAPI request-scoped or in-memory store). No DB for chat; optional anonymized export of 50 responses for qualitative evaluation.

**Rationale**: Clarifications: multi-user anonymous, session-isolated; chat lost on refresh; no persistence layer for chat.

**Alternatives considered**: Redis/DB for chat rejected per spec.

---

## 9. Error and Empty-State Messaging

**Decision**: STT failure: show "Couldn't hear that—try again" in chat, allow retry. Empty search: show "No products match that. Try different keywords or filters." (optional TTS). RAG/LLM/TTS failure: show "Something went wrong. Please try again.", no TTS for error. Status messages: "Listening…", "Searching…", "Generating…" in chat or near mic.

**Rationale**: All from spec clarifications; consistent pattern for testability and UX.

---

## 10. Model Loading and Consistency

**Decision**: Load all models (embedding, Whisper, LLM, TTS) once at app startup (global/singleton). Inference uses same tokenizers/transformers as training pipeline for embedding model.

**Rationale**: Technical constraint in requirements; avoids latency spikes and version drift.

**Alternatives considered**: Lazy loading rejected for latency and consistency.

---

## 11. MLOps Scope for MVP

**Decision**: Plan and contracts include Airflow, MLflow, Ray Tune, Evidently AI, Grafana per constitution; implementation order can phase MLOps after core voice+RAG pipeline. Docker Compose for local dev with FastAPI, Streamlit, ChromaDB (and optionally Airflow/MLflow).

**Rationale**: Requirements list full MLOps; MVP can deliver pipeline first, then add orchestration and monitoring.

**Alternatives considered**: Dropping MLOps rejected (grading requirement); full MLOps before app rejected for risk.
