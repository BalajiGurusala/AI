# Tasks: ShopTalk AI Assistant

**Input**: Design documents from `.specify/memory/`  
**Prerequisites**: plan.md, requirements.md (as spec), research.md, data-model.md, contracts/

**Tests**: Included per test_strategy.md (unit, integration, contract for pipeline and API).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/`, `backend/tests/` (repository root = Shop_Talk_Assistant or parent)
- **Frontend**: `frontend/src/`, `frontend/tests/`
- **Data/Models**: `data/`, `models/` at repo root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and structure per plan.md

- [ ] T001 Create backend directory structure (backend/src/models, backend/src/services, backend/src/api, backend/tests/unit, backend/tests/integration, backend/tests/contract)
- [ ] T002 Create frontend directory structure (frontend/src/components, frontend/src/pages, frontend/src/services, frontend/tests)
- [ ] T003 Create repository root directories (data, models) and .env.example with OPENAI_API_KEY, ELEVENLABS_API_KEY, CHROMA_PATH placeholders
- [ ] T004 Initialize backend Python project with pyproject.toml or requirements.txt (FastAPI, uvicorn, LangChain, chromadb, openai-whisper, pydantic, python-dotenv, httpx)
- [ ] T005 Initialize frontend Python project with requirements.txt (Streamlit, httpx) in frontend/
- [ ] T006 [P] Configure ruff and black for backend in backend/pyproject.toml or backend/ruff.toml
- [ ] T007 [P] Add backend pytest configuration (backend/pytest.ini or backend/pyproject.toml) with markers for unit, integration, contract

**Checkpoint**: Project skeleton and tooling ready

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure required before any user story. Models load once at startup.

- [ ] T008 Implement pydantic models for Product, ChatMessage, VoiceQueryRequest, VoiceQueryResponse, SearchRequest, SearchResponse in backend/src/models/schemas.py
- [ ] T009 Implement ChromaDB client wrapper (connect, query with metadata filter) in backend/src/services/vector_store.py with configurable path from env
- [ ] T010 Implement embedding loader (HuggingFace all-MiniLM-L6-v2 or env override) in backend/src/services/embeddings.py; load once at module/singleton level
- [ ] T011 Create FastAPI app and router skeleton in backend/src/api/main.py with CORS for Streamlit origin
- [ ] T012 Implement GET /health in backend/src/api/main.py returning status, embedding_loaded, chroma_connected
- [ ] T013 Implement centralized error handling and pipeline status mapping (stt_failed, no_results, pipeline_error) in backend/src/api/errors.py
- [ ] T014 Add .env loading and settings (pydantic BaseSettings or dotenv) in backend/src/config.py for API keys, CHROMA_PATH, model names
- [ ] T015 [P] Unit test for VectorStore.search() returning correct shape and handling empty results in backend/tests/unit/test_vector_store.py
- [ ] T016 [P] Unit test for audio conversion helpers (wav to float32) in backend/tests/unit/test_voice_utils.py
- [ ] T017 Contract test for GET /health in backend/tests/contract/test_health.py

**Checkpoint**: Foundation ready – health works, vector store and embeddings available; user story implementation can begin

---

## Phase 3: User Story 1 – Voice + RAG Pipeline (Priority: P1) – MVP

**Goal**: User submits voice audio; app transcribes, runs hybrid search, generates natural language response, optionally returns TTS audio. Handles STT failure, empty results, and pipeline errors with spec messages.

**Independent Test**: POST audio to /api/v1/voice/query; receive response_text and product_ids or error status; GET /health returns ok when models and Chroma are ready.

### Tests for User Story 1

- [ ] T018 [P] [US1] Contract test for POST /api/v1/voice/query request/response schema in backend/tests/contract/test_voice_api.py
- [ ] T019 [P] [US1] Contract test for POST /api/v1/search request/response schema in backend/tests/contract/test_search_api.py
- [ ] T020 [US1] Integration test: Audio file → Whisper → text in backend/tests/integration/test_voice_pipeline.py
- [ ] T021 [US1] Integration test: Query "red shirt" → vector store returns at least one result with shirt in metadata in backend/tests/integration/test_rag_pipeline.py

### Implementation for User Story 1

- [ ] T022 [P] [US1] Implement STT service (Whisper base, load once) in backend/src/services/stt.py with error handling returning None or raising for timeout/API error
- [ ] T023 [P] [US1] Implement TTS service (ElevenLabs primary, gTTS fallback) in backend/src/services/tts.py; optional audio output for empty/error (no TTS per spec)
- [ ] T024 [US1] Implement hybrid search service (semantic + metadata filters price_max, category) in backend/src/services/search.py using ChromaDB and embedding model
- [ ] T025 [US1] Implement RAG generation service (retrieve top-k, build prompt, call LLM) in backend/src/services/rag.py using LangChain; handle zero results with spec message text
- [ ] T026 [US1] Implement voice pipeline orchestrator (STT → search → RAG → TTS) in backend/src/services/voice_pipeline.py with status stt_failed, no_results, pipeline_error and spec response messages
- [ ] T027 [US1] Implement POST /api/v1/voice/query in backend/src/api/routes/voice.py (accept multipart or JSON with audio_base64, filters); return VoiceQueryResponse
- [ ] T028 [US1] Implement POST /api/v1/search in backend/src/api/routes/search.py (SearchRequest → SearchResponse with product_ids, products, total)
- [ ] T029 [US1] Wire voice and search routes into backend/src/api/main.py and ensure models load once at startup
- [ ] T030 [US1] Add logging for voice and RAG pipeline steps in backend/src/services/voice_pipeline.py and backend/src/api/routes/voice.py

**Checkpoint**: US1 complete – voice endpoint returns transcript + response + product_ids or error; search endpoint returns products; integration tests pass

---

## Phase 4: User Story 2 – Streamlit Chat UI (Priority: P2)

**Goal**: User sees Streamlit UI with sidebar (filters, Record), chat area (user/assistant bubbles, status "Listening…", "Searching…", "Generating…", error messages), and product cards (image, title, price, Add to Cart mock). Session-scoped in-memory chat history; context-aware follow-ups.

**Independent Test**: Open Streamlit app; click Record and speak; see status then reply and product cards; change filters and search again; see empty-state or error message when applicable.

### Implementation for User Story 2

- [ ] T031 [P] [US2] Implement API client for backend (health, voice/query, search) in frontend/src/services/api_client.py
- [ ] T032 [US2] Implement session state for chat messages (list of role, content, message_type, product_ids) in frontend/src/services/chat_state.py
- [ ] T033 [US2] Build chat message component (user bubble, assistant bubble, status, error) in frontend/src/components/chat_message.py
- [ ] T034 [US2] Build product card component (image, title, price, Add to Cart mock button) in frontend/src/components/product_card.py
- [ ] T035 [US2] Build sidebar (Price filter, Category filter, Record button) in frontend/src/components/sidebar.py
- [ ] T036 [US2] Implement main chat page: render messages, show status during request, call voice API on Record and append user + assistant messages in frontend/src/pages/chat.py
- [ ] T037 [US2] Implement product grid: display product cards from last assistant message product_ids in frontend/src/pages/chat.py or frontend/src/components/product_grid.py
- [ ] T038 [US2] Wire Record button to record audio, send to POST /api/v1/voice/query, display "Listening…" then "Searching…" then "Generating…" then result or error in frontend
- [ ] T039 [US2] Pass session chat history as context to backend for follow-up queries (e.g. "show me the blue one") in frontend/src/services/api_client.py and backend if needed
- [ ] T040 [US2] Handle empty search and pipeline error messages in UI (show assistant bubble with spec text, no TTS for error)

**Checkpoint**: US2 complete – full voice-driven chat UI with filters, status, product cards, and error/empty states

---

## Phase 5: User Story 3 – Data Ingestion (Priority: P3)

**Goal**: Load ABO (or sample) data; chunk text; generate embeddings; upsert into ChromaDB so hybrid search returns real products.

**Independent Test**: Run ingestion script; then POST /api/v1/search with query_text; receive products from vector store.

### Implementation for User Story 3

- [ ] T041 [US3] Add data processing utilities (clean text, chunk) in backend/src/services/data_processing.py; unit test DataCleaner.clean_text() in backend/tests/unit/test_data_processing.py
- [ ] T042 [US3] Implement ingestion script: load ABO CSV (or sample), build Product documents, chunk description+title+caption, embed, upsert ChromaDB in backend/scripts/ingest_abo.py or pipelines/ingest.py
- [ ] T043 [US3] Document ingestion in quickstart (optional image captioning on Kaggle, then merge captions and re-run embed/upsert) in .specify/memory/quickstart.md
- [ ] T044 [US3] Add minimal sample dataset or fixture for dev (small CSV + ChromaDB seed) so backend/tests/integration can run without full ABO

**Checkpoint**: US3 complete – vector store populated; search returns real products; ingestion repeatable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and optional evaluation path

- [ ] T045 [P] Add README with project overview, prerequisites, and link to .specify/memory/quickstart.md at repository root
- [ ] T046 Run quickstart.md steps and fix any path or env issues (backend, frontend, data dir)
- [ ] T047 [P] Optional: Add 50-query evaluation export (anonymized transcript + response + product_ids) in backend/src/services/evaluation_export.py and document in test_strategy.md
- [ ] T048 Add docker-compose.yml skeleton (FastAPI, Streamlit, ChromaDB) for local dev per requirements
- [ ] T049 Code cleanup: Google-style docstrings for public modules in backend/src and frontend/src
- [ ] T050 Ensure all pydantic models and API request/response match .specify/memory/contracts/api-openapi.yaml

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies – start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 – blocks all user stories
- **Phase 3 (US1)**: Depends on Phase 2 – MVP voice + RAG pipeline
- **Phase 4 (US2)**: Depends on Phase 3 – needs voice and search API
- **Phase 5 (US3)**: Depends on Phase 2 (can run in parallel with US1/US2 for backend-only; for full E2E, after US1)
- **Phase 6 (Polish)**: Depends on Phase 3–5 as needed

### User Story Dependencies

- **US1 (P1)**: After Foundational – no dependency on US2/US3 (can use fixture or minimal Chroma data for integration tests)
- **US2 (P2)**: After US1 – depends on voice and search endpoints
- **US3 (P3)**: After Foundational – can run parallel to US1 for backend; provides real data for E2E

### Within Each User Story

- Contract/integration tests before or alongside implementation
- Models/schemas before services; services before routes
- US1: STT, TTS, search, RAG, then orchestrator and routes

### Parallel Opportunities

- T006, T007 (Setup) can run in parallel
- T015, T016, T017 (Foundational tests) can run in parallel
- T018, T019 (US1 contract tests) can run in parallel
- T022, T023 (US1 STT, TTS) can run in parallel
- T031, T033, T034, T035 (US2 client and components) can run in parallel after T032
- T045, T047, T049 (Polish) marked [P] where independent

---

## Parallel Example: User Story 1

```bash
# Contract tests in parallel:
Task T018: backend/tests/contract/test_voice_api.py
Task T019: backend/tests/contract/test_search_api.py

# Services in parallel (after search/RAG design):
Task T022: backend/src/services/stt.py
Task T023: backend/src/services/tts.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup  
2. Complete Phase 2: Foundational  
3. Complete Phase 3: User Story 1 (voice + RAG pipeline)  
4. **STOP and VALIDATE**: Call /health and POST /api/v1/voice/query (and /api/v1/search) with test data  
5. Demo backend API; add minimal Streamlit caller if needed for demo  

### Incremental Delivery

1. Setup + Foundational → health and vector store ready  
2. US1 → Voice + RAG API testable (MVP)  
3. US2 → Full Streamlit UI (voice-driven chat + product cards)  
4. US3 → Real ABO ingestion  
5. Polish → Quickstart, Docker, docs  

### Suggested MVP Scope

- **MVP = Phase 1 + Phase 2 + Phase 3 (US1)**  
- Delivers: backend with health, voice/query, search; contract and integration tests; no UI required for MVP definition, but US2 is needed for end-to-end user flow.

---

## Notes

- [P] = parallelizable (different files, no ordering dependency within phase)
- [USn] = task belongs to that user story for traceability
- Each user story is independently testable at its checkpoint
- Models must load once at startup (plan/constitution); no lazy load in inference path
- Commit after each task or logical group; run tests after Phase 2 and Phase 3
