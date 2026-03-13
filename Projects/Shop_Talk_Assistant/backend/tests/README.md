# Tests (`backend/tests/`)

Test suite for the ShopTalk AI Shopping Assistant backend.
Covers unit, integration, and contract testing layers aligned with
[test_strategy.md](../../.spec/test_strategy.md) and [api-openapi.yaml](../../.specify/memory/contracts/api-openapi.yaml).

**170 tests | 3 layers | 11 modules | < 1 second runtime**

---

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ShopTalk Test Pyramid                            │
│                                                                         │
│                        ┌───────────────┐                                │
│                        │  Integration  │  11 tests                      │
│                        │   (RAG E2E)   │  Synthetic data fixtures       │
│                        └───────┬───────┘                                │
│                    ┌───────────┴───────────┐                            │
│                    │     Contract Tests    │  31 tests                   │
│                    │   (OpenAPI schemas)   │  API shape validation       │
│                    └───────────┬───────────┘                            │
│          ┌─────────────────────┴─────────────────────┐                  │
│          │              Unit Tests                    │  128 tests       │
│          │  Search │ Schemas │ Voice │ Config │ Spec  │  Fast, isolated  │
│          └─────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

```
conftest.py  ──────────────────────  Shared fixtures (20 synthetic products,
    │                                 embedding indexes, mock encoders)
    │
    ├── unit/
    │   ├── test_search.py ─────────  src/search.py
    │   │   (tokenizer, overlap, head-nouns, gender/color intent,
    │   │    dynamic alpha, L2 norm, retrieval, reranking, hybrid_search)
    │   │
    │   ├── test_schemas.py ────────  backend/src/models/schemas.py
    │   │   (Filters, ChatMessage, Product, ChatRequest, ChatResponse,
    │   │    SearchRequest, SearchResponse, VoiceQueryResponse, HealthResponse)
    │   │
    │   ├── test_voice_utils.py ────  backend/src/services/voice.py
    │   │   (VoiceService lifecycle, Whisper STT mocked, gTTS TTS,
    │   │    WAV generation, temp file cleanup)
    │   │
    │   ├── test_data_processing.py   src/search.py (text cleaning)
    │   │   (emojis, unicode, HTML entities, long input, punctuation,
    │   │    hyphens, tabs/newlines)
    │   │
    │   ├── test_config.py ─────────  backend/src/config.py
    │   │   (default values, env overrides, API key handling)
    │   │
    │   └── test_spec_compliance.py   Cross-cutting spec validation
    │       ├── constitution.md  (pydantic, hardware detect, singleton)
    │       ├── requirements.md  (endpoints, CORS, error messages)
    │       ├── data-model.md    (Product fields, ChatMessage roles)
    │       ├── architecture.md  (FastAPI, Streamlit, lifespan)
    │       └── test_strategy.md (test file coverage meta-check)
    │
    ├── contract/
    │   ├── test_health.py ─────────  GET /health
    │   │   (response shape, types, serialization roundtrip)
    │   │
    │   ├── test_search_api.py ─────  POST /api/v1/search + /api/v1/chat
    │   │   (request validation, top_k bounds, filters, response shapes,
    │   │    Product schema fields, status enum)
    │   │
    │   └── test_voice_api.py ──────  POST /api/v1/voice/query
    │       (stt_failed, no_results, pipeline_error status,
    │        audio_base64, products list shape)
    │
    └── integration/
        └── test_rag_pipeline.py ───  Full search + RAG pipeline
            ├── Hybrid search with synthetic data
            ├── Category & price filter validation
            ├── Zero-results edge case
            ├── Context formatting for LLM
            ├── DataFrame → Product conversion
            └── LLM manager lifecycle
```

---

## Prerequisites

1. **Python 3.10+** with the project virtual environment activated
2. **pytest** installed (already in dev dependencies)

```bash
cd Shop_Talk_Assistant
source .venv/bin/activate
pip install pytest   # if not already installed
```

> No ML models, API keys, data files, or running services are needed.
> All tests use synthetic fixtures and mocks.

---

## How to Run Tests

### Run Everything

```bash
cd Shop_Talk_Assistant
PYTHONPATH=. python -m pytest backend/tests/ -v
```

### Run by Layer

```bash
# Unit tests only (128 tests, < 0.5s)
PYTHONPATH=. python -m pytest backend/tests/unit/ -v

# Contract tests only (31 tests)
PYTHONPATH=. python -m pytest backend/tests/contract/ -v

# Integration tests only (11 tests)
PYTHONPATH=. python -m pytest backend/tests/integration/ -v
```

### Run a Single Test File

```bash
# Search pipeline tests
PYTHONPATH=. python -m pytest backend/tests/unit/test_search.py -v

# Schema validation tests
PYTHONPATH=. python -m pytest backend/tests/unit/test_schemas.py -v

# Spec compliance checks
PYTHONPATH=. python -m pytest backend/tests/unit/test_spec_compliance.py -v

# Voice service tests
PYTHONPATH=. python -m pytest backend/tests/unit/test_voice_utils.py -v

# RAG pipeline integration
PYTHONPATH=. python -m pytest backend/tests/integration/test_rag_pipeline.py -v
```

### Run a Single Test

```bash
PYTHONPATH=. python -m pytest backend/tests/unit/test_search.py::TestColorIntent::test_detects_synonym -v
```

### Useful Options

```bash
# Show full tracebacks on failure
PYTHONPATH=. python -m pytest backend/tests/ -v --tb=long

# Stop on first failure
PYTHONPATH=. python -m pytest backend/tests/ -v -x

# Show slowest 5 tests
PYTHONPATH=. python -m pytest backend/tests/ -v --durations=5

# Run with print output visible
PYTHONPATH=. python -m pytest backend/tests/ -v -s
```

---

## Test Inventory

### Unit Tests (128 tests)

| File | Tests | What It Validates |
|------|------:|-------------------|
| `test_search.py` | 31 | Tokenization, field overlap, head-noun extraction, gender/color intent detection, dynamic alpha calculation, L2 normalization, in-memory retrieval shape/sorting, reranking with penalties, full `hybrid_search` entry point, metadata filters, alpha attribute |
| `test_schemas.py` | 19 | All 9 pydantic models: required fields, optional defaults, validation rejection, type constraints (top_k bounds 1-20, role enum) |
| `test_data_processing.py` | 13 | Text cleaning edge cases: emojis, unicode, HTML entities, special chars, numeric strings, very long input, punctuation-only, tabs/newlines, hyphenated compounds |
| `test_voice_utils.py` | 11 | VoiceService singleton lifecycle, STT-without-load guard, TTS unavailable fallback, WAV byte generation, mocked Whisper transcription (success, empty, exception), temp file cleanup |
| `test_config.py` | 12 | Settings defaults (data_dir, model IDs, ports, CORS), environment variable overrides, API key handling |
| `test_spec_compliance.py` | 21 | Cross-references 5 spec docs: constitution (pydantic, hardware detect, singleton, .env), requirements (endpoints, CORS, error messages, session context), data-model (Product fields, ChatMessage roles, status enum, top_k bounds, filters), architecture (FastAPI, Streamlit, lifespan startup), test_strategy (test file coverage meta-check) |

### Contract Tests (31 tests)

| File | Tests | What It Validates |
|------|------:|-------------------|
| `test_health.py` | 6 | `GET /health` response: required fields (status, embedding_loaded, chroma_connected), types, optional extended fields, serialization roundtrip |
| `test_search_api.py` | 18 | `POST /api/v1/search` + `POST /api/v1/chat`: query_text required, top_k default=5 with bounds [1,20], filters shape, Product schema has all 12 fields from OpenAPI, ChatResponse status values, session_context support |
| `test_voice_api.py` | 7 | `POST /api/v1/voice/query`: response_text required, status enum (ok/stt_failed/no_results/pipeline_error), audio_base64 presence, products list shape |

### Integration Tests (11 tests)

| File | Tests | What It Validates |
|------|------:|-------------------|
| `test_rag_pipeline.py` | 11 | Full hybrid search with synthetic 20-product catalog, result metadata columns, category filter correctness, price filter cap, zero-results edge case, LLM context formatting, `_row_to_product` conversion with missing fields, LLMManager lifecycle and mocked generation |

---

## Fixtures & Synthetic Data

All tests run against synthetic fixtures defined in `conftest.py` -- no real data files, ML models, or API keys required.

| Fixture | Description |
|---------|-------------|
| `sample_df` | 20 synthetic products spanning 10 categories (SHOES, SHIRT, WATCH, LIGHTING, etc.) with realistic titles, brands, colors, prices, and descriptions |
| `text_index` | Random normalised float32 array (20 x 384) simulating SentenceTransformer embeddings |
| `image_index` | Random normalised float32 array (20 x 512) simulating CLIP embeddings |
| `encode_text_fn` | Deterministic mock text encoder (query string -> 1x384 unit vector) |
| `encode_clip_fn` | Deterministic mock CLIP encoder (query string -> 1x512 unit vector) |

---

## Mapping to Spec Documents

| Spec Document | Coverage |
|---------------|----------|
| `test_strategy.md` | Unit tests for data processing, vector search shape, voice utils; Integration tests for voice pipeline and RAG pipeline |
| `data-model.md` | All entity schemas validated: Product (12 fields), ChatMessage (role enum), VoiceQueryResponse (status enum), SearchRequest (top_k bounds), Filters |
| `api-openapi.yaml` | Contract tests for all 4 endpoints: health, chat, search, voice/query |
| `constitution.md` | Pydantic validation, hardware detection, singleton load-once pattern, .env configuration |
| `requirements.md` | API endpoints exist, CORS configured, error messages match spec, session context support |
| `architecture.md` | FastAPI app structure, Streamlit app exists, lifespan model loading, shared search module |

---

## Adding New Tests

1. **Unit test**: Add to `unit/test_<module>.py`. Use synthetic data from `conftest.py` fixtures.
2. **Contract test**: Add to `contract/test_<endpoint>.py`. Validate schema shapes against `api-openapi.yaml`.
3. **Integration test**: Add to `integration/test_<pipeline>.py`. Test cross-module flows with fixtures.

Follow existing naming conventions:

```python
class TestFeatureName:
    def test_expected_behavior(self):
        ...
    def test_edge_case(self):
        ...
    def test_failure_mode(self):
        ...
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'backend'` | Run with `PYTHONPATH=.` from project root |
| `ModuleNotFoundError: No module named 'src'` | Run with `PYTHONPATH=.` from project root |
| Tests hang or are slow | All tests should complete in < 1s. If hanging, check for accidental network calls |
| Import errors in `schemas.py` | Ensure `backend/src/models/__init__.py` exists |
