# Data Model: ShopTalk

**Feature**: ShopTalk AI Assistant | **Date**: 2025-02-08

Entities and validation rules derived from [requirements.md](requirements.md) and [architecture.md](architecture.md).

---

## 1. Product (ABO / Vector Store)

Represents an item in the catalog; stored in ChromaDB/OpenSearch with metadata and optional image caption.

| Field | Type | Validation | Notes |
|-------|------|------------|--------|
| id | string | required, unique | ABO product ID |
| title | string | required | Product title |
| description | string | optional | Text description |
| price | decimal | optional, >= 0 | Price (currency from dataset) |
| category | string | optional | Product category (filter dimension) |
| image_caption | string | optional | BLIP/CLIP caption (stored for search) |
| image_url | string | optional | Path or URL to image |
| embedding_id | string | optional | Vector store document ID (if different from id) |

**Relationships**: None (flat document). Used by hybrid search (semantic on description+title+caption; keyword/filter on price, category).

**Lifecycle**: Ingested by pipeline (Airflow); updated daily. No application-level state transitions.

---

## 2. Chat Message (In-Memory Only)

A single message in the session chat. Not persisted except optional export for 50 qualitative responses.

| Field | Type | Validation | Notes |
|-------|------|------------|--------|
| role | enum | "user" \| "assistant" \| "system" | Who sent the message |
| content | string | required | Text (user query or assistant response) |
| timestamp | datetime | optional | Client or server time |
| message_type | enum | "text" \| "status" \| "error" | status = "Listening…" etc.; error = STT/RAG/TTS failure |
| product_ids | list[string] | optional | For assistant: product IDs referenced in this turn (for product cards) |

**Relationships**: Part of a logical Session (no persisted session entity; identified by browser/session). Order by timestamp for context in RAG.

**Lifecycle**: Created when user speaks or assistant responds; discarded on tab close/refresh. Optional: anonymized export of last N assistant messages for evaluation (50-query set).

---

## 3. Voice Request / Response (API)

Payloads for the voice pipeline. Not stored; request/response only.

**VoiceQueryRequest** (e.g. from Streamlit to FastAPI):

| Field | Type | Validation |
|-------|------|------------|
| audio_base64 | string | optional | Recorded audio (e.g. base64 WAV) |
| session_id | string | optional | For session isolation (if multi-request) |
| filters | object | optional | { "price_max": number, "category": string } |

**VoiceQueryResponse**:

| Field | Type | Validation |
|-------|------|------------|
| transcript | string | optional | STT output (empty on STT failure) |
| response_text | string | required | Assistant reply (or error message) |
| product_ids | list[string] | optional | Top-K product IDs for product cards |
| status | string | optional | "ok" \| "stt_failed" \| "no_results" \| "pipeline_error" |
| audio_base64 | string | optional | TTS audio (optional; e.g. for empty/error, no TTS) |

---

## 4. Search Request / Result (Internal or API)

**SearchRequest** (keyword + semantic inputs):

| Field | Type | Validation |
|-------|------|------------|
| query_text | string | required |
| top_k | int | optional, default 5, 1–20 |
| filters | object | optional | price range, category |
| session_context | list[ChatMessage] | optional | For "show me the blue one" style follow-ups |

**SearchResult**:

| Field | Type | Validation |
|-------|------|------------|
| product_ids | list[string] | Ordered by relevance |
| products | list[Product] | optional | Full product records for display |
| total | int | Count (0 for empty search) |

---

## 5. Evaluation Export (50-Query Set)

For qualitative scoring only; stored only when explicitly exporting.

| Field | Type | Validation |
|-------|------|------------|
| query_id | string | optional, unique per export |
| query_text | string | Anonymized user query text |
| response_text | string | Assistant response |
| product_ids | list[string] | Retrieved product IDs |
| scores | object | optional | Helpfulness, Naturalness (filled later) |

**Retention**: Anonymized transcripts for 50 query responses only; no voice recordings; no general chat persistence.

---

## 6. Identity and Uniqueness

- **Product.id**: Unique per ABO dataset; primary key in vector store.
- **Chat messages**: No persistent ID; in-memory list keyed by session.
- **Session**: No stored entity; identified by Streamlit session_id or browser cookie/lifetime.

---

## 7. State Transitions

- **Product**: Ingested → Indexed (in vector store). No app-level states.
- **Chat**: Message appended to in-memory list; no state machine.
- **Voice pipeline**: Recording → Transcribing → Searching → Generating → Speaking; any step can transition to error state (response_text = error message, status = stt_failed | no_results | pipeline_error).
