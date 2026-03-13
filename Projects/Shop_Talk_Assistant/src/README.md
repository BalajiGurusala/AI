# Shared Core (`src/`)

Shared runtime logic used by both notebooks and serving services.

## Main Module

- `src/search.py`: hybrid retrieval + reranking engine.

## Why This Exists

Keeps one canonical implementation of search behavior:
- avoids copy/paste drift between notebooks and app/backend
- makes offline evaluation and online serving consistent

## What `hybrid_search` Does

1. Encode query (text + CLIP text).
2. Retrieve candidates (in-memory dot-product; optionally Chroma collections).
3. Fuse scores (`alpha` text/image weighting, optionally dynamic).
4. Apply metadata filters (`price_max`, `category`).
5. Stage-2 rerank with lexical/type/gender heuristics.
6. Return ranked DataFrame with `hybrid_score` and `_rank`.

## Consumers

- Notebooks in `notebooks/`
- Backend RAG service: `backend/src/services/rag.py`
- Streamlit standalone mode: `app/streamlit_app.py`
