# Data Artifacts (`data/`)

This folder stores runtime artifacts exported from notebooks and loaded by serving code.

`data/` is intentionally not versioned with large binary artifacts. Populate it after running notebooks.

## Expected Directory Layout

```text
data/
├── rag_products.pkl
├── rag_text_index.npy
├── rag_image_index.npy
├── rag_config.json
├── finetuned_text_index.npy                  # optional but recommended
├── llm_evaluation.csv                        # optional analysis output
├── llm_comparison.csv                        # optional analysis output
├── llm_config.json                           # optional LLM settings snapshot
├── eval_queries.json                         # optional, used in eval workflows
└── models/
    └── finetuned-shoptalk-emb/              # optional fine-tuned ST model directory
```

## Required vs Optional

### Required (minimum to serve search/chat)
- `rag_products.pkl` (or `products_with_prices.pkl`)
- `rag_text_index.npy` (or `finetuned_text_index.npy`)
- `rag_image_index.npy`
- `rag_config.json`

### Optional (recommended for best quality)
- `finetuned_text_index.npy`
- `models/finetuned-shoptalk-emb/`

### Optional (evaluation/debug artifacts)
- `llm_evaluation.csv`
- `llm_comparison.csv`
- `llm_config.json`
- `eval_queries.json`
- `finetune_results.json`
- `finetune_evaluation.csv`
- `finetune_visualisations.png`

## Producer -> Consumer Mapping

- `03-rag-prototype.ipynb` -> baseline retrieval artifacts (`rag_*`).
- `04-llm-integration.ipynb` -> LLM evaluation/config artifacts.
- `05-fine-tuning.ipynb` -> fine-tuned index/model and finetune eval outputs.

Serving consumers:
- `backend/src/services/embeddings.py`
- `app/streamlit_app.py` (standalone mode)

## Model Resolution Rules (Serving)

SentenceTransformer source preference:
1. `FINETUNED_MODEL_PATH` env var (if path exists)
2. `data/models/finetuned-shoptalk-emb/`
3. base model id (`all-MiniLM-L6-v2`)

Text index preference:
1. `finetuned_text_index.npy`
2. `rag_text_index.npy`

## How to Populate

From notebook environment, copy exports into this folder while preserving filenames.

Example:
- local copy into repo `data/`
- or upload to EC2 `~/shoptalk/data/` before running `deploy/deploy.sh`

## Troubleshooting

- Backend fails at startup with `No product data in /app/data`:
  - missing `rag_products.pkl` / `products_with_prices.pkl`.
- Search returns empty or poor results:
  - check index/model mismatch; ensure fine-tuned index + fine-tuned encoder path are aligned.
- Container starts but model load fails:
  - verify `models/finetuned-shoptalk-emb/` exists and is readable.
