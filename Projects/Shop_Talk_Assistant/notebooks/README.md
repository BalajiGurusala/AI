# Notebooks (`notebooks/`)

This folder contains the offline ML pipeline for ShopTalk: data understanding, enrichment, retrieval system construction, LLM integration, and embedding fine-tuning.

These notebooks are the **artifact producers** for the serving stack (`backend/`, `app/`).

## Recommended Execution Order

1. `01-shoptalk-eda.ipynb`
2. `02-image-captioning.ipynb`
3. `03-rag-prototype.ipynb`
4. `04-llm-integration.ipynb`
5. `05-fine-tuning.ipynb`

Run in order because later notebooks consume artifacts exported by earlier ones.

## Notebook-by-Notebook Guide

## `01-shoptalk-eda.ipynb`

**Purpose**
- Explore ABO data quality and distribution.
- Understand schema, category spread, missingness, and candidate fields for retrieval.

**Inputs**
- Raw ABO metadata and image pointers.

**Outputs**
- Analysis only (primarily exploratory figures/tables).

---

## `02-image-captioning.ipynb`

**Purpose**
- Generate or validate image-level captions for products.
- Create text enrichment that helps multimodal retrieval.

**Inputs**
- Product metadata + images.

**Typical Outputs**
- Caption-enriched product records (intermediate to feed NB03).

---

## `03-rag-prototype.ipynb`

**Purpose**
- Build the core hybrid search prototype.
- Create text/image embedding indexes and baseline retrieval behavior.

**Inputs**
- Enriched products from earlier steps.

**Key Exports (consumed by serving and later notebooks)**
- `rag_products.pkl`
- `rag_text_index.npy`
- `rag_image_index.npy`
- `rag_config.json`

---

## `04-llm-integration.ipynb`

**Purpose**
- Integrate hybrid retrieval with LLM response generation.
- Evaluate generation quality and retrieval-response behavior.

**Current evaluation design**
- LLM-as-judge metrics on 0-1 scale:
  - Faithfulness
  - Relevance
  - Helpfulness
  - Naturalness
- Includes query-level tracking (`query_id`, `query_text`) and visual diagnostics.

**Key Exports**
- `llm_evaluation.csv`
- `llm_comparison.csv`
- `llm_config.json`
- `eval_queries.json`
- `rag_pipeline_config.json`

> Note: price fields are optional/pass-through only; synthetic price generation is removed.

---

## `05-fine-tuning.ipynb`

**Purpose**
- Fine-tune embedding model with triplet training.
- Compare base vs full fine-tuned vs LoRA experiment.
- Benchmark retrieval and export serving-ready artifacts.

**Prereqs**
- NB03 artifacts (catalog + base indexes/config)
- NB04 `eval_queries.json` for standardized evaluation

**Key Exports**
- `finetuned_text_index.npy`
- `models/finetuned-shoptalk-emb/` (SentenceTransformer directory)
- `finetune_results.json`
- `finetune_evaluation.csv`
- `finetune_visualisations.png`

## Artifact Contract for Serving

For `backend/` and `app/` to run, copy exported files to project `data/`.

Minimum required:
- `rag_products.pkl` (or `products_with_prices.pkl`)
- `rag_text_index.npy` (or `finetuned_text_index.npy`)
- `rag_image_index.npy`
- `rag_config.json`

Recommended for fine-tuned serving:
- `finetuned_text_index.npy`
- `models/finetuned-shoptalk-emb/`

## Reproducibility Tips

- Keep package versions stable across notebook and serving envs.
- Avoid changing column names in exported artifacts unless you update serving code.
- Re-export full artifact set after retraining/fine-tuning.
- Validate with quick checks:
  - load all expected files,
  - run one retrieval query,
  - run one `/api/v1/chat` call.
