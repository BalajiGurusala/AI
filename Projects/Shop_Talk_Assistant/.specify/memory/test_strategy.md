# Test Strategy & Evaluation Plan

## 1. Automated Testing Layers (CI/CD)
* **Unit Tests (`tests/unit/`):**
    * **Data Processing:** Test `DataCleaner.clean_text()` handles special characters and emojis.
    * **Vector Logic:** Test `VectorStore.search()` returns correct shape and handles empty results.
    * **Voice Utils:** Test audio file conversion functions (wav -> mp3 -> float32).
* **Integration Tests (`tests/integration/`):**
    * **Voice Pipeline:** Verify `Audio File -> Whisper -> Text` matches expected transcript.
    * **RAG Pipeline:** Verify `Query ("red shirt") -> Database` returns at least 1 result with "shirt" in metadata.

## 2. Research Evaluation (The Capstone Metrics)
* **Retrieval Evaluation:**
    * **Metric:** `Precision@5` (Are the top 5 retrieved products actually relevant?).
    * **Method:** Create a "Gold Standard" dataset of 50 queries + expected product IDs. Run script to compare.
* **Generation Evaluation (RAGAS / TruLens):**
    * **Faithfulness:** Does the answer contradict the retrieved product data? (Score 0-1).
    * **Answer Relevance:** Does the answer actually address the user's voice query? (Score 0-1).
* **Latency Benchmarking:**
    * Measure `Time-to-First-Token` (TTFT).
    * Measure `Total-Round-Trip` (Voice Start to Audio Response).
    * **Success Criteria:** Total interaction < 5 seconds.

## 3. MLOps Validation
* **Drift Detection (Evidently AI):**
    * **Embedding Drift:** Compare "Training Data Embeddings" vs. "Live Query Embeddings" daily.
    * **Alert Threshold:** Trigger warning if drift > 0.1 cosine distance.
* **Model Versioning:**
    * Ensure the `finetuned_model.pt` loaded in Inference matches the SHA256 hash in MLflow Registry.