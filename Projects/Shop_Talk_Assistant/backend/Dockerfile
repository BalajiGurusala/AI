# ShopTalk Backend — FastAPI + ML Models
FROM python:3.11-slim

WORKDIR /app
ENV FINETUNED_MODEL_PATH=/app/data/models/finetuned-shoptalk-emb

# System deps (ffmpeg required by Whisper STT)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy source code
COPY src/ /app/src/
COPY backend/ /app/backend/

# Best-effort model pre-download (non-fatal; runtime will still load if unavailable)
RUN python - <<'PY'
import logging
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO)

try:
    SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Cached all-MiniLM-L6-v2")
except Exception as e:
    logging.warning(f"Could not cache SentenceTransformer at build time: {e}")

try:
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logging.info("Cached CLIP model/processor")
except Exception as e:
    logging.warning(f"Could not cache CLIP at build time: {e}")

print("Model cache step completed")
PY

EXPOSE 8000

CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
