"""
ShopTalk Backend — FastAPI Application.

Models load once at startup (per constitution.md).
CORS configured for Streamlit frontend.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.src.config import settings
from backend.src.services.embeddings import embedding_service
from backend.src.services.rag import llm_manager
from backend.src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("ShopTalk Backend — Starting")
    logger.info("=" * 60)

    # Load embedding models + product data
    embedding_service.load(
        data_dir=settings.data_dir,
        text_model_id=settings.text_model_id,
        clip_model_id=settings.clip_model_id,
    )

    # Connect to LLM
    llm_manager.initialize(
        ollama_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        openai_key=settings.openai_api_key,
        groq_key=settings.groq_api_key,
    )

    logger.info("=" * 60)
    logger.info(f"ShopTalk Backend — Ready ({embedding_service.df.shape[0]:,} products)")
    logger.info(f"  LLM: {llm_manager.name}")
    logger.info(f"  Device: {embedding_service.device}")
    logger.info("=" * 60)

    yield  # App is running

    logger.info("ShopTalk Backend — Shutting down")


# Create app
app = FastAPI(
    title="ShopTalk Backend API",
    description="Voice-driven product search and RAG responses.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve product images as static files (if available)
image_path = settings.image_base_path or settings.data_dir / "images" / "small"
if image_path.exists():
    app.mount("/images", StaticFiles(directory=str(image_path)), name="images")
    logger.info(f"Serving images from {image_path}")

# Include routes
app.include_router(router)
