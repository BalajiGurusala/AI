"""
ShopTalk Backend — Embedding & Search Model Loader.

Models load ONCE at startup (constitution.md: "Models must load once at startup,
NEVER during inference"). This module provides a singleton.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Loads and holds all search models + product data. Singleton."""

    def __init__(self):
        self.device: str = ""
        self.st_model = None
        self.clip_model = None
        self.clip_processor = None
        self.df: Optional[pd.DataFrame] = None
        self.text_index: Optional[np.ndarray] = None
        self.image_index: Optional[np.ndarray] = None
        self.config: dict = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, data_dir: Path, text_model_id: str, clip_model_id: str):
        """Load all models and data. Call once at startup."""
        if self._loaded:
            logger.info("Models already loaded — skipping")
            return

        # Device detection
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
            "cpu"
        )
        gpu = torch.cuda.get_device_name(0) if self.device == "cuda" else self.device
        logger.info(f"Device: {self.device} ({gpu})")

        # Load product data
        pkl_candidates = ["products_with_prices.pkl", "rag_products.pkl"]
        for pkl in pkl_candidates:
            p = data_dir / pkl
            if p.exists():
                self.df = pd.read_pickle(p)
                logger.info(f"Loaded {pkl}: {len(self.df):,} products")
                break
        if self.df is None:
            raise FileNotFoundError(f"No product data in {data_dir}")

        # Load embeddings
        # Prefer fine-tuned if available
        ft_path = data_dir / "finetuned_text_index.npy"
        base_path = data_dir / "rag_text_index.npy"
        if ft_path.exists():
            self.text_index = np.load(ft_path)
            logger.info(f"Loaded fine-tuned text index: {self.text_index.shape}")
        elif base_path.exists():
            self.text_index = np.load(base_path)
            logger.info(f"Loaded base text index: {self.text_index.shape}")
        else:
            raise FileNotFoundError(f"No text index in {data_dir}")

        self.image_index = np.load(data_dir / "rag_image_index.npy")
        logger.info(f"Image index: {self.image_index.shape}")

        # Load config
        cfg_path = data_dir / "rag_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.config = json.load(f)

        # Load SentenceTransformer
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer: {text_model_id}")
        t0 = time.time()
        self.st_model = SentenceTransformer(text_model_id, device=self.device)
        logger.info(f"  Loaded in {time.time()-t0:.1f}s")

        # Load CLIP
        from transformers import CLIPModel, CLIPProcessor
        logger.info(f"Loading CLIP: {clip_model_id}")
        t0 = time.time()
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        logger.info(f"  Loaded in {time.time()-t0:.1f}s")

        self._loaded = True
        logger.info(f"All models loaded ({len(self.df):,} products indexed)")

    def encode_text(self, query: str) -> np.ndarray:
        """Encode query with SentenceTransformer."""
        emb = self.st_model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        return emb.astype(np.float32)

    def encode_clip(self, query: str) -> np.ndarray:
        """Encode query with CLIP text encoder."""
        inputs = self.clip_processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        emb = features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.where(norms == 0, 1, norms)


# Singleton instance
embedding_service = EmbeddingService()
