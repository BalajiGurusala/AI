"""
ShopTalk Backend — Configuration (pydantic BaseSettings).

Loads from environment variables / .env file per constitution.md.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # --- Data paths ---
    data_dir: Path = Path("/app/data")
    chroma_path: Optional[Path] = None
    image_base_path: Optional[Path] = None

    # --- Model IDs ---
    text_model_id: str = "all-MiniLM-L6-v2"
    clip_model_id: str = "openai/clip-vit-base-patch32"

    # --- LLM ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    default_llm: str = "ollama"  # "ollama" | "openai" | "groq"

    # --- Search ---
    default_top_k: int = 5
    alpha_default: float = 0.6

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:8501,http://frontend:8501"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
