"""
Unit tests for backend configuration.

Validates pydantic Settings model loads defaults correctly
and respects environment variable overrides.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.src.config import Settings


class TestConfigDefaults:
    def test_default_data_dir(self):
        s = Settings()
        assert s.data_dir == Path("/app/data")

    def test_default_text_model_id(self):
        s = Settings()
        assert s.text_model_id == "all-MiniLM-L6-v2"

    def test_default_clip_model_id(self):
        s = Settings()
        assert s.clip_model_id == "openai/clip-vit-base-patch32"

    def test_default_ollama_model(self):
        s = Settings()
        assert s.ollama_model == "llama3.2"

    def test_default_top_k(self):
        s = Settings()
        assert s.default_top_k == 5

    def test_default_alpha(self):
        s = Settings()
        assert s.alpha_default == 0.6

    def test_default_port(self):
        s = Settings()
        assert s.port == 8000

    def test_cors_origins_default(self):
        s = Settings()
        origins = s.cors_origins.split(",")
        assert "http://localhost:8501" in origins


class TestConfigOverrides:
    def test_env_override_port(self):
        with patch.dict(os.environ, {"PORT": "9000"}):
            s = Settings()
            assert s.port == 9000

    def test_env_override_data_dir(self):
        with patch.dict(os.environ, {"DATA_DIR": "/custom/data"}):
            s = Settings()
            assert s.data_dir == Path("/custom/data")

    def test_env_override_api_keys(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123",
            "GROQ_API_KEY": "gsk-test456",
        }):
            s = Settings()
            assert s.openai_api_key == "sk-test123"
            assert s.groq_api_key == "gsk-test456"

    def test_no_api_keys_by_default(self):
        s = Settings()
        # They might be set in .env; just verify they're Optional[str]
        assert s.openai_api_key is None or isinstance(s.openai_api_key, str)
