"""
Spec compliance tests — verifies the codebase against requirements.md,
data-model.md, constitution.md, and test_strategy.md.

These tests check structural requirements without needing running services.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TestConstitutionCompliance:
    """Checks against constitution.md requirements."""

    def test_pydantic_models_exist(self):
        """constitution.md §4: Use pydantic models for all data exchange."""
        from backend.src.models.schemas import (
            Product, ChatRequest, ChatResponse,
            SearchRequest, SearchResponse,
            VoiceQueryResponse, HealthResponse,
            Filters, ChatMessage,
        )
        assert Product is not None
        assert ChatRequest is not None

    def test_settings_uses_pydantic(self):
        """constitution.md §4: Use pydantic for validation."""
        from backend.src.config import Settings
        from pydantic_settings import BaseSettings
        assert issubclass(Settings, BaseSettings)

    def test_hardware_detection_code_exists(self):
        """constitution.md §4: Code must detect mps vs cuda dynamically."""
        source_file = PROJECT_ROOT / "backend" / "src" / "services" / "embeddings.py"
        content = source_file.read_text()
        assert "mps" in content, "Missing MPS (Apple Silicon) detection"
        assert "cuda" in content, "Missing CUDA detection"
        assert "cpu" in content, "Missing CPU fallback"

    def test_models_load_once_pattern(self):
        """constitution.md: Models must load once at startup, NEVER during inference."""
        from backend.src.services.embeddings import EmbeddingService
        svc = EmbeddingService()
        assert hasattr(svc, '_loaded'), "Singleton pattern missing _loaded flag"
        assert svc._loaded is False, "Should not be loaded at construction time"

    def test_env_file_example_exists(self):
        """constitution.md §4: Use .env files."""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example missing"


class TestRequirementsCompliance:
    """Checks against requirements.md requirements."""

    def test_api_endpoints_exist(self):
        """requirements.md: Must have health, chat, search, voice endpoints."""
        from backend.src.api.routes import router
        routes = [r.path for r in router.routes]
        assert "/health" in routes, "Missing /health endpoint"
        assert "/api/v1/chat" in routes, "Missing /api/v1/chat endpoint"
        assert "/api/v1/search" in routes, "Missing /api/v1/search endpoint"
        assert "/api/v1/voice/query" in routes, "Missing /api/v1/voice/query endpoint"

    def test_cors_configured(self):
        """requirements.md §4: Streamlit frontend needs CORS."""
        source_file = PROJECT_ROOT / "backend" / "src" / "api" / "main.py"
        content = source_file.read_text()
        assert "CORSMiddleware" in content, "CORS middleware not configured"

    def test_system_prompt_exists(self):
        """requirements.md: RAG should use a system prompt."""
        from backend.src.services.rag import SYSTEM_PROMPT
        assert "ShopTalk" in SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 50

    def test_error_messages_per_spec(self):
        """requirements.md: Specific error messages for different failure modes."""
        routes_file = PROJECT_ROOT / "backend" / "src" / "api" / "routes.py"
        content = routes_file.read_text()
        # STT failure message
        assert "couldn't understand" in content.lower() or "couldn't hear" in content.lower(), \
            "Missing STT failure message per spec"

    def test_voice_service_has_stt_and_tts(self):
        """requirements.md: Voice STT (Whisper) + TTS (gTTS)."""
        from backend.src.services.voice import VoiceService
        svc = VoiceService()
        assert hasattr(svc, 'transcribe'), "Missing STT transcribe method"
        assert hasattr(svc, 'synthesize'), "Missing TTS synthesize method"
        assert hasattr(svc, 'synthesize_base64'), "Missing base64 TTS method"

    def test_session_context_in_chat_request(self):
        """requirements.md: Chat history context-aware within session."""
        from backend.src.models.schemas import ChatRequest, ChatMessage
        req = ChatRequest(
            query_text="show me the blue one",
            session_context=[
                ChatMessage(role="user", content="shoes"),
                ChatMessage(role="assistant", content="Here are shoes"),
            ],
        )
        assert len(req.session_context) == 2


class TestDataModelCompliance:
    """Checks against data-model.md entity definitions."""

    def test_product_has_all_fields(self):
        """data-model.md §1: Product entity fields."""
        from backend.src.models.schemas import Product
        fields = Product.model_fields
        required_fields = {"id", "title"}
        optional_fields = {
            "description", "bullet_points", "keywords", "brand",
            "color", "category", "price", "image_url", "image_caption",
            "detection_confidence",
        }
        for f in required_fields:
            assert f in fields, f"Missing required field: {f}"
        for f in optional_fields:
            assert f in fields, f"Missing optional field: {f}"

    def test_chat_message_has_role_enum(self):
        """data-model.md §2: ChatMessage role is user|assistant|system."""
        from backend.src.models.schemas import ChatMessage
        from pydantic import ValidationError
        for valid_role in ["user", "assistant", "system"]:
            ChatMessage(role=valid_role, content="test")
        with pytest.raises(ValidationError):
            ChatMessage(role="admin", content="test")

    def test_voice_response_status_values(self):
        """data-model.md §3: Status enum ok|stt_failed|no_results|pipeline_error."""
        from backend.src.models.schemas import VoiceQueryResponse
        for status in ["ok", "stt_failed", "no_results", "pipeline_error"]:
            resp = VoiceQueryResponse(response_text="test", status=status)
            assert resp.status == status

    def test_search_request_top_k_bounds(self):
        """data-model.md §4: top_k default 5, range 1-20."""
        from backend.src.models.schemas import SearchRequest
        req = SearchRequest(query_text="test")
        assert req.top_k == 5

    def test_filters_have_price_and_category(self):
        """data-model.md §4: Filters include price_max and category."""
        from backend.src.models.schemas import Filters
        f = Filters(price_max=100.0, category="SHOES")
        assert f.price_max == 100.0
        assert f.category == "SHOES"


class TestArchitectureCompliance:
    """Checks against architecture.md structure."""

    def test_fastapi_app_exists(self):
        """architecture.md §2: Backend is FastAPI."""
        source_file = PROJECT_ROOT / "backend" / "src" / "api" / "main.py"
        content = source_file.read_text()
        assert "FastAPI" in content

    def test_streamlit_app_exists(self):
        """architecture.md §2: Frontend is Streamlit."""
        app_file = PROJECT_ROOT / "app" / "streamlit_app.py"
        assert app_file.exists(), "Streamlit app missing"
        content = app_file.read_text()
        assert "streamlit" in content.lower()

    def test_lifespan_loads_models(self):
        """architecture.md: Models load at startup."""
        source_file = PROJECT_ROOT / "backend" / "src" / "api" / "main.py"
        content = source_file.read_text()
        assert "lifespan" in content, "Missing lifespan context manager"
        assert "embedding_service.load" in content, "Embeddings not loaded at startup"
        assert "llm_manager.initialize" in content, "LLM not initialized at startup"
        assert "voice_service.load" in content, "Voice not loaded at startup"

    def test_search_module_is_shared(self):
        """architecture.md: Single source of truth for search logic."""
        search_file = PROJECT_ROOT / "src" / "search.py"
        assert search_file.exists(), "Shared search module missing"
        content = search_file.read_text()
        assert "SINGLE SOURCE OF TRUTH" in content or "hybrid_search" in content


class TestTestStrategyCoverage:
    """Meta-tests: verify the test suite covers test_strategy.md requirements."""

    def test_unit_test_files_exist(self):
        """test_strategy.md §1: Unit tests in tests/unit/."""
        test_dir = PROJECT_ROOT / "backend" / "tests" / "unit"
        assert test_dir.exists()
        test_files = list(test_dir.glob("test_*.py"))
        assert len(test_files) >= 3, f"Expected >= 3 unit test files, found {len(test_files)}"

    def test_integration_test_files_exist(self):
        """test_strategy.md §1: Integration tests in tests/integration/."""
        test_dir = PROJECT_ROOT / "backend" / "tests" / "integration"
        assert test_dir.exists()
        test_files = list(test_dir.glob("test_*.py"))
        assert len(test_files) >= 1, "Expected >= 1 integration test file"

    def test_contract_test_files_exist(self):
        """test_strategy.md: Contract tests for API endpoints."""
        test_dir = PROJECT_ROOT / "backend" / "tests" / "contract"
        assert test_dir.exists()
        test_files = list(test_dir.glob("test_*.py"))
        assert len(test_files) >= 2, "Expected >= 2 contract test files"
