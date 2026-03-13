"""
Contract tests for GET /health endpoint.

Validates response shape matches api-openapi.yaml contract:
  - status: string
  - embedding_loaded: boolean
  - chroma_connected: boolean
"""

import pytest
from backend.src.models.schemas import HealthResponse


class TestHealthContract:
    """Verify HealthResponse model conforms to OpenAPI contract."""

    def test_response_has_required_fields(self):
        """Contract: /health must return status, embedding_loaded, chroma_connected."""
        resp = HealthResponse(
            status="ok",
            embedding_loaded=True,
            chroma_connected=False,
        )
        data = resp.model_dump()
        assert "status" in data
        assert "embedding_loaded" in data
        assert "chroma_connected" in data

    def test_status_is_string(self):
        resp = HealthResponse(status="degraded")
        assert isinstance(resp.status, str)

    def test_embedding_loaded_is_bool(self):
        resp = HealthResponse(embedding_loaded=True)
        assert isinstance(resp.embedding_loaded, bool)

    def test_chroma_connected_is_bool(self):
        resp = HealthResponse(chroma_connected=False)
        assert isinstance(resp.chroma_connected, bool)

    def test_extended_fields_are_optional(self):
        """Extended fields (llm_available, etc.) should not break minimal contract."""
        resp = HealthResponse(
            status="ok",
            embedding_loaded=True,
            chroma_connected=True,
        )
        data = resp.model_dump()
        assert data["llm_available"] is False
        assert data["stt_available"] is False
        assert data["tts_available"] is False
        assert data["product_count"] == 0

    def test_serialization_roundtrip(self):
        resp = HealthResponse(
            status="ok", embedding_loaded=True, chroma_connected=False,
            llm_available=True, stt_available=True, tts_available=True,
            product_count=5000,
        )
        data = resp.model_dump()
        restored = HealthResponse(**data)
        assert restored == resp
