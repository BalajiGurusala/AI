"""
Contract tests for POST /api/v1/voice/query.

Validates request/response schemas match api-openapi.yaml:
  VoiceQueryResponse:
    - response_text: string (required)
    - transcript: string (optional)
    - product_ids: array[string] (optional)
    - status: enum [ok, stt_failed, no_results, pipeline_error]
    - audio_base64: string (optional)
"""

import pytest
from pydantic import ValidationError

from backend.src.models.schemas import (
    VoiceQueryResponse,
    Product,
)


class TestVoiceQueryResponseContract:
    """Verify VoiceQueryResponse matches the OpenAPI VoiceQueryResponse schema."""

    def test_response_text_required(self):
        with pytest.raises(ValidationError):
            VoiceQueryResponse()

    def test_minimal_valid_response(self):
        """Contract minimum: response_text is required."""
        resp = VoiceQueryResponse(response_text="No results found.")
        data = resp.model_dump()
        assert "response_text" in data
        assert isinstance(data["response_text"], str)

    def test_stt_failed_status(self):
        """When STT fails, status should be stt_failed per spec."""
        resp = VoiceQueryResponse(
            transcript="",
            response_text="Couldn't hear that—try again",
            status="stt_failed",
        )
        assert resp.status == "stt_failed"
        assert resp.audio_base64 is None

    def test_successful_response_shape(self):
        resp = VoiceQueryResponse(
            transcript="red shoes",
            response_text="Here are some red shoes for you.",
            product_ids=["P001", "P002"],
            products=[
                Product(id="P001", title="Red Running Shoes", price=89.99),
                Product(id="P002", title="Red Casual Shoes", price=59.99),
            ],
            status="ok",
            audio_base64="dGVzdA==",
        )
        data = resp.model_dump()
        assert isinstance(data["transcript"], str)
        assert isinstance(data["product_ids"], list)
        assert all(isinstance(pid, str) for pid in data["product_ids"])
        assert isinstance(data["audio_base64"], str)

    def test_no_results_status(self):
        resp = VoiceQueryResponse(
            transcript="purple helicopter",
            response_text="No products match that. Try different keywords or filters.",
            status="no_results",
        )
        assert resp.status == "no_results"

    def test_pipeline_error_status(self):
        resp = VoiceQueryResponse(
            transcript="shoes",
            response_text="Something went wrong. Please try again.",
            status="pipeline_error",
        )
        assert resp.status == "pipeline_error"

    def test_products_list_contains_valid_products(self):
        resp = VoiceQueryResponse(
            response_text="Found products.",
            products=[Product(id="P001", title="Shoes", price=50.0)],
        )
        assert len(resp.products) == 1
        assert resp.products[0].id == "P001"
