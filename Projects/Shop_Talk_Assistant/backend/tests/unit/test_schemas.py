"""
Unit tests for backend/src/models/schemas.py — Pydantic Schemas.

Validates all request/response models match the data-model.md
and api-openapi.yaml contract definitions.
"""

import pytest
from pydantic import ValidationError

from backend.src.models.schemas import (
    Filters,
    ChatMessage,
    Product,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    VoiceQueryResponse,
    HealthResponse,
)


class TestFilters:
    def test_empty_filters(self):
        f = Filters()
        assert f.price_max is None
        assert f.category is None

    def test_with_values(self):
        f = Filters(price_max=50.0, category="SHOES")
        assert f.price_max == 50.0
        assert f.category == "SHOES"


class TestChatMessage:
    def test_valid_roles(self):
        for role in ["user", "assistant", "system"]:
            msg = ChatMessage(role=role, content="Hello")
            assert msg.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello")

    def test_content_required(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user")


class TestProduct:
    def test_minimal_product(self):
        p = Product(id="P001", title="Red Shoes")
        assert p.id == "P001"
        assert p.title == "Red Shoes"
        assert p.price is None
        assert p.image_url is None

    def test_full_product(self):
        p = Product(
            id="P001", title="Red Shoes", description="Running shoes",
            bullet_points="Lightweight, Breathable", keywords="shoes, running",
            brand="Nike", color="Red", price=89.99, category="SHOES",
            image_url="/images/p001.jpg", image_caption="red running shoes",
            detection_confidence=0.95,
        )
        assert p.price == 89.99
        assert p.detection_confidence == 0.95

    def test_id_required(self):
        with pytest.raises(ValidationError):
            Product(title="Shoes")

    def test_title_required(self):
        with pytest.raises(ValidationError):
            Product(id="P001")


class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(query_text="red shoes")
        assert req.query_text == "red shoes"
        assert req.filters is None
        assert req.session_context is None

    def test_with_filters(self):
        req = ChatRequest(
            query_text="shoes",
            filters=Filters(price_max=100.0, category="SHOES"),
        )
        assert req.filters.price_max == 100.0

    def test_with_session_context(self):
        req = ChatRequest(
            query_text="show me the blue one",
            session_context=[
                ChatMessage(role="user", content="shoes"),
                ChatMessage(role="assistant", content="Here are some shoes"),
            ],
        )
        assert len(req.session_context) == 2

    def test_query_text_required(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    def test_minimal_response(self):
        resp = ChatResponse(response_text="Here are the products.")
        assert resp.response_text == "Here are the products."
        assert resp.product_ids == []
        assert resp.products == []
        assert resp.status == "ok"

    def test_with_products(self):
        resp = ChatResponse(
            response_text="Found these:",
            product_ids=["P001", "P002"],
            products=[
                Product(id="P001", title="Red Shoes"),
                Product(id="P002", title="Blue Shoes"),
            ],
            status="ok",
        )
        assert len(resp.products) == 2

    def test_valid_status_values(self):
        for status in ["ok", "no_results", "pipeline_error"]:
            resp = ChatResponse(response_text="test", status=status)
            assert resp.status == status


class TestSearchRequest:
    def test_defaults(self):
        req = SearchRequest(query_text="shoes")
        assert req.top_k == 5
        assert req.filters is None

    def test_top_k_bounds(self):
        req = SearchRequest(query_text="shoes", top_k=20)
        assert req.top_k == 20

    def test_top_k_too_high(self):
        with pytest.raises(ValidationError):
            SearchRequest(query_text="shoes", top_k=21)

    def test_top_k_too_low(self):
        with pytest.raises(ValidationError):
            SearchRequest(query_text="shoes", top_k=0)


class TestSearchResponse:
    def test_empty_response(self):
        resp = SearchResponse()
        assert resp.product_ids == []
        assert resp.total == 0

    def test_with_results(self):
        resp = SearchResponse(
            product_ids=["P001"],
            products=[Product(id="P001", title="Shoes")],
            total=1,
        )
        assert resp.total == 1


class TestVoiceQueryResponse:
    def test_stt_failed(self):
        resp = VoiceQueryResponse(
            transcript="",
            response_text="I couldn't understand the audio. Please try again.",
            status="stt_failed",
        )
        assert resp.status == "stt_failed"
        assert resp.audio_base64 is None

    def test_success_with_audio(self):
        resp = VoiceQueryResponse(
            transcript="red shoes",
            response_text="Here are red shoes.",
            product_ids=["P001"],
            status="ok",
            audio_base64="base64encodedaudio==",
        )
        assert resp.transcript == "red shoes"
        assert resp.audio_base64 is not None

    def test_response_text_required(self):
        with pytest.raises(ValidationError):
            VoiceQueryResponse()


class TestHealthResponse:
    def test_defaults(self):
        health = HealthResponse()
        assert health.status == "ok"
        assert health.embedding_loaded is False
        assert health.chroma_connected is False

    def test_full_health(self):
        health = HealthResponse(
            status="ok",
            embedding_loaded=True,
            chroma_connected=False,
            llm_available=True,
            stt_available=True,
            tts_available=True,
            product_count=1000,
        )
        assert health.product_count == 1000
        assert health.llm_available is True
