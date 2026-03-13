"""
Contract tests for POST /api/v1/search and POST /api/v1/chat.

Validates request/response schemas match api-openapi.yaml.
"""

import pytest
from pydantic import ValidationError

from backend.src.models.schemas import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    Product,
    Filters,
    ChatMessage,
)


class TestSearchRequestContract:
    """POST /api/v1/search request body."""

    def test_query_text_required(self):
        with pytest.raises(ValidationError):
            SearchRequest()

    def test_minimal_request(self):
        req = SearchRequest(query_text="red shirt")
        data = req.model_dump()
        assert data["query_text"] == "red shirt"
        assert data["top_k"] == 5

    def test_top_k_default(self):
        req = SearchRequest(query_text="shoes")
        assert req.top_k == 5

    def test_top_k_range_1_to_20(self):
        SearchRequest(query_text="x", top_k=1)
        SearchRequest(query_text="x", top_k=20)

        with pytest.raises(ValidationError):
            SearchRequest(query_text="x", top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query_text="x", top_k=21)

    def test_filters_shape(self):
        req = SearchRequest(
            query_text="shoes",
            filters=Filters(price_max=100.0, category="SHOES"),
        )
        assert req.filters.price_max == 100.0
        assert req.filters.category == "SHOES"


class TestSearchResponseContract:
    """POST /api/v1/search response body."""

    def test_empty_response(self):
        resp = SearchResponse(product_ids=[], products=[], total=0)
        data = resp.model_dump()
        assert data["total"] == 0
        assert data["product_ids"] == []

    def test_response_with_products(self):
        resp = SearchResponse(
            product_ids=["P001", "P002"],
            products=[
                Product(id="P001", title="Red Shoes"),
                Product(id="P002", title="Blue Shoes"),
            ],
            total=2,
        )
        data = resp.model_dump()
        assert data["total"] == 2
        assert len(data["products"]) == 2
        assert all("id" in p for p in data["products"])
        assert all("title" in p for p in data["products"])

    def test_product_schema_fields(self):
        """Verify Product schema has all fields from OpenAPI spec."""
        p = Product(
            id="P001", title="Test Product",
            description="A test", bullet_points="Good quality",
            keywords="test", brand="TestBrand", color="Red",
            price=9.99, category="TEST",
            image_url="/images/test.jpg",
            image_caption="a red product",
            detection_confidence=0.88,
        )
        data = p.model_dump()
        expected_fields = {
            "id", "title", "description", "bullet_points", "keywords",
            "brand", "color", "price", "category", "image_url",
            "image_caption", "detection_confidence",
        }
        assert expected_fields.issubset(set(data.keys()))


class TestChatRequestContract:
    """POST /api/v1/chat request body."""

    def test_query_text_required(self):
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_minimal(self):
        req = ChatRequest(query_text="red shoes")
        data = req.model_dump()
        assert data["query_text"] == "red shoes"

    def test_with_session_context(self):
        req = ChatRequest(
            query_text="show me the blue one",
            session_context=[
                ChatMessage(role="user", content="shoes"),
                ChatMessage(role="assistant", content="Here are shoes"),
            ],
        )
        data = req.model_dump()
        assert len(data["session_context"]) == 2
        assert data["session_context"][0]["role"] == "user"

    def test_filters_optional(self):
        req = ChatRequest(query_text="shoes")
        assert req.filters is None


class TestChatResponseContract:
    """POST /api/v1/chat response body."""

    def test_response_text_required(self):
        with pytest.raises(ValidationError):
            ChatResponse()

    def test_minimal(self):
        resp = ChatResponse(response_text="Here are the products.")
        assert resp.status == "ok"
        assert resp.product_ids == []

    def test_status_values(self):
        for status in ["ok", "no_results", "pipeline_error"]:
            resp = ChatResponse(response_text="test", status=status)
            assert resp.status == status

    def test_full_response(self):
        resp = ChatResponse(
            response_text="I found great shoes for you!",
            product_ids=["P001", "P002", "P003"],
            products=[
                Product(id="P001", title="Shoe 1", price=59.99),
                Product(id="P002", title="Shoe 2", price=79.99),
                Product(id="P003", title="Shoe 3", price=99.99),
            ],
            status="ok",
        )
        data = resp.model_dump()
        assert len(data["product_ids"]) == 3
        assert len(data["products"]) == 3
