"""
Integration tests for the RAG pipeline.

Per test_strategy.md:
  - RAG Pipeline: Verify Query ("red shirt") → Database returns at least 1
    result with "shirt" in metadata.

These tests use synthetic data (conftest fixtures) to validate the full
search → context → LLM generation pipeline without requiring real models.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.search import hybrid_search


class TestRAGSearchPipeline:
    """Test the search portion of the RAG pipeline with synthetic data."""

    def test_query_returns_results(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="red shirt",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
        )
        assert not results.empty, "Search should return at least 1 result"
        assert len(results) <= 5

    def test_results_have_product_metadata(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
        )
        required_cols = ["item_id", "item_name_flat", "product_type_flat", "hybrid_score"]
        for col in required_cols:
            assert col in results.columns, f"Missing required column: {col}"

    def test_category_filter_returns_only_matching(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="product",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=20,
            category="SHOES",
        )
        if not results.empty:
            for _, row in results.iterrows():
                assert "SHOES" in str(row["product_type_flat"]).upper(), \
                    f"Category filter broken: got {row['product_type_flat']}"

    def test_price_filter_caps_results(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="product",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=20,
            price_max=50.0,
        )
        if not results.empty:
            assert all(results["price"] <= 50.0), \
                f"Price filter broken: max was {results['price'].max()}"

    def test_zero_results_on_impossible_filter(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        """When no products match, search should return empty DataFrame."""
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
            price_max=0.001,
        )
        assert results.empty, "Should return empty for impossible price filter"


class TestRAGContextFormatting:
    """Test context formatting for LLM input."""

    def test_format_context_with_products(self):
        from backend.src.services.rag import _format_context
        df = pd.DataFrame([{
            "item_id": "P001",
            "item_name_flat": "Red Shoes",
            "brand_flat": "Nike",
            "product_type_flat": "SHOES",
            "color_flat": "Red",
            "price": 89.99,
            "bullet_point_flat": "Lightweight running shoes",
            "image_caption": "red athletic shoes",
        }])
        context = _format_context(df)
        assert "Red Shoes" in context
        assert "Nike" in context
        assert "$89.99" in context

    def test_format_context_empty(self):
        from backend.src.services.rag import _format_context
        context = _format_context(pd.DataFrame())
        assert context == "No products found."


class TestRowToProduct:
    """Test DataFrame row to Product schema conversion."""

    def test_converts_row(self):
        from backend.src.services.rag import _row_to_product
        row = pd.Series({
            "item_id": "P001",
            "item_name_flat": "Red Shoes",
            "bullet_point_flat": "Lightweight running shoes",
            "price": 89.99,
            "product_type_flat": "SHOES",
            "brand_flat": "Nike",
            "color_flat": "Red",
            "path": "images/small/p001.jpg",
            "image_caption": "red shoes",
        })
        product = _row_to_product(row)
        assert product.id == "P001"
        assert product.title == "Red Shoes"
        assert product.price == 89.99
        assert product.image_url is not None

    def test_handles_missing_fields(self):
        from backend.src.services.rag import _row_to_product
        row = pd.Series({
            "item_id": "P002",
            "item_name_flat": "Unknown Product",
        })
        product = _row_to_product(row)
        assert product.id == "P002"
        assert product.price is None
        assert product.image_url is None


class TestLLMManager:
    """Test LLM manager initialization and fallback logic."""

    def test_initial_state(self):
        from backend.src.services.rag import LLMManager
        mgr = LLMManager()
        assert not mgr.is_available
        assert mgr.name == "none"

    def test_generate_without_llm_raises(self):
        from backend.src.services.rag import LLMManager
        mgr = LLMManager()
        with pytest.raises(RuntimeError, match="No LLM available"):
            mgr.generate([{"role": "user", "content": "hello"}])

    def test_generate_with_mocked_llm(self):
        from backend.src.services.rag import LLMManager
        mgr = LLMManager()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Here are some shoes.")
        mgr._llm = mock_llm
        mgr._name = "mock"

        result = mgr.generate([{"role": "user", "content": "shoes"}])
        assert result == "Here are some shoes."
