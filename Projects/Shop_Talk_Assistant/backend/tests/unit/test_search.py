"""
Unit tests for src/search.py — Hybrid Search Pipeline.

Covers: tokenization, field overlap, head-noun extraction, gender/color intent,
dynamic alpha, L2 normalisation, in-memory retrieval, reranking, metadata
filters, and the full hybrid_search entry point.

Per test_strategy.md:
  - Vector Logic: Test VectorStore.search() returns correct shape and handles empty results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.search import (
    tokenize,
    field_overlap,
    extract_head_nouns,
    infer_expected_types,
    extract_gender_intent,
    extract_color_intent,
    compute_dynamic_alpha,
    l2_normalize,
    retrieve_inmemory,
    apply_rerank,
    hybrid_search,
    ALPHA_DEFAULT,
)


# ══════════════════════════════════════════════════════════════════════
# Tokenizer & helpers
# ══════════════════════════════════════════════════════════════════════

class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("Red Shoes for Women")
        assert "red" in tokens
        assert "shoes" in tokens
        assert "women" in tokens
        assert "for" not in tokens  # stopword

    def test_removes_special_characters(self):
        tokens = tokenize("t-shirt $50!")
        assert "t-shirt" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_stopwords(self):
        assert tokenize("for the a an") == []


class TestFieldOverlap:
    def test_full_overlap(self):
        assert field_overlap("red shoes", "red running shoes") == 1.0

    def test_partial_overlap(self):
        score = field_overlap("red shoes", "blue running shoes")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert field_overlap("laptop computer", "red running shoes") == 0.0

    def test_empty_query(self):
        assert field_overlap("", "some text") == 0.0


class TestExtractHeadNouns:
    def test_known_category_noun(self):
        nouns = extract_head_nouns("running shoes for women")
        assert "shoes" in nouns

    def test_compound_token(self):
        nouns = extract_head_nouns("red t-shirt")
        assert "shirt" in nouns or "t-shirt" in nouns

    def test_no_match(self):
        nouns = extract_head_nouns("comfortable beautiful")
        assert len(nouns) == 0


class TestInferExpectedTypes:
    def test_shoes_maps_to_type(self):
        types = infer_expected_types("running shoes")
        assert "SHOES" in types

    def test_shirt_maps(self):
        types = infer_expected_types("polo shirt")
        assert "SHIRT" in types or "T_SHIRT" in types

    def test_no_mapping(self):
        types = infer_expected_types("nice comfortable product")
        assert len(types) == 0


class TestGenderIntent:
    def test_female_intent(self):
        assert extract_gender_intent("shoes for women") == "female"

    def test_male_intent(self):
        assert extract_gender_intent("men's dress shoes") == "male"

    def test_no_gender(self):
        assert extract_gender_intent("red running shoes") is None

    def test_both_genders_returns_none(self):
        assert extract_gender_intent("shoes for men and women") is None


class TestColorIntent:
    def test_detects_primary_color(self):
        assert extract_color_intent("red shoes") == "red"

    def test_detects_synonym(self):
        assert extract_color_intent("navy polo shirt") == "blue"

    def test_no_color(self):
        assert extract_color_intent("running shoes") is None

    def test_detects_brown_synonym(self):
        assert extract_color_intent("tan leather bag") == "brown"


class TestDynamicAlpha:
    def test_default_range(self):
        alpha = compute_dynamic_alpha("red shoes")
        assert 0.2 <= alpha <= 0.9

    def test_visual_cue_lowers_alpha(self):
        baseline = compute_dynamic_alpha("shoes")
        visual = compute_dynamic_alpha("colorful patterned shoes")
        assert visual < baseline

    def test_technical_cue_raises_alpha(self):
        baseline = compute_dynamic_alpha("cable")
        technical = compute_dynamic_alpha("12 inch cable mount")
        assert technical > baseline


class TestL2Normalize:
    def test_unit_vectors(self):
        x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normed = l2_normalize(x)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_safe(self):
        x = np.array([[0.0, 0.0]], dtype=np.float32)
        normed = l2_normalize(x)
        assert np.all(np.isfinite(normed))


# ══════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════

class TestRetrieveInmemory:
    def test_returns_correct_shape(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        q_text = encode_text_fn("red shoes")
        q_clip = encode_clip_fn("red shoes")
        results = retrieve_inmemory(q_text, q_clip, text_index, image_index, sample_df, alpha=0.6, n_fetch=5)
        assert len(results) == 5
        assert "hybrid_score" in results.columns
        assert "text_sim" in results.columns
        assert "image_sim" in results.columns

    def test_scores_are_sorted_descending(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        q_text = encode_text_fn("shoes")
        q_clip = encode_clip_fn("shoes")
        results = retrieve_inmemory(q_text, q_clip, text_index, image_index, sample_df, alpha=0.6, n_fetch=10)
        scores = results["hybrid_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_respects_n_fetch(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        q_text = encode_text_fn("shoes")
        q_clip = encode_clip_fn("shoes")
        results = retrieve_inmemory(q_text, q_clip, text_index, image_index, sample_df, alpha=0.6, n_fetch=3)
        assert len(results) == 3


# ══════════════════════════════════════════════════════════════════════
# Reranking
# ══════════════════════════════════════════════════════════════════════

class TestApplyRerank:
    def _make_results(self, sample_df):
        results = sample_df.copy()
        rng = np.random.default_rng(42)
        results["hybrid_score"] = rng.uniform(0.3, 0.9, len(results))
        return results

    def test_returns_top_k(self, sample_df):
        results = self._make_results(sample_df)
        reranked = apply_rerank(results, query="red shoes", top_k=5)
        assert len(reranked) == 5

    def test_has_rank_column(self, sample_df):
        results = self._make_results(sample_df)
        reranked = apply_rerank(results, query="red shoes", top_k=5)
        assert "_rank" in reranked.columns
        assert list(reranked["_rank"]) == [1, 2, 3, 4, 5]

    def test_empty_df(self):
        empty = pd.DataFrame()
        result = apply_rerank(empty, query="shoes", top_k=5)
        assert result.empty

    def test_gender_penalty_applied(self, sample_df):
        results = self._make_results(sample_df)
        results["hybrid_score"] = 0.5
        reranked = apply_rerank(results, query="shoes for women", top_k=len(results))
        mens_rows = reranked[reranked["item_name_flat"].str.contains("Men's", case=False)]
        womens_rows = reranked[reranked["item_name_flat"].str.contains("Women", case=False)]
        if not mens_rows.empty and not womens_rows.empty:
            assert mens_rows["hybrid_score"].mean() <= womens_rows["hybrid_score"].mean()


# ══════════════════════════════════════════════════════════════════════
# Full hybrid_search
# ══════════════════════════════════════════════════════════════════════

class TestHybridSearch:
    def test_returns_results(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="red shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
        )
        assert not results.empty
        assert len(results) <= 5

    def test_empty_results_with_impossible_filter(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
            price_max=0.01,
        )
        assert results.empty

    def test_category_filter(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
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
            assert all("SHOES" in str(r).upper() for r in results["product_type_flat"])

    def test_price_filter(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="product",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=20,
            price_max=30.0,
        )
        if not results.empty:
            assert all(p <= 30.0 for p in results["price"])

    def test_has_required_columns(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
        )
        assert "hybrid_score" in results.columns
        assert "_rank" in results.columns

    def test_alpha_used_attribute(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
        )
        assert "alpha_used" in results.attrs

    def test_no_rerank_mode(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
            rerank=False,
        )
        assert not results.empty
        assert "_rank" in results.columns

    def test_explicit_alpha(self, sample_df, text_index, image_index, encode_text_fn, encode_clip_fn):
        results = hybrid_search(
            query="shoes",
            df=sample_df,
            text_index=text_index,
            image_index=image_index,
            encode_text_fn=encode_text_fn,
            encode_clip_fn=encode_clip_fn,
            top_k=5,
            alpha=1.0,
        )
        assert results.attrs["alpha_used"] == 1.0
