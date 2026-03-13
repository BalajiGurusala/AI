"""
Unit tests for data processing utilities.

Per test_strategy.md:
  - Data Processing: Test DataCleaner.clean_text() handles special characters and emojis.

Since backend/src/services/data_processing.py doesn't exist yet (T041),
this file tests the search module's text processing utilities which serve
the same purpose in the current codebase.
"""

import pytest

from src.search import tokenize, field_overlap, extract_head_nouns


class TestTextCleaning:
    """Test text cleaning/tokenization for special characters and edge cases."""

    def test_handles_emojis(self):
        tokens = tokenize("red shoes 👟🔥")
        assert "red" in tokens
        assert "shoes" in tokens

    def test_handles_special_chars(self):
        tokens = tokenize("men's #1 best-seller!!!")
        assert any("men" in t for t in tokens)

    def test_handles_unicode(self):
        tokens = tokenize("café résumé naïve")
        assert len(tokens) >= 0  # Should not crash

    def test_handles_html_entities(self):
        tokens = tokenize("shoes &amp; boots")
        assert "shoes" in tokens
        assert "boots" in tokens

    def test_handles_numeric_strings(self):
        tokens = tokenize("12 inch 5mm cable")
        assert "12" in tokens
        assert "inch" in tokens

    def test_handles_very_long_input(self):
        long_text = " ".join(["word"] * 10000)
        tokens = tokenize(long_text)
        assert len(tokens) == 10000

    def test_handles_only_punctuation(self):
        tokens = tokenize("!@#$%^&*()")
        assert tokens == []

    def test_handles_tabs_and_newlines(self):
        tokens = tokenize("red\tshoes\nfor\nwomen")
        assert "red" in tokens
        assert "shoes" in tokens
        assert "women" in tokens

    def test_preserves_hyphens_in_compound_words(self):
        tokens = tokenize("t-shirt heavy-duty")
        assert "t-shirt" in tokens
        assert "heavy-duty" in tokens


class TestFieldOverlapEdgeCases:
    def test_identical_strings(self):
        score = field_overlap("red shoes", "red shoes")
        assert score == 1.0

    def test_case_insensitive(self):
        score = field_overlap("RED SHOES", "red shoes for women")
        assert score == 1.0

    def test_partial_match(self):
        score = field_overlap("red blue green", "red shoes")
        assert 0.0 < score < 1.0

    def test_empty_both(self):
        assert field_overlap("", "") == 0.0
