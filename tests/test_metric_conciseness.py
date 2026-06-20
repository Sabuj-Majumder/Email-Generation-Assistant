"""
Tests for Metric 3 — Structural Conciseness / Signal Density.

Covers:
  - Length ratio penalty curve at boundary values: 0.0, 0.5, 1.0, 1.5, and outside the band.
  - Lexical density calculation (unique content words / total words).
  - Final score formula: 0.5 * lexical_density + 0.5 * length_penalty_score.
  - Pure Python — zero API calls, all tests fully synchronous and deterministic.
"""

from __future__ import annotations

import pytest

from email_eval.evaluation.metric_conciseness import (
    BAND_HIGH,
    BAND_LOW,
    LENGTH_PENALTY_WEIGHT,
    LEXICAL_DENSITY_WEIGHT,
    STOPWORDS,
    _length_penalty_score,
    _lexical_density,
    _tokenize,
    compute_conciseness,
)
from email_eval.schemas import GeneratedEmail, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_scenario(reference: str) -> Scenario:
    return Scenario(
        id="test-conciseness",
        intent="Test scenario",
        key_facts=["Fact one"],
        tone="neutral",
        human_reference_email=reference,
    )


def _make_generated(email: str, scenario_id: str = "test-conciseness") -> GeneratedEmail:
    return GeneratedEmail(
        scenario_id=scenario_id,
        model_id="model_a",
        raw_model_output=email,
        parsed_email=email,
        cot_scratchpad=None,
        parse_fallback_used=False,
    )


# ── _tokenize ─────────────────────────────────────────────────────────────────

def test_tokenize_basic() -> None:
    tokens = _tokenize("Hello, World! This is a test.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_tokenize_empty() -> None:
    assert _tokenize("") == []


def test_tokenize_strips_punctuation() -> None:
    tokens = _tokenize("re: meeting—follow-up")
    assert "re" in tokens
    assert "meeting" in tokens
    assert "follow" in tokens
    assert "up" in tokens


# ── _lexical_density ──────────────────────────────────────────────────────────

def test_lexical_density_no_repetition() -> None:
    # All unique content words (excluding stopwords)
    words = ["innovation", "drives", "productivity", "quality", "results"]
    density = _lexical_density(words)
    # All are content words, all unique → density = 5/5 = 1.0
    assert abs(density - 1.0) < 1e-6


def test_lexical_density_all_stopwords() -> None:
    # All stopwords → no content words → density = 0.0
    words = list(STOPWORDS)[:5]
    density = _lexical_density(words)
    assert density == 0.0


def test_lexical_density_with_repetition() -> None:
    # "product product product" → 1 unique content word out of 3 total
    words = ["product", "product", "product"]
    density = _lexical_density(words)
    assert abs(density - 1 / 3) < 1e-6


def test_lexical_density_empty() -> None:
    assert _lexical_density([]) == 0.0


# ── _length_penalty_score (boundary value tests) ──────────────────────────────

def test_length_penalty_at_ratio_zero() -> None:
    """Ratio 0.0 (empty email) → score 0.0."""
    assert _length_penalty_score(0.0) == 0.0


def test_length_penalty_at_band_low_boundary() -> None:
    """Ratio exactly at BAND_LOW → score 1.0 (inside band)."""
    assert _length_penalty_score(BAND_LOW) == 1.0


def test_length_penalty_at_ratio_one() -> None:
    """Ratio 1.0 (same length as reference) → score 1.0 (inside band)."""
    assert _length_penalty_score(1.0) == 1.0


def test_length_penalty_at_band_high_boundary() -> None:
    """Ratio exactly at BAND_HIGH → score 1.0 (inside band)."""
    assert _length_penalty_score(BAND_HIGH) == 1.0


def test_length_penalty_below_band() -> None:
    """Ratio 0.25 (half of BAND_LOW=0.5) → linear decay, score = 0.25 / 0.5 = 0.5."""
    score = _length_penalty_score(0.25)
    assert abs(score - 0.5) < 1e-6


def test_length_penalty_above_band() -> None:
    """Ratio 2.0 (BAND_HIGH=1.5 + 0.5) → score = 1 - (2.0-1.5)/1.5 = 1 - 0.333 = 0.667."""
    score = _length_penalty_score(2.0)
    expected = 1.0 - (2.0 - BAND_HIGH) / BAND_HIGH
    assert abs(score - expected) < 1e-6


def test_length_penalty_far_above_band() -> None:
    """Ratio >= 2*BAND_HIGH = 3.0 → score clamped to 0.0."""
    score = _length_penalty_score(3.0)
    assert score == 0.0


def test_length_penalty_clamped_at_zero() -> None:
    """Extreme ratio → score never goes below 0.0."""
    score = _length_penalty_score(100.0)
    assert score == 0.0


# ── compute_conciseness (integration) ────────────────────────────────────────

def test_compute_conciseness_same_length_as_reference() -> None:
    """
    When generated email is identical to the reference, length_ratio=1.0 (perfect).
    Conciseness score depends on lexical density of that text.
    """
    text = "The quarterly results demonstrate strong operational efficiency and growth."
    scenario = _make_scenario(text)
    generated = _make_generated(text)

    result = compute_conciseness(scenario, generated)

    assert result.metric_name == "conciseness"
    assert result.raw_detail["length_ratio"] == 1.0
    assert result.raw_detail["length_penalty_score"] == 1.0
    assert 0.0 <= result.score <= 1.0


def test_compute_conciseness_too_short_penalized() -> None:
    """
    Generated email much shorter than reference → length_ratio < BAND_LOW → penalty.
    """
    reference = " ".join(["word"] * 100)  # 100 words
    generated_text = " ".join(["innovation", "drives", "results"])  # 3 words → ratio 0.03
    scenario = _make_scenario(reference)
    generated = _make_generated(generated_text)

    result = compute_conciseness(scenario, generated)

    assert result.raw_detail["length_ratio"] < BAND_LOW
    assert result.raw_detail["length_penalty_score"] < 1.0
    # Score must be less than it would be at the ideal ratio
    assert result.score < 1.0


def test_compute_conciseness_too_long_penalized() -> None:
    """
    Generated email much longer than reference → length_ratio > BAND_HIGH → penalty.
    """
    reference = " ".join(["word"] * 10)  # 10 words
    generated_text = " ".join(["innovation", "drives", "quality", "results", "performance"] * 20)  # 100 words
    scenario = _make_scenario(reference)
    generated = _make_generated(generated_text)

    result = compute_conciseness(scenario, generated)

    assert result.raw_detail["length_ratio"] > BAND_HIGH
    assert result.raw_detail["length_penalty_score"] < 1.0


def test_compute_conciseness_formula_applied_correctly() -> None:
    """
    Verify the formula: score = LEXICAL_DENSITY_WEIGHT * lex_density + LENGTH_PENALTY_WEIGHT * length_penalty
    """
    reference = "We deliver excellent results through innovative strategies and focused execution."
    generated_text = "We deliver excellent results through innovative strategies and focused execution."
    scenario = _make_scenario(reference)
    generated = _make_generated(generated_text)

    result = compute_conciseness(scenario, generated)
    detail = result.raw_detail

    expected = (
        LEXICAL_DENSITY_WEIGHT * detail["lexical_density"]
        + LENGTH_PENALTY_WEIGHT * detail["length_penalty_score"]
    )
    assert abs(result.score - expected) < 1e-6


def test_compute_conciseness_raw_detail_complete() -> None:
    """Verify raw_detail contains all required fields."""
    scenario = _make_scenario("Reference email text with some content here.")
    generated = _make_generated("Generated email text.")
    result = compute_conciseness(scenario, generated)

    required_keys = {
        "generated_word_count", "reference_word_count", "length_ratio",
        "lexical_density", "length_penalty_score", "formula",
        "band_low", "band_high", "final_score",
    }
    assert required_keys.issubset(result.raw_detail.keys())
