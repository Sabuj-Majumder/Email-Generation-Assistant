"""
Tests for Metric 1 — Fact Inclusion Rate.

Covers:
  - A fact clearly present in the email scores a hit (is_hit=True).
  - A fact with ANSWER: yes but QUOTE: NONE is correctly downgraded to a miss
    (judge-hallucination guard — the key correctness property of this metric).
  - A fact with ANSWER: no is a clean miss.
  - Score calculation: hits / total_facts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from email_eval.client.groq_client import GroqClient

import pytest

from email_eval.evaluation.metric_fact_inclusion import (
    _is_quote_grounded,
    _parse_extraction_response,
    compute_fact_inclusion,
)
from email_eval.schemas import GeneratedEmail, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def scenario() -> Scenario:
    return Scenario(
        id="test-s01",
        intent="Follow up after a meeting",
        key_facts=[
            "The meeting was on Monday June 10th",
            "The budget is $50,000",
            "Next step is a demo call",
        ],
        tone="formal",
        human_reference_email="Dear Alice, following our meeting on Monday June 10th...",
    )


@pytest.fixture()
def generated() -> GeneratedEmail:
    return GeneratedEmail(
        scenario_id="metric-test-01",
        model_id="model_a",
        raw_model_output="The meeting on Monday June 10th went well. The next step is a demo call.",
        parsed_email="The meeting on Monday June 10th went well. The next step is a demo call.",
        cot_scratchpad="Planning...",
        parse_fallback_used=False,
    )


def _make_client(responses: list[str]) -> MagicMock:
    """Build a mock GrokClient that returns responses in order."""
    client = MagicMock()
    client.complete = AsyncMock(side_effect=responses)
    return client


# ── _parse_extraction_response ────────────────────────────────────────────────

def test_parse_answer_yes_with_quote() -> None:
    response = "ANSWER: yes\nQUOTE: Monday June 10th"
    answer, quote = _parse_extraction_response(response)
    assert answer == "yes"
    assert quote == "Monday June 10th"


def test_parse_answer_no() -> None:
    response = "ANSWER: no\nQUOTE: NONE"
    answer, quote = _parse_extraction_response(response)
    assert answer == "no"
    assert quote == "NONE"


def test_parse_malformed_returns_defaults() -> None:
    answer, quote = _parse_extraction_response("Something unexpected")
    assert answer == "unknown"
    assert quote == "NONE"


# ── _is_quote_grounded ────────────────────────────────────────────────────────

def test_quote_grounded_exact_match() -> None:
    email = "The meeting was on Monday June 10th and went well."
    assert _is_quote_grounded("Monday June 10th", email) is True


def test_quote_grounded_partial_window_match() -> None:
    email = "We discussed the budget allocation in detail."
    assert _is_quote_grounded("budget allocation in detail", email) is True


def test_quote_none_string_not_grounded() -> None:
    assert _is_quote_grounded("NONE", "Any email text here") is False


def test_quote_empty_string_not_grounded() -> None:
    assert _is_quote_grounded("", "Any email text here") is False


def test_quote_hallucinated_not_in_email() -> None:
    email = "Thank you for the opportunity to work together."
    assert _is_quote_grounded("The budget is $50,000", email) is False


# ── compute_fact_inclusion: hit ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fact_present_scores_hit(scenario: Scenario, generated: GeneratedEmail) -> None:
    """A fact that is genuinely present in the email with a grounded quote = hit."""
    # Facts: ["The meeting was on Monday June 10th", "$50,000", "demo call"]
    # Email: "The meeting on Monday June 10th went well. The next step is a demo call."
    # Fact 1: present (June 10th), Fact 2: not present ($50,000), Fact 3: present (demo call)
    responses = [
        "ANSWER: yes\nQUOTE: Monday June 10th",       # fact 1 — hit
        "ANSWER: no\nQUOTE: NONE",                    # fact 2 — miss
        "ANSWER: yes\nQUOTE: next step is a demo call",  # fact 3 — hit
    ]
    client = _make_client(responses)
    score = await compute_fact_inclusion(scenario, generated, client, "llama-3.3-70b-versatile")
    assert score.metric_name == "fact_inclusion"
    assert score.raw_detail["facts_found"] == 2
    assert score.raw_detail["facts_missed"] == 1
    assert score.raw_detail["total_facts"] == 3
    assert abs(score.score - 2 / 3) < 1e-6


# ── compute_fact_inclusion: hallucination guard ───────────────────────────────

@pytest.mark.asyncio
async def test_yes_with_none_quote_downgraded_to_miss(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """
    CRITICAL: ANSWER=yes but QUOTE=NONE must be downgraded to a miss.
    This is the judge-hallucination guard — the primary correctness invariant
    of Metric 1. If this test fails, the metric is broken.
    """
    responses = [
        "ANSWER: yes\nQUOTE: NONE",   # hallucinated — should be downgraded
        "ANSWER: yes\nQUOTE: NONE",   # hallucinated — should be downgraded
        "ANSWER: yes\nQUOTE: NONE",   # hallucinated — should be downgraded
    ]
    client = _make_client(responses)
    result = await compute_fact_inclusion(scenario, generated, client, "llama-3.3-70b-versatile")

    assert result.raw_detail["facts_found"] == 0
    assert result.score == 0.0

    # Verify the downgrade flag is recorded in raw_detail for every fact
    for fact_entry in result.raw_detail["per_fact"]:
        assert fact_entry["downgraded_to_miss"] is True
        assert fact_entry["is_hit"] is False


@pytest.mark.asyncio
async def test_yes_with_hallucinated_quote_not_in_email(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """ANSWER=yes with a quote that doesn't appear in the email = downgraded to miss."""
    responses = [
        "ANSWER: yes\nQUOTE: the quarterly forecast is excellent",  # not in email
        "ANSWER: no\nQUOTE: NONE",
        "ANSWER: no\nQUOTE: NONE",
    ]
    client = _make_client(responses)
    result = await compute_fact_inclusion(scenario, generated, client, "llama-3.3-70b-versatile")

    assert result.raw_detail["facts_found"] == 0
    per_fact = result.raw_detail["per_fact"]
    assert per_fact[0]["downgraded_to_miss"] is True


# ── compute_fact_inclusion: all hits ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_facts_present_score_one(scenario: Scenario, generated: GeneratedEmail) -> None:
    responses = [
        "ANSWER: yes\nQUOTE: Monday June 10th",
        "ANSWER: yes\nQUOTE: demo call",
        "ANSWER: yes\nQUOTE: next step is a demo call",
    ]
    client = _make_client(responses)
    score = await compute_fact_inclusion(scenario, generated, client, "llama-3.3-70b-versatile")
    assert score.score == 1.0


# ── compute_fact_inclusion: all misses ────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_facts_absent_score_zero(scenario: Scenario, generated: GeneratedEmail) -> None:
    responses = [
        "ANSWER: no\nQUOTE: NONE",
        "ANSWER: no\nQUOTE: NONE",
        "ANSWER: no\nQUOTE: NONE",
    ]
    client = _make_client(responses)
    score = await compute_fact_inclusion(scenario, generated, client, "llama-3.3-70b-versatile")
    assert score.score == 0.0
    assert score.raw_detail["facts_found"] == 0
