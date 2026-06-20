"""
Integration tests for evaluator.py.

Covers:
  - EvalResult assembly with mocked clients for all three metrics.
  - overall_score is the correct unweighted mean of the 3 metric scores.
  - Metrics 1 & 2 make API calls; Metric 3 is pure Python (no mock needed for it).
  - result contains exactly 3 MetricScore objects.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock
from email_eval.client.groq_client import GroqClient

import pytest

from email_eval.config import Settings
from email_eval.evaluation.evaluator import evaluate
from email_eval.schemas import EvalResult, GeneratedEmail, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def settings() -> Settings:
    """Minimal Settings with judge_samples=2 for fast test execution."""
    return Settings(
        grok_api_key="test-key-not-real",
        model_name="grok-test",
        judge_samples=2,
    )


@pytest.fixture()
def scenario() -> Scenario:
    return Scenario(
        id="eval-test-01",
        intent="Notify the board of Q3 financial results",
        key_facts=[
            "Revenue grew 18% year-over-year",
            "Operating margin improved to 24%",
        ],
        tone="formal",
        human_reference_email=(
            "Dear Board Members, I am pleased to report that Q3 revenue grew 18% "
            "year-over-year. Operating margin improved to 24%, reflecting disciplined "
            "cost management and strong top-line growth. We look forward to discussing "
            "these results at the upcoming board meeting."
        ),
    )


@pytest.fixture()
def generated() -> GeneratedEmail:
    return GeneratedEmail(
        scenario_id="eval-test-01",
        model_id="model_a",
        raw_model_output=(
            "REASONING:\nPlan: formal opening, include both facts.\n\n===EMAIL===\n"
            "Dear Board Members, Q3 revenue grew 18% year-over-year. "
            "Operating margin improved to 24%."
        ),
        parsed_email=(
            "Dear Board Members, Q3 revenue grew 18% year-over-year. "
            "Operating margin improved to 24%."
        ),
        cot_scratchpad="Plan: formal opening, include both facts.",
        parse_fallback_used=False,
    )


def _make_client(responses: list[str]) -> MagicMock:
    """
    Build a mock GrokClient that returns responses in order across all calls.

    The evaluator will call:
      - Metric 1: N calls (one per key_fact) → fact inclusion
      - Metric 2: judge_samples calls → tone fidelity
    Total calls = len(key_facts) + judge_samples
    """
    client = MagicMock()
    client.complete = AsyncMock(side_effect=responses)
    return client


# ── overall_score math ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_overall_score_is_unweighted_mean(
    scenario: Scenario, generated: GeneratedEmail, settings: Settings
) -> None:
    """
    overall_score must be the unweighted mean of the 3 metric scores.
    We fix the mock responses so we can predict exact metric scores and verify.
    """
    # Metric 1 (fact_inclusion): 2 facts
    #   Fact 1: ANSWER=yes, QUOTE grounded → hit
    #   Fact 2: ANSWER=yes, QUOTE grounded → hit
    #   → fact_inclusion score = 2/2 = 1.0
    #
    # Metric 2 (tone_fidelity): judge_samples=2
    #   Scores: 4, 4 → mean=4.0, normalized=0.8
    #
    # Metric 3 (conciseness): pure Python, computed from actual text
    #   We don't fix this — just verify it's included in the mean correctly.

    fact_responses = [
        "ANSWER: yes\nQUOTE: revenue grew 18% year-over-year",   # fact 1 hit
        "ANSWER: yes\nQUOTE: Operating margin improved to 24%",   # fact 2 hit
    ]
    tone_responses = [
        "SCORE: 4\nJUSTIFICATION: Good formal tone.",
        "SCORE: 4\nJUSTIFICATION: Matches formal register.",
    ]
    # Total: 2 fact calls + 2 tone calls = 4 responses
    client = _make_client(fact_responses + tone_responses)

    result = await evaluate(scenario, generated, client, settings)

    assert isinstance(result, EvalResult)
    assert result.scenario_id == scenario.id
    assert result.model_id == generated.model_id

    # Verify exact scores we can predict
    fact_metric = result.get_metric("fact_inclusion")
    assert fact_metric is not None
    assert abs(fact_metric.score - 1.0) < 1e-6

    tone_metric = result.get_metric("tone_fidelity")
    assert tone_metric is not None
    assert abs(tone_metric.score - 0.8) < 1e-6

    conciseness_metric = result.get_metric("conciseness")
    assert conciseness_metric is not None

    # overall_score must be the unweighted mean
    expected_overall = (fact_metric.score + tone_metric.score + conciseness_metric.score) / 3
    assert abs(result.overall_score - expected_overall) < 1e-4


@pytest.mark.asyncio
async def test_eval_result_has_exactly_three_metrics(
    scenario: Scenario, generated: GeneratedEmail, settings: Settings
) -> None:
    """EvalResult must contain exactly 3 MetricScore objects."""
    responses = [
        "ANSWER: no\nQUOTE: NONE",
        "ANSWER: no\nQUOTE: NONE",
        "SCORE: 3\nJUSTIFICATION: Adequate.",
        "SCORE: 3\nJUSTIFICATION: Adequate.",
    ]
    client = _make_client(responses)
    result = await evaluate(scenario, generated, client, settings)

    assert result.model_id == "model_a"
    assert len(result.metric_scores) == 3
    metric_names = {m.metric_name for m in result.metric_scores}
    assert metric_names == {"fact_inclusion", "tone_fidelity", "conciseness"}


@pytest.mark.asyncio
async def test_overall_score_bounds(
    scenario: Scenario, generated: GeneratedEmail, settings: Settings
) -> None:
    """overall_score is always in [0.0, 1.0]."""
    responses = [
        "ANSWER: no\nQUOTE: NONE",
        "ANSWER: no\nQUOTE: NONE",
        "SCORE: 1\nJUSTIFICATION: Very poor.",
        "SCORE: 1\nJUSTIFICATION: Very poor.",
    ]
    client = _make_client(responses)
    result = await evaluate(scenario, generated, client, settings)

    assert 0.0 <= result.overall_score <= 1.0


@pytest.mark.asyncio
async def test_generated_email_preserved_in_result(
    scenario: Scenario, generated: GeneratedEmail, settings: Settings
) -> None:
    """The GeneratedEmail object is embedded unchanged in EvalResult."""
    responses = [
        "ANSWER: yes\nQUOTE: revenue grew 18%",
        "ANSWER: yes\nQUOTE: margin improved",
        "SCORE: 5\nJUSTIFICATION: Perfect.",
        "SCORE: 5\nJUSTIFICATION: Perfect.",
    ]
    client = _make_client(responses)
    result = await evaluate(scenario, generated, client, settings)

    assert result.generated_email.scenario_id == generated.scenario_id
    assert result.generated_email.model_id == generated.model_id
    assert result.generated_email.parsed_email == generated.parsed_email
    assert result.generated_email.parse_fallback_used == generated.parse_fallback_used


@pytest.mark.asyncio
async def test_overall_score_all_perfect(
    scenario: Scenario, generated: GeneratedEmail, settings: Settings
) -> None:
    """All metrics at max → overall_score approaches 1.0."""
    responses = [
        "ANSWER: yes\nQUOTE: revenue grew 18% year-over-year",
        "ANSWER: yes\nQUOTE: Operating margin improved to 24%",
        "SCORE: 5\nJUSTIFICATION: Perfect match.",
        "SCORE: 5\nJUSTIFICATION: Perfect match.",
    ]
    client = _make_client(responses)
    result = await evaluate(scenario, generated, client, settings)

    # fact=1.0, tone=1.0, conciseness=<= 1.0 → overall < 1.0 only if conciseness < 1.0
    assert result.overall_score > 0.8


@pytest.mark.asyncio
async def test_get_metric_helper(
    scenario: Scenario, generated: GeneratedEmail, settings: Settings
) -> None:
    """EvalResult.get_metric() returns the correct MetricScore by name."""
    responses = [
        "ANSWER: no\nQUOTE: NONE",
        "ANSWER: no\nQUOTE: NONE",
        "SCORE: 3\nJUSTIFICATION: Average.",
        "SCORE: 3\nJUSTIFICATION: Average.",
    ]
    client = _make_client(responses)
    result = await evaluate(scenario, generated, client, settings)

    assert result.get_metric("fact_inclusion") is not None
    assert result.get_metric("tone_fidelity") is not None
    assert result.get_metric("conciseness") is not None
    assert result.get_metric("nonexistent_metric") is None
