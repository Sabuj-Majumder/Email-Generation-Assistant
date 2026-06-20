"""
Tests for Metric 2 — Tone Fidelity Score.

Covers:
  - Mock client returning fixed scores: verify mean and std-dev computed correctly.
  - Low reliability flag triggered when std-dev > threshold.
  - Normalized score = mean / 5.0.
  - Handles partial parse failures gracefully (None scores excluded from stats).
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock
from email_eval.client.groq_client import GroqClient

import pytest

from email_eval.evaluation.metric_tone_fidelity import (
    LOW_RELIABILITY_STD_THRESHOLD,
    _parse_judge_response,
    compute_tone_fidelity,
)
from email_eval.schemas import GeneratedEmail, Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def scenario() -> Scenario:
    return Scenario(
        id="test-tone-01",
        intent="Notify team of policy change",
        key_facts=["Policy effective date is July 1st"],
        tone="formal",
        human_reference_email="Dear Team, effective July 1st, the following policy applies...",
    )


@pytest.fixture()
def generated() -> GeneratedEmail:
    return GeneratedEmail(
        scenario_id="metric-test-01",
        model_id="model_a",
        raw_model_output="Here is the email:\n\n===EMAIL===\nDear Sir, the report is attached.",
        parsed_email="Dear Sir, the report is attached.",
        cot_scratchpad=None,
        parse_fallback_used=False,
    )


def _make_client(score_strings: list[str]) -> MagicMock:
    """Build a mock GrokClient returning one score response per call."""
    responses = [f"SCORE: {s}\nJUSTIFICATION: Sample justification." for s in score_strings]
    client = MagicMock()
    client.complete = AsyncMock(side_effect=responses)
    return client


# ── _parse_judge_response ─────────────────────────────────────────────────────

def test_parse_integer_score() -> None:
    response = "SCORE: 4\nJUSTIFICATION: Well matched."
    score, justification = _parse_judge_response(response)
    assert score == 4.0
    assert justification == "Well matched."


def test_parse_float_score() -> None:
    score, _ = _parse_judge_response("SCORE: 3.5\nJUSTIFICATION: Partially matched.")
    assert score == 3.5


def test_parse_score_out_of_range_rejected() -> None:
    # Score of 6 is outside 1-5, should be rejected → None
    score, _ = _parse_judge_response("SCORE: 6\nJUSTIFICATION: Too high.")
    assert score is None


def test_parse_malformed_returns_none() -> None:
    score, justification = _parse_judge_response("No valid format here.")
    assert score is None
    assert justification == ""


# ── Mean and std-dev computation ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mean_and_stddev_computed_correctly(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """
    Fixed scores [4, 4, 4] → mean=4.0, std=0.0, normalized=0.8, reliable.
    """
    client = _make_client(["4", "4", "4"])
    score = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)

    assert score.metric_name == "tone_fidelity"
    assert abs(score.score - 0.8) < 1e-6
    assert abs(score.raw_detail["mean_raw_score"] - 4.0) < 1e-6
    assert abs(score.raw_detail["std_dev"] - 0.0) < 1e-6
    assert score.raw_detail["low_judge_reliability"] is False


@pytest.mark.asyncio
async def test_varying_scores_std_dev_correct(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """
    Fixed scores [2, 4, 4] → mean=10/3, sample std-dev should be correct.
    """
    client = _make_client(["2", "4", "4"])
    result = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)

    expected_mean = (2 + 4 + 4) / 3
    expected_variance = ((2 - expected_mean) ** 2 + (4 - expected_mean) ** 2 + (4 - expected_mean) ** 2) / 2
    expected_std = math.sqrt(expected_variance)

    assert abs(result.raw_detail["mean_raw_score"] - expected_mean) < 1e-4
    assert abs(result.raw_detail["std_dev"] - expected_std) < 1e-4
    assert abs(result.score - (expected_mean / 5.0)) < 1e-4


@pytest.mark.asyncio
async def test_low_reliability_flag_triggered(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """
    Scores [1, 5, 1] → high std-dev → low_judge_reliability=True.
    """
    client = _make_client(["1", "5", "1"])
    result = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)

    assert result.raw_detail["std_dev"] > LOW_RELIABILITY_STD_THRESHOLD
    assert result.raw_detail["low_judge_reliability"] is True


@pytest.mark.asyncio
async def test_high_reliability_flag_not_triggered(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """
    Scores [4, 5, 4] → low std-dev → low_judge_reliability=False.
    """
    client = _make_client(["4", "5", "4"])
    score = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)

    assert score.raw_detail["low_judge_reliability"] is False


# ── Normalized score bounds ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_normalized_score_max(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """All 5s → normalized score = 1.0."""
    client = _make_client(["5", "5", "5"])
    result = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)
    assert abs(result.score - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_normalized_score_min(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """All 1s → normalized score = 0.2."""
    client = _make_client(["1", "1", "1"])
    result = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)
    assert abs(result.score - 0.2) < 1e-6


# ── Sample count in raw_detail ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_raw_detail_contains_all_samples(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    client = _make_client(["3", "4", "5"])
    result = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=3)
    samples = result.raw_detail["samples"]
    assert len(samples) == 3
    assert all("raw_score" in s for s in samples)
    assert all("justification" in s for s in samples)


@pytest.mark.asyncio
async def test_single_sample_std_dev_zero(
    scenario: Scenario, generated: GeneratedEmail
) -> None:
    """Single sample → std_dev=0.0 (undefined, set to 0 by convention)."""
    client = _make_client(["3"])
    score = await compute_tone_fidelity(scenario, generated, client, "llama-3.3-70b-versatile", judge_samples=1)

    assert score.raw_detail["std_dev"] == 0.0
    assert abs(score.score - 0.6) < 1e-6
