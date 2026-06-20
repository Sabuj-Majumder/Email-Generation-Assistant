"""
Tests for generator.py — CoT parsing and graceful fallback.

Covers:
  - Correct parsing when ===EMAIL=== delimiter is present.
  - Graceful fallback (parse_fallback_used=True, cot_scratchpad=None) when delimiter absent.
  - REASONING: label stripped from cot_scratchpad.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from email_eval.generation.generator import generate_email
from email_eval.prompts.strategy_a_cot_fewshot import COT_DELIMITER
from email_eval.schemas import Scenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def scenario() -> Scenario:
    return Scenario(
        id="gen-test-01",
        intent="Request a project status update",
        key_facts=["Project deadline is August 15th", "Budget is on track"],
        tone="formal",
        human_reference_email="Dear Team, please provide a status update on the project...",
    )


def _make_client(response: str) -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    return client


# ── Strategy A: delimiter present ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strategy_a_parses_delimiter_correctly(scenario: Scenario) -> None:
    """When ===EMAIL=== is present, cot_scratchpad and parsed_email are split correctly."""
    reasoning = "1. Place deadline fact in opening.\n2. Use formal tone markers."
    email_body = "Subject: Project Status Update\n\nDear Team,\n\nCould you please provide..."
    raw_output = f"REASONING:\n{reasoning}\n\n{COT_DELIMITER}\n{email_body}"

    client = _make_client(raw_output)
    result = await generate_email(scenario, "model_a", "llama-3.3-70b-versatile", client)

    assert result.parse_fallback_used is False
    assert result.cot_scratchpad is not None
    assert "Place deadline fact" in result.cot_scratchpad
    assert "REASONING:" not in result.cot_scratchpad  # label stripped
    assert result.parsed_email == email_body
    assert result.raw_model_output == raw_output
    assert result.model_id == "model_a"
    assert result.scenario_id == scenario.id


@pytest.mark.asyncio
async def test_strategy_a_reasoning_label_stripped(scenario: Scenario) -> None:
    """The 'REASONING:' label is removed from cot_scratchpad."""
    raw_output = f"REASONING:\nMy plan is X.\n\n{COT_DELIMITER}\nThe email."
    client = _make_client(raw_output)
    result = await generate_email(scenario, "model_a", "llama-3.3-70b-versatile", client)

    assert result.cot_scratchpad is not None
    assert result.cot_scratchpad.upper().startswith("REASONING:") is False
    assert "My plan is X." in result.cot_scratchpad


@pytest.mark.asyncio
async def test_strategy_a_whitespace_trimmed(scenario: Scenario) -> None:
    """Parsed email and scratchpad should be stripped of leading/trailing whitespace."""
    raw_output = f"REASONING:\n  plan  \n\n{COT_DELIMITER}\n\n  email body  \n"
    client = _make_client(raw_output)
    result = await generate_email(scenario, "model_a", "llama-3.3-70b-versatile", client)

    assert result.parsed_email == "email body"
    assert result.cot_scratchpad is not None
    assert result.cot_scratchpad.strip() == result.cot_scratchpad


# ── Strategy A: delimiter missing (graceful fallback) ─────────────────────────

@pytest.mark.asyncio
async def test_strategy_a_fallback_when_delimiter_missing(scenario: Scenario) -> None:
    """
    CRITICAL: When ===EMAIL=== is absent, the generator must NOT raise.
    It must set parse_fallback_used=True and use the full output as parsed_email.
    This ensures the batch run never breaks on non-compliant model responses.
    """
    raw_output = "Here is the email without any delimiter: Dear Team, please update us."
    client = _make_client(raw_output)
    result = await generate_email(scenario, "model_a", "llama-3.3-70b-versatile", client)

    assert result.parse_fallback_used is True
    assert result.cot_scratchpad is None
    assert result.parsed_email == raw_output.strip()
    assert result.raw_model_output == raw_output


@pytest.mark.asyncio
async def test_strategy_a_fallback_does_not_raise_on_empty_output(scenario: Scenario) -> None:
    """Fallback must handle edge case of completely empty model response."""
    client = _make_client("")
    result = await generate_email(scenario, "model_a", "llama-3.3-70b-versatile", client)

    assert result.parse_fallback_used is True
    assert result.parsed_email == ""
    assert result.cot_scratchpad is None


# ── GeneratedEmail schema fields ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generated_email_fields_populated(scenario: Scenario) -> None:
    """All GeneratedEmail fields are correctly populated from generate_email."""
    email_body = "Dear Team, please send the status report."
    raw_output = f"REASONING:\nUse formal tone.\n\n{COT_DELIMITER}\n{email_body}"
    client = _make_client(raw_output)
    result = await generate_email(scenario, "model_a", "llama-3.3-70b-versatile", client)

    assert result.scenario_id == scenario.id
    assert result.model_id == "model_a"
    assert result.raw_model_output == raw_output
    assert result.parsed_email == email_body
    assert isinstance(result.cot_scratchpad, str)
    assert result.parse_fallback_used is False
