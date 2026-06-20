from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Scenario(BaseModel):
    """One test scenario: the input provided to the email generator."""

    id: str
    intent: str
    key_facts: list[str] = Field(..., min_length=1)
    tone: str
    human_reference_email: str


class GeneratedEmail(BaseModel):
    """Output of the generation layer for one (scenario, model) pair."""

    scenario_id: str
    model_id: Literal["model_a", "model_b"]
    raw_model_output: str
    parsed_email: str
    cot_scratchpad: str | None = None
    parse_fallback_used: bool = False


class MetricScore(BaseModel):
    """Normalised score (0.0–1.0) for one metric on one (scenario, model) pair."""

    metric_name: Literal["fact_inclusion", "tone_fidelity", "conciseness"]
    score: float = Field(..., ge=0.0, le=1.0)
    raw_detail: dict[str, Any]

    @model_validator(mode="after")
    def clamp_score(self) -> MetricScore:
        self.score = max(0.0, min(1.0, self.score))
        return self


class EvalResult(BaseModel):
    """Complete evaluation record for one (scenario, model) pair."""

    scenario_id: str
    model_id: str
    generated_email: GeneratedEmail
    metric_scores: list[MetricScore] = Field(..., min_length=3, max_length=3)
    overall_score: float = Field(..., ge=0.0, le=1.0)

    def get_metric(self, name: str) -> MetricScore | None:
        """Return a metric score by name."""
        return next((m for m in self.metric_scores if m.metric_name == name), None)
