from __future__ import annotations

import asyncio
import logging

from email_eval.client.groq_client import GroqClient
from email_eval.config import Settings
from email_eval.evaluation.metric_conciseness import compute_conciseness
from email_eval.evaluation.metric_fact_inclusion import compute_fact_inclusion
from email_eval.evaluation.metric_tone_fidelity import compute_tone_fidelity
from email_eval.schemas import EvalResult, GeneratedEmail, Scenario

logger = logging.getLogger(__name__)


async def evaluate(
    scenario: Scenario,
    generated: GeneratedEmail,
    client: GroqClient,
    settings: Settings,
) -> EvalResult:
    """
    Run all three metrics over one (scenario, generated_email) pair.

    Metrics 1 (fact_inclusion) and 2 (tone_fidelity) are run concurrently.
    Metric 3 (conciseness) is pure-Python with no I/O.
    Overall score is the unweighted mean of the three normalised metric scores.
    """
    logger.info(
        "Starting evaluation | scenario=%s model_id=%s",
        scenario.id, generated.model_id,
    )

    fact_score, tone_score = await asyncio.gather(
        compute_fact_inclusion(scenario, generated, client, settings.judge_model_name),
        compute_tone_fidelity(
            scenario, generated, client, settings.judge_model_name,
            judge_samples=settings.judge_samples,
        ),
    )

    conciseness_score = compute_conciseness(scenario, generated)

    metric_scores = [fact_score, tone_score, conciseness_score]
    overall_score = sum(m.score for m in metric_scores) / len(metric_scores)

    logger.info(
        "Evaluation complete | scenario=%s model_id=%s "
        "fact=%.3f tone=%.3f conciseness=%.3f overall=%.3f",
        scenario.id, generated.model_id,
        fact_score.score, tone_score.score, conciseness_score.score, overall_score,
    )

    return EvalResult(
        scenario_id=scenario.id,
        model_id=generated.model_id,
        generated_email=generated,
        metric_scores=metric_scores,
        overall_score=round(overall_score, 4),
    )
