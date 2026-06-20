from __future__ import annotations

import logging
import math
import re

from email_eval.schemas import GeneratedEmail, MetricScore, Scenario

logger = logging.getLogger(__name__)

METRIC_NAME = "tone_fidelity"

# Std-dev threshold above which judge reliability is flagged.
# 0.75 on a 1-5 scale represents meaningful disagreement across samples.
LOW_RELIABILITY_STD_THRESHOLD = 0.75

JUDGE_TEMPERATURE = 0.7


def _parse_judge_response(response: str) -> tuple[float | None, str]:
    """
    Parse judge response into (score, justification).

    Expected format:
        SCORE: <1-5>
        JUSTIFICATION: <one sentence>

    Returns (None, '') on parse failure — caller treats as a missing sample.
    """
    score: float | None = None
    justification = ""

    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            raw = line.split(":", 1)[1].strip()
            match = re.search(r"(\d+(?:\.\d+)?)", raw)
            if match:
                candidate = float(match.group(1))
                if 1.0 <= candidate <= 5.0:
                    score = candidate
        elif line.upper().startswith("JUSTIFICATION:"):
            justification = line.split(":", 1)[1].strip()

    return score, justification


async def compute_tone_fidelity(
    scenario: Scenario,
    generated: GeneratedEmail,
    client,
    judge_model_name: str,
    judge_samples: int = 3,
) -> MetricScore:
    """
    Compute the Tone Fidelity Score for one (scenario, generated email) pair.

    Score = mean_raw_score / 5.0 (normalised 0.0–1.0).

    The judge is called `judge_samples` times at temperature > 0 to capture
    variance. If std_dev > LOW_RELIABILITY_STD_THRESHOLD, the result is
    flagged as low_judge_reliability in raw_detail.
    """
    email_text = generated.parsed_email
    tone = scenario.tone

    system_prompt = (
        "You are an expert writing coach specializing in professional communication. "
        "Evaluate tone with precision. Respond only in the exact format requested."
    )
    user_prompt = (
        f'Target tone: "{tone}"\n'
        f'Email: """{email_text}"""\n\n'
        "Rate how well this email's tone matches the target tone on a 1-5 scale, "
        "considering word choice, sentence length, and formality markers.\n"
        "1 = completely wrong tone\n"
        "3 = partially matches tone\n"
        "5 = perfectly matches tone\n\n"
        "Respond in this exact format:\n"
        "SCORE: <1-5>\n"
        "JUSTIFICATION: <one sentence explaining your rating>"
    )

    samples: list[dict] = []
    valid_scores: list[float] = []

    for i in range(judge_samples):
        try:
            response = await client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=judge_model_name,
                temperature=JUDGE_TEMPERATURE,
            )
            score, justification = _parse_judge_response(response)
        except Exception as exc:
            logger.error(
                "Tone judge call %d/%d failed | scenario=%s error=%s",
                i + 1, judge_samples, scenario.id, exc,
            )
            score, justification = None, f"ERROR: {exc}"

        samples.append({"sample_index": i + 1, "raw_score": score, "justification": justification})

        if score is not None:
            valid_scores.append(score)

        logger.debug(
            "Tone judge sample %d/%d | scenario=%s score=%s",
            i + 1, judge_samples, scenario.id, score,
        )

    if valid_scores:
        mean_score = sum(valid_scores) / len(valid_scores)
        if len(valid_scores) > 1:
            variance = sum((s - mean_score) ** 2 for s in valid_scores) / (len(valid_scores) - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0
    else:
        mean_score = 0.0
        std_dev = 0.0

    normalized_score = mean_score / 5.0
    low_reliability = std_dev > LOW_RELIABILITY_STD_THRESHOLD

    if low_reliability:
        logger.warning(
            "Tone fidelity: low judge reliability | scenario=%s model_id=%s std_dev=%.3f",
            scenario.id, generated.model_id, std_dev,
        )

    logger.info(
        "Tone fidelity | scenario=%s model_id=%s score=%.3f mean=%.2f std=%.3f",
        scenario.id, generated.model_id, normalized_score, mean_score, std_dev,
    )

    return MetricScore(
        metric_name=METRIC_NAME,
        score=normalized_score,
        raw_detail={
            "target_tone": tone,
            "samples": samples,
            "valid_sample_count": len(valid_scores),
            "mean_raw_score": round(mean_score, 4),
            "std_dev": round(std_dev, 4),
            "normalized_score": round(normalized_score, 4),
            "low_judge_reliability": low_reliability,
            "reliability_threshold": LOW_RELIABILITY_STD_THRESHOLD,
        },
    )
