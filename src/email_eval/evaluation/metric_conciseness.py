from __future__ import annotations

import logging
import re

from email_eval.schemas import GeneratedEmail, MetricScore, Scenario

logger = logging.getLogger(__name__)

METRIC_NAME = "conciseness"

# Formula weights — must sum to 1.0.
LEXICAL_DENSITY_WEIGHT: float = 0.5
LENGTH_PENALTY_WEIGHT: float = 0.5
assert abs(LEXICAL_DENSITY_WEIGHT + LENGTH_PENALTY_WEIGHT - 1.0) < 1e-9

# Length ratio band: emails within [BAND_LOW, BAND_HIGH] relative to the
# reference score 1.0. Outside the band, the score decays linearly to 0.
BAND_LOW: float = 0.5
BAND_HIGH: float = 1.5

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
        "in", "on", "at", "to", "by", "of", "up", "as", "is", "am", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "shall", "should", "may", "might",
        "must", "can", "could", "i", "me", "my", "we", "us", "our", "you",
        "your", "he", "him", "his", "she", "her", "it", "its", "they",
        "them", "their", "this", "that", "these", "those", "with", "from",
        "not", "if", "then", "than", "also", "about", "more", "any", "all",
        "no", "each", "who", "what", "which", "how", "when", "where",
        "there", "here", "just", "very", "well", "please", "dear", "re",
        "sincerely", "regards", "best", "thank", "thanks", "hi", "hello",
    }
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _lexical_density(words: list[str]) -> float:
    """unique_content_words / total_words. Returns 0.0 for empty input."""
    if not words:
        return 0.0
    content_words = [w for w in words if w not in STOPWORDS]
    if not content_words:
        return 0.0
    return len(set(content_words)) / len(words)


def _length_penalty_score(length_ratio: float) -> float:
    """
    Linear tent function scoring email length relative to the human reference.

    ratio in [BAND_LOW, BAND_HIGH] → 1.0 (ideal)
    ratio < BAND_LOW              → linear decay toward 0.0 at ratio=0
    ratio > BAND_HIGH             → linear decay toward 0.0 at ratio=2*BAND_HIGH
    """
    if BAND_LOW <= length_ratio <= BAND_HIGH:
        return 1.0
    if length_ratio < BAND_LOW:
        return max(0.0, length_ratio / BAND_LOW)
    return max(0.0, 1.0 - (length_ratio - BAND_HIGH) / BAND_HIGH)


def compute_conciseness(
    scenario: Scenario,
    generated: GeneratedEmail,
) -> MetricScore:
    """
    Compute the Structural Conciseness / Signal Density metric.

    Formula: 0.5 * lexical_density + 0.5 * length_penalty_score
    Pure Python — zero API calls, fully deterministic.
    """
    gen_words = _tokenize(generated.parsed_email)
    ref_words = _tokenize(scenario.human_reference_email)

    gen_word_count = len(gen_words)
    ref_word_count = len(ref_words)

    lex_density = _lexical_density(gen_words)

    if ref_word_count == 0:
        length_ratio = 1.0
        length_penalty = 1.0
    else:
        length_ratio = gen_word_count / ref_word_count
        length_penalty = _length_penalty_score(length_ratio)

    final_score = max(0.0, min(1.0, LEXICAL_DENSITY_WEIGHT * lex_density + LENGTH_PENALTY_WEIGHT * length_penalty))

    logger.info(
        "Conciseness | scenario=%s model_id=%s score=%.3f lex_density=%.3f length_ratio=%.3f",
        scenario.id, generated.model_id, final_score, lex_density, length_ratio,
    )

    return MetricScore(
        metric_name=METRIC_NAME,
        score=final_score,
        raw_detail={
            "generated_word_count": gen_word_count,
            "reference_word_count": ref_word_count,
            "length_ratio": round(length_ratio, 4),
            "lexical_density": round(lex_density, 4),
            "length_penalty_score": round(length_penalty, 4),
            "formula": f"{LEXICAL_DENSITY_WEIGHT} * lexical_density + {LENGTH_PENALTY_WEIGHT} * length_penalty_score",
            "band_low": BAND_LOW,
            "band_high": BAND_HIGH,
            "final_score": round(final_score, 4),
        },
    )
