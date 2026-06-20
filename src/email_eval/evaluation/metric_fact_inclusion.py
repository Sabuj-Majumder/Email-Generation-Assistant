from __future__ import annotations

import logging
import re

from email_eval.schemas import GeneratedEmail, MetricScore, Scenario

logger = logging.getLogger(__name__)

METRIC_NAME = "fact_inclusion"

# Minimum character window for sliding-window quote grounding check.
_MIN_QUOTE_WINDOW = 10


def _is_quote_grounded(quote: str, email_text: str) -> bool:
    """
    Check whether the judge's supporting quote actually appears in the email.

    Uses a sliding-window approach to handle minor whitespace or punctuation
    differences. Returns True only if a meaningful substring of the quote is
    found verbatim (case-insensitive) in the email text.
    """
    if not quote or quote.strip().upper() == "NONE":
        return False

    quote_clean = quote.strip().lower()
    email_clean = email_text.lower()

    if quote_clean in email_clean:
        return True

    window = _MIN_QUOTE_WINDOW
    if len(quote_clean) < window:
        return False

    for i in range(len(quote_clean) - window + 1):
        if quote_clean[i : i + window] in email_clean:
            return True

    return False


def _parse_extraction_response(response: str) -> tuple[str, str]:
    """
    Parse the constrained extraction response into (answer, quote).

    Expected format:
        ANSWER: yes|no
        QUOTE: <text or NONE>

    Returns ('unknown', 'NONE') on parse failure, treated as a miss.
    """
    answer = "unknown"
    quote = "NONE"

    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("QUOTE:"):
            quote = line.split(":", 1)[1].strip()

    return answer, quote


async def compute_fact_inclusion(
    scenario: Scenario,
    generated: GeneratedEmail,
    client,
    judge_model_name: str,
) -> MetricScore:
    """
    Compute the Fact Inclusion Rate for one (scenario, generated email) pair.

    Score = facts_found / total_facts (0.0–1.0).

    Each fact is checked independently via one LLM extraction call. A
    deterministic quote-grounding guard then verifies the judge's evidence,
    downgrading ungrounded "yes" claims to misses to prevent hallucination.
    """
    email_text = generated.parsed_email
    facts = scenario.key_facts
    per_fact_results: list[dict] = []
    hits = 0

    for fact in facts:
        system_prompt = (
            "You are a precise fact-checking assistant. "
            "Answer only in the exact format requested — no additional commentary."
        )
        user_prompt = (
            f'Fact to check: "{fact}"\n'
            f'Email: """{email_text}"""\n\n'
            "Does the email contain this fact, even if paraphrased? "
            "Respond in this exact format:\n"
            "ANSWER: yes|no\n"
            "QUOTE: <the exact span from the email that supports your answer, "
            'or "NONE" if answer is no>'
        )

        try:
            response = await client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=judge_model_name,
                temperature=0.0,
            )
            answer, quote = _parse_extraction_response(response)
        except Exception as exc:
            logger.error(
                "Fact extraction call failed | scenario=%s fact='%.40s' error=%s",
                scenario.id, fact, exc,
            )
            answer, quote = "unknown", "NONE"

        claimed_present = answer == "yes"
        quote_grounded = _is_quote_grounded(quote, email_text)

        if claimed_present and not quote_grounded:
            logger.warning(
                "Fact inclusion: ANSWER=yes but quote not grounded — downgrading to miss "
                "| scenario=%s fact='%.40s'",
                scenario.id, fact,
            )
            is_hit = False
            downgraded = True
        else:
            is_hit = claimed_present and quote_grounded
            downgraded = False

        if is_hit:
            hits += 1

        per_fact_results.append(
            {
                "fact": fact,
                "llm_answer": answer,
                "llm_quote": quote,
                "quote_grounded": quote_grounded,
                "downgraded_to_miss": downgraded,
                "is_hit": is_hit,
            }
        )

        logger.debug(
            "Fact check | scenario=%s hit=%s downgraded=%s fact='%.40s'",
            scenario.id, is_hit, downgraded, fact,
        )

    total = len(facts)
    score = hits / total if total > 0 else 0.0

    logger.info(
        "Fact inclusion | scenario=%s model_id=%s score=%.3f hits=%d/%d",
        scenario.id, generated.model_id, score, hits, total,
    )

    return MetricScore(
        metric_name=METRIC_NAME,
        score=score,
        raw_detail={
            "total_facts": total,
            "facts_found": hits,
            "facts_missed": total - hits,
            "per_fact": per_fact_results,
        },
    )
