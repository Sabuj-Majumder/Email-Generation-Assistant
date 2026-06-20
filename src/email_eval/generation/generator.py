from __future__ import annotations

import logging

from email_eval.client.groq_client import GroqClient
from email_eval.prompts.strategy_a_cot_fewshot import COT_DELIMITER, build_strategy_a_prompt
from email_eval.schemas import GeneratedEmail, Scenario

logger = logging.getLogger(__name__)


async def generate_email(
    scenario: Scenario,
    model_id: str,
    model_name: str,
    client: GroqClient,
) -> GeneratedEmail:
    """
    Generate one email for the given scenario using Strategy A prompting.

    Returns a GeneratedEmail with parsed_email, cot_scratchpad, and
    parse_fallback_used set. Never raises — degrades gracefully when the
    model omits the CoT delimiter.
    """
    logger.info(
        "Generating email | scenario=%s model_id=%s model_name=%s",
        scenario.id, model_id, model_name,
    )

    system_prompt, user_prompt = build_strategy_a_prompt(scenario)

    raw_output = await client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=0.7,
    )

    parsed_email, cot_scratchpad, parse_fallback_used = _parse_strategy_a(
        raw_output, scenario.id
    )

    logger.info(
        "Generation complete | scenario=%s model_id=%s fallback=%s email_len=%d",
        scenario.id, model_id, parse_fallback_used, len(parsed_email),
    )

    return GeneratedEmail(
        scenario_id=scenario.id,
        model_id=model_id,  # type: ignore[arg-type]
        raw_model_output=raw_output,
        parsed_email=parsed_email,
        cot_scratchpad=cot_scratchpad,
        parse_fallback_used=parse_fallback_used,
    )


def _parse_strategy_a(
    raw_output: str,
    scenario_id: str,
) -> tuple[str, str | None, bool]:
    """
    Split Strategy A output on COT_DELIMITER (===EMAIL===).

    Returns (parsed_email, cot_scratchpad, parse_fallback_used).
    If the delimiter is missing, the full raw output is used as parsed_email
    and parse_fallback_used is set to True.
    """
    if COT_DELIMITER in raw_output:
        parts = raw_output.split(COT_DELIMITER, maxsplit=1)
        reasoning_block = parts[0].strip()
        email_body = parts[1].strip()

        if reasoning_block.upper().startswith("REASONING:"):
            reasoning_block = reasoning_block[len("REASONING:"):].strip()

        return email_body, reasoning_block, False

    logger.warning(
        "CoT delimiter not found in output | scenario=%s — using full output as parsed_email",
        scenario_id,
    )
    return raw_output.strip(), None, True
