from __future__ import annotations

from email_eval.prompts.few_shot_examples import format_examples_for_prompt
from email_eval.schemas import Scenario

# Contract between this module and generator.py — do not change without updating both.
COT_DELIMITER = "===EMAIL==="


def build_strategy_a_prompt(scenario: Scenario) -> tuple[str, str]:
    """Build the Strategy A (CoT + Few-Shot + Role) system and user prompts."""
    examples_block = format_examples_for_prompt()
    facts_str = "\n".join(f"- {f}" for f in scenario.key_facts)

    system_prompt = f"""\
You are a senior executive communications specialist who has ghostwritten emails \
for C-suite leaders for 15 years. You write with precision, appropriate tone \
calibration, and never waste the reader's time. Every sentence earns its place.

════════════════════════════════════════
REFERENCE EXAMPLES
════════════════════════════════════════
Study these examples carefully. They show you the expected quality, structure, \
and tone calibration for different scenarios.

{examples_block}

════════════════════════════════════════
YOUR TASK
════════════════════════════════════════
You will receive an Intent, Key Facts, and a Tone. Before writing, think through \
your approach systematically:

1. List which Key Facts you will place in which paragraph.
2. Identify 2-3 concrete tone markers (word choices, sentence length patterns, \
   formality signals) you will use to hit the target tone.
3. Decide on an email length appropriate to the intent — neither too terse to \
   omit context nor too verbose to disrespect the reader's time.

Output your thinking under "REASONING:" — this is your private scratchpad.
Then output the final email, exactly as the reader should receive it, under \
"{COT_DELIMITER}" — no extra commentary or explanation after the email.

FORMAT (follow exactly):
REASONING:
<your step-by-step planning>

{COT_DELIMITER}
<the complete, polished email>"""

    user_prompt = f"""\
Intent: {scenario.intent}

Key Facts to include:
{facts_str}

Tone: {scenario.tone}

Remember: output REASONING: block first, then {COT_DELIMITER}, then the email."""

    return system_prompt, user_prompt
