from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    intent: str
    key_facts: list[str]
    tone: str
    email: str


FEW_SHOT_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        intent="Follow up after a product demo meeting and propose next steps",
        key_facts=[
            "Demo occurred on Tuesday, June 10th",
            "Client expressed interest in the analytics dashboard feature",
            "Pilot program would start in Q3",
            "Next step is a technical deep-dive call",
        ],
        tone="formal",
        email="""\
Subject: Follow-Up: Product Demo & Proposed Next Steps — [Your Company]

Dear Ms. Hartwell,

Thank you for your time on Tuesday, June 10th. It was a pleasure walking your team through our platform, and I appreciated the insightful questions raised during the analytics dashboard segment.

Based on our conversation, I understand that the dashboard's real-time reporting capabilities align well with your Q3 objectives. To advance the discussion, I would like to propose a focused technical deep-dive call with your engineering team, which would allow us to address integration requirements and confirm readiness for the pilot program launch.

Please let me know your availability over the next two weeks, and I will arrange the session accordingly.

Best regards,
Jordan Mills
Senior Account Executive""",
    ),
    FewShotExample(
        intent="Notify the team that Friday's sprint review is rescheduled",
        key_facts=[
            "Sprint review was originally scheduled for Friday at 2 PM",
            "New time is Monday at 10 AM",
            "Same video conference link will be used",
            "Reason: key stakeholder has a conflict",
        ],
        tone="casual",
        email="""\
Subject: Quick Update: Sprint Review Moved to Monday

Hey everyone,

Quick heads-up — we're moving this Friday's 2 PM sprint review to Monday at 10 AM. One of our key stakeholders has a conflict that came up last minute, so shifting gives everyone a chance to join.

Good news: the same video link still works, so no new calendar invite needed. Just hop on Monday morning when you're ready.

Thanks for being flexible — looking forward to showing off what we've shipped!

Cheers,
Alex""",
    ),
    FewShotExample(
        intent="Apologize to a client for a billing error and explain corrective action",
        key_facts=[
            "Client was overcharged by $340 on their April invoice",
            "Error was caused by a system configuration issue now resolved",
            "Refund will be processed within 5 business days",
            "A 10% discount will be applied to their next invoice as goodwill",
        ],
        tone="apologetic",
        email="""\
Subject: Sincere Apologies Regarding Your April Invoice

Dear Mr. Okafor,

I am writing to sincerely apologize for the $340 overcharge that appeared on your April invoice. Upon investigation, we identified the root cause as a system configuration issue that has now been fully resolved to prevent recurrence.

We understand that billing errors erode trust, and we take full responsibility. A refund of $340 has been initiated and will appear in your account within 5 business days. Additionally, as a token of our commitment to your satisfaction, a 10% discount will be automatically applied to your next invoice.

Please do not hesitate to reach out directly if you have any questions or concerns. We value your partnership greatly and are committed to earning back your confidence.

Sincerely,
Priya Nair
Customer Success Manager""",
    ),
]


def format_examples_for_prompt() -> str:
    """Render all few-shot examples as a formatted string for system prompt injection."""
    lines: list[str] = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        facts_str = "\n".join(f"  - {f}" for f in ex.key_facts)
        lines.append(
            f"--- EXAMPLE {i} ---\n"
            f"Intent: {ex.intent}\n"
            f"Key Facts:\n{facts_str}\n"
            f"Tone: {ex.tone}\n\n"
            f"Output:\n{ex.email}"
        )
    return "\n\n".join(lines)
