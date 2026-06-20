from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from email_eval.config import get_settings
from email_eval.orchestration.run_eval import run_pipeline, MODELS_TESTED

_VALID_MODELS = tuple(MODELS_TESTED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--scenario-id",
    default=None,
    metavar="ID",
    help="Run only the scenario with this ID (e.g. s03).",
)
@click.option(
    "--model-id",
    default=None,
    type=click.Choice(_VALID_MODELS, case_sensitive=True),
    help="Run only this model. Runs both if omitted.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip (scenario_id, model_id) pairs already in results/raw_results.json.",
)
def cli(scenario_id: str | None, model_id: str | None, resume: bool) -> None:
    """Email Eval Assistant — run the generation + evaluation pipeline."""
    logger.info(
        "Starting pipeline | scenario_id=%s model_id=%s resume=%s",
        scenario_id or "all", model_id or "all", resume,
    )

    try:
        settings = get_settings()
    except Exception as exc:
        logger.error("Failed to load settings — is GROQ_API_KEY set in .env? Error: %s", exc)
        sys.exit(1)

    try:
        results = asyncio.run(
            run_pipeline(
                settings=settings,
                scenario_id_filter=scenario_id,
                model_filter=model_id,
                resume=resume,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Run interrupted. Re-run with --resume to continue.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info(
        "Run complete. %d result(s) produced. Check results/ directory.",
        len(results),
    )


if __name__ == "__main__":
    cli()
