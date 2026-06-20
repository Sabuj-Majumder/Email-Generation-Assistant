from __future__ import annotations

import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from email_eval.client.groq_client import GroqClient
from email_eval.config import Settings
from email_eval.evaluation.evaluator import evaluate
from email_eval.generation.generator import generate_email
from email_eval.schemas import EvalResult, Scenario

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "data" / "scenarios.json"
_RESULTS_DIR = _PROJECT_ROOT / "results"
_RAW_JSON = _RESULTS_DIR / "raw_results.json"
_RAW_CSV = _RESULTS_DIR / "raw_results.csv"
_REPORT_MD = _RESULTS_DIR / "comparative_report.md"

MODELS_TESTED = ["model_a", "model_b"]

_LOW_RELIABILITY_THRESHOLD = 0.75


def load_scenarios(scenario_id_filter: str | None = None) -> list[Scenario]:
    """Load scenarios.json, optionally filtering to a single scenario_id."""
    with _SCENARIOS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    scenarios = [Scenario(**s) for s in raw]
    if scenario_id_filter:
        scenarios = [s for s in scenarios if s.id == scenario_id_filter]
        if not scenarios:
            raise ValueError(
                f"No scenario found with id='{scenario_id_filter}'. "
                f"Available: {[s.id for s in [Scenario(**s) for s in raw]]}"
            )
    logger.info("Loaded %d scenario(s)", len(scenarios))
    return scenarios


def load_completed_pairs(resume: bool) -> set[tuple[str, str]]:
    """Return (scenario_id, model_id) pairs already recorded in raw_results.json."""
    if not resume or not _RAW_JSON.exists():
        return set()
    with _RAW_JSON.open("r", encoding="utf-8") as fh:
        existing = json.load(fh)
    pairs = {(r["scenario_id"], r["model_id"]) for r in existing}
    logger.info("Resume mode: %d completed pairs found, will skip them", len(pairs))
    return pairs


def load_existing_results(resume: bool) -> list[EvalResult]:
    """Load existing EvalResult objects when resuming a partial run."""
    if not resume or not _RAW_JSON.exists():
        return []
    with _RAW_JSON.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [EvalResult(**r) for r in raw]


async def run_pipeline(
    settings: Settings,
    scenario_id_filter: str | None = None,
    model_filter: str | None = None,
    resume: bool = False,
) -> list[EvalResult]:
    """
    Full evaluation pipeline.

    For each (scenario, model) pair: generate email via Strategy A, evaluate
    on all three metrics, persist results incrementally, then write the
    comparative report.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(scenario_id_filter)
    completed_pairs = load_completed_pairs(resume)
    all_results: list[EvalResult] = load_existing_results(resume)

    models_to_run = [model_filter] if model_filter else MODELS_TESTED
    client = GroqClient(settings)

    total_pairs = sum(
        1
        for s in scenarios
        for mod in models_to_run
        if (s.id, mod) not in completed_pairs
    )
    logger.info(
        "Pipeline starting | scenarios=%d models=%d pairs_to_run=%d",
        len(scenarios), len(models_to_run), total_pairs,
    )

    for scenario in scenarios:
        for model_id in models_to_run:
            pair = (scenario.id, model_id)
            if pair in completed_pairs:
                logger.info("Skipping completed pair | scenario=%s model_id=%s", *pair)
                continue

            logger.info("Processing | scenario=%s model_id=%s", scenario.id, model_id)

            try:
                actual_model_name = getattr(settings, model_id)
                generated = await generate_email(scenario, model_id, actual_model_name, client)
                result = await evaluate(scenario, generated, client, settings)
                all_results.append(result)
                _persist_json(all_results)
                logger.info(
                    "Pair complete | scenario=%s model_id=%s overall=%.3f",
                    scenario.id, model_id, result.overall_score,
                )
            except Exception as exc:
                logger.error(
                    "Pipeline error | scenario=%s model_id=%s error=%s",
                    scenario.id, model_id, exc, exc_info=True,
                )
                continue

    if all_results:
        _persist_json(all_results)
        _persist_csv(all_results)
        _generate_report(all_results, settings)
        logger.info(
            "Pipeline complete | total_results=%d | outputs written to %s",
            len(all_results), _RESULTS_DIR,
        )
    else:
        logger.warning("No results to persist — all pairs may have failed or been skipped")

    return all_results


def _persist_json(results: list[EvalResult]) -> None:
    payload = [r.model_dump() for r in results]
    with _RAW_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.debug("Persisted %d results to %s", len(results), _RAW_JSON)


def _persist_csv(results: list[EvalResult]) -> None:
    """Write a flattened CSV: one row per (scenario × model × metric)."""
    rows: list[dict[str, Any]] = []
    for result in results:
        base = {
            "scenario_id": result.scenario_id,
            "model_id": result.model_id,
            "overall_score": result.overall_score,
            "parse_fallback_used": result.generated_email.parse_fallback_used,
        }
        for metric in result.metric_scores:
            row = {**base, "metric_name": metric.metric_name, "metric_score": metric.score}
            detail = metric.raw_detail
            if metric.metric_name == "fact_inclusion":
                row["facts_found"] = detail.get("facts_found")
                row["facts_missed"] = detail.get("facts_missed")
                row["total_facts"] = detail.get("total_facts")
            elif metric.metric_name == "tone_fidelity":
                row["tone_mean_raw"] = detail.get("mean_raw_score")
                row["tone_std_dev"] = detail.get("std_dev")
                row["low_judge_reliability"] = detail.get("low_judge_reliability")
            elif metric.metric_name == "conciseness":
                row["generated_word_count"] = detail.get("generated_word_count")
                row["reference_word_count"] = detail.get("reference_word_count")
                row["length_ratio"] = detail.get("length_ratio")
                row["lexical_density"] = detail.get("lexical_density")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(_RAW_CSV, index=False)
    logger.info("CSV written | rows=%d path=%s", len(rows), _RAW_CSV)


def _model_label(model_id: str, settings: Settings) -> str:
    return (
        f"Model A (`{settings.model_a}`)"
        if model_id == "model_a"
        else f"Model B (`{settings.model_b}`)"
    )


def _generate_report(results: list[EvalResult], settings: Settings) -> None:
    """Generate comparative_report.md from evaluation results."""
    agg: dict[str, dict[str, list[float]]] = {"model_a": {}, "model_b": {}}
    overall: dict[str, list[float]] = {"model_a": [], "model_b": []}
    fallback_counts: dict[str, int] = {"model_a": 0, "model_b": 0}
    low_reliability_counts: dict[str, int] = {"model_a": 0, "model_b": 0}
    fact_drop_detail: dict[str, list[dict]] = {"model_a": [], "model_b": []}

    for result in results:
        mod = result.model_id
        if mod not in agg:
            continue

        overall[mod].append(result.overall_score)
        if result.generated_email.parse_fallback_used:
            fallback_counts[mod] += 1

        for ms in result.metric_scores:
            agg[mod].setdefault(ms.metric_name, []).append(ms.score)

            if ms.metric_name == "tone_fidelity" and ms.raw_detail.get("low_judge_reliability"):
                low_reliability_counts[mod] += 1

            if ms.metric_name == "fact_inclusion":
                per_fact = ms.raw_detail.get("per_fact", [])
                missed = [f for f in per_fact if not f.get("is_hit")]
                fact_drop_detail[mod].append(
                    {
                        "scenario_id": result.scenario_id,
                        "missed_facts": [m["fact"] for m in missed],
                        "facts_found": ms.raw_detail.get("facts_found", 0),
                        "total_facts": ms.raw_detail.get("total_facts", 0),
                    }
                )

    def avg(lst: list[float]) -> float:
        return statistics.mean(lst) if lst else 0.0

    metrics = ["fact_inclusion", "tone_fidelity", "conciseness"]
    a_avgs = {m: avg(agg["model_a"].get(m, [])) for m in metrics}
    b_avgs = {m: avg(agg["model_b"].get(m, [])) for m in metrics}
    a_overall = avg(overall["model_a"])
    b_overall = avg(overall["model_b"])

    winner_overall = "model_a" if a_overall >= b_overall else "model_b"
    loser_overall = "model_b" if winner_overall == "model_a" else "model_a"

    loser_fact_details = fact_drop_detail.get(loser_overall, [])
    total_loser_facts_missed = sum(len(d["missed_facts"]) for d in loser_fact_details)
    total_loser_scenarios = len(loser_fact_details) if loser_fact_details else 1
    avg_facts_dropped = total_loser_facts_missed / total_loser_scenarios
    all_missed = [f for d in loser_fact_details for f in d["missed_facts"]]
    scenarios_with_miss = sum(1 for d in loser_fact_details if d["missed_facts"])

    setup_note = (
        f"Both models were evaluated using the same prompt (Strategy A: CoT+Role+Few-Shot). "
        f"The judge model was `{settings.judge_model_name}`."
    )

    report_lines = [
        "# Email Generation — Comparative Evaluation Report",
        "",
        f"> **Comparison type**: Model A vs Model B — same prompt (Strategy A).",
        f"> {setup_note}",
        "",
        "---",
        "",
        "## 1. Summary Results",
        "",
        "| Model | Fact Inclusion | Tone Fidelity | Conciseness | **Overall** |",
        "|---|---|---|---|---|",
        f"| {_model_label('model_a', settings)} | {a_avgs['fact_inclusion']:.3f} | {a_avgs['tone_fidelity']:.3f} | {a_avgs['conciseness']:.3f} | **{a_overall:.3f}** |",
        f"| {_model_label('model_b', settings)} | {b_avgs['fact_inclusion']:.3f} | {b_avgs['tone_fidelity']:.3f} | {b_avgs['conciseness']:.3f} | **{b_overall:.3f}** |",
        "",
        "---",
        "",
        "## 2. Per-Metric Analysis",
        "",
        "### 2.1 Fact Inclusion Rate",
        "",
        f"- **{_model_label('model_a', settings)}**: `{a_avgs['fact_inclusion']:.3f}`",
        f"- **{_model_label('model_b', settings)}**: `{b_avgs['fact_inclusion']:.3f}`",
        f"- **Winner**: {'Model A' if a_avgs['fact_inclusion'] >= b_avgs['fact_inclusion'] else 'Model B'}",
        "",
        "_Method: Hybrid LLM-extractor with deterministic quote-grounding guard. "
        "Each fact is checked independently; model claims without grounding are downgraded to misses._",
        "",
        "### 2.2 Tone Fidelity Score",
        "",
        f"- **{_model_label('model_a', settings)}**: `{a_avgs['tone_fidelity']:.3f}`",
        f"- **{_model_label('model_b', settings)}**: `{b_avgs['tone_fidelity']:.3f}`",
        f"- **Winner**: {'Model A' if a_avgs['tone_fidelity'] >= b_avgs['tone_fidelity'] else 'Model B'}",
        f"- Low judge reliability flags (std dev > {_LOW_RELIABILITY_THRESHOLD}): "
        f"Model A: {low_reliability_counts['model_a']}, "
        f"Model B: {low_reliability_counts['model_b']}",
        "",
        f"_Method: LLM-as-judge (using `{settings.judge_model_name}`), {settings.judge_samples} samples per evaluation. "
        "Mean/std-dev computed per scenario. High std-dev flags indicate unstable scoring._",
        "",
        "### 2.3 Structural Conciseness / Signal Density",
        "",
        f"- **{_model_label('model_a', settings)}**: `{a_avgs['conciseness']:.3f}`",
        f"- **{_model_label('model_b', settings)}**: `{b_avgs['conciseness']:.3f}`",
        f"- **Winner**: {'Model A' if a_avgs['conciseness'] >= b_avgs['conciseness'] else 'Model B'}",
        "",
        "_Method: Pure-Python deterministic. Formula: `0.5 × lexical_density + 0.5 × length_penalty`. "
        "Zero API calls. Fully reproducible._",
        "",
        "---",
        "",
        "## 3. Failure Mode Analysis",
        "",
        f"The lower-performing model overall is **{_model_label(loser_overall, settings)}** "
        f"(overall score: `{b_overall if loser_overall == 'model_b' else a_overall:.3f}`).",
        "",
        f"**Fact Inclusion failure pattern**: "
        f"{_model_label(loser_overall, settings)} dropped an average of "
        f"**{avg_facts_dropped:.1f} key facts per scenario**. "
        f"Out of {total_loser_scenarios} scenarios evaluated, "
        f"**{scenarios_with_miss}** had at least one missed fact "
        f"(total facts missed: {total_loser_facts_missed}).",
    ]

    if all_missed:
        report_lines += [
            "",
            "Sample of missed facts (from raw_detail audit trail):",
            "",
        ]
        for fact in all_missed[:5]:
            report_lines.append(f"- `{fact[:100]}`")
        if len(all_missed) > 5:
            report_lines.append(
                f"- _(and {len(all_missed) - 5} more — see `raw_results.json` for full detail)_"
            )

    if fallback_counts[loser_overall] > 0:
        report_lines += [
            "",
            f"**CoT parse fallback**: {_model_label(loser_overall, settings)} triggered `parse_fallback_used=True` "
            f"in {fallback_counts[loser_overall]} case(s), meaning the model omitted the `===EMAIL===` delimiter.",
        ]

    report_lines += [
        "",
        "---",
        "",
        "## 4. Production Recommendation",
        "",
        f"**Recommended model: {_model_label(winner_overall, settings)}**",
        "",
        f"Based on the evaluation results, {_model_label(winner_overall, settings)} outperforms on "
        f"overall score (`{a_overall:.3f}` vs `{b_overall:.3f}`), "
        f"fact inclusion (`{a_avgs['fact_inclusion']:.3f}` vs `{b_avgs['fact_inclusion']:.3f}`), "
        f"and tone fidelity (`{a_avgs['tone_fidelity']:.3f}` vs `{b_avgs['tone_fidelity']:.3f}`).",
        "",
        "**Caveat — Self-Preference Bias**: The model acting as the judge for Tone Fidelity is "
        f"`{settings.judge_model_name}`. If this is one of the tested models, it may exhibit "
        "self-preference bias in the Tone Fidelity metric. Keep this in mind when interpreting "
        "narrow margins.",
        "",
        "---",
        "",
        "## 5. Evaluation Methodology Notes",
        "",
        "- **Metric 1 (Fact Inclusion)**: Hybrid — LLM extraction per fact + deterministic quote-grounding guard.",
        "- **Metric 2 (Tone Fidelity)**: LLM-as-judge, multi-sampled (N=3 by default). "
        "  Std-dev > 0.75 flagged as low reliability.",
        "- **Metric 3 (Conciseness)**: Pure Python, zero API calls, fully deterministic.",
        "- **Overall score**: Unweighted mean of the 3 normalised (0.0–1.0) metric scores.",
        "",
        "_Full audit trail in `results/raw_results.json`. Flat CSV in `results/raw_results.csv`._",
    ]

    report_text = "\n".join(report_lines)
    with _REPORT_MD.open("w", encoding="utf-8") as fh:
        fh.write(report_text)

    logger.info("Comparative report written to %s", _REPORT_MD)
