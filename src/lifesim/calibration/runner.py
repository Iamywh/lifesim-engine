from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import AgentState
from lifesim.canonical import build_canonical_transitions
from lifesim.config import LifeSimConfig, load_config
from lifesim.events.model import EventHistory
from lifesim.rng import create_rng
from lifesim.weekly import WeeklyContext, WeeklyPipeline

DEFAULT_CHECKPOINTS = (12, 26, 52, 156)
CALIBRATION_SEEDS = tuple(range(200))
HOLDOUT_SEEDS = tuple(range(1000, 1050))


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    label: str
    seeds: tuple[int, ...]
    duration_weeks: int
    checkpoints: tuple[int, ...]
    run_count: int
    failure_count: int
    metrics: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seeds": list(self.seeds),
            "duration_weeks": self.duration_weeks,
            "checkpoints": list(self.checkpoints),
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
        }


def run_calibration(
    *,
    label: str,
    config_path: Path = Path("configs/canonical/maya_v1.toml"),
    agent_scenario_path: Path = Path("configs/scenarios/maya_start.toml"),
    seeds: tuple[int, ...] = CALIBRATION_SEEDS,
    duration_weeks: int = 156,
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
    **catalog_paths: Path,
) -> CalibrationResult:
    base_config = load_config(config_path)
    valid_checkpoints = tuple(week for week in checkpoints if week <= duration_weeks)
    summaries: list[dict[str, Any]] = []
    failures = 0
    for seed in seeds:
        config = _with_seed_and_duration(base_config, seed, duration_weeks)
        agent = load_agent_state(agent_scenario_path)
        pipeline = WeeklyPipeline(build_canonical_transitions(**catalog_paths))
        try:
            summaries.append(_run_summary(config, agent, pipeline, valid_checkpoints))
        except (RuntimeError, TypeError, ValueError):
            failures += 1
    metrics = _aggregate_metrics(summaries, valid_checkpoints)
    warnings = _warnings(metrics, failures)
    return CalibrationResult(
        label=label,
        seeds=tuple(seeds),
        duration_weeks=duration_weeks,
        checkpoints=valid_checkpoints,
        run_count=len(summaries),
        failure_count=failures,
        metrics=metrics,
        warnings=warnings,
    )


def write_markdown_report(
    calibration: CalibrationResult,
    holdout: CalibrationResult | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M12 Calibration Report",
        "",
        "M12 adds explicit financial charges, canonical engine construction, and a deterministic calibration harness.",
        "",
        _section(calibration),
    ]
    if holdout is not None:
        lines.extend(["", _section(holdout)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _section(result: CalibrationResult) -> str:
    metrics = result.metrics
    warnings = ", ".join(result.warnings) if result.warnings else "none"
    checkpoint_lines = [
        (
            f"- week {week}: avg liquid {values['avg_liquid']:.2f}, "
            f"avg arrears {values['avg_arrears']:.2f}, employed {values['employment_rate']:.2f}, "
            f"avg education progress {values['avg_education_progress']:.2f}, "
            f"avg connections {values['avg_connections']:.2f}"
        )
        for week, values in metrics["checkpoints"].items()
    ]
    return "\n".join(
        [
            f"## {result.label}",
            "",
            f"- runs: {result.run_count}",
            f"- failures: {result.failure_count}",
            f"- warnings: {warnings}",
            f"- avg events per run: {metrics['avg_events_per_run']:.2f}",
            f"- avg financial charges per run: {metrics['avg_financial_charges_per_run']:.2f}",
            f"- avg unpaid charge amount per run: {metrics['avg_unpaid_charge_amount_per_run']:.2f}",
            "",
            "### Checkpoints",
            *checkpoint_lines,
        ]
    )


def _with_seed_and_duration(config: LifeSimConfig, seed: int, duration_weeks: int) -> LifeSimConfig:
    return replace(
        config,
        simulation=replace(
            config.simulation,
            seed=seed,
            duration_weeks=duration_weeks,
        ),
    )


def _run_summary(
    config: LifeSimConfig,
    initial_agent: AgentState,
    pipeline: WeeklyPipeline,
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    rng = create_rng(config.simulation.seed)
    agent = initial_agent
    checkpoint_summaries = {0: _agent_checkpoint(agent)} if 0 in checkpoints else {}
    event_history: Any = EventHistory()
    decision_history = None
    consequence_runtime = None
    learning_runtime = None
    passive_runtime = None
    employment_runtime = None
    development_runtime = None
    social_runtime = None
    adaptation_runtime = None
    for week in range(1, config.simulation.duration_weeks + 1):
        context = WeeklyContext(
            week=week,
            config=config,
            rng=rng,
            event_history=event_history,
            decision_history=decision_history,
            consequence_runtime=consequence_runtime,
            learning_runtime=learning_runtime,
            passive_runtime=passive_runtime,
            employment_runtime=employment_runtime,
            development_runtime=development_runtime,
            social_runtime=social_runtime,
            adaptation_runtime=adaptation_runtime,
        )
        result = pipeline.advance(agent, context)
        agent = result.agent_state
        if result.event_history is not None:
            event_history = result.event_history
        if result.decision_history is not None:
            decision_history = result.decision_history
        if result.consequence_runtime is not None:
            consequence_runtime = result.consequence_runtime
        if result.learning_runtime is not None:
            learning_runtime = result.learning_runtime
        if result.passive_runtime is not None:
            passive_runtime = result.passive_runtime
        if result.employment_runtime is not None:
            employment_runtime = result.employment_runtime
        if result.development_runtime is not None:
            development_runtime = result.development_runtime
        if result.social_runtime is not None:
            social_runtime = result.social_runtime
        if result.adaptation_runtime is not None:
            adaptation_runtime = result.adaptation_runtime
        if week in checkpoints:
            checkpoint_summaries[week] = _agent_checkpoint(agent)

    records = consequence_runtime.history.records if consequence_runtime is not None else ()
    charge_applications = [
        application
        for record in records
        for application in record.financial_charge_applications
    ]
    return {
        "checkpoints": checkpoint_summaries,
        "event_count": (
            len(event_history.occurrences) if event_history is not None else 0
        ),
        "charge_count": len(charge_applications),
        "unpaid_charge_amount": sum(
            (application.amount_unpaid for application in charge_applications),
            Decimal("0.00"),
        ),
    }


def _aggregate_metrics(
    summaries: list[dict[str, Any]],
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    checkpoint_metrics = {
        week: _checkpoint_metrics(summaries, week)
        for week in checkpoints
        if any(week in summary["checkpoints"] for summary in summaries)
    }
    event_counts = [summary["event_count"] for summary in summaries]
    charge_counts = [summary["charge_count"] for summary in summaries]
    unpaid_totals = [summary["unpaid_charge_amount"] for summary in summaries]
    return {
        "checkpoints": checkpoint_metrics,
        "avg_events_per_run": _avg(event_counts),
        "avg_financial_charges_per_run": _avg(charge_counts),
        "avg_unpaid_charge_amount_per_run": _avg_decimal(unpaid_totals),
    }


def _agent_checkpoint(agent: Any) -> dict[str, Decimal | float | int]:
    return {
        "liquid": (
            agent.financial.cash
            + agent.financial.bank_balance
            + agent.financial.savings
            + agent.financial.emergency_fund
        ),
        "arrears": sum((arrear.balance for arrear in agent.financial.arrears), Decimal("0.00")),
        "employed": 1.0 if agent.employment.status in {"employed", "student_part_time"} else 0.0,
        "education_progress": agent.education.progress,
        "connections": len(agent.social.connections),
    }


def _checkpoint_metrics(summaries: list[dict[str, Any]], week: int) -> dict[str, float]:
    snapshots = [
        summary["checkpoints"][week]
        for summary in summaries
        if week in summary["checkpoints"]
    ]
    return {
        "avg_liquid": _avg_decimal([snapshot["liquid"] for snapshot in snapshots]),
        "avg_arrears": _avg_decimal([snapshot["arrears"] for snapshot in snapshots]),
        "employment_rate": _avg([snapshot["employed"] for snapshot in snapshots]),
        "avg_education_progress": _avg([
            snapshot["education_progress"] for snapshot in snapshots
        ]),
        "avg_connections": _avg([snapshot["connections"] for snapshot in snapshots]),
    }


def _warnings(metrics: dict[str, Any], failures: int) -> tuple[str, ...]:
    warnings: list[str] = []
    if failures:
        warnings.append("RUN_FAILURES")
    events = metrics["avg_events_per_run"]
    if events < 5.0:
        warnings.append("EVENT_DOMINANCE")
    if metrics["avg_financial_charges_per_run"] < 1.0:
        warnings.append("NO_FINANCIAL_PRESSURE")
    if metrics["avg_unpaid_charge_amount_per_run"] > 300.0:
        warnings.append("UNIVERSAL_ARREARS")
    final = metrics["checkpoints"].get(156)
    if final is not None:
        if final["avg_education_progress"] >= 99.0:
            warnings.append("UNIVERSAL_GRADUATION")
        if final["avg_education_progress"] <= 1.0:
            warnings.append("NEAR_ZERO_GRADUATION")
        if final["employment_rate"] >= 0.98:
            warnings.append("EMPLOYMENT_TOO_EASY")
        if final["employment_rate"] <= 0.05:
            warnings.append("EMPLOYMENT_TOO_HARD")
    return tuple(warnings)


def _avg(values: list[float | int]) -> float:
    return float(mean(values)) if values else 0.0


def _avg_decimal(values: list[Decimal]) -> float:
    return float(sum(values, Decimal("0.00")) / Decimal(len(values))) if values else 0.0
