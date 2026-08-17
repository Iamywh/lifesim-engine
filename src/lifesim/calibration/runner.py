from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
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
ROUTINE_PROFILES = (
    "balanced_week",
    "austerity_home_week",
    "low_cost_active_week",
    "recovery_focus_week",
    "social_week",
)
HEALTH_MENTAL_FIELDS = (
    "health.energy",
    "health.physical_health",
    "mental.stress",
    "mental.mental_load",
    "mental.recovery_need",
    "mental.mood",
    "mental.loneliness",
    "health.sleep_debt",
)
BOUNDED_FIELDS = tuple(field for field in HEALTH_MENTAL_FIELDS if field != "health.sleep_debt")
MONEY_ACCOUNTS = ("cash", "bank_balance", "savings", "emergency_fund")
PERSONALITY_WEEKLY_CAP = 0.0015 + 1e-12
PERSONALITY_ANCHOR_CAP = 0.12 + 1e-12


class HardInvariantError(RuntimeError):
    """Raised when a completed simulation step violates calibration invariants."""


@dataclass(frozen=True, slots=True)
class CalibrationRunRecord:
    seed: int
    completed: bool
    failure_week: int | None
    failure_type: str
    failure_message: str
    checkpoints: dict[int, dict[str, Any]]
    metrics: dict[str, Any]
    hard_invariant_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "completed": self.completed,
            "failure_week": self.failure_week,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "checkpoints": {
                str(week): _json_value(values) for week, values in self.checkpoints.items()
            },
            "metrics": _json_value(self.metrics),
            "hard_invariant_failures": list(self.hard_invariant_failures),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    label: str
    seeds: tuple[int, ...]
    duration_weeks: int
    checkpoints: tuple[int, ...]
    run_records: tuple[CalibrationRunRecord, ...]
    run_count: int
    successful_run_count: int
    failure_count: int
    hard_invariant_failure_count: int
    metrics: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seeds": list(self.seeds),
            "duration_weeks": self.duration_weeks,
            "checkpoints": list(self.checkpoints),
            "run_count": self.run_count,
            "successful_run_count": self.successful_run_count,
            "failure_count": self.failure_count,
            "hard_invariant_failure_count": self.hard_invariant_failure_count,
            "metrics": _json_value(self.metrics),
            "warnings": list(self.warnings),
            "run_records": [record.to_dict() for record in self.run_records],
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
    pipeline = WeeklyPipeline(build_canonical_transitions(**catalog_paths))
    records = tuple(
        _run_record(
            config=_with_seed_and_duration(base_config, seed, duration_weeks),
            agent=load_agent_state(agent_scenario_path),
            pipeline=pipeline,
            checkpoints=valid_checkpoints,
        )
        for seed in seeds
    )
    metrics = aggregate_run_records(records, valid_checkpoints, duration_weeks)
    warnings = diagnose_warnings(metrics)
    return CalibrationResult(
        label=label,
        seeds=tuple(seeds),
        duration_weeks=duration_weeks,
        checkpoints=valid_checkpoints,
        run_records=records,
        run_count=len(seeds),
        successful_run_count=sum(1 for record in records if record.completed),
        failure_count=sum(1 for record in records if not record.completed),
        hard_invariant_failure_count=sum(
            1 for record in records if record.hard_invariant_failures
        ),
        metrics=metrics,
        warnings=warnings,
    )


def aggregate_run_records(
    records: tuple[CalibrationRunRecord, ...],
    checkpoints: tuple[int, ...],
    duration_weeks: int,
) -> dict[str, Any]:
    successes = tuple(record for record in records if record.completed)
    failures = tuple(record for record in records if not record.completed)
    finance = _finance_aggregates(successes, checkpoints)
    employment = _employment_aggregates(successes, checkpoints)
    education = _education_aggregates(successes, checkpoints)
    health = _health_aggregates(successes, duration_weeks)
    social = _social_aggregates(successes, duration_weeks)
    routine = _routine_aggregates(successes)
    events = _event_aggregates(successes, duration_weeks)
    adaptation = _adaptation_aggregates(successes)
    return {
        "requested_runs": len(records),
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "failure_types": dict(sorted(Counter(record.failure_type for record in failures).items())),
        "failure_weeks": dict(
            sorted(Counter(record.failure_week for record in failures).items())
        ),
        "hard_invariant_failures": sum(
            len(record.hard_invariant_failures) for record in records
        ),
        "finance": finance,
        "employment": employment,
        "education": education,
        "health_mental": health,
        "social": social,
        "routine": routine,
        "events": events,
        "adaptation": adaptation,
    }


def diagnose_warnings(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    warnings: list[str] = []
    if metrics["failed_runs"]:
        warnings.append("RUN_FAILURES")
    if metrics["hard_invariant_failures"]:
        warnings.append("HARD_INVARIANT_FAILURES")

    events = metrics["events"]
    if events["dominant_event_share"] >= 0.35 or events["dominant_category_share"] >= 0.55:
        warnings.append("EVENT_DOMINANCE")
    if events["event_week_rate"] < 0.25:
        warnings.append("EVENTS_TOO_RARE")
    if events["event_week_rate"] > 0.70:
        warnings.append("EVENTS_TOO_FREQUENT")

    if metrics["routine"]["dominant_share"] >= 0.75:
        warnings.append("ROUTINE_LOCK_IN")
    if metrics["education"]["dominant_development_profile_share"] >= 0.75:
        warnings.append("DEVELOPMENT_LOCK_IN")

    employment = metrics["employment"]
    ever_26 = employment["ever_employed_rate_by_checkpoint"].get("26", 0.0)
    ever_52 = employment["ever_employed_rate_by_checkpoint"].get("52", 0.0)
    first_employment = employment["first_employment_week_distribution"]
    if ever_26 >= 0.95 and first_employment.get("p50", 999.0) <= 6.0:
        warnings.append("EMPLOYMENT_TOO_EASY")
    if ever_52 <= 0.35:
        warnings.append("EMPLOYMENT_TOO_HARD")

    education = metrics["education"]
    graduation_156 = education["graduation_rate_by_checkpoint"].get("156", 0.0)
    if graduation_156 >= 0.98:
        warnings.append("UNIVERSAL_GRADUATION")
    if graduation_156 < 0.45:
        warnings.append("EDUCATION_TOO_SLOW")

    finance = metrics["finance"]
    if finance["arrear_incidence_rate"] >= 0.95:
        warnings.append("UNIVERSAL_ARREARS")
    if finance["charge_count_distribution"].get("p50", 0.0) < 1.0:
        warnings.append("NO_FINANCIAL_PRESSURE")
    if finance["final_liquid_distribution"].get("p95", 0.0) > 12000.0:
        warnings.append("RESOURCE_EXPLOSION")

    if metrics["health_mental"]["boundary_saturation_rate"] > 0.10:
        warnings.append("STATE_BOUNDARY_SATURATION")
    social = metrics["social"]
    if (
        social["choice_shares"].get("keep_social_light", 0.0) >= 0.85
        or social["dominant_contact_share"] >= 0.65
    ):
        warnings.append("SOCIAL_LOCK_IN")
    if (
        metrics["adaptation"]["final_habit_strength_distribution"].get("p95", 0.0) >= 95.0
        or (
            metrics["adaptation"]["final_habit_strength_distribution"].get("p95", 0.0) >= 75.0
            and metrics["adaptation"]["max_habit_familiarity_distribution"].get("p95", 0.0) >= 0.17
        )
    ):
        warnings.append("HABIT_LOCK_IN")
    if (
        metrics["adaptation"]["max_weekly_personality_delta_distribution"].get("p95", 0.0)
        > 0.0015
        or metrics["adaptation"]["max_anchor_displacement_distribution"].get("p95", 0.0)
        > 0.12
    ):
        warnings.append("PERSONALITY_DRIFT_HIGH")
    return tuple(warnings)


def write_markdown_report(
    calibration: CalibrationResult,
    holdout: CalibrationResult | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M12 Calibration Report",
        "",
        "## Goals",
        "",
        "M12 validates the canonical Maya v1 simulator after adding exact financial charges, richer world events, canonical engine construction, and deterministic calibration diagnostics.",
        "",
        "## Baseline M11 Failure",
        "",
        "The post-M11 architecture could fail when M4 affordability reasoned over total liquid resources while an M5 direct monetary effect targeted only one account. Perceived and actual costs could also disagree. The reproducible targeted baseline is `financial.bank_balance` underflow from a direct monetary delta even when other liquid accounts exist.",
        "",
        "A full old-code cohort was not run from this branch; keeping that isolated historical run would require checking out pre-M12 code and catalogs outside the runtime package. The targeted regression is retained in `tests/test_consequences.py::test_monetary_underflow_fails_without_partial_mutation`.",
        "",
        "## Structural Financial Fix",
        "",
        "M12 keeps strict direct monetary effects for legacy/state-delta semantics, but ordinary event costs now use explicit `FinancialChargeDefinition` records. Charges settle exact `Decimal` amounts across a declared funding order, audit every transfer, and use either `require_full` or `arrear` shortfall policy. Scheduled financial charges preserve decision/event/outcome provenance and execute exactly once.",
        "",
        "## Event Additions",
        "",
        "The starter world expands from 5 to 12 events with modest city, finance, social, health, housing, technology, bureaucracy, education, and refund opportunities. The global world event probability remains 0.45 and max events per week remains 1.",
        "",
        "## Parameter Change Log",
        "",
        "- `pay_for_faster_transport` estimated cost: 14.00 -> 16.00, aligned with actual charge.",
        "- `handle_immediately` estimated cost: 22.00 -> 24.00, aligned with actual charge.",
        "- `accept_invitation` estimated cost: 18.00 -> 20.00, aligned with actual charge.",
        "- Direct event-cost bank deltas for transport, small expenses, and social invitations were replaced by explicit financial charges.",
        "- Added behavior tags to starter event options for M11 evidence.",
        "- Added 7 starter events: phone/device problem, household maintenance issue, free local activity, bureaucratic errand, minor health setback, university admin deadline, and small refund opportunity.",
        "- No M8 employment, M7 routine, M9 development, M10 social, M11 adaptation rates, or global M3 event probability were tuned in this hardening pass.",
        "",
        _result_section("200-Seed Calibration Summary", calibration),
    ]
    if holdout is not None:
        lines.extend(["", _result_section("50-Seed Holdout Comparison", holdout)])
    lines.extend(["", _warnings_section(calibration, holdout)])
    lines.extend(
        [
            "",
            "## Canonical Seed-42 Validation",
            "",
            "Canonical input paths are frozen below. Validation requires 12/26/52-week runs to match the corresponding prefixes of the 156-week run, and two 156-week runs with seed 42 to produce identical complete JSON.",
            "",
            "## Known Limitations",
            "",
            "- Indefinite starter jobs can make current week-156 employment sticky after acquisition. Employment warnings therefore use acquisition funnel timing/rates rather than current employment alone.",
            "- Calibration diagnostics are broad warnings, not automatic tuning instructions.",
            "- No layoffs, career ladders, relationship life-cycle arcs, or long-term macroeconomic systems are implemented in M12.",
            "",
            "## Frozen Canonical Input Paths",
            "",
            "- `configs/canonical/maya_v1.toml`",
            "- `configs/scenarios/maya_start.toml`",
            "- `configs/events/starter.toml`",
            "- `configs/consequences/starter.toml`",
            "- `configs/routines/starter.toml`",
            "- `configs/employment/starter.toml`",
            "- `configs/development/starter.toml`",
            "- `configs/social/starter.toml`",
            "- `configs/adaptation/starter.toml`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result_section(title: str, result: CalibrationResult) -> str:
    metrics = result.metrics
    finance = metrics["finance"]
    employment = metrics["employment"]
    education = metrics["education"]
    health = metrics["health_mental"]
    social = metrics["social"]
    routine = metrics["routine"]
    events = metrics["events"]
    adaptation = metrics["adaptation"]
    warnings = ", ".join(result.warnings) if result.warnings else "none"
    return "\n".join(
        [
            f"## {title}",
            "",
            f"- requested runs: {result.run_count}",
            f"- successful runs: {result.successful_run_count}",
            f"- failures: {result.failure_count}",
            f"- hard invariant failures: {result.hard_invariant_failure_count}",
            f"- warnings: {warnings}",
            "",
            "### Finance Distributions",
            _stats_line("final liquid", finance["final_liquid_distribution"], precision=2),
            _stats_line("maximum arrears", finance["max_arrear_distribution"], precision=2),
            _stats_line("final arrears", finance["final_arrear_distribution"], precision=2),
            _stats_line("final debt", finance["final_debt_distribution"], precision=2),
            f"- arrear incidence rate: {finance['arrear_incidence_rate']:.3f}",
            f"- arrear recovery rate: {finance['arrear_recovery_rate']:.3f}",
            f"- liquid p05/p50/p95 by checkpoint: {_checkpoint_triplets(finance['liquid_by_checkpoint'])}",
            "",
            "### Employment Funnel",
            f"- ever employed by checkpoint: {employment['ever_employed_rate_by_checkpoint']}",
            f"- current employed by checkpoint: {employment['current_employed_rate_by_checkpoint']}",
            f"- submissions/skips/invitations/attended/offers/accepted: {employment['application_submission_count']} / {employment['application_skip_count']} / {employment['interview_invitation_count']} / {employment['interview_attended_count']} / {employment['offer_count']} / {employment['accepted_offer_count']}",
            _stats_line("first employment week", employment["first_employment_week_distribution"], precision=2),
            f"- application submission -> interview invitation rate: {employment['application_to_interview_rate']:.3f}",
            f"- interview attended -> offer rate: {employment['interview_to_offer_rate']:.3f}",
            f"- offer -> acceptance rate: {employment['offer_acceptance_rate']:.3f}",
            "",
            "### Education / Development",
            f"- progress p05/p50/p95 by checkpoint: {_checkpoint_triplets(education['progress_by_checkpoint'])}",
            f"- graduation rate by checkpoint: {education['graduation_rate_by_checkpoint']}",
            _stats_line("graduation week", education["graduation_week_distribution"], precision=2),
            f"- development profile shares: {education['development_profile_shares']}",
            _stats_line("final efficiency", education["final_efficiency_distribution"], precision=4),
            f"- efficiency factors: {_efficiency_factor_summary(education['efficiency_factor_distributions'])}",
            "",
            "### Health / Mental Saturation",
            f"- boundary saturation rate: {health['boundary_saturation_rate']:.4f}",
            f"- saturated fields: {health['saturated_fields']}",
            f"- boundary direction by field: {_boundary_direction_summary(health['fields'])}",
            "",
            "### Social",
            f"- focal opportunity rate: {social['focal_opportunity_rate']:.3f}",
            f"- no-opportunity rate: {social['no_opportunity_rate']:.3f}",
            f"- new connection incidence rate: {social['new_connection_incidence_rate']:.3f}",
            _stats_line("final connections", social["final_connection_distribution"], precision=2),
            f"- choice shares: {social['choice_shares']}",
            f"- dominant contact/share: {social['dominant_contact_id']} / {social['dominant_contact_share']:.3f}",
            f"- social outcome counts: {social['outcome_counts']}",
            "",
            "### Routine",
            f"- profile shares: {routine['profile_shares']}",
            f"- dominant profile/share: {routine['dominant_profile']} / {routine['dominant_share']:.3f}",
            "",
            "### Events",
            f"- event-week rate: {events['event_week_rate']:.3f}",
            f"- dominant event/share: {events['dominant_event_id']} / {events['dominant_event_share']:.3f}",
            f"- dominant category/share: {events['dominant_category']} / {events['dominant_category_share']:.3f}",
            "",
            "### Adaptation",
            _stats_line("habits formed", adaptation["habits_formed_distribution"], precision=2),
            _stats_line("final habit strength", adaptation["final_habit_strength_distribution"], precision=2),
            _stats_line("max habit familiarity", adaptation["max_habit_familiarity_distribution"], precision=4),
            _stats_line("max weekly personality delta", adaptation["max_weekly_personality_delta_distribution"], precision=6),
            _stats_line("max anchor displacement", adaptation["max_anchor_displacement_distribution"], precision=6),
        ]
    )


def _warnings_section(
    calibration: CalibrationResult,
    holdout: CalibrationResult | None,
) -> str:
    calibration_warnings = ", ".join(calibration.warnings) if calibration.warnings else "none"
    lines = [
        "## Remaining Warnings",
        "",
        f"- calibration: {calibration_warnings}",
    ]
    if holdout is not None:
        holdout_warnings = ", ".join(holdout.warnings) if holdout.warnings else "none"
        lines.append(f"- holdout: {holdout_warnings}")
    lines.extend(
        [
            "- warnings are diagnostics only; no M8 employment, M7 routine, M9 development, M10 social, or M11 adaptation tuning was performed in M12.",
        ]
    )
    return "\n".join(lines)


def _run_record(
    *,
    config: LifeSimConfig,
    agent: AgentState,
    pipeline: WeeklyPipeline,
    checkpoints: tuple[int, ...],
) -> CalibrationRunRecord:
    seed = config.simulation.seed
    tracker = _Tracker(agent, checkpoints)
    last_completed_week = 0
    event_history: Any = EventHistory()
    decision_history = None
    consequence_runtime = None
    learning_runtime = None
    passive_runtime = None
    employment_runtime = None
    development_runtime = None
    social_runtime = None
    adaptation_runtime = None
    rng = create_rng(seed)
    try:
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
            event_history = result.event_history if result.event_history is not None else event_history
            decision_history = (
                result.decision_history if result.decision_history is not None else decision_history
            )
            consequence_runtime = (
                result.consequence_runtime
                if result.consequence_runtime is not None
                else consequence_runtime
            )
            learning_runtime = (
                result.learning_runtime if result.learning_runtime is not None else learning_runtime
            )
            passive_runtime = (
                result.passive_runtime if result.passive_runtime is not None else passive_runtime
            )
            employment_runtime = (
                result.employment_runtime
                if result.employment_runtime is not None
                else employment_runtime
            )
            development_runtime = (
                result.development_runtime
                if result.development_runtime is not None
                else development_runtime
            )
            social_runtime = (
                result.social_runtime if result.social_runtime is not None else social_runtime
            )
            adaptation_runtime = (
                result.adaptation_runtime
                if result.adaptation_runtime is not None
                else adaptation_runtime
            )
            tracker.observe_week(
                week=week,
                agent=agent,
                result=result,
                consequence_runtime=consequence_runtime,
                adaptation_runtime=adaptation_runtime,
            )
            last_completed_week = week
    except (RuntimeError, TypeError, ValueError) as error:
        return CalibrationRunRecord(
            seed=seed,
            completed=False,
            failure_week=last_completed_week + 1,
            failure_type=type(error).__name__,
            failure_message=str(error),
            checkpoints=tracker.checkpoints,
            metrics=tracker.final_metrics(agent),
            hard_invariant_failures=(
                (str(error),) if isinstance(error, HardInvariantError) else ()
            ),
        )

    try:
        tracker.validate_final(
            config.simulation.duration_weeks,
            consequence_runtime,
        )
    except HardInvariantError as error:
        return CalibrationRunRecord(
            seed=seed,
            completed=False,
            failure_week=config.simulation.duration_weeks,
            failure_type=type(error).__name__,
            failure_message=str(error),
            checkpoints=tracker.checkpoints,
            metrics=tracker.final_metrics(agent),
            hard_invariant_failures=(str(error),),
        )

    return CalibrationRunRecord(
        seed=seed,
        completed=True,
        failure_week=None,
        failure_type="",
        failure_message="",
        checkpoints=tracker.checkpoints,
        metrics=tracker.final_metrics(agent),
        hard_invariant_failures=(),
    )


class _Tracker:
    def __init__(self, agent: AgentState, checkpoints: tuple[int, ...]) -> None:
        self._checkpoints = set(checkpoints)
        self.checkpoints: dict[int, dict[str, Any]] = {}
        self.initial_liquid = _liquid(agent)
        self.initial_debt = _debt_balance(agent)
        self.minimum_liquid = self.initial_liquid
        self.maximum_arrears = Decimal("0.00")
        self.weeks_with_arrears = 0
        self.first_arrear_week: int | None = None
        self._ever_arrear_ids: set[str] = set()
        self._final_arrear_ids: set[str] = set()
        self.arrears_created = 0
        self.charge_count = 0
        self.charge_due = Decimal("0.00")
        self.charge_paid = Decimal("0.00")
        self.charge_unpaid = Decimal("0.00")
        self.recurring_income_received = Decimal("0.00")
        self.employment_wage_income = Decimal("0.00")
        self.routine_spending = Decimal("0.00")
        self.first_job_discovery_week: int | None = None
        self.first_application_week: int | None = None
        self.first_interview_week: int | None = None
        self.first_offer_week: int | None = None
        self.first_employment_start_week: int | None = None
        self.total_employed_weeks = 0
        self.jobs_held: set[str] = set()
        self.contract_endings = 0
        self.application_submissions = 0
        self.application_skips = 0
        self.interview_invitations = 0
        self.interviews_attended = 0
        self.interviews_declined = 0
        self.offers_produced = 0
        self.offers_accepted = 0
        self.offers_declined = 0
        self.development_profiles: Counter[str] = Counter()
        self.effective_study_hours = 0.0
        self.effective_practice_hours = 0.0
        self.final_efficiencies: list[float] = []
        self.energy_factors: list[float] = []
        self.stress_factors: list[float] = []
        self.mental_load_factors: list[float] = []
        self.recovery_factors: list[float] = []
        self.workload_factors: list[float] = []
        self.skill_level_deltas: dict[str, float] = {}
        self.new_skills_created = 0
        self.graduation_week: int | None = None
        self.health_mental = {
            field: {
                "min": _path_value(agent, field),
                "max": _path_value(agent, field),
                "weeks_at_0": 0,
                "weeks_at_100": 0,
            }
            for field in HEALTH_MENTAL_FIELDS
        }
        self.starting_connections = len(agent.social.connections)
        self.final_connections = self.starting_connections
        self.social_focal_opportunity_count = 0
        self.social_no_opportunity_weeks = 0
        self.social_executed_focal_choices = 0
        self.social_choices: Counter[str] = Counter()
        self.social_contact_counts: Counter[str] = Counter()
        self.social_outcomes: Counter[str] = Counter()
        self.support_network_strength_checkpoints: dict[int, float] = {}
        self.routine_profiles: Counter[str] = Counter()
        self.event_weeks = 0
        self.event_free_weeks = 0
        self.event_counts: Counter[str] = Counter()
        self.event_category_counts: Counter[str] = Counter()
        self.option_counts: Counter[str] = Counter()
        self.outcome_counts: Counter[str] = Counter()
        self.habits_formed = 0
        self.final_habit_strengths: list[float] = []
        self.max_habit_familiarity = 0.0
        self.max_weekly_personality_delta = 0.0
        self.max_anchor_displacement = 0.0
        self._provenance_ids: set[str] = set()
        if 0 in self._checkpoints:
            self.checkpoints[0] = _checkpoint(agent)

    def observe_week(
        self,
        *,
        week: int,
        agent: AgentState,
        result: Any,
        consequence_runtime: Any,
        adaptation_runtime: Any,
    ) -> None:
        self.minimum_liquid = min(self.minimum_liquid, _liquid(agent))
        total_arrears = _arrear_balance(agent)
        if total_arrears > Decimal("0.00"):
            self.weeks_with_arrears += 1
            self.first_arrear_week = self.first_arrear_week or week
        self.maximum_arrears = max(self.maximum_arrears, total_arrears)
        current_arrear_ids = {arrear.obligation_id for arrear in agent.financial.arrears}
        new_arrears = current_arrear_ids - self._ever_arrear_ids
        self.arrears_created += len(new_arrears)
        self._ever_arrear_ids.update(new_arrears)
        self._final_arrear_ids = current_arrear_ids
        if _is_currently_employed(agent):
            self.total_employed_weeks += 1
            if self.first_employment_start_week is None:
                self.first_employment_start_week = agent.employment.start_week or week
            if agent.employment.contract_id:
                self.jobs_held.add(agent.employment.contract_id)
        self.final_connections = len(agent.social.connections)
        self._observe_events(result.events)
        self._observe_decisions(result.decisions)
        self._observe_consequences(result.consequences)
        self._observe_passive(result.passive_records)
        self._observe_employment(result.employment_records)
        self._observe_development(result.development_records)
        self._observe_social(result.social_records)
        self._observe_adaptation(result.adaptation_records, adaptation_runtime)
        self._observe_health_mental(agent)
        if week in self._checkpoints:
            self.checkpoints[week] = _checkpoint(agent)
            self.support_network_strength_checkpoints[week] = (
                agent.social.support_network_strength
            )
        _assert_hard_invariants(
            week=week,
            agent=agent,
            consequence_records=result.consequences,
            consequence_runtime=consequence_runtime,
            adaptation_records=result.adaptation_records,
            provenance_ids=self._provenance_ids,
        )

    def validate_final(self, duration_weeks: int, consequence_runtime: Any) -> None:
        if consequence_runtime is None:
            return
        for scheduled in consequence_runtime.pending_scheduled_effects:
            if scheduled.due_week <= duration_weeks:
                raise HardInvariantError("Past-due scheduled effect remains pending.")
        for scheduled in consequence_runtime.pending_scheduled_financial_charges:
            if scheduled.due_week <= duration_weeks:
                raise HardInvariantError("Past-due scheduled financial charge remains pending.")

    def final_metrics(self, agent: AgentState) -> dict[str, Any]:
        self.final_habit_strengths = [habit.strength for habit in agent.habits.items]
        final_debt = _debt_balance(agent)
        final_arrears = _arrear_balance(agent)
        final_skill_levels = {skill.name: skill.level for skill in agent.skills.items}
        return {
            "initial_liquid": self.initial_liquid,
            "minimum_liquid": self.minimum_liquid,
            "final_liquid": _liquid(agent),
            "initial_debt_balance": self.initial_debt,
            "final_debt_balance": final_debt,
            "any_arrear": bool(self._ever_arrear_ids),
            "first_arrear_week": self.first_arrear_week,
            "maximum_total_arrear_balance": self.maximum_arrears,
            "final_total_arrear_balance": final_arrears,
            "weeks_with_arrears": self.weeks_with_arrears,
            "arrears_created": self.arrears_created,
            "arrears_fully_recovered": len(self._ever_arrear_ids - self._final_arrear_ids),
            "charge_count": self.charge_count,
            "charge_amount_due": self.charge_due,
            "charge_amount_paid": self.charge_paid,
            "charge_amount_unpaid": self.charge_unpaid,
            "recurring_income_received": self.recurring_income_received,
            "employment_wage_income": self.employment_wage_income,
            "routine_spending": self.routine_spending,
            "first_job_discovery_week": self.first_job_discovery_week,
            "first_application_week": self.first_application_week,
            "first_interview_week": self.first_interview_week,
            "first_offer_week": self.first_offer_week,
            "first_employment_start_week": self.first_employment_start_week,
            "ever_employed": self.first_employment_start_week is not None
            or self.total_employed_weeks > 0,
            "currently_employed_final": _is_currently_employed(agent),
            "total_employed_weeks": self.total_employed_weeks,
            "jobs_held": len(self.jobs_held),
            "contract_endings": self.contract_endings,
            "application_submissions": self.application_submissions,
            "application_skips": self.application_skips,
            "interview_invitations": self.interview_invitations,
            "interviews_attended": self.interviews_attended,
            "interviews_declined": self.interviews_declined,
            "offers_produced": self.offers_produced,
            "offers_accepted": self.offers_accepted,
            "offers_declined": self.offers_declined,
            "application_count": self.application_submissions,
            "interview_count": self.interviews_attended,
            "offer_count": self.offers_produced,
            "accepted_offers": self.offers_accepted,
            "graduated": agent.education.status == "completed",
            "graduation_week": self.graduation_week,
            "development_profile_counts": dict(self.development_profiles),
            "effective_study_hours": self.effective_study_hours,
            "effective_practice_hours": self.effective_practice_hours,
            "final_efficiencies": tuple(self.final_efficiencies),
            "energy_factors": tuple(self.energy_factors),
            "stress_factors": tuple(self.stress_factors),
            "mental_load_factors": tuple(self.mental_load_factors),
            "recovery_factors": tuple(self.recovery_factors),
            "workload_factors": tuple(self.workload_factors),
            "skill_level_deltas": dict(self.skill_level_deltas),
            "new_skills_created": self.new_skills_created,
            "final_skill_levels": final_skill_levels,
            "health_mental": self.health_mental,
            "starting_connection_count": self.starting_connections,
            "final_connection_count": self.final_connections,
            "new_persistent_connections": max(0, self.final_connections - self.starting_connections),
            "social_focal_opportunity_count": self.social_focal_opportunity_count,
            "social_no_opportunity_weeks": self.social_no_opportunity_weeks,
            "social_executed_focal_choices": self.social_executed_focal_choices,
            "social_choice_counts": dict(self.social_choices),
            "social_contact_counts": dict(self.social_contact_counts),
            "social_outcome_counts": dict(self.social_outcomes),
            "support_network_strength_checkpoints": dict(self.support_network_strength_checkpoints),
            "final_support_network_strength": agent.social.support_network_strength,
            "routine_profile_counts": dict(self.routine_profiles),
            "event_weeks": self.event_weeks,
            "event_free_weeks": self.event_free_weeks,
            "event_id_counts": dict(self.event_counts),
            "event_category_counts": dict(self.event_category_counts),
            "chosen_option_counts": dict(self.option_counts),
            "outcome_counts": dict(self.outcome_counts),
            "habits_formed": self.habits_formed,
            "final_managed_habit_strengths": self.final_habit_strengths,
            "max_habit_familiarity": self.max_habit_familiarity,
            "max_weekly_personality_delta": self.max_weekly_personality_delta,
            "max_anchor_displacement": self.max_anchor_displacement,
        }

    def _observe_events(self, events: tuple[Any, ...]) -> None:
        world_events = [event for event in events if not _synthetic_event_id(event.event_id)]
        if world_events:
            self.event_weeks += 1
        else:
            self.event_free_weeks += 1
        for event in world_events:
            self.event_counts[event.event_id] += 1
            self.event_category_counts[event.category] += 1

    def _observe_decisions(self, decisions: tuple[Any, ...]) -> None:
        for decision in decisions:
            if decision.chosen_option_id:
                self.option_counts[f"{decision.source_event_id}:{decision.chosen_option_id}"] += 1
            for evaluation in decision.evaluations:
                if not evaluation.available:
                    continue
                for component in evaluation.components:
                    if component.name == "habit_familiarity":
                        self.max_habit_familiarity = max(
                            self.max_habit_familiarity,
                            abs(component.contribution),
                        )

    def _observe_consequences(self, records: tuple[Any, ...]) -> None:
        for record in records:
            if record.selected_outcome_id:
                self.outcome_counts[
                    f"{record.source_event_id}:{record.selected_outcome_id}"
                ] += 1
            for application in record.financial_charge_applications:
                if application.skipped:
                    continue
                self.charge_count += 1
                self.charge_due += application.amount_due
                self.charge_paid += application.amount_paid
                self.charge_unpaid += application.amount_unpaid

    def _observe_passive(self, records: tuple[Any, ...]) -> None:
        for record in records:
            entries = getattr(record, "entries", ())
            for entry in entries:
                if entry.kind == "income":
                    self.recurring_income_received += entry.amount_paid
                    if entry.name.lower().startswith("wages:"):
                        self.employment_wage_income += entry.amount_paid
                if entry.kind.startswith("routine_"):
                    self.routine_spending += entry.amount_paid
            spending = getattr(record, "spending", ())
            for entry in spending:
                self.routine_spending += entry.amount_paid
            profile_id = getattr(record, "profile_id", "")
            if profile_id:
                self.routine_profiles[profile_id] += 1

    def _observe_employment(self, records: tuple[Any, ...]) -> None:
        for record in records:
            if (
                hasattr(record, "discovered_job_keys")
                and record.discovered_job_keys
                and self.first_job_discovery_week is None
            ):
                self.first_job_discovery_week = record.week
            event_ids_created = getattr(record, "event_ids_created", ())
            for event_id in event_ids_created:
                if event_id.startswith("employment_interview:"):
                    self.interview_invitations += 1
                elif event_id.startswith("employment_offer:"):
                    self.offers_produced += 1
                    self.first_offer_week = self.first_offer_week or record.week
            stage = getattr(record, "stage", "")
            detail = getattr(record, "detail", "")
            status_after = getattr(record, "status_after", "")
            if stage == "APPLICATION_DECISION":
                if detail == "submitted" and status_after == "SUBMITTED":
                    self.application_submissions += 1
                    self.first_application_week = self.first_application_week or record.week
                elif detail == "skipped" and status_after == "SKIPPED":
                    self.application_skips += 1
            elif stage == "APPLICATION_RESOLVED":
                if detail == "interview_invited" and status_after == "INTERVIEW_INVITED":
                    self.interview_invitations += 1
            elif stage == "INTERVIEW_DECISION":
                if detail == "interview_attended" and status_after == "INTERVIEW_ATTENDED":
                    self.interviews_attended += 1
                    self.first_interview_week = self.first_interview_week or record.week
                elif detail == "interview_declined" and status_after == "DECLINED":
                    self.interviews_declined += 1
            elif stage == "INTERVIEW_RESOLVED":
                if detail == "offer_available" and status_after == "OFFER_AVAILABLE":
                    self.offers_produced += 1
                    self.first_offer_week = self.first_offer_week or record.week
            elif stage == "OFFER_DECISION":
                if detail == "accepted_start_scheduled" and status_after == "ACCEPTED":
                    self.offers_accepted += 1
                elif detail == "declined" and status_after == "DECLINED":
                    self.offers_declined += 1
            action = getattr(record, "action", "")
            if action == "contract_started":
                self.jobs_held.add(record.contract_id)
                self.first_employment_start_week = self.first_employment_start_week or record.week
            elif action == "contract_ended":
                self.contract_endings += 1

    def _observe_development(self, records: tuple[Any, ...]) -> None:
        for record in records:
            profile_id = getattr(record, "profile_id", "")
            if not profile_id:
                continue
            self.development_profiles[profile_id] += 1
            self.effective_study_hours += record.efficiency.effective_study_hours
            self.effective_practice_hours += record.efficiency.effective_practice_hours
            self.final_efficiencies.append(record.efficiency.final_efficiency)
            self.energy_factors.append(record.efficiency.energy_factor)
            self.stress_factors.append(record.efficiency.stress_factor)
            self.mental_load_factors.append(record.efficiency.mental_load_factor)
            self.recovery_factors.append(record.efficiency.recovery_factor)
            self.workload_factors.append(record.efficiency.workload_factor)
            if record.education_progress is not None and record.education_progress.completed:
                self.graduation_week = self.graduation_week or record.week
            for skill in record.skill_developments:
                self.skill_level_deltas[skill.skill_name] = (
                    self.skill_level_deltas.get(skill.skill_name, 0.0) + skill.level_delta
                )
                if skill.level_before == 0.0:
                    self.new_skills_created += 1

    def _observe_social(self, records: tuple[Any, ...]) -> None:
        for record in records:
            option_ids = getattr(record, "option_ids", ())
            if option_ids:
                self.social_focal_opportunity_count += 1
            interaction_type = getattr(record, "interaction_type", "")
            option_id = getattr(record, "option_id", "")
            if interaction_type == "no_opportunity":
                self.social_no_opportunity_weeks += 1
            choice = _social_choice(option_id)
            if choice:
                self.social_choices[choice] += 1
                self.social_executed_focal_choices += 1
            contact_id = getattr(record, "contact_id", "")
            if contact_id:
                self.social_contact_counts[contact_id] += 1
            outcome = getattr(record, "outcome", None)
            if outcome is not None:
                self.social_outcomes[outcome.selected_outcome_id] += 1

    def _observe_adaptation(self, records: tuple[Any, ...], runtime: Any) -> None:
        for record in records:
            for change in record.habit_strength_changes:
                if change.before == 0.0 and change.after > 0.0:
                    self.habits_formed += 1
            for change in record.personality_changes:
                self.max_weekly_personality_delta = max(
                    self.max_weekly_personality_delta,
                    abs(change.delta),
                )
                self.max_anchor_displacement = max(
                    self.max_anchor_displacement,
                    abs(change.after - change.anchor),
                )
        if runtime is not None and runtime.personality_anchor is not None:
            for accumulator in runtime.trait_accumulators:
                self.max_anchor_displacement = max(self.max_anchor_displacement, 0.0)

    def _observe_health_mental(self, agent: AgentState) -> None:
        for field, metrics in self.health_mental.items():
            value = _path_value(agent, field)
            metrics["min"] = min(metrics["min"], value)
            metrics["max"] = max(metrics["max"], value)
            if field in BOUNDED_FIELDS and value == 0.0:
                metrics["weeks_at_0"] += 1
            if field in BOUNDED_FIELDS and value == 100.0:
                metrics["weeks_at_100"] += 1


def _finance_aggregates(
    records: tuple[CalibrationRunRecord, ...],
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    metrics = [record.metrics for record in records]
    arrears_created = sum(item["arrears_created"] for item in metrics)
    arrears_recovered = sum(item["arrears_fully_recovered"] for item in metrics)
    return {
        "initial_liquid_distribution": stats([item["initial_liquid"] for item in metrics]),
        "minimum_liquid_distribution": stats([item["minimum_liquid"] for item in metrics]),
        "final_liquid_distribution": stats([item["final_liquid"] for item in metrics]),
        "liquid_by_checkpoint": {
            str(week): stats([
                record.checkpoints[week]["liquid"]
                for record in records
                if week in record.checkpoints
            ])
            for week in checkpoints
        },
        "account_by_checkpoint": {
            str(week): {
                account: stats([
                    record.checkpoints[week][account]
                    for record in records
                    if week in record.checkpoints
                ])
                for account in MONEY_ACCOUNTS
            }
            for week in checkpoints
        },
        "final_debt_distribution": stats([item["final_debt_balance"] for item in metrics]),
        "initial_debt_distribution": stats([item["initial_debt_balance"] for item in metrics]),
        "arrear_incidence_rate": _rate(item["any_arrear"] for item in metrics),
        "arrear_recovery_rate": (
            arrears_recovered / arrears_created if arrears_created else 0.0
        ),
        "first_arrear_week_distribution": stats_optional([
            item["first_arrear_week"] for item in metrics
        ]),
        "max_arrear_distribution": stats([
            item["maximum_total_arrear_balance"] for item in metrics
        ]),
        "final_arrear_distribution": stats([
            item["final_total_arrear_balance"] for item in metrics
        ]),
        "weeks_with_arrears_distribution": stats([
            item["weeks_with_arrears"] for item in metrics
        ]),
        "arrears_created_distribution": stats([item["arrears_created"] for item in metrics]),
        "arrears_recovered_distribution": stats([
            item["arrears_fully_recovered"] for item in metrics
        ]),
        "charge_count_distribution": stats([item["charge_count"] for item in metrics]),
        "charge_due_distribution": stats([item["charge_amount_due"] for item in metrics]),
        "charge_paid_distribution": stats([item["charge_amount_paid"] for item in metrics]),
        "charge_unpaid_distribution": stats([item["charge_amount_unpaid"] for item in metrics]),
        "recurring_income_distribution": stats([
            item["recurring_income_received"] for item in metrics
        ]),
        "employment_wage_income_distribution": stats([
            item["employment_wage_income"] for item in metrics
        ]),
        "routine_spending_distribution": stats([item["routine_spending"] for item in metrics]),
    }


def _employment_aggregates(
    records: tuple[CalibrationRunRecord, ...],
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    metrics = [record.metrics for record in records]
    applications = sum(
        item.get("application_submissions", item.get("application_count", 0))
        for item in metrics
    )
    application_skips = sum(item.get("application_skips", 0) for item in metrics)
    invitations = sum(
        item.get("interview_invitations", item.get("interview_count", 0))
        for item in metrics
    )
    interviews = sum(
        item.get("interviews_attended", item.get("interview_count", 0))
        for item in metrics
    )
    interviews_declined = sum(item.get("interviews_declined", 0) for item in metrics)
    offers = sum(item.get("offers_produced", item.get("offer_count", 0)) for item in metrics)
    accepted = sum(
        item.get("offers_accepted", item.get("accepted_offers", 0))
        for item in metrics
    )
    declined = sum(item.get("offers_declined", 0) for item in metrics)
    return {
        "ever_employed_rate_by_checkpoint": {
            str(week): _rate(
                item["first_employment_start_week"] is not None
                and item["first_employment_start_week"] <= week
                for item in metrics
            )
            for week in checkpoints
        },
        "current_employed_rate_by_checkpoint": {
            str(week): _rate(
                record.checkpoints.get(week, {}).get("currently_employed", False)
                for record in records
            )
            for week in checkpoints
        },
        "first_job_discovery_week_distribution": stats_optional([
            item["first_job_discovery_week"] for item in metrics
        ]),
        "first_application_week_distribution": stats_optional([
            item["first_application_week"] for item in metrics
        ]),
        "first_interview_week_distribution": stats_optional([
            item["first_interview_week"] for item in metrics
        ]),
        "first_offer_week_distribution": stats_optional([
            item["first_offer_week"] for item in metrics
        ]),
        "first_employment_week_distribution": stats_optional([
            item["first_employment_start_week"] for item in metrics
        ]),
        "total_employed_weeks_distribution": stats([
            item["total_employed_weeks"] for item in metrics
        ]),
        "jobs_held_distribution": stats([item["jobs_held"] for item in metrics]),
        "contract_endings_distribution": stats([item["contract_endings"] for item in metrics]),
        "application_submission_count": applications,
        "application_skip_count": application_skips,
        "interview_invitation_count": invitations,
        "interview_attended_count": interviews,
        "interview_declined_count": interviews_declined,
        "offer_count": offers,
        "accepted_offer_count": accepted,
        "declined_offer_count": declined,
        "application_count": applications,
        "interview_count": interviews,
        "application_to_interview_rate": invitations / applications if applications else 0.0,
        "interview_to_offer_rate": offers / interviews if interviews else 0.0,
        "offer_acceptance_rate": accepted / offers if offers else 0.0,
    }


def _education_aggregates(
    records: tuple[CalibrationRunRecord, ...],
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    metrics = [record.metrics for record in records]
    development_counts = _sum_counters(item["development_profile_counts"] for item in metrics)
    total_development = sum(development_counts.values())
    return {
        "progress_by_checkpoint": {
            str(week): stats([
                record.checkpoints[week]["education_progress"]
                for record in records
                if week in record.checkpoints
            ])
            for week in checkpoints
        },
        "graduation_rate_by_checkpoint": {
            str(week): _rate(
                item["graduation_week"] is not None and item["graduation_week"] <= week
                for item in metrics
            )
            for week in checkpoints
        },
        "graduation_week_distribution": stats_optional([
            item["graduation_week"] for item in metrics
        ]),
        "development_profile_counts": dict(sorted(development_counts.items())),
        "development_profile_shares": _shares(development_counts),
        "dominant_development_profile_share": (
            max(development_counts.values()) / total_development if total_development else 0.0
        ),
        "effective_study_hours_distribution": stats([
            item["effective_study_hours"] for item in metrics
        ]),
        "effective_practice_hours_distribution": stats([
            item["effective_practice_hours"] for item in metrics
        ]),
        "final_efficiency_distribution": stats([
            value
            for item in metrics
            for value in item.get("final_efficiencies", ())
        ]),
        "efficiency_factor_distributions": {
            "energy_factor": stats([
                value for item in metrics for value in item.get("energy_factors", ())
            ]),
            "stress_factor": stats([
                value for item in metrics for value in item.get("stress_factors", ())
            ]),
            "mental_load_factor": stats([
                value for item in metrics for value in item.get("mental_load_factors", ())
            ]),
            "recovery_factor": stats([
                value for item in metrics for value in item.get("recovery_factors", ())
            ]),
            "workload_factor": stats([
                value for item in metrics for value in item.get("workload_factors", ())
            ]),
        },
        "new_skills_created_distribution": stats([
            item["new_skills_created"] for item in metrics
        ]),
    }


def _health_aggregates(
    records: tuple[CalibrationRunRecord, ...],
    duration_weeks: int,
) -> dict[str, Any]:
    field_metrics: dict[str, Any] = {}
    total_boundary_weeks = 0
    total_bounded_observations = max(1, len(records) * len(BOUNDED_FIELDS))
    saturated_fields: list[str] = []
    for field in HEALTH_MENTAL_FIELDS:
        values = [record.metrics["health_mental"][field] for record in records]
        weeks_at_boundary = [
            item["weeks_at_0"] + item["weeks_at_100"]
            for item in values
        ]
        field_metrics[field] = {
            "minimum_distribution": stats([item["min"] for item in values]),
            "maximum_distribution": stats([item["max"] for item in values]),
            "boundary_weeks_distribution": stats(weeks_at_boundary),
            "weeks_at_0_distribution": stats([item["weeks_at_0"] for item in values]),
            "weeks_at_100_distribution": stats([item["weeks_at_100"] for item in values]),
        }
        if field in BOUNDED_FIELDS:
            total_boundary_weeks += sum(weeks_at_boundary)
            if stats(weeks_at_boundary).get("p95", 0.0) >= 8.0:
                saturated_fields.append(field)
    return {
        "fields": field_metrics,
        "boundary_saturation_rate": total_boundary_weeks
        / total_bounded_observations
        / max(1, duration_weeks),
        "saturated_fields": saturated_fields,
    }


def _social_aggregates(
    records: tuple[CalibrationRunRecord, ...],
    duration_weeks: int,
) -> dict[str, Any]:
    metrics = [record.metrics for record in records]
    social_choices = _sum_counters(item["social_choice_counts"] for item in metrics)
    contact_counts = _sum_counters(item.get("social_contact_counts", {}) for item in metrics)
    total_choices = sum(social_choices.values())
    total_contacts = sum(contact_counts.values())
    dominant_contact_id, dominant_contact_count = _dominant(contact_counts)
    return {
        "starting_connection_distribution": stats([
            item["starting_connection_count"] for item in metrics
        ]),
        "final_connection_distribution": stats([
            item["final_connection_count"] for item in metrics
        ]),
        "new_connection_incidence_rate": _rate(
            item["new_persistent_connections"] > 0 for item in metrics
        ),
        "new_persistent_connections_distribution": stats([
            item["new_persistent_connections"] for item in metrics
        ]),
        "social_focal_opportunity_distribution": stats([
            item["social_focal_opportunity_count"] for item in metrics
        ]),
        "social_executed_focal_choice_distribution": stats([
            item.get("social_executed_focal_choices", 0) for item in metrics
        ]),
        "social_no_opportunity_distribution": stats([
            item.get("social_no_opportunity_weeks", 0) for item in metrics
        ]),
        "focal_opportunity_rate": (
            sum(item["social_focal_opportunity_count"] for item in metrics)
            / max(1, len(records) * duration_weeks)
        ),
        "no_opportunity_rate": (
            sum(item.get("social_no_opportunity_weeks", 0) for item in metrics)
            / max(1, len(records) * duration_weeks)
        ),
        "choice_counts": dict(sorted(social_choices.items())),
        "choice_shares": _shares(social_choices),
        "dominant_voluntary_choice_share": (
            max(social_choices.values()) / total_choices if total_choices else 0.0
        ),
        "dominant_social_choice_share": (
            max(social_choices.values()) / total_choices if total_choices else 0.0
        ),
        "contact_counts": dict(sorted(contact_counts.items())),
        "contact_shares": _shares(contact_counts),
        "dominant_contact_id": dominant_contact_id,
        "dominant_contact_share": (
            dominant_contact_count / total_contacts if total_contacts else 0.0
        ),
        "outcome_counts": dict(sorted(_sum_counters(item["social_outcome_counts"] for item in metrics).items())),
        "final_support_strength_distribution": stats([
            item["final_support_network_strength"] for item in metrics
        ]),
    }


def _routine_aggregates(records: tuple[CalibrationRunRecord, ...]) -> dict[str, Any]:
    counts = _sum_counters(record.metrics["routine_profile_counts"] for record in records)
    for profile in ROUTINE_PROFILES:
        counts.setdefault(profile, 0)
    total = sum(counts.values())
    dominant_profile, dominant_count = _dominant(counts)
    return {
        "profile_counts": dict(sorted(counts.items())),
        "profile_shares": _shares(counts),
        "dominant_profile": dominant_profile,
        "dominant_share": dominant_count / total if total else 0.0,
    }


def _event_aggregates(
    records: tuple[CalibrationRunRecord, ...],
    duration_weeks: int,
) -> dict[str, Any]:
    event_counts = _sum_counters(record.metrics["event_id_counts"] for record in records)
    category_counts = _sum_counters(
        record.metrics["event_category_counts"] for record in records
    )
    option_counts = _sum_counters(record.metrics["chosen_option_counts"] for record in records)
    outcome_counts = _sum_counters(record.metrics["outcome_counts"] for record in records)
    total_events = sum(event_counts.values())
    total_categories = sum(category_counts.values())
    dominant_event_id, dominant_event_count = _dominant(event_counts)
    dominant_category, dominant_category_count = _dominant(category_counts)
    return {
        "event_week_rate": (
            sum(record.metrics["event_weeks"] for record in records)
            / max(1, len(records) * duration_weeks)
        ),
        "event_free_weeks_distribution": stats([
            record.metrics["event_free_weeks"] for record in records
        ]),
        "event_counts": dict(sorted(event_counts.items())),
        "event_shares": _shares(event_counts),
        "event_category_counts": dict(sorted(category_counts.items())),
        "event_category_shares": _shares(category_counts),
        "chosen_option_counts": dict(sorted(option_counts.items())),
        "chosen_option_shares": _shares(option_counts),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "outcome_shares": _shares(outcome_counts),
        "dominant_event_id": dominant_event_id,
        "dominant_event_share": (
            dominant_event_count / total_events if total_events else 0.0
        ),
        "dominant_category": dominant_category,
        "dominant_category_share": (
            dominant_category_count / total_categories if total_categories else 0.0
        ),
    }


def _adaptation_aggregates(records: tuple[CalibrationRunRecord, ...]) -> dict[str, Any]:
    habit_strengths = [
        strength
        for record in records
        for strength in record.metrics["final_managed_habit_strengths"]
    ]
    return {
        "habits_formed_distribution": stats([
            record.metrics["habits_formed"] for record in records
        ]),
        "final_habit_strength_distribution": stats(habit_strengths),
        "max_habit_familiarity_distribution": stats([
            record.metrics["max_habit_familiarity"] for record in records
        ]),
        "max_weekly_personality_delta_distribution": stats([
            record.metrics["max_weekly_personality_delta"] for record in records
        ]),
        "max_anchor_displacement_distribution": stats([
            record.metrics["max_anchor_displacement"] for record in records
        ]),
    }


def stats(values: Iterable[Any]) -> dict[str, float]:
    numeric = sorted(_to_float(value) for value in values)
    if not numeric:
        return _empty_stats()
    return {
        "min": numeric[0],
        "p05": percentile(numeric, 0.05),
        "p25": percentile(numeric, 0.25),
        "p50": percentile(numeric, 0.50),
        "p75": percentile(numeric, 0.75),
        "p95": percentile(numeric, 0.95),
        "max": numeric[-1],
        "mean": float(mean(numeric)),
    }


def stats_optional(values: Iterable[Any]) -> dict[str, float]:
    return stats(value for value in values if value is not None)


def percentile(sorted_values: list[float], quantile: float) -> float:
    """Linear interpolation between closest ranks, deterministic and dependency-free."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _assert_hard_invariants(
    *,
    week: int,
    agent: AgentState,
    consequence_records: tuple[Any, ...],
    consequence_runtime: Any,
    adaptation_records: tuple[Any, ...],
    provenance_ids: set[str],
) -> None:
    for account in MONEY_ACCOUNTS:
        if getattr(agent.financial, account) < Decimal("0.00"):
            raise HardInvariantError(f"Negative money account '{account}' at week {week}.")
    if _contains_nonfinite(agent):
        raise HardInvariantError(f"Non-finite state value at week {week}.")
    for record in consequence_records:
        _unique_id(f"consequence:{record.consequence_id}", provenance_ids)
        for application in record.effect_applications:
            if application.scheduled_effect_id:
                _unique_id(f"effect_application:{application.scheduled_effect_id}", provenance_ids)
        for application in record.financial_charge_applications:
            if application.amount_due != application.amount_paid + application.amount_unpaid:
                raise HardInvariantError("Financial charge due != paid + unpaid.")
            if sum((transfer.amount for transfer in application.funding), Decimal("0.00")) != application.amount_paid:
                raise HardInvariantError("Financial charge funding transfers do not sum to paid.")
            if application.scheduled_charge_id:
                _unique_id(f"charge_application:{application.scheduled_charge_id}", provenance_ids)
        for scheduled in record.scheduled_effects_created:
            _unique_id(f"scheduled_effect:{scheduled.scheduled_effect_id}", provenance_ids)
        for scheduled in record.scheduled_financial_charges_created:
            _unique_id(f"scheduled_charge:{scheduled.scheduled_charge_id}", provenance_ids)
    if consequence_runtime is not None:
        for scheduled in consequence_runtime.pending_scheduled_effects:
            if scheduled.due_week <= week:
                raise HardInvariantError("Past-due scheduled effect remains pending.")
        for scheduled in consequence_runtime.pending_scheduled_financial_charges:
            if scheduled.due_week <= week:
                raise HardInvariantError("Past-due scheduled financial charge remains pending.")
    for record in adaptation_records:
        for change in record.personality_changes:
            if abs(change.delta) > PERSONALITY_WEEKLY_CAP:
                raise HardInvariantError("M11 weekly personality cap exceeded.")
            if abs(change.after - change.anchor) > PERSONALITY_ANCHOR_CAP:
                raise HardInvariantError("M11 personality anchor cap exceeded.")


def _checkpoint(agent: AgentState) -> dict[str, Any]:
    output = {
        "liquid": _liquid(agent),
        "debt_balance": _debt_balance(agent),
        "arrear_balance": _arrear_balance(agent),
        "currently_employed": _is_currently_employed(agent),
        "education_progress": agent.education.progress,
        "education_status": agent.education.status,
        "connection_count": len(agent.social.connections),
        "support_network_strength": agent.social.support_network_strength,
    }
    for account in MONEY_ACCOUNTS:
        output[account] = getattr(agent.financial, account)
    return output


def _with_seed_and_duration(config: LifeSimConfig, seed: int, duration_weeks: int) -> LifeSimConfig:
    return replace(
        config,
        simulation=replace(
            config.simulation,
            seed=seed,
            duration_weeks=duration_weeks,
        ),
    )


def _liquid(agent: AgentState) -> Decimal:
    return sum((getattr(agent.financial, account) for account in MONEY_ACCOUNTS), Decimal("0.00"))


def _debt_balance(agent: AgentState) -> Decimal:
    return sum((debt.balance for debt in agent.financial.debts), Decimal("0.00"))


def _arrear_balance(agent: AgentState) -> Decimal:
    return sum((arrear.balance for arrear in agent.financial.arrears), Decimal("0.00"))


def _is_currently_employed(agent: AgentState) -> bool:
    return agent.employment.status in {"employed", "student_part_time"}


def _path_value(agent: AgentState, path: str) -> float:
    section, field = path.split(".", 1)
    return float(getattr(getattr(agent, section), field))


def _synthetic_event_id(event_id: str) -> bool:
    return event_id.startswith(
        (
            "weekly_",
            "employment_",
            "social_",
        )
    )


def _social_choice(option_id: str) -> str:
    if option_id.startswith("connect:"):
        return "connect"
    if option_id.startswith("seek_support:"):
        return "seek_support"
    if option_id.startswith("engage:"):
        return "engage"
    if option_id == "keep_social_light":
        return "keep_social_light"
    return ""


def _sum_counters(items: Iterable[Mapping[str, int]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(item)
    return counter


def _shares(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def _dominant(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return "", 0
    return max(sorted(counter.items()), key=lambda item: item[1])


def _rate(values: Iterable[Any]) -> float:
    materialized = tuple(values)
    if not materialized:
        return 0.0
    return sum(1 for value in materialized if value) / len(materialized)


def _to_float(value: Any) -> float:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _empty_stats() -> dict[str, float]:
    return {
        "min": 0.0,
        "p05": 0.0,
        "p25": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "p95": 0.0,
        "max": 0.0,
        "mean": 0.0,
    }


def _unique_id(value: str, seen: set[str]) -> None:
    if value in seen:
        raise HardInvariantError(f"Duplicate provenance id '{value}'.")
    seen.add(value)


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, Decimal):
        return not value.is_finite()
    if isinstance(value, float):
        return not math.isfinite(value)
    if value is None or isinstance(value, int | str | bool):
        return False
    if isinstance(value, tuple | list):
        return any(_contains_nonfinite(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return any(
            _contains_nonfinite(getattr(value, field))
            for field in value.__dataclass_fields__
        )
    return False


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _stats_line(
    label: str,
    values: Mapping[str, float],
    *,
    precision: int = 2,
) -> str:
    formatted = {
        name: f"{values.get(name, 0.0):.{precision}f}"
        for name in ("p05", "p50", "p95", "mean")
    }
    return (
        f"- {label}: p05 {formatted['p05']}, "
        f"p50 {formatted['p50']}, p95 {formatted['p95']}, "
        f"mean {formatted['mean']}"
    )


def _checkpoint_triplets(
    values: Mapping[str, Mapping[str, float]],
    *,
    precision: int = 2,
) -> dict[str, str]:
    return {
        week: (
            f"{stats_values.get('p05', 0.0):.{precision}f}/"
            f"{stats_values.get('p50', 0.0):.{precision}f}/"
            f"{stats_values.get('p95', 0.0):.{precision}f}"
        )
        for week, stats_values in values.items()
    }


def _boundary_direction_summary(fields: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        field: (
            f"0 weeks p50/p95 "
            f"{metrics['weeks_at_0_distribution'].get('p50', 0.0):.1f}/"
            f"{metrics['weeks_at_0_distribution'].get('p95', 0.0):.1f}; "
            f"100 weeks p50/p95 "
            f"{metrics['weeks_at_100_distribution'].get('p50', 0.0):.1f}/"
            f"{metrics['weeks_at_100_distribution'].get('p95', 0.0):.1f}"
        )
        for field, metrics in fields.items()
    }


def _efficiency_factor_summary(factors: Mapping[str, Mapping[str, float]]) -> dict[str, str]:
    return {
        name: (
            f"p05/p50/p95 "
            f"{values.get('p05', 0.0):.4f}/"
            f"{values.get('p50', 0.0):.4f}/"
            f"{values.get('p95', 0.0):.4f}"
        )
        for name, values in factors.items()
    }
