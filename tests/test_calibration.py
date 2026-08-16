from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.calibration import CalibrationResult, CalibrationRunRecord, run_calibration
from lifesim.calibration.runner import (
    HardInvariantError,
    _assert_hard_invariants,
    aggregate_run_records,
    diagnose_warnings,
    percentile,
    stats,
    write_markdown_report,
)
from lifesim.canonical import build_canonical_engine, build_canonical_transitions
from lifesim.config import load_config
from lifesim.weekly import WeeklyTransitionResult

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")


def test_failed_run_is_retained_with_seed_week_type_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingTransition:
        def apply(self, state, context):
            raise RuntimeError("synthetic calibration failure")

    from lifesim.calibration import runner

    monkeypatch.setattr(runner, "build_canonical_transitions", lambda **_: (FailingTransition(),))

    result = run_calibration(label="failure-test", seeds=(7, 8), duration_weeks=3, checkpoints=(1,))

    assert result.run_count == 2
    assert result.successful_run_count == 0
    assert result.failure_count == 2
    assert [record.seed for record in result.run_records] == [7, 8]
    assert {record.failure_week for record in result.run_records} == {1}
    assert {record.failure_type for record in result.run_records} == {"RuntimeError"}
    assert all("synthetic" in record.failure_message for record in result.run_records)


def test_percentile_statistics_are_deterministic_without_numpy() -> None:
    values = [10.0, 0.0, 30.0, 20.0]

    assert percentile(sorted(values), 0.50) == 15.0
    assert stats(values)["p25"] == 7.5
    assert stats(reversed(values)) == stats(values)


def test_aggregate_statistics_are_order_independent_and_include_shares() -> None:
    first = run_record(
        1,
        routine_profile_counts={"balanced_week": 2},
        event_id_counts={"a": 1, "b": 1},
        event_category_counts={"city": 2},
    )
    second = run_record(
        2,
        routine_profile_counts={"social_week": 2},
        event_id_counts={"a": 2},
        event_category_counts={"finance": 2},
    )

    forward = aggregate_run_records((first, second), (12,), 12)
    reversed_result = aggregate_run_records((second, first), (12,), 12)

    assert forward == reversed_result
    assert forward["routine"]["profile_shares"]["balanced_week"] == 0.5
    assert forward["events"]["event_shares"]["a"] == 0.75


def test_employment_funnel_and_arrear_metrics_are_extracted_from_run_records() -> None:
    result = aggregate_run_records(
        (
            run_record(1, first_employment_start_week=10, first_application_week=4),
            run_record(
                2,
                any_arrear=True,
                first_arrear_week=3,
                maximum_total_arrear_balance=Decimal("120.00"),
                final_total_arrear_balance=Decimal("0.00"),
                arrears_created=1,
                arrears_fully_recovered=1,
                first_employment_start_week=None,
            ),
        ),
        (12, 52),
        52,
    )

    assert result["employment"]["ever_employed_rate_by_checkpoint"]["12"] == 0.5
    assert result["employment"]["first_employment_week_distribution"]["p50"] == 10.0
    assert result["finance"]["arrear_incidence_rate"] == 0.5
    assert result["finance"]["arrear_recovery_rate"] == 1.0
    assert result["finance"]["max_arrear_distribution"]["p95"] > 100.0


def test_warning_diagnostics_use_real_dominance_and_lock_in_semantics() -> None:
    metrics = aggregate_run_records(
        (
            run_record(1, routine_profile_counts={"balanced_week": 52}, event_weeks=30, event_id_counts={"dominant": 40}, event_category_counts={"city": 40}),
            run_record(2, routine_profile_counts={"balanced_week": 52}, event_weeks=30, event_id_counts={"other": 2}, event_category_counts={"city": 2}),
        ),
        (52,),
        52,
    )

    warnings = diagnose_warnings(metrics)

    assert "EVENT_DOMINANCE" in warnings
    assert "EVENTS_TOO_RARE" not in warnings
    assert "ROUTINE_LOCK_IN" in warnings


def test_state_saturation_warning_detects_boundary_pin() -> None:
    record = run_record(1)
    saturated = dict(record.metrics)
    saturated["health_mental"] = {
        field: {"min": 0.0, "max": 100.0, "weeks_at_0": 20, "weeks_at_100": 20}
        for field in saturated["health_mental"]
    }
    metrics = aggregate_run_records(
        (replace(record, metrics=saturated),),
        (52,),
        52,
    )

    assert "STATE_BOUNDARY_SATURATION" in diagnose_warnings(metrics)


def test_hard_financial_invariant_detection() -> None:
    agent = load_agent_state(MAYA_SCENARIO)
    object.__setattr__(agent.financial, "cash", Decimal("-1.00"))

    with pytest.raises(HardInvariantError, match="Negative money account"):
        _assert_hard_invariants(
            week=1,
            agent=agent,
            consequence_records=(),
            consequence_runtime=None,
            adaptation_records=(),
            provenance_ids=set(),
        )


def test_hard_charge_and_past_due_invariant_detection() -> None:
    agent = load_agent_state(MAYA_SCENARIO)
    bad_application = SimpleNamespace(
        amount_due=Decimal("3.00"),
        amount_paid=Decimal("1.00"),
        amount_unpaid=Decimal("1.00"),
        funding=(),
        scheduled_charge_id="",
    )
    bad_record = SimpleNamespace(
        consequence_id="bad",
        effect_applications=(),
        financial_charge_applications=(bad_application,),
        scheduled_effects_created=(),
        scheduled_financial_charges_created=(),
    )

    with pytest.raises(HardInvariantError, match="due != paid"):
        _assert_hard_invariants(
            week=1,
            agent=agent,
            consequence_records=(bad_record,),
            consequence_runtime=None,
            adaptation_records=(),
            provenance_ids=set(),
        )

    runtime = SimpleNamespace(
        pending_scheduled_effects=(SimpleNamespace(due_week=1),),
        pending_scheduled_financial_charges=(),
    )
    with pytest.raises(HardInvariantError, match="Past-due scheduled effect"):
        _assert_hard_invariants(
            week=1,
            agent=agent,
            consequence_records=(),
            consequence_runtime=runtime,
            adaptation_records=(),
            provenance_ids=set(),
        )


def test_report_serialization_is_deterministic_and_readable(tmp_path: Path) -> None:
    result = CalibrationResult(
        label="fixture",
        seeds=(1,),
        duration_weeks=52,
        checkpoints=(52,),
        run_records=(run_record(1),),
        run_count=1,
        successful_run_count=1,
        failure_count=0,
        hard_invariant_failure_count=0,
        metrics=aggregate_run_records((run_record(1),), (52,), 52),
        warnings=(),
    )
    path = tmp_path / "report.md"

    write_markdown_report(result, None, path)
    first = path.read_text(encoding="utf-8")
    write_markdown_report(result, None, path)

    assert path.read_text(encoding="utf-8") == first
    assert "Baseline M11 Failure" in first
    assert "Parameter Change Log" in first


def test_canonical_builder_is_deterministic_and_includes_arrear_settlement() -> None:
    transitions = build_canonical_transitions()
    names = tuple(type(transition).__name__ for transition in transitions)

    assert "ArrearSettlementTransition" in names
    assert names.index("RoutineExecutionTransition") < names.index("ArrearSettlementTransition")
    assert names.index("ArrearSettlementTransition") < names.index("AdaptationTransition")

    config = load_config("configs/default.toml")
    maya = load_agent_state(MAYA_SCENARIO)
    engine = build_canonical_engine(config)

    assert engine.run(initial_agent=maya).to_dict() == engine.run(initial_agent=maya).to_dict()


def test_demo_cli_uses_canonical_engine_for_weekly_maya_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--config",
            "configs/default.toml",
            "--agent-scenario",
            str(MAYA_SCENARIO),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    output = json.loads(completed.stdout)

    assert output["states"][1]["agent"]["identity"]["agent_id"] == "maya"
    assert output["passive_history"]["arrear_settlement_records"]


def test_calibration_runs_fixed_seed_smoke_cohort() -> None:
    result = run_calibration(
        label="smoke",
        seeds=(0, 1),
        duration_weeks=12,
        checkpoints=(12,),
    )

    assert result.run_count == 2
    assert result.successful_run_count == 2
    assert result.failure_count == 0
    assert 12 in result.run_records[0].checkpoints
    assert "finance" in result.metrics


def test_employment_tracking_counts_observed_contract_starts() -> None:
    result = run_calibration(
        label="employment-tracker",
        seeds=(0,),
        duration_weeks=52,
        checkpoints=(12, 26, 52),
    )
    employment = result.metrics["employment"]

    assert employment["first_employment_week_distribution"]["p50"] == 13.0
    assert employment["ever_employed_rate_by_checkpoint"]["26"] == 1.0
    assert employment["current_employed_rate_by_checkpoint"]["26"] == 1.0
    assert employment["jobs_held_distribution"]["p50"] == 1.0


def test_seed_42_full_156_canonical_smoke_and_prefix_determinism() -> None:
    config = load_config("configs/canonical/maya_v1.toml")
    maya = load_agent_state(MAYA_SCENARIO)
    full = build_canonical_engine(config).run(initial_agent=maya).to_dict()

    for weeks in (12, 26, 52):
        short_config = replace(
            config,
            simulation=replace(config.simulation, duration_weeks=weeks),
        )
        short = build_canonical_engine(short_config).run(initial_agent=maya).to_dict()
        assert short["states"] == full["states"][: weeks + 1]


class NoopTransition:
    def apply(self, state, context):
        return WeeklyTransitionResult(agent_state=state)


def run_record(seed: int, **overrides) -> CalibrationRunRecord:
    metrics = base_metrics()
    metrics.update(overrides)
    return CalibrationRunRecord(
        seed=seed,
        completed=True,
        failure_week=None,
        failure_type="",
        failure_message="",
        checkpoints={12: checkpoint(), 52: checkpoint(), 156: checkpoint()},
        metrics=metrics,
    )


def checkpoint() -> dict[str, object]:
    return {
        "liquid": Decimal("100.00"),
        "cash": Decimal("10.00"),
        "bank_balance": Decimal("70.00"),
        "savings": Decimal("20.00"),
        "emergency_fund": Decimal("0.00"),
        "debt_balance": Decimal("50.00"),
        "arrear_balance": Decimal("0.00"),
        "currently_employed": True,
        "education_progress": 60.0,
        "education_status": "enrolled",
        "connection_count": 3,
        "support_network_strength": 35.0,
    }


def base_metrics() -> dict[str, object]:
    return {
        "initial_liquid": Decimal("100.00"),
        "minimum_liquid": Decimal("80.00"),
        "final_liquid": Decimal("120.00"),
        "initial_debt_balance": Decimal("50.00"),
        "final_debt_balance": Decimal("25.00"),
        "any_arrear": False,
        "first_arrear_week": None,
        "maximum_total_arrear_balance": Decimal("0.00"),
        "final_total_arrear_balance": Decimal("0.00"),
        "weeks_with_arrears": 0,
        "arrears_created": 0,
        "arrears_fully_recovered": 0,
        "charge_count": 2,
        "charge_amount_due": Decimal("20.00"),
        "charge_amount_paid": Decimal("20.00"),
        "charge_amount_unpaid": Decimal("0.00"),
        "recurring_income_received": Decimal("100.00"),
        "employment_wage_income": Decimal("50.00"),
        "routine_spending": Decimal("40.00"),
        "first_job_discovery_week": 2,
        "first_application_week": 3,
        "first_interview_week": 5,
        "first_offer_week": 8,
        "first_employment_start_week": 10,
        "ever_employed": True,
        "currently_employed_final": True,
        "total_employed_weeks": 40,
        "jobs_held": 1,
        "contract_endings": 0,
        "application_count": 2,
        "interview_count": 1,
        "offer_count": 1,
        "accepted_offers": 1,
        "graduated": False,
        "graduation_week": None,
        "development_profile_counts": {"steady_study": 10},
        "effective_study_hours": 20.0,
        "effective_practice_hours": 10.0,
        "skill_level_deltas": {"spreadsheets": 2.0},
        "new_skills_created": 0,
        "final_skill_levels": {"spreadsheets": 33.0},
        "health_mental": {
            field: {"min": 25.0, "max": 80.0, "weeks_at_0": 0, "weeks_at_100": 0}
            for field in (
                "health.energy",
                "health.physical_health",
                "mental.stress",
                "mental.mental_load",
                "mental.recovery_need",
                "mental.mood",
                "mental.loneliness",
                "health.sleep_debt",
            )
        },
        "starting_connection_count": 2,
        "final_connection_count": 3,
        "new_persistent_connections": 1,
        "social_focal_opportunity_count": 3,
        "social_choice_counts": {"connect": 2, "keep_social_light": 1},
        "social_outcome_counts": {"warm": 1},
        "support_network_strength_checkpoints": {12: 35.0},
        "final_support_network_strength": 40.0,
        "routine_profile_counts": {"balanced_week": 5, "social_week": 5},
        "event_weeks": 6,
        "event_free_weeks": 6,
        "event_id_counts": {"event_a": 3, "event_b": 3},
        "event_category_counts": {"city": 3, "finance": 3},
        "chosen_option_counts": {"event_a:option": 3},
        "outcome_counts": {"event_a:outcome": 1},
        "habits_formed": 1,
        "final_managed_habit_strengths": [20.0],
        "max_habit_familiarity": 0.05,
        "max_weekly_personality_delta": 0.001,
        "max_anchor_displacement": 0.04,
    }
