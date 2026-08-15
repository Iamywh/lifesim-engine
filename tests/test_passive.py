from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import (
    Debt,
    FinancialState,
    IncomeStream,
    RecurringCommitment,
    RoutineState,
)
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.consequences import (
    ConsequenceCatalog,
    ConsequenceEngine,
    DecisionConsequenceTransition,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    StateEffectDefinition,
)
from lifesim.decisions import (
    DecisionEngine,
    DecisionEngineTransition,
    DecisionHistory,
    DecisionRecord,
    DecisionScoreComponent,
    OptionEvaluation,
)
from lifesim.engine import LifeSimEngine
from lifesim.events import (
    EventCatalog,
    EventDefinition,
    EventEngine,
    EventEngineTransition,
    EventOption,
)
from lifesim.passive import (
    PassiveCashflowEngine,
    PassiveCashflowTransition,
    PassiveLifeRuntimeState,
    RoutineEngine,
    RoutineExecutionTransition,
    RoutinePlanningTransition,
    load_routine_catalog,
    parse_routine_catalog,
)
from lifesim.weekly import WeeklyContext

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")
ROUTINES = Path("configs/routines/starter.toml")


def test_existing_maya_scenario_loads_with_routine_defaults() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    assert isinstance(maya.routine, RoutineState)
    assert maya.routine.current_profile_id == "balanced_week"
    assert maya.financial.income_streams[0].due_day == 7


def test_routine_state_backward_compatible_default() -> None:
    assert RoutineState().current_profile_id == "balanced_week"


def test_routine_catalog_validation_and_decimal_money() -> None:
    catalog = load_routine_catalog(ROUTINES)

    assert catalog.get("balanced_week").food_budget == Decimal("55.00")
    try:
        parse_routine_catalog(
            {
                "profiles": [
                    routine_raw("duplicate"),
                    routine_raw("duplicate"),
                ]
            }
        )
    except ValueError as error:
        assert "profile_id" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected duplicate routine profile IDs to fail.")

    bad = routine_raw("bad")
    bad["social_contact"] = 2.0
    try:
        parse_routine_catalog({"profiles": [bad]})
    except ValueError as error:
        assert "social_contact" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid routine range to fail.")


def test_weekly_and_monthly_cadence_use_calendar_months() -> None:
    maya = with_financial(
        bank=Decimal("0.00"),
        income_streams=(
            IncomeStream("weekly income", Decimal("10.00"), "weekly", 1.0),
            IncomeStream("monthly income", Decimal("100.00"), "monthly", 1.0, due_day=1),
        ),
    )

    week1 = run_cashflow(maya, context(start_date="2026-01-26"))
    week2 = run_cashflow(week1[0], context(week=2, start_date="2026-01-26"))

    assert [entry.name for entry in week1[2].entries] == ["weekly income", "monthly income"]
    assert [entry.name for entry in week2[2].entries] == ["weekly income"]
    assert week2[0].financial.bank_balance == Decimal("120.00")


def test_cross_year_monthly_cadence() -> None:
    maya = with_financial(
        bank=Decimal("0.00"),
        income_streams=(IncomeStream("new year income", Decimal("50.00"), "monthly", 1.0, due_day=1),),
    )

    next_state, _, record = run_cashflow(maya, context(start_date="2026-12-28"))

    assert record.entries[0].due_date == "2027-01-01"
    assert next_state.financial.bank_balance == Decimal("50.00")


def test_income_processed_before_required_outflows_and_decimal_exactness() -> None:
    maya = with_financial(
        bank=Decimal("0.00"),
        income_streams=(IncomeStream("grant", Decimal("100.10"), "weekly", 1.0),),
        commitments=(RecurringCommitment("rent", Decimal("80.05"), "weekly", "housing"),),
    )

    next_state, _, record = run_cashflow(maya, context())

    assert next_state.financial.bank_balance == Decimal("20.05")
    assert record.entries[0].kind == "income"
    assert record.entries[1].paid is True


def test_income_reliability_zero_one_and_deterministic_intermediate() -> None:
    maya = with_financial(
        bank=Decimal("0.00"),
        income_streams=(
            IncomeStream("never", Decimal("10.00"), "weekly", 0.0),
            IncomeStream("always", Decimal("10.00"), "weekly", 1.0),
            IncomeStream("maybe", Decimal("10.00"), "weekly", 0.5),
        ),
    )

    first = run_cashflow(maya, context(seed=7))
    second = run_cashflow(maya, context(seed=7))

    assert first[2].to_dict() == second[2].to_dict()
    assert first[2].entries[0].paid is False
    assert first[2].entries[1].paid is True
    assert first[2].entries[2].roll is not None


def test_partial_obligatory_payment_creates_and_reinforces_arrear_without_negative_money() -> None:
    maya = with_financial(
        bank=Decimal("30.00"),
        commitments=(RecurringCommitment("rent", Decimal("100.00"), "weekly", "housing"),),
    )

    first, runtime, record = run_cashflow(maya, context())
    second, _, _ = run_cashflow(first, context(week=2), runtime)

    assert record.entries[0].amount_paid == Decimal("30.00")
    assert first.financial.bank_balance == Decimal("0.00")
    assert first.financial.arrears[0].balance == Decimal("70.00")
    assert second.financial.arrears[0].balance == Decimal("170.00")
    assert second.financial.arrears[0].missed_occurrences == 2
    assert second.needs.housing_security < maya.needs.housing_security


def test_debt_interest_minimum_payment_cap_and_missed_audit() -> None:
    maya = with_financial(
        bank=Decimal("0.00"),
        debts=(Debt("loan", Decimal("10.00"), Decimal("25.00"), Decimal("0.10"), due_day=1),),
    )

    next_state, _, record = run_cashflow(maya, context(start_date="2026-01-01"))

    assert record.entries[0].kind == "debt_interest"
    assert record.entries[0].amount_due == Decimal("0.02")
    assert record.entries[1].kind == "debt_payment"
    assert record.entries[1].amount_due == Decimal("10.02")
    assert record.entries[1].amount_paid == Decimal("0.00")
    assert next_state.financial.debts[0].balance == Decimal("10.02")
    assert next_state.financial.debts[0].consecutive_missed_payments == 1


def test_routine_planning_uses_decision_engine_and_avoids_duplicate_decisions() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(
        config(duration_weeks=1, seed=2),
        transitions=(
            RoutinePlanningTransition(RoutineEngine(load_routine_catalog(ROUTINES)), DecisionEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
        ),
    ).run(initial_agent=maya)

    decisions = result.states[1].decisions
    assert [decision.source_event_id for decision in decisions].count("weekly_routine") == 1
    assert decisions[0].evaluations[0].components
    assert result.states[1].events[0].event_id == "weekly_routine"


def test_routine_decision_does_not_perturb_event_rng() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    base = LifeSimEngine(
        config(duration_weeks=1, seed=4),
        transitions=(EventEngineTransition(EventEngine(event_catalog())),),
    ).run(initial_agent=maya)
    with_routine = LifeSimEngine(
        config(duration_weeks=1, seed=4),
        transitions=(
            RoutinePlanningTransition(RoutineEngine(load_routine_catalog(ROUTINES)), DecisionEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
        ),
    ).run(initial_agent=maya)

    assert base.states[1].event_traces[0].to_dict() == with_routine.states[1].event_traces[0].to_dict()


def test_routine_execution_tradeoffs_and_mutation_boundary() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    before = maya.to_dict()

    next_state, _, record = execute_profile(maya, "austerity_home_week")
    after = next_state.to_dict()

    for key in ("identity", "personality", "skills", "employment", "goals", "memory"):
        assert before[key] == after[key]
    assert next_state.financial.bank_balance > Decimal("900.00")
    assert record.profile_id == "austerity_home_week"
    assert next_state.routine.low_social_streak == 1


def test_single_home_week_not_harmful_but_repeated_low_social_weeks_accumulate_loneliness() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    one_week, _, _ = execute_profile(maya, "austerity_home_week")
    repeated = maya
    for _ in range(5):
        repeated, _, _ = execute_profile(repeated, "austerity_home_week")

    assert one_week.mental.mood >= maya.mental.mood
    assert repeated.mental.loneliness > one_week.mental.loneliness
    assert repeated.needs.belonging < one_week.needs.belonging


def test_recovery_social_and_active_profiles_have_expected_tradeoffs() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    depleted = replace(
        maya,
        health=replace(maya.health, energy=35.0, sleep_debt=12.0),
        mental=replace(maya.mental, loneliness=70.0),
    )

    recovery, _, _ = execute_profile(depleted, "recovery_focus_week")
    social, _, _ = execute_profile(depleted, "social_week")
    active, _, _ = execute_profile(depleted, "low_cost_active_week")

    assert recovery.health.energy > depleted.health.energy
    assert recovery.health.sleep_debt < depleted.health.sleep_debt
    assert social.mental.loneliness < depleted.mental.loneliness
    assert active.health.mobility > depleted.health.mobility
    assert active.health.energy < depleted.health.energy


def test_routine_spending_cannot_make_money_negative_and_food_shortfall_is_audited() -> None:
    maya = with_financial(bank=Decimal("10.00"))

    next_state, _, record = execute_profile(maya, "balanced_week")

    assert next_state.financial.bank_balance == Decimal("0.00")
    assert next_state.financial.cash == Decimal("0.00")
    assert next_state.financial.savings == Decimal("0.00")
    assert next_state.financial.emergency_fund == Decimal("0.00")
    assert record.spending[0].amount_paid == Decimal("10.00")
    assert any(effect.reason == "food_shortfall" for effect in record.effects)


def test_no_event_week_still_produces_ordinary_life_changes_and_repeated_runs_match() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    transitions = (
        PassiveCashflowTransition(PassiveCashflowEngine()),
        RoutinePlanningTransition(RoutineEngine(load_routine_catalog(ROUTINES)), DecisionEngine()),
        EventEngineTransition(EventEngine(EventCatalog((), event_probability=0.0))),
        RoutineExecutionTransition(RoutineEngine(load_routine_catalog(ROUTINES))),
    )
    engine = LifeSimEngine(config(duration_weeks=2, seed=8), transitions=transitions)

    first = engine.run(initial_agent=maya)
    second = engine.run(initial_agent=maya)

    assert first.to_dict() == second.to_dict()
    assert first.states[1].agent_state != maya
    assert first.states[1].event_traces[0].no_event is True
    assert first.states[1].passive_records


def test_austerity_emergence_changes_social_incentive_without_scripted_switch() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    routine_engine = RoutineEngine(load_routine_catalog(ROUTINES))
    initial_social_score = routine_score(maya, routine_engine, "social_week", week=1)
    austerity = maya
    for _ in range(5):
        austerity, _, _ = execute_profile(austerity, "austerity_home_week")
    later_social_score = routine_score(austerity, routine_engine, "social_week", week=6)
    social_streak = maya
    for _ in range(5):
        social_streak, _, _ = execute_profile(social_streak, "social_week")

    assert austerity.financial.bank_balance > social_streak.financial.bank_balance
    assert austerity.mental.loneliness > maya.mental.loneliness
    assert later_social_score > initial_social_score


def test_low_cost_active_preserves_more_money_than_social_and_improves_physical_state() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    active, _, _ = execute_profile(maya, "low_cost_active_week")
    social, _, _ = execute_profile(maya, "social_week")

    assert active.financial.bank_balance > social.financial.bank_balance
    assert active.health.physical_health > maya.health.physical_health
    assert active.health.energy < maya.health.energy


def test_m5_outcome_rng_isolation_with_routine_enabled() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    consequence = ConsequenceCatalog(
        (
            OptionConsequenceDefinition(
                event_id="choice_event",
                event_version="1",
                option_id="chosen",
                outcomes=(
                    OutcomeDefinition(
                        "normal",
                        0.8,
                        effects=(StateEffectDefinition(path="mental.stress", delta=1.0),),
                    ),
                    OutcomeDefinition(
                        "bad",
                        0.2,
                        effects=(StateEffectDefinition(path="mental.stress", delta=8.0),),
                    ),
                ),
            ),
        )
    )
    base = LifeSimEngine(
        config(duration_weeks=1, seed=5),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence)),
        ),
    ).run(initial_agent=maya)
    with_routine = LifeSimEngine(
        config(duration_weeks=1, seed=5),
        transitions=(
            RoutinePlanningTransition(RoutineEngine(load_routine_catalog(ROUTINES)), DecisionEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence)),
        ),
    ).run(initial_agent=maya)

    base_consequence = base.states[1].consequences[0].to_dict()
    routine_consequence = next(
        consequence.to_dict()
        for consequence in with_routine.states[1].consequences
        if consequence.source_event_id == "choice_event"
    )
    assert routine_consequence == base_consequence


def test_cli_demo_exposes_passive_audit_json() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--config",
            "configs/default.toml",
            "--agent-scenario",
            "configs/scenarios/maya_start.toml",
            "--event-catalog",
            "configs/events/starter.toml",
            "--consequence-catalog",
            "configs/consequences/starter.toml",
            "--routine-catalog",
            "configs/routines/starter.toml",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    output = json.loads(completed.stdout)

    assert output["states"][1]["passive_records"]
    assert output["passive_history"]["cashflow_records"]
    assert output["passive_history"]["routine_records"]


def run_cashflow(
    state,
    weekly_context: WeeklyContext,
    runtime: PassiveLifeRuntimeState | None = None,
):
    return PassiveCashflowEngine().apply(
        state,
        weekly_context,
        runtime or PassiveLifeRuntimeState(),
    )


def execute_profile(state, profile_id: str):
    engine = RoutineEngine(load_routine_catalog(ROUTINES))
    profile = engine.catalog.get(profile_id)
    runtime = PassiveLifeRuntimeState(
        planned_routine_profile_id=profile.profile_id,
        planned_routine_decision=routine_decision(profile.profile_id),
    )
    return engine.execute(state, context(), runtime)


def routine_score(state, engine: RoutineEngine, profile_id: str, *, week: int) -> float:
    runtime, _, _, decision = engine.plan(
        state,
        context(week=week),
        PassiveLifeRuntimeState(),
        DecisionEngine(),
        DecisionHistory(),
    )
    assert runtime.planned_routine_profile_id
    evaluation = next(item for item in decision.evaluations if item.option_id == profile_id)
    assert evaluation.final_score is not None
    return evaluation.final_score


def with_financial(
    *,
    bank: Decimal = Decimal("0.00"),
    cash: Decimal = Decimal("0.00"),
    savings: Decimal = Decimal("0.00"),
    emergency: Decimal = Decimal("0.00"),
    debts: tuple[Debt, ...] = (),
    income_streams: tuple[IncomeStream, ...] = (),
    commitments: tuple[RecurringCommitment, ...] = (),
):
    maya = load_agent_state(MAYA_SCENARIO)
    return replace(
        maya,
        financial=FinancialState(
            currency="EUR",
            cash=cash,
            bank_balance=bank,
            savings=savings,
            emergency_fund=emergency,
            debts=debts,
            income_streams=income_streams,
            recurring_commitments=commitments,
        ),
    )


def config(*, duration_weeks: int = 1, seed: int = 1, start_date: str = "2026-01-05") -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(
            name="passive-test",
            seed=seed,
            duration_weeks=duration_weeks,
            start_date=__import__("datetime").date.fromisoformat(start_date),
        ),
        city=CityConfig(name="Veyra"),
    )


def context(*, week: int = 1, seed: int = 1, start_date: str = "2026-01-05") -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=config(seed=seed, start_date=start_date),
        rng=__import__("random").Random(seed),
    )


def event_catalog() -> EventCatalog:
    return EventCatalog((event_definition(),), event_probability=1.0)


def event_definition() -> EventDefinition:
    return EventDefinition(
        event_id="choice_event",
        version="1",
        category="test",
        base_weight=1.0,
        conditions=(),
        weight_modifiers=(),
        cooldown_weeks=0,
        tags=("test",),
        title="Choice event",
        summary="A deterministic event.",
        options=(
            EventOption(
                option_id="chosen",
                label="Chosen",
                summary="Chosen option.",
                short_term_value=0.2,
            ),
        ),
    )


def routine_decision(profile_id: str) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"decision_routine_{profile_id}",
        agent_id="maya",
        week=1,
        source_event_id="weekly_routine",
        source_event_version="1",
        time_pressure=0.0,
        available_option_ids=(profile_id,),
        unavailable_option_ids=(),
        chosen_option_id=profile_id,
        evaluations=(
            OptionEvaluation(
                option_id=profile_id,
                available=True,
                unavailable_reason="",
                deterministic_score=1.0,
                controlled_noise=0.0,
                final_score=1.0,
                components=(
                    DecisionScoreComponent(
                        name="fixture",
                        signal=1.0,
                        weight=1.0,
                        contribution=1.0,
                    ),
                ),
            ),
        ),
        strongest_positive_factors=("fixture",),
        strongest_negative_factors=(),
    )


def routine_raw(profile_id: str) -> dict[str, object]:
    return {
        "profile_id": profile_id,
        "label": profile_id,
        "summary": "Synthetic routine.",
        "estimated_cost": "10.00",
        "time_cost_hours": 1.0,
        "energy_cost": 1.0,
        "short_term_value": 0.0,
        "future_value": 0.0,
        "perceived_risk": 0.0,
        "uncertainty": 0.0,
        "social_value": 0.0,
        "social_pressure": 0.0,
        "autonomy_value": 0.0,
        "learning_value": 0.0,
        "health_value": 0.0,
        "comfort_value": 0.0,
        "goal_tags": [],
        "food_budget": "5.00",
        "transport_budget": "3.00",
        "discretionary_budget": "2.00",
        "social_contact": 0.5,
        "physical_activity": 0.5,
        "recovery_intensity": 0.5,
    }
