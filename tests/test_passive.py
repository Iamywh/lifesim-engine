from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

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
    parse_event_catalog,
)
from lifesim.passive import (
    PassiveCashflowEngine,
    PassiveCashflowTransition,
    PassiveLifeRuntimeState,
    RoutineCatalog,
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
    assert catalog.get("balanced_week").minimum_food_budget == Decimal("38.00")
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

    bad_food = routine_raw("bad_food")
    bad_food["minimum_food_budget"] = "6.00"
    bad_food["food_budget"] = "5.00"
    with pytest.raises(ValueError, match="minimum_food_budget"):
        parse_routine_catalog({"profiles": [bad_food]})


def test_ordinary_event_cost_gate_remains_all_or_nothing_by_default() -> None:
    maya = with_financial(bank=Decimal("0.00"))
    parsed = parse_event_catalog(
        {
            "events": [
                {
                    "event_id": "parsed",
                    "version": "1",
                    "category": "test",
                    "base_weight": 1.0,
                    "title": "Parsed",
                    "summary": "Parsed event.",
                    "options": [
                        {
                            "option_id": "partial",
                            "label": "Partial",
                            "summary": "Partial option.",
                            "estimated_cost": "1.00",
                            "requires_full_estimated_cost": False,
                            "time_cost_hours": 0.0,
                            "energy_cost": 0.0,
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
                        }
                    ],
                }
            ],
        }
    )
    occurrence = event_definition(
        EventOption(
            option_id="expensive",
            label="Expensive",
            summary="Requires full payment.",
            estimated_cost=Decimal("1.00"),
            short_term_value=1.0,
        )
    ).to_occurrence(week=1, effective_weight=1.0)

    assert parsed.definitions[0].options[0].requires_full_estimated_cost is False
    assert EventOption("default", "Default", "Default.").requires_full_estimated_cost is True
    decision = DecisionEngine().decide_event(maya, context(), occurrence)

    assert decision.chosen_option_id is None
    assert decision.unavailable_option_ids == ("expensive",)
    assert decision.evaluations[0].unavailable_reason == "insufficient_liquid_resources"


def test_routine_options_remain_evaluable_under_scarcity_and_choose_real_profile() -> None:
    broke = with_financial(bank=Decimal("0.00"))
    engine = RoutineEngine(load_routine_catalog(ROUTINES))

    runtime, occurrence, _, decision = engine.plan(
        broke,
        context(),
        PassiveLifeRuntimeState(),
        DecisionEngine(),
        DecisionHistory(),
    )

    assert all(option.requires_full_estimated_cost is False for option in occurrence.options)
    assert decision.chosen_option_id is not None
    assert runtime.planned_routine_profile_id == decision.chosen_option_id
    assert set(decision.available_option_ids) == {
        "balanced_week",
        "austerity_home_week",
        "low_cost_active_week",
        "recovery_focus_week",
        "social_week",
    }


def test_scarcity_can_make_lower_cost_routines_score_relatively_better() -> None:
    scarce = with_financial(bank=Decimal("40.00"))
    engine = RoutineEngine(load_routine_catalog(ROUTINES))
    _, _, _, decision = engine.plan(
        scarce,
        context(),
        PassiveLifeRuntimeState(),
        DecisionEngine(),
        DecisionHistory(),
    )

    scores = {evaluation.option_id: evaluation for evaluation in decision.evaluations}
    assert scores["austerity_home_week"].available is True
    assert scores["social_week"].available is True
    austerity_cost = component(
        scores["austerity_home_week"],
        "financial_cost",
    ).contribution
    social_cost = component(scores["social_week"], "financial_cost").contribution
    assert austerity_cost > social_cost


def test_routine_planning_rejects_missing_choice_instead_of_falling_back() -> None:
    engine = RoutineEngine(RoutineCatalog(()))

    with pytest.raises(ValueError, match="choose an available routine profile"):
        engine.plan(
            load_agent_state(MAYA_SCENARIO),
            context(),
            PassiveLifeRuntimeState(),
            DecisionEngine(),
            DecisionHistory(),
        )


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


def test_accumulated_isolation_can_create_negative_mood_pressure_without_cliff() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    strained = replace(
        maya,
        mental=replace(maya.mental, loneliness=90.0),
        personality=replace(maya.personality, social_need=1.0),
        routine=replace(maya.routine, low_social_streak=8),
    )

    first_week, _, first_record = execute_profile(maya, "austerity_home_week")
    strained_week, _, strained_record = execute_profile(strained, "austerity_home_week")

    first_mood = next(effect for effect in first_record.effects if effect.path == "mental.mood")
    strained_mood = next(effect for effect in strained_record.effects if effect.path == "mental.mood")

    assert first_week.mental.mood >= maya.mental.mood
    assert first_mood.delta > 0
    assert strained_mood.delta < 0
    assert strained_week.mental.mood < strained.mental.mood
    assert strained_week.mental.loneliness > strained.mental.loneliness


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


def test_food_security_uses_minimum_adequate_food_not_full_planned_food_budget() -> None:
    maya = with_financial(bank=Decimal("38.00"))

    next_state, _, record = execute_profile(maya, "social_week")

    assert record.spending[0].amount_due == Decimal("58.00")
    assert record.spending[0].amount_paid == Decimal("38.00")
    assert next_state.needs.food_security == maya.needs.food_security
    assert not any(effect.reason == "food_shortfall" for effect in record.effects)


def test_routine_spending_uses_only_bank_and_cash_when_liquid_funds_are_sufficient() -> None:
    maya = with_financial(
        bank=Decimal("200.00"),
        savings=Decimal("100.00"),
        emergency=Decimal("100.00"),
    )

    next_state, _, record = execute_profile(maya, "social_week")

    assert next_state.financial.savings == Decimal("100.00")
    assert next_state.financial.emergency_fund == Decimal("100.00")
    assert record.spending[0].amount_paid == Decimal("58.00")
    assert {transfer.source for entry in record.spending for transfer in entry.funding} == {
        "bank_balance",
    }


def test_routine_minimum_food_can_use_savings_but_optional_spending_cannot() -> None:
    maya = with_financial(
        savings=Decimal("100.00"),
        emergency=Decimal("100.00"),
    )

    next_state, _, record = execute_profile(maya, "social_week")

    food, transport, discretionary = record.spending
    assert food.amount_due == Decimal("58.00")
    assert food.amount_paid == Decimal("38.00")
    assert tuple(transfer.source for transfer in food.funding) == ("savings",)
    assert transport.amount_paid == Decimal("0.00")
    assert discretionary.amount_paid == Decimal("0.00")
    assert next_state.financial.savings == Decimal("62.00")
    assert next_state.financial.emergency_fund == Decimal("100.00")
    assert next_state.needs.food_security == maya.needs.food_security


def test_routine_minimum_food_uses_emergency_only_after_other_sources() -> None:
    maya = with_financial(
        savings=Decimal("20.00"),
        emergency=Decimal("100.00"),
    )

    next_state, _, record = execute_profile(maya, "social_week")

    food = record.spending[0]
    assert food.amount_paid == Decimal("38.00")
    assert [(transfer.source, transfer.amount) for transfer in food.funding] == [
        ("savings", Decimal("20.00")),
        ("emergency_fund", Decimal("18.00")),
    ]
    assert next_state.financial.emergency_fund == Decimal("82.00")
    assert next_state.needs.food_security == maya.needs.food_security


def test_social_week_does_not_exhaust_reserves_for_discretionary_spending() -> None:
    maya = with_financial(
        savings=Decimal("1000.00"),
        emergency=Decimal("1000.00"),
    )

    next_state, _, record = execute_profile(maya, "social_week")

    assert record.spending[2].kind == "routine_discretionary"
    assert record.spending[2].amount_due == Decimal("55.00")
    assert record.spending[2].amount_paid == Decimal("0.00")
    assert next_state.financial.savings == Decimal("962.00")
    assert next_state.financial.emergency_fund == Decimal("1000.00")


def test_contractual_obligations_keep_full_reserve_funding_order() -> None:
    maya = with_financial(
        savings=Decimal("30.00"),
        emergency=Decimal("100.00"),
        commitments=(RecurringCommitment("rent", Decimal("50.00"), "weekly", "housing"),),
    )

    next_state, _, record = run_cashflow(maya, context())

    assert record.entries[0].paid is True
    assert [(transfer.source, transfer.amount) for transfer in record.entries[0].funding] == [
        ("savings", Decimal("30.00")),
        ("emergency_fund", Decimal("20.00")),
    ]
    assert next_state.financial.savings == Decimal("0.00")
    assert next_state.financial.emergency_fund == Decimal("80.00")
    assert not next_state.financial.arrears


def test_food_security_penalty_scales_with_minimum_shortfall_without_negative_balances() -> None:
    low_food, _, low_record = execute_profile(with_financial(bank=Decimal("10.00")), "social_week")
    less_low_food, _, less_low_record = execute_profile(
        with_financial(bank=Decimal("28.00")),
        "social_week",
    )

    low_penalty = next(effect for effect in low_record.effects if effect.reason == "food_shortfall")
    less_low_penalty = next(
        effect for effect in less_low_record.effects if effect.reason == "food_shortfall"
    )

    assert low_penalty.delta == pytest.approx(-2.8)
    assert less_low_penalty.delta == pytest.approx(-1.0)
    assert low_food.needs.food_security < less_low_food.needs.food_security
    for state in (low_food, less_low_food):
        assert state.financial.cash == Decimal("0.00")
        assert state.financial.bank_balance == Decimal("0.00")
        assert state.financial.savings == Decimal("0.00")
        assert state.financial.emergency_fund == Decimal("0.00")


def test_passive_cashflow_transition_fails_if_processed_twice_for_same_week() -> None:
    maya = with_financial(
        bank=Decimal("100.00"),
        commitments=(RecurringCommitment("rent", Decimal("20.00"), "weekly", "housing"),),
    )
    transition = PassiveCashflowTransition(PassiveCashflowEngine())
    first = transition.apply(maya, context())

    with pytest.raises(ValueError, match="already processed"):
        transition.apply(first.agent_state, replace(context(), passive_runtime=first.passive_runtime))


def test_routine_planning_and_execution_fail_if_processed_twice_for_same_week() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    routine_engine = RoutineEngine(load_routine_catalog(ROUTINES))
    planning = RoutinePlanningTransition(routine_engine, DecisionEngine())
    execution = RoutineExecutionTransition(routine_engine)
    planned = planning.apply(maya, context())

    with pytest.raises(ValueError, match="already processed"):
        planning.apply(maya, replace(context(), passive_runtime=planned.passive_runtime))

    executed = execution.apply(maya, replace(context(), passive_runtime=planned.passive_runtime))
    with pytest.raises(ValueError, match="already processed"):
        execution.apply(
            executed.agent_state,
            replace(context(), passive_runtime=executed.passive_runtime),
        )


def test_passive_history_rejects_duplicate_weeks_and_runtime_inconsistency() -> None:
    _, runtime, record = run_cashflow(
        with_financial(
            bank=Decimal("10.00"),
            commitments=(RecurringCommitment("rent", Decimal("20.00"), "weekly", "housing"),),
        ),
        context(),
    )

    with pytest.raises(ValueError, match="cashflow week"):
        runtime.history.record_cashflow(record)
    with pytest.raises(ValueError, match="exactly match cashflow history"):
        PassiveLifeRuntimeState(history=runtime.history)
    with pytest.raises(ValueError, match="exactly match cashflow history"):
        PassiveLifeRuntimeState(processed_cashflow_weeks=(1,))
    with pytest.raises(ValueError, match="exactly match routine history"):
        PassiveLifeRuntimeState(processed_routine_execution_weeks=(1,))


def test_missed_obligation_audits_state_effects_and_reconciles_funding() -> None:
    maya = with_financial(
        bank=Decimal("30.00"),
        commitments=(RecurringCommitment("rent", Decimal("100.00"), "weekly", "housing"),),
    )

    next_state, _, record = run_cashflow(maya, context())

    entry = record.entries[0]
    assert entry.arrear_balance_after == Decimal("70.00")
    assert entry.funding[0].amount == Decimal("30.00")
    assert {effect.path for effect in record.effects} == {
        "mental.stress",
        "needs.housing_security",
    }
    assert {effect.source for effect in record.effects} == {"commitment:rent"}
    assert next_state.mental.stress > maya.mental.stress
    assert next_state.needs.housing_security < maya.needs.housing_security


def test_passive_audit_invariants_reject_contradictory_records() -> None:
    from lifesim.passive import CashflowEntry, FundingTransfer, RoutineEffectApplication

    with pytest.raises(ValueError, match="funding transfer total"):
        CashflowEntry(
            entry_id="bad",
            kind="commitment",
            name="bad",
            amount_due=Decimal("10.00"),
            amount_paid=Decimal("5.00"),
            cadence="weekly",
            due_date="2026-01-05",
            paid=False,
            funding=(FundingTransfer("bank_balance", Decimal("4.00")),),
        )
    with pytest.raises(ValueError, match="outflow amount_paid"):
        CashflowEntry(
            entry_id="overpay",
            kind="commitment",
            name="overpay",
            amount_due=Decimal("10.00"),
            amount_paid=Decimal("11.00"),
            cadence="weekly",
            due_date="2026-01-05",
            paid=True,
        )
    with pytest.raises(ValueError, match="paid flag"):
        CashflowEntry(
            entry_id="bad_paid",
            kind="commitment",
            name="bad paid",
            amount_due=Decimal("10.00"),
            amount_paid=Decimal("5.00"),
            cadence="weekly",
            due_date="2026-01-05",
            paid=True,
        )
    with pytest.raises(ValueError, match="delta"):
        RoutineEffectApplication(
            path="mental.stress",
            before=10.0,
            after=12.0,
            delta=3.0,
            clamped=False,
            reason="bad_delta",
        )


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


def test_probabilistic_passive_income_rng_isolated_from_event_decision_and_consequence_rng() -> None:
    base_state = with_financial(bank=Decimal("100.00"))
    income_state = with_financial(
        bank=Decimal("100.00"),
        income_streams=(IncomeStream("zero gig", Decimal("0.00"), "weekly", 0.5),),
    )
    consequence = ConsequenceCatalog(
        (
            OptionConsequenceDefinition(
                event_id="choice_event",
                event_version="1",
                option_id="chosen",
                outcomes=(
                    OutcomeDefinition(
                        "normal",
                        0.55,
                        effects=(StateEffectDefinition(path="mental.stress", delta=1.0),),
                    ),
                    OutcomeDefinition(
                        "hard",
                        0.45,
                        effects=(StateEffectDefinition(path="mental.stress", delta=6.0),),
                    ),
                ),
            ),
        )
    )
    base = LifeSimEngine(
        config(duration_weeks=1, seed=33),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence)),
        ),
    ).run(initial_agent=base_state)
    with_income = LifeSimEngine(
        config(duration_weeks=1, seed=33),
        transitions=(
            PassiveCashflowTransition(PassiveCashflowEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence)),
        ),
    ).run(initial_agent=income_state)

    assert with_income.states[1].passive_records[0].entries[0].roll is not None
    assert base.states[1].event_traces[0].to_dict() == with_income.states[1].event_traces[0].to_dict()
    base_decision = base.states[1].decisions[0]
    income_decision = with_income.states[1].decisions[0]
    assert base_decision.evaluations[0].controlled_noise == (
        income_decision.evaluations[0].controlled_noise
    )
    assert base.states[1].consequences[0].to_dict() == with_income.states[1].consequences[0].to_dict()


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
        planned_routine_week=1,
        processed_routine_planning_weeks=(1,),
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


def event_definition(*options: EventOption) -> EventDefinition:
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
        options=options
        or (
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
        "minimum_food_budget": "4.00",
        "transport_budget": "3.00",
        "discretionary_budget": "2.00",
        "social_contact": 0.5,
        "physical_activity": 0.5,
        "recovery_intensity": 0.5,
    }


def component(evaluation: OptionEvaluation, name: str) -> DecisionScoreComponent:
    return next(item for item in evaluation.components if item.name == name)
