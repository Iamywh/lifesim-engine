from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lifesim.adaptation.catalog import load_adaptation_catalog, parse_adaptation_catalog
from lifesim.adaptation.engine import AdaptationEngine, AdaptationTransition
from lifesim.adaptation.model import AdaptationRuntimeState, HabitDefinition, TraitEvidenceMapping
from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import Habit, HabitsState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.decisions.engine import DecisionEngine, DecisionEngineTransition
from lifesim.development.model import DevelopmentEfficiencyAudit, DevelopmentWeekRecord
from lifesim.engine import LifeSimEngine
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.passive.catalog import load_routine_catalog
from lifesim.passive.model import RoutineWeekRecord
from lifesim.rng import create_rng
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

ROOT = Path(__file__).resolve().parents[1]
MAYA = ROOT / "configs" / "scenarios" / "maya_start.toml"
ADAPTATION = ROOT / "configs" / "adaptation" / "starter.toml"
ROUTINES = ROOT / "configs" / "routines" / "starter.toml"


def test_starter_catalog_loads_and_validates() -> None:
    catalog = load_adaptation_catalog(ADAPTATION)

    assert catalog.habit("study_consistency").behavior_tags == ("study",)
    with pytest.raises(ValueError, match="unique"):
        parse_adaptation_catalog(
            {
                "habits": [
                    habit_raw("study_consistency"),
                    habit_raw("study_consistency"),
                ]
            }
        )
    with pytest.raises(ValueError, match="trait"):
        TraitEvidenceMapping("behavior_tag", "study", "not_a_trait", 0.1)
    with pytest.raises(ValueError, match="between"):
        HabitDefinition("bad", "bad", "weekly", ("study",), -1.0, 1.0, 0.1, 8.0, 3)


def test_backward_compatibility_and_maya_initial_values() -> None:
    old_habit = Habit("morning planning", "most weekdays", 35.0)
    old_option = EventOption("old", "Old", "Old.")
    maya = load_agent_state(MAYA)

    assert old_habit.habit_id == "morning_planning"
    assert old_option.behavior_tags == ()
    assert maya.personality.discipline == 0.55
    assert maya.personality.frugality == 0.63
    assert maya.habits.routine_stability == 34.0
    assert {habit.habit_id: habit.strength for habit in maya.habits.items} == {
        "evening_expense_notes": 29.0,
        "morning_planning": 35.0,
    }


def test_behavior_tags_propagate_to_generated_options() -> None:
    routine_catalog = load_routine_catalog(ROUTINES)
    option = next(
        option
        for option in __import__("lifesim.passive.engine").passive.engine._routine_occurrence(
            context(),
            routine_catalog,
        ).options
        if option.option_id == "low_cost_active_week"
    )

    assert option.behavior_tags == ("low_spend", "active_mobility")


def test_merely_offered_option_creates_no_behavior_evidence() -> None:
    state = load_agent_state(MAYA)
    event = tagged_event()
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    next_state, _, record = engine().adapt(
        state,
        context(events=(event,), decisions=(decision,)),
        AdaptationRuntimeState(),
    )

    assert record.behavior_evidence == ()
    assert next_state.habits == state.habits


def test_routine_execution_produces_behavior_evidence_and_latent_habit() -> None:
    state = load_agent_state(MAYA)
    event = tagged_event()
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    routine = routine_record(decision.decision_id, week=1, profile_id="study")

    _, runtime, record = engine().adapt(
        state,
        context(events=(event,), decisions=(decision,), passive_records=(routine,)),
        AdaptationRuntimeState(),
    )

    assert record.behavior_evidence[0].behavior_tags == ("study",)
    assert runtime.habit_candidates[0].latent_strength > 0.0
    assert "study_consistency" not in {habit.habit_id for habit in state.habits.items}


def test_execution_provenance_requires_real_same_week_decision() -> None:
    state = load_agent_state(MAYA)
    event = tagged_event()
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)

    with pytest.raises(ValueError, match="real M4 decision"):
        engine().adapt(
            state,
            context(passive_records=(routine_record("missing-decision", profile_id="study"),)),
            AdaptationRuntimeState(),
        )
    with pytest.raises(ValueError, match="current adaptation week"):
        engine().adapt(
            state,
            context(week=2, events=(event,), decisions=(decision,), passive_records=(routine_record(decision.decision_id, week=2, profile_id="study"),)),
            AdaptationRuntimeState(),
        )


def test_duplicate_source_decision_counts_once_and_preserves_execution_order() -> None:
    state = load_agent_state(MAYA)
    routine_event = tagged_event(week=1)
    development_event = tagged_event(week=1)
    development_event = replace(development_event, event_id="development_choice")
    routine_decision = DecisionEngine().decide_event(state, context(events=(routine_event,)), routine_event)
    development_decision = DecisionEngine().decide_event(
        state,
        context(events=(development_event,)),
        development_event,
    )

    _, runtime, record = engine().adapt(
        state,
        context(
            events=(routine_event, development_event),
            decisions=(routine_decision, development_decision),
            passive_records=(
                routine_record(routine_decision.decision_id, profile_id="study"),
                routine_record(routine_decision.decision_id, profile_id="study"),
            ),
            development_records=(development_record(development_decision.decision_id),),
        ),
        AdaptationRuntimeState(),
    )

    assert tuple(item.decision_id for item in record.behavior_evidence) == (
        routine_decision.decision_id,
        development_decision.decision_id,
    )
    assert runtime.processed_behavior_decision_ids == (
        routine_decision.decision_id,
        development_decision.decision_id,
    )


def test_repeated_distinct_weeks_can_form_modest_habit() -> None:
    state = load_agent_state(MAYA)
    runtime = AdaptationRuntimeState()
    for week in range(1, 5):
        event = tagged_event(week=week)
        decision = DecisionEngine().decide_event(state, context(week=week, events=(event,)), event)
        state, runtime, _ = engine().adapt(
            state,
            context(
                week=week,
                events=(event,),
                decisions=(decision,),
                passive_records=(routine_record(decision.decision_id, week=week, profile_id="study"),),
            ),
            runtime,
        )

    habit = next(habit for habit in state.habits.items if habit.habit_id == "study_consistency")
    assert 5.0 <= habit.strength <= 15.0
    assert habit.formed_week >= 3


def test_unobservable_legacy_habits_do_not_decay() -> None:
    state = load_agent_state(MAYA)
    next_state, _, _ = engine().adapt(state, context(), AdaptationRuntimeState())

    assert {
        habit.habit_id: habit.strength
        for habit in next_state.habits.items
        if habit.habit_id in {"morning_planning", "evening_expense_notes"}
    } == {"morning_planning": 35.0, "evening_expense_notes": 29.0}


def test_routine_stability_repetition_and_switch_are_bounded() -> None:
    state = load_agent_state(MAYA)
    event = tagged_event()
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    repeat = routine_record(decision.decision_id, week=1, profile_id="balanced_week", weeks=3)
    next_state, _, record = engine().adapt(
        state,
        context(events=(event,), decisions=(decision,), passive_records=(repeat,)),
        AdaptationRuntimeState(),
    )
    switch_event = tagged_event(week=2)
    switch_decision = DecisionEngine().decide_event(next_state, context(week=2, events=(switch_event,)), switch_event)
    switched = routine_record(switch_decision.decision_id, week=2, profile_id="social_week", previous="balanced_week", weeks=1)
    switched_state, _, switched_record = engine().adapt(
        next_state,
        context(week=2, events=(switch_event,), decisions=(switch_decision,), passive_records=(switched,)),
        AdaptationRuntimeState(),
    )

    assert record.routine_stability.after > record.routine_stability.before
    assert switched_record.routine_stability.after < switched_record.routine_stability.before
    assert abs(switched_state.habits.routine_stability - next_state.habits.routine_stability) < 2.0


def test_habit_familiarity_component_is_modest_and_tag_driven() -> None:
    state = replace(
        load_agent_state(MAYA),
        habits=HabitsState(
            routine_stability=50.0,
            items=(Habit("study consistency", "weekly", 80.0, "study_consistency", ("study",)),),
        ),
    )
    event = EventOccurrence(
        "choice",
        "1",
        1,
        "test",
        1.0,
        "Choice",
        "Choice.",
        ("test",),
        options=(
            EventOption("study", "Study", "Study.", behavior_tags=("study",), requires_full_estimated_cost=False),
            EventOption("other", "Other", "Other.", requires_full_estimated_cost=False),
        ),
    )
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    study_eval = next(item for item in decision.evaluations if item.option_id == "study")
    other_eval = next(item for item in decision.evaluations if item.option_id == "other")
    study_component = next(item for item in study_eval.components if item.name == "habit_familiarity")
    other_component = next(item for item in other_eval.components if item.name == "habit_familiarity")

    assert 0.0 < study_component.contribution <= 0.18
    assert other_component.contribution == 0.0


def test_personality_moves_only_after_accumulated_evidence_and_with_cap() -> None:
    state = load_agent_state(MAYA)
    runtime = AdaptationRuntimeState()
    for week in range(1, 12):
        event = tagged_event(week=week)
        decision = DecisionEngine().decide_event(state, context(week=week, events=(event,)), event)
        state, runtime, record = engine().adapt(
            state,
            context(
                week=week,
                events=(event,),
                decisions=(decision,),
                passive_records=(routine_record(decision.decision_id, week=week, profile_id="study"),),
            ),
            runtime,
        )

    discipline_changes = [change for change in record.personality_changes if change.trait == "discipline"]
    assert state.personality.discipline > 0.55
    assert discipline_changes
    assert max(abs(change.delta) for change in record.personality_changes) <= 0.0015
    assert state.personality.discipline - runtime.personality_anchor["discipline"] < 0.12


def test_adaptation_once_week_and_mutation_boundary() -> None:
    state = load_agent_state(MAYA)
    next_state, runtime, _ = engine().adapt(state, context(), AdaptationRuntimeState())

    with pytest.raises(ValueError, match="already processed"):
        engine().adapt(next_state, context(), runtime)
    assert replace(next_state, habits=state.habits, personality=state.personality) == state


def test_same_week_decisions_unchanged_when_adaptation_runs_last() -> None:
    state = load_agent_state(MAYA)
    event = tagged_event()
    baseline = LifeSimEngine(
        config(),
        transitions=(EmitEventTransition(event), DecisionEngineTransition(DecisionEngine())),
    ).run(initial_agent=state)
    with_m11 = LifeSimEngine(
        config(),
        transitions=(
            EmitEventTransition(event),
            DecisionEngineTransition(DecisionEngine()),
            AdaptationTransition(engine()),
        ),
    ).run(initial_agent=state)

    assert baseline.states[1].decisions[0].to_dict() == with_m11.states[1].decisions[0].to_dict()
    assert with_m11.states[1].agent_state.personality == state.personality


def test_cli_demo_outputs_adaptation_json_and_is_deterministic() -> None:
    args = [
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
        "--employment-catalog",
        "configs/employment/starter.toml",
        "--development-catalog",
        "configs/development/starter.toml",
        "--social-catalog",
        "configs/social/starter.toml",
        "--adaptation-catalog",
        "configs/adaptation/starter.toml",
    ]
    first = subprocess.check_output([sys.executable, *args], text=True)
    second = subprocess.check_output([sys.executable, *args], text=True)
    payload = json.loads(first)

    assert first == second
    assert payload["adaptation_history"]["records"]
    assert payload["adaptation_runtime"]["habit_candidates"]
    assert payload["states"][1]["adaptation_records"]


class EmitEventTransition:
    def __init__(self, event: EventOccurrence) -> None:
        self.event = event

    def apply(self, state, context):
        return WeeklyTransitionResult(agent_state=state, events=(self.event,))


def engine() -> AdaptationEngine:
    return AdaptationEngine(load_adaptation_catalog(ADAPTATION))


def context(*, week: int = 1, seed: int = 7, **overrides) -> WeeklyContext:
    return WeeklyContext(week=week, config=config(seed=seed), rng=create_rng(seed), **overrides)


def config(*, duration_weeks: int = 1, seed: int = 7) -> LifeSimConfig:
    return LifeSimConfig(SimulationConfig("adaptation-test", seed, duration_weeks), CityConfig("Test City"))


def tagged_event(*, week: int = 1) -> EventOccurrence:
    return EventOccurrence(
        "weekly_routine",
        "1",
        week,
        "routine",
        1.0,
        "Routine",
        "Routine.",
        ("routine",),
        options=(
            EventOption(
                "study",
                "Study",
                "Study.",
                future_value=0.4,
                learning_value=0.4,
                goal_tags=("education",),
                behavior_tags=("study",),
                requires_full_estimated_cost=False,
            ),
            EventOption("rest", "Rest", "Rest.", health_value=0.4, behavior_tags=("recovery",), requires_full_estimated_cost=False),
        ),
    )


def routine_record(
    decision_id: str,
    *,
    week: int = 1,
    profile_id: str,
    previous: str = "",
    weeks: int = 1,
) -> RoutineWeekRecord:
    return RoutineWeekRecord(
        week=week,
        profile_id=profile_id,
        previous_profile_id=previous,
        weeks_in_current_profile=weeks,
        low_social_streak=0,
        decision_id=decision_id,
        spending=(),
        effects=(),
    )


def development_record(decision_id: str, *, week: int = 1, profile_id: str = "study") -> DevelopmentWeekRecord:
    return DevelopmentWeekRecord(
        week=week,
        profile_id=profile_id,
        decision_id=decision_id,
        education_hours=8.0,
        practice=(),
        efficiency=DevelopmentEfficiencyAudit(
            energy_factor=1.0,
            stress_factor=1.0,
            mental_load_factor=1.0,
            recovery_factor=1.0,
            workload_factor=1.0,
            combined_workload_hours=8.0,
            study_ratio=1.0,
            effective_study_hours=8.0,
            effective_practice_hours=0.0,
            final_efficiency=1.0,
        ),
        education_progress=None,
        skill_developments=(),
        effects=(),
    )


def habit_raw(habit_id: str) -> dict:
    return {
        "habit_id": habit_id,
        "name": habit_id,
        "cadence": "weekly",
        "behavior_tags": ["study"],
        "formation_rate": 2.0,
        "reinforcement_rate": 1.0,
        "nonuse_decay_rate": 0.1,
        "formation_threshold": 8.0,
        "minimum_reinforcing_weeks": 3,
    }
