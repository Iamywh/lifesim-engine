from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path

import pytest

from lifesim.adaptation.catalog import load_adaptation_catalog, parse_adaptation_catalog
from lifesim.adaptation.engine import AdaptationEngine, AdaptationTransition
from lifesim.adaptation.model import (
    AdaptationHistory,
    AdaptationRuntimeState,
    AdaptationWeekRecord,
    BehaviorEvidenceRecord,
    HabitDefinition,
    PersonalityAnchor,
    TraitEvidenceAccumulator,
    TraitEvidenceMapping,
)
from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import Habit, HabitsState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.decisions.engine import DecisionEngine, DecisionEngineTransition
from lifesim.development.model import DevelopmentEfficiencyAudit, DevelopmentWeekRecord
from lifesim.employment.model import ApplicationStageRecord
from lifesim.engine import LifeSimEngine
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.learning.model import ExperienceEvaluation, LearningRecord
from lifesim.passive.catalog import load_routine_catalog
from lifesim.passive.model import RoutineWeekRecord
from lifesim.rng import create_rng
from lifesim.social.model import (
    SocialInteractionOutcomeAudit,
    SocialInteractionRecord,
    SocialOutcomeProbability,
)
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
    development_event = replace(tagged_event(week=1), event_id="weekly_development")
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
                routine_record(routine_decision.decision_id, profile_id=routine_decision.chosen_option_id or ""),
                routine_record(routine_decision.decision_id, profile_id=routine_decision.chosen_option_id or ""),
            ),
            development_records=(
                development_record(
                    development_decision.decision_id,
                    profile_id=development_decision.chosen_option_id or "",
                ),
            ),
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


def test_source_specific_provenance_rejects_mismatched_records() -> None:
    state = load_agent_state(MAYA)
    routine_event = tagged_event()
    routine_decision = DecisionEngine().decide_event(state, context(events=(routine_event,)), routine_event)
    wrong_profile = "rest" if routine_decision.chosen_option_id != "rest" else "study"
    with pytest.raises(ValueError, match="profile_id"):
        engine().adapt(
            state,
            context(
                events=(routine_event,),
                decisions=(routine_decision,),
                passive_records=(routine_record(routine_decision.decision_id, profile_id=wrong_profile),),
            ),
            AdaptationRuntimeState(),
        )

    development_event = replace(tagged_event(), event_id="weekly_development")
    development_decision = DecisionEngine().decide_event(
        state,
        context(events=(development_event,)),
        development_event,
    )
    wrong_development = "rest" if development_decision.chosen_option_id != "rest" else "study"
    with pytest.raises(ValueError, match="profile_id"):
        engine().adapt(
            state,
            context(
                events=(development_event,),
                decisions=(development_decision,),
                development_records=(
                    development_record(development_decision.decision_id, profile_id=wrong_development),
                ),
            ),
            AdaptationRuntimeState(),
        )

    social_event = social_event_with_options(EventOption("connect:lina", "Connect", "Connect.", behavior_tags=("social_outreach",), social_value=0.6))
    social_decision = DecisionEngine().decide_event(state, context(events=(social_event,)), social_event)
    with pytest.raises(ValueError, match="interaction_type"):
        engine().adapt(
            state,
            context(
                events=(social_event,),
                decisions=(social_decision,),
                social_records=(
                    social_record(
                        social_decision.decision_id,
                        "connect:lina",
                        interaction_type="seek_support",
                        contact_id="lina",
                    ),
                ),
            ),
            AdaptationRuntimeState(),
        )

    employment_event = employment_event_with_options()
    employment_decision = DecisionEngine().decide_event(state, context(events=(employment_event,)), employment_event)
    assert employment_decision.chosen_option_id == "apply"
    with pytest.raises(ValueError, match="stage/detail/status"):
        engine().adapt(
            state,
            context(
                events=(employment_event,),
                decisions=(employment_decision,),
                employment_records=(
                    employment_record(
                        employment_decision.decision_id,
                        stage="APPLICATION_DECISION",
                        detail="skipped",
                        status_after="SKIPPED",
                    ),
                ),
            ),
            AdaptationRuntimeState(),
        )


def test_trait_evidence_decay_is_once_per_elapsed_week() -> None:
    state = load_agent_state(MAYA)
    runtime = AdaptationRuntimeState(
        trait_accumulators=(
            TraitEvidenceAccumulator(
                "discipline",
                signed_evidence=1.0,
                evidence_weight=1.0,
                distinct_weeks=(1,),
                source_families=("development",),
                last_evidence_week=1,
                last_updated_week=1,
            ),
        )
    )

    for week in range(2, 5):
        state, runtime, _ = engine().adapt(state, context(week=week), runtime)

    accumulator = next(item for item in runtime.trait_accumulators if item.trait == "discipline")
    assert accumulator.signed_evidence == pytest.approx(0.92**3)
    assert accumulator.evidence_weight == pytest.approx(0.92**3)
    assert accumulator.last_evidence_week == 1
    assert accumulator.last_updated_week == 4

    event = tagged_event(week=5)
    decision = DecisionEngine().decide_event(state, context(week=5, events=(event,)), event)
    state, runtime, _ = engine().adapt(
        state,
        context(
            week=5,
            events=(event,),
            decisions=(decision,),
            passive_records=(routine_record(decision.decision_id, week=5, profile_id=decision.chosen_option_id or ""),),
        ),
        runtime,
    )
    renewed = next(item for item in runtime.trait_accumulators if item.trait == "discipline")
    assert renewed.last_evidence_week == 5
    assert renewed.last_updated_week == 5
    assert renewed.signed_evidence > accumulator.signed_evidence * 0.92


def test_cost_restraint_evidence_is_weaker_when_financially_constrained() -> None:
    rich = funded_state(Decimal("10000.00"))
    constrained = funded_state(Decimal("10.00"))

    rich_record = adapt_cost_choice(rich)
    constrained_record = adapt_cost_choice(constrained)
    rich_frugality = next(item for item in rich_record.trait_evidence if item.evidence_key == "cost_restraint")
    constrained_frugality = next(
        (item for item in constrained_record.trait_evidence if item.evidence_key == "cost_restraint"),
        None,
    )
    constrained_weight = 0.0 if constrained_frugality is None else constrained_frugality.weight

    assert rich_frugality.weight > constrained_weight
    assert constrained_weight < rich_frugality.weight / 2


def test_source_family_diversity_counts_causal_systems_not_analytical_views() -> None:
    state = load_agent_state(MAYA)
    runtime = AdaptationRuntimeState()
    development_event = replace(tagged_event(week=1), event_id="weekly_development")
    development_decision = DecisionEngine().decide_event(
        state,
        context(week=1, events=(development_event,)),
        development_event,
    )
    state, runtime, record = engine().adapt(
        state,
        context(
            week=1,
            events=(development_event,),
            decisions=(development_decision,),
            development_records=(
                development_record(
                    development_decision.decision_id,
                    profile_id=development_decision.chosen_option_id or "",
                ),
            ),
        ),
        runtime,
    )

    assert {item.source_type for item in record.trait_evidence if item.trait == "discipline"} >= {
        "behavior",
        "choice_metric",
    }
    assert {item.source_family for item in record.trait_evidence if item.trait == "discipline"} == {"development"}
    discipline = next(item for item in runtime.trait_accumulators if item.trait == "discipline")
    assert discipline.source_families == ("development",)

    social = social_event_with_options(
        EventOption(
            "engage:lina",
            "Engage",
            "Engage.",
            behavior_tags=("social_exploration",),
            social_value=0.6,
            learning_value=0.1,
            requires_full_estimated_cost=False,
        ),
        week=2,
    )
    social_decision = DecisionEngine().decide_event(state, context(week=2, events=(social,)), social)
    state, runtime, _ = engine().adapt(
        state,
        context(
            week=2,
            events=(social,),
            decisions=(social_decision,),
            social_records=(
                social_record(
                    social_decision.decision_id,
                    social_decision.chosen_option_id or "",
                    interaction_type="engage",
                    contact_id="lina",
                    week=2,
                ),
            ),
        ),
        runtime,
    )
    curiosity = next(item for item in runtime.trait_accumulators if item.trait == "curiosity")
    assert curiosity.source_families == ("development", "social")


def test_unavailable_behavior_does_not_create_nonuse_weakening() -> None:
    state = replace(
        funded_state(Decimal("0.00")),
        habits=HabitsState(
            routine_stability=34.0,
            items=(
                Habit(
                    "study consistency",
                    "weekly",
                    50.0,
                    "study_consistency",
                    ("study",),
                    formed_week=1,
                    last_reinforced_week=1,
                ),
            ),
        ),
    )
    event = EventOccurrence(
        "weekly_routine",
        "1",
        5,
        "routine",
        1.0,
        "Routine",
        "Routine.",
        ("routine",),
        options=(
            EventOption("study", "Study", "Study.", estimated_cost=Decimal("100.00"), behavior_tags=("study",)),
            EventOption("rest", "Rest", "Rest.", health_value=0.4, behavior_tags=("recovery",)),
        ),
    )
    decision = DecisionEngine().decide_event(state, context(week=5, events=(event,)), event)

    next_state, _, record = engine().adapt(
        state,
        context(week=5, events=(event,), decisions=(decision,)),
        AdaptationRuntimeState(),
    )

    assert decision.unavailable_option_ids == ("study",)
    assert next_state.habits.items[0].strength == 50.0
    assert not record.habit_strength_changes


def test_non_moral_evidence_constraints() -> None:
    exhausted = replace(
        load_agent_state(MAYA),
        health=replace(load_agent_state(MAYA).health, energy=20.0),
        mental=replace(load_agent_state(MAYA).mental, recovery_need=80.0),
    )
    recovery_event = EventOccurrence(
        "weekly_routine",
        "1",
        1,
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
                future_value=0.6,
                energy_cost=35.0,
                behavior_tags=("study",),
                requires_full_estimated_cost=False,
            ),
            EventOption(
                "rest",
                "Rest",
                "Rest.",
                health_value=0.7,
                comfort_value=0.5,
                behavior_tags=("recovery",),
                requires_full_estimated_cost=False,
            ),
        ),
    )
    recovery_decision = DecisionEngine().decide_event(exhausted, context(events=(recovery_event,)), recovery_event)
    assert recovery_decision.chosen_option_id == "rest"
    _, _, recovery_record = engine().adapt(
        exhausted,
        context(
            events=(recovery_event,),
            decisions=(recovery_decision,),
            passive_records=(routine_record(recovery_decision.decision_id, profile_id="rest"),),
        ),
        AdaptationRuntimeState(),
    )
    assert not [
        item
        for item in recovery_record.trait_evidence
        if item.trait == "discipline" and item.signal < 0.0
    ]

    state = load_agent_state(MAYA)
    support_event = social_event_with_options(
        EventOption(
            "seek_support:lina",
            "Seek support",
            "Seek support.",
            behavior_tags=("support_seeking", "social_outreach"),
            social_value=0.5,
        )
    )
    support_decision = DecisionEngine().decide_event(state, context(events=(support_event,)), support_event)
    _, _, support_record = engine().adapt(
        state,
        context(
            events=(support_event,),
            decisions=(support_decision,),
            social_records=(
                social_record(
                    support_decision.decision_id,
                    "seek_support:lina",
                    interaction_type="seek_support",
                    contact_id="lina",
                ),
            ),
        ),
        AdaptationRuntimeState(),
    )
    assert all(item.trait != "independence" for item in support_record.trait_evidence)

    _, _, adversity_record = engine().adapt(
        state,
        context(learning_records=(learning_record("negative"),)),
        AdaptationRuntimeState(),
    )
    assert all(item.trait != "resilience" for item in adversity_record.trait_evidence)

    _, _, work_record = engine().adapt(
        state,
        context(employment_records=(employment_record("", stage="WORK_WEEK", detail="worked", status_after="ACTIVE"),)),
        AdaptationRuntimeState(),
    )
    assert all(item.trait != "conscientiousness" for item in work_record.trait_evidence)


def test_adaptation_runtime_rejects_orphan_processed_ids_and_anchor_is_immutable() -> None:
    evidence = BehaviorEvidenceRecord(
        week=1,
        decision_id="decision-1",
        source_event_id="event",
        source_event_version="1",
        source_option_id="option",
        source_system="routine",
        behavior_tags=("study",),
        executed=True,
        source_record_ids=("routine:1:decision-1:option",),
    )
    record = AdaptationWeekRecord(
        week=1,
        behavior_evidence=(evidence,),
        processed_experience_ids=("consequence-1",),
    )
    history = AdaptationHistory((record,))
    with pytest.raises(ValueError, match="processed behavior ids"):
        AdaptationRuntimeState(
            history=history,
            processed_weeks=(1,),
            processed_behavior_decision_ids=("decision-1", "orphan"),
            processed_experience_ids=("consequence-1",),
        )
    with pytest.raises(ValueError, match="processed experience ids"):
        AdaptationRuntimeState(
            history=history,
            processed_weeks=(1,),
            processed_behavior_decision_ids=("decision-1",),
            processed_experience_ids=("consequence-1", "orphan"),
        )

    _, runtime, _ = engine().adapt(load_agent_state(MAYA), context(), AdaptationRuntimeState())
    assert isinstance(runtime.personality_anchor, PersonalityAnchor)
    assert runtime.to_dict()["personality_anchor"]["discipline"] == 0.55
    with pytest.raises(FrozenInstanceError):
        runtime.personality_anchor.discipline = 0.1


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
    repeat = routine_record(decision.decision_id, week=1, profile_id=decision.chosen_option_id or "", weeks=3)
    next_state, _, record = engine().adapt(
        state,
        context(events=(event,), decisions=(decision,), passive_records=(repeat,)),
        AdaptationRuntimeState(),
    )
    switch_event = tagged_event(week=2)
    switch_decision = DecisionEngine().decide_event(next_state, context(week=2, events=(switch_event,)), switch_event)
    switched = routine_record(
        switch_decision.decision_id,
        week=2,
        profile_id=switch_decision.chosen_option_id or "",
        previous=decision.chosen_option_id or "",
        weeks=1,
    )
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
    assert runtime.personality_anchor is not None
    assert state.personality.discipline - runtime.personality_anchor.value("discipline") < 0.12


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


def social_event_with_options(*options: EventOption, week: int = 1) -> EventOccurrence:
    return EventOccurrence(
        "weekly_social_focus",
        "1",
        week,
        "social",
        1.0,
        "Social",
        "Social.",
        ("social",),
        options=options,
    )


def employment_event_with_options(*, week: int = 1) -> EventOccurrence:
    return EventOccurrence(
        "employment_opening:cafe_assistant:1",
        "1",
        week,
        "employment",
        1.0,
        "Opening",
        "Opening.",
        ("employment",),
        options=(
            EventOption(
                "apply",
                "Apply",
                "Apply.",
                future_value=0.7,
                learning_value=0.1,
                behavior_tags=("job_search",),
                requires_full_estimated_cost=False,
            ),
            EventOption("skip", "Skip", "Skip.", future_value=-0.2, requires_full_estimated_cost=False),
        ),
    )


def cost_choice_event(*, week: int = 1) -> EventOccurrence:
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
                "cheap",
                "Cheap",
                "Cheap.",
                estimated_cost=Decimal("0.00"),
                short_term_value=0.2,
                behavior_tags=("low_spend",),
                requires_full_estimated_cost=False,
            ),
            EventOption(
                "expensive",
                "Expensive",
                "Expensive.",
                estimated_cost=Decimal("100.00"),
                short_term_value=-0.2,
                behavior_tags=("spending",),
                requires_full_estimated_cost=False,
            ),
        ),
    )


def adapt_cost_choice(state):
    event = cost_choice_event()
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    assert decision.chosen_option_id == "cheap"
    _, _, record = engine().adapt(
        state,
        context(
            events=(event,),
            decisions=(decision,),
            passive_records=(routine_record(decision.decision_id, profile_id="cheap"),),
        ),
        AdaptationRuntimeState(),
    )
    return record


def funded_state(resources: Decimal):
    state = load_agent_state(MAYA)
    return replace(
        state,
        financial=replace(
            state.financial,
            cash=resources,
            bank_balance=Decimal("0.00"),
            savings=Decimal("0.00"),
            emergency_fund=Decimal("0.00"),
        ),
    )


def social_record(
    decision_id: str,
    option_id: str,
    *,
    interaction_type: str,
    contact_id: str,
    week: int = 1,
) -> SocialInteractionRecord:
    return SocialInteractionRecord(
        week=week,
        decision_id=decision_id,
        option_id=option_id,
        contact_id=contact_id,
        interaction_type=interaction_type,
        outcome=SocialInteractionOutcomeAudit((SocialOutcomeProbability("ok", 1.0),), 0.0, "ok"),
    )


def employment_record(
    decision_id: str,
    *,
    week: int = 1,
    stage: str = "APPLICATION_DECISION",
    detail: str = "submitted",
    status_after: str = "SUBMITTED",
) -> ApplicationStageRecord:
    return ApplicationStageRecord(
        application_id="application-1",
        week=week,
        stage=stage,
        status_after=status_after,
        job_id="cafe_assistant",
        job_version="1",
        decision_id=decision_id,
        detail=detail,
    )


def learning_record(consequence_id: str) -> LearningRecord:
    return LearningRecord(
        consequence_id=consequence_id,
        source_decision_id="decision",
        source_event_id="event",
        source_event_version="1",
        source_option_id="option",
        week_learned=1,
        evaluation=ExperienceEvaluation(
            valence=-0.8,
            salience=0.8,
            affected_domains=("health",),
            strongest_negative_effects=("health.energy",),
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
