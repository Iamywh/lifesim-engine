from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import AgentState, IdentityState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.consequences import (
    ConsequenceApplicationError,
    ConsequenceCatalog,
    ConsequenceEngine,
    ConsequenceRuntimeState,
    DecisionConsequenceTransition,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    ScheduledConsequenceTransition,
    ScheduledEffect,
    StateEffectDefinition,
    load_consequence_catalog,
    parse_consequence_catalog,
)
from lifesim.decisions import DecisionEngine, DecisionEngineTransition
from lifesim.decisions.model import DecisionRecord, DecisionScoreComponent, OptionEvaluation
from lifesim.engine import LifeSimEngine
from lifesim.events import (
    EventCatalog,
    EventCondition,
    EventDefinition,
    EventEngine,
    EventEngineTransition,
    EventOccurrence,
    EventOption,
    load_event_catalog,
)
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")
STARTER_EVENTS = Path("configs/events/starter.toml")
STARTER_CONSEQUENCES = Path("configs/consequences/starter.toml")


def make_config(*, duration_weeks: int = 1, seed: int = 42) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(
            name="consequence-test",
            seed=seed,
            duration_weeks=duration_weeks,
        ),
        city=CityConfig(name="Veyra"),
    )


def test_consequence_catalog_loading_and_reference_validation() -> None:
    events = load_event_catalog(STARTER_EVENTS)
    catalog = load_consequence_catalog(STARTER_CONSEQUENCES, event_catalog=events)

    assert len(catalog.definitions) == 10
    assert catalog.find(
        event_id="social_invitation",
        event_version="1",
        option_id="decline_and_rest",
    )


def test_invalid_catalog_data_references_and_duplicate_keys_fail_fast() -> None:
    events = EventCatalog((event_definition(options=(option("known"),)),), event_probability=1.0)

    with pytest.raises(ValueError, match="Unsupported consequence write path"):
        StateEffectDefinition(path="identity.agent_id", delta=1.0)

    with pytest.raises(TypeError, match="monetary"):
        StateEffectDefinition(path="financial.bank_balance", delta=-1.0)

    with pytest.raises(TypeError, match="integer"):
        StateEffectDefinition(path="mental.stress", delta=1.0, delay_weeks=True)

    with pytest.raises(ValueError, match="unknown"):
        parse_consequence_catalog(
            {
                "consequences": [
                    consequence_raw(
                        event_id="unknown",
                        option_id="known",
                    )
                ]
            },
            event_catalog=events,
        )

    with pytest.raises(ValueError, match="unique"):
        ConsequenceCatalog(
            (
                consequence_definition(option_id="known"),
                consequence_definition(option_id="known"),
            )
        )


def test_exact_decimal_deltas_serialize_to_strings() -> None:
    effect = StateEffectDefinition(
        path="financial.bank_balance",
        delta=Decimal("-14.00"),
    )

    assert effect.to_dict()["delta"] == "-14.00"


def test_deterministic_immediate_effects_change_state_and_record_trace() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = ConsequenceEngine(
        ConsequenceCatalog(
            (
                consequence_definition(
                    effects=(
                        StateEffectDefinition(
                            path="financial.bank_balance",
                            delta=Decimal("-14.00"),
                        ),
                        StateEffectDefinition(path="mental.stress", delta=3.0),
                    ),
                ),
            )
        )
    )

    next_state, runtime, records = engine.resolve_decisions(
        maya,
        context(),
        (occurrence(),),
        (decision_record(),),
        ConsequenceRuntimeState(),
    )

    assert next_state.financial.bank_balance == Decimal("966.00")
    assert next_state.mental.stress == 60.0
    assert records[0].effect_applications[0].before == Decimal("980.00")
    assert records[0].effect_applications[0].after == Decimal("966.00")
    assert runtime.history.records == records


def test_bounded_float_clamping_is_audited() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    record = apply_single_effect(maya, StateEffectDefinition(path="health.energy", delta=100.0))

    application = record.effect_applications[0]
    assert application.before == 64.0
    assert application.after == 100.0
    assert application.clamped is True


def test_monetary_underflow_fails_without_partial_mutation() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = ConsequenceEngine(
        ConsequenceCatalog(
            (
                consequence_definition(
                    effects=(
                        StateEffectDefinition(path="mental.stress", delta=10.0),
                        StateEffectDefinition(
                            path="financial.bank_balance",
                            delta=Decimal("-5000.00"),
                        ),
                    ),
                ),
            )
        )
    )

    with pytest.raises(ConsequenceApplicationError, match="monetary"):
        engine.resolve_decisions(
            maya,
            context(),
            (occurrence(),),
            (decision_record(),),
            ConsequenceRuntimeState(),
        )

    assert maya.mental.stress == 57.0
    assert maya.financial.bank_balance == Decimal("980.00")


def test_chosen_option_only_unchosen_has_no_effect() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = ConsequenceEngine(
        ConsequenceCatalog(
            (
                consequence_definition(
                    option_id="chosen",
                    effects=(StateEffectDefinition(path="mental.stress", delta=3.0),),
                ),
                consequence_definition(
                    option_id="unchosen",
                    effects=(StateEffectDefinition(path="mental.stress", delta=40.0),),
                ),
            )
        )
    )

    next_state, _, _ = engine.resolve_decisions(
        maya,
        context(),
        (occurrence(options=(option("chosen"), option("unchosen"))),),
        (decision_record(chosen_option_id="chosen"),),
        ConsequenceRuntimeState(),
    )

    assert next_state.mental.stress == 60.0


def test_deterministic_weighted_actual_outcomes_and_rng_isolation() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    consequence = consequence_definition(
        outcomes=(
            OutcomeDefinition(
                "normal",
                weight=0.8,
                effects=(StateEffectDefinition(path="mental.stress", delta=1.0),),
            ),
            OutcomeDefinition(
                "traffic",
                weight=0.2,
                effects=(StateEffectDefinition(path="mental.stress", delta=8.0),),
            ),
        )
    )
    engine = ConsequenceEngine(ConsequenceCatalog((consequence,)))

    first = engine.resolve_decisions(
        maya,
        context(seed=5),
        (occurrence(),),
        (decision_record(),),
        ConsequenceRuntimeState(),
    )
    second = engine.resolve_decisions(
        maya,
        context(seed=5),
        (occurrence(),),
        (decision_record(),),
        ConsequenceRuntimeState(),
    )

    assert first[2][0].to_dict() == second[2][0].to_dict()
    assert first[2][0].outcome_roll is not None
    assert first[2][0].outcome_total_weight == 1.0


def test_delayed_effect_scheduling_due_week_and_conditional_skip() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    effect = StateEffectDefinition(
        path="mental.stress",
        delta=-5.0,
        delay_weeks=1,
        conditions=(EventCondition("string_equals", path="education.status", value="not_enrolled"),),
    )
    engine = ConsequenceEngine(ConsequenceCatalog((consequence_definition(effects=(effect,)),)))

    unchanged, runtime, records = engine.resolve_decisions(
        maya,
        context(week=1),
        (occurrence(),),
        (decision_record(),),
        ConsequenceRuntimeState(),
    )
    after_due, runtime, due_records = engine.apply_due_scheduled_effects(
        unchanged,
        context(week=2),
        runtime,
    )

    scheduled = records[0].scheduled_effects_created[0]
    application = due_records[0].effect_applications[0]
    assert unchanged == maya
    assert scheduled.created_week == 1
    assert scheduled.due_week == 2
    assert application.scheduled_effect_id == scheduled.scheduled_effect_id
    assert application.skipped is True
    assert application.skip_reason == "condition_not_met"
    assert after_due == maya
    assert runtime.pending_scheduled_effects == ()


def test_delayed_effects_due_before_same_week_event_eligibility() -> None:
    maya = replace(load_agent_state(MAYA_SCENARIO), health=replace(load_agent_state(MAYA_SCENARIO).health, energy=64.0))
    pending = ScheduledEffect(
        scheduled_effect_id="scheduled_energy",
        source_decision_id="decision_previous",
        source_consequence_id="consequence_previous",
        source_event_id="previous",
        source_event_version="1",
        chosen_option_id="rest",
        source_outcome_id=None,
        created_week=0,
        due_week=1,
        effect=StateEffectDefinition(path="health.energy", delta=10.0),
    )
    consequence_engine = ConsequenceEngine(ConsequenceCatalog())
    event_catalog = EventCatalog(
        (
            event_definition(
                event_id="energy_gate",
                conditions=(EventCondition("numeric_gte", path="health.energy", value=70.0),),
                options=(),
            ),
        ),
        event_probability=1.0,
    )
    result = LifeSimEngine(
        make_config(duration_weeks=1, seed=1),
        transitions=(
            RuntimeSeedTransition(
                ConsequenceRuntimeState(pending_scheduled_effects=(pending,))
            ),
            ScheduledConsequenceTransition(consequence_engine),
            EventEngineTransition(EventEngine(event_catalog)),
        ),
    ).run(initial_agent=maya)

    assert result.states[1].agent_state.health.energy == 74.0
    assert result.states[1].events[0].event_id == "energy_gate"


def test_decision_cannot_be_processed_twice() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = ConsequenceEngine(ConsequenceCatalog((consequence_definition(),)))
    runtime = ConsequenceRuntimeState()
    _, runtime, _ = engine.resolve_decisions(
        maya,
        context(),
        (occurrence(),),
        (decision_record(),),
        runtime,
    )

    with pytest.raises(ValueError, match="already been processed"):
        engine.resolve_decisions(
            maya,
            context(),
            (occurrence(),),
            (decision_record(),),
            runtime,
        )


def test_multiple_same_week_decisions_apply_in_deterministic_sequence() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = ConsequenceEngine(
        ConsequenceCatalog(
            (
                consequence_definition(
                    event_id="a",
                    option_id="chosen",
                    effects=(StateEffectDefinition(path="mental.stress", delta=2.0),),
                ),
                consequence_definition(
                    event_id="b",
                    option_id="chosen",
                    effects=(StateEffectDefinition(path="mental.stress", delta=3.0),),
                ),
            )
        )
    )

    next_state, _, records = engine.resolve_decisions(
        maya,
        context(),
        (
            occurrence(event_id="a"),
            occurrence(event_id="b"),
        ),
        (
            decision_record(decision_id="b_decision", event_id="b"),
            decision_record(decision_id="a_decision", event_id="a"),
        ),
        ConsequenceRuntimeState(),
    )

    assert [record.source_event_id for record in records] == ["a", "b"]
    assert next_state.mental.stress == 62.0


def test_generic_agent_input_immutability_and_repeated_run_isolation() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    generic = replace(
        maya,
        identity=IdentityState(
            agent_id="alex",
            display_name="Alex",
            age_years=25,
            pronouns="they/them",
            life_stage="young_adult",
            origin_city="Dublin",
            current_city="Veyra",
            background="Generic consequence test agent.",
        ),
    )
    before = generic.to_dict()
    engine = LifeSimEngine(
        make_config(duration_weeks=1, seed=1),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog_with_options())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(
                ConsequenceEngine(ConsequenceCatalog((consequence_definition(),)))
            ),
        ),
    )

    first = engine.run(initial_agent=generic)
    second = engine.run(initial_agent=generic)

    assert generic.to_dict() == before
    assert first.to_dict() == second.to_dict()
    assert first.consequence_history is not second.consequence_history
    assert first.states[1].agent_state.identity.agent_id == "alex"


def test_pending_effects_beyond_duration_remain_serialized() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(
        make_config(duration_weeks=1, seed=1),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog_with_options())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(
                ConsequenceEngine(
                    ConsequenceCatalog(
                        (
                            consequence_definition(
                                effects=(
                                    StateEffectDefinition(
                                        path="mental.stress",
                                        delta=2.0,
                                        delay_weeks=3,
                                    ),
                                )
                            ),
                        )
                    )
                )
            ),
        ),
    ).run(initial_agent=maya)

    assert engine.pending_scheduled_effects[0].due_week == 4
    assert engine.to_dict()["pending_scheduled_effects"][0]["due_week"] == 4


def test_noop_consequence_configuration_preserves_m3_m4_timelines() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event_catalog = event_catalog_with_options()
    base = LifeSimEngine(
        make_config(duration_weeks=2, seed=9),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog)),
            DecisionEngineTransition(DecisionEngine()),
        ),
    ).run(initial_agent=maya)
    with_noop = LifeSimEngine(
        make_config(duration_weeks=2, seed=9),
        transitions=(
            ScheduledConsequenceTransition(ConsequenceEngine(ConsequenceCatalog())),
            EventEngineTransition(EventEngine(event_catalog)),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(ConsequenceCatalog())),
        ),
    ).run(initial_agent=maya)

    assert [state.to_dict()["events"] for state in base.states[1:]] == [
        state.to_dict()["events"] for state in with_noop.states[1:]
    ]
    assert [state.to_dict()["decisions"] for state in base.states[1:]] == [
        state.to_dict()["decisions"] for state in with_noop.states[1:]
    ]


def test_real_consequences_can_alter_future_event_eligibility() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event_catalog = EventCatalog(
        (
            event_definition(
                event_id="first",
                options=(option("chosen"),),
                cooldown_weeks=10,
            ),
            event_definition(
                event_id="second",
                conditions=(EventCondition("numeric_gte", path="mental.stress", value=60.0),),
                options=(option("chosen"),),
            ),
        ),
        event_probability=1.0,
        max_events_per_week=1,
    )
    consequence_catalog = ConsequenceCatalog(
        (
            consequence_definition(
                event_id="first",
                effects=(StateEffectDefinition(path="mental.stress", delta=5.0),),
            ),
        )
    )

    result = LifeSimEngine(
        make_config(duration_weeks=2, seed=1),
        transitions=(
            ScheduledConsequenceTransition(ConsequenceEngine(consequence_catalog)),
            EventEngineTransition(EventEngine(event_catalog)),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence_catalog)),
        ),
    ).run(initial_agent=maya)

    assert result.states[1].events[0].event_id == "first"
    assert result.states[2].events[0].event_id == "second"


def test_cli_outputs_consequences_history_and_pending_json(tmp_path: Path) -> None:
    event_path = tmp_path / "events.toml"
    consequence_path = tmp_path / "consequences.toml"
    event_path.write_text(event_catalog_text(), encoding="utf-8")
    consequence_path.write_text(
        """
[[consequences]]
event_id = "choice_event"
event_version = "1"
option_id = "chosen"

[[consequences.effects]]
path = "mental.stress"
delta = 2.0
delay_weeks = 0

[[consequences.effects]]
path = "mental.stress"
delta = -1.0
delay_weeks = 10
""".strip(),
        encoding="utf-8",
    )
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
            str(event_path),
            "--consequence-catalog",
            str(consequence_path),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    output = json.loads(completed.stdout)

    assert output["states"][1]["consequences"][0]["chosen_option_id"] == "chosen"
    assert output["consequence_history"]["records"]
    assert output["pending_scheduled_effects"][0]["due_week"] == 11


def apply_single_effect(agent: AgentState, effect: StateEffectDefinition) -> Any:
    next_state, _, records = ConsequenceEngine(
        ConsequenceCatalog((consequence_definition(effects=(effect,)),))
    ).resolve_decisions(
        agent,
        context(),
        (occurrence(),),
        (decision_record(),),
        ConsequenceRuntimeState(),
    )
    assert next_state is not agent
    return records[0]


def context(*, week: int = 1, seed: int = 1) -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=make_config(seed=seed),
        rng=__import__("random").Random(seed),
    )


def event_definition(
    *,
    event_id: str = "choice_event",
    conditions: tuple[EventCondition, ...] = (),
    options: tuple[EventOption, ...] = (EventOption(
        option_id="chosen",
        label="Chosen",
        summary="Chosen option.",
    ),),
    cooldown_weeks: int = 0,
) -> EventDefinition:
    return EventDefinition(
        event_id=event_id,
        version="1",
        category="test",
        base_weight=1.0,
        conditions=conditions,
        weight_modifiers=(),
        cooldown_weeks=cooldown_weeks,
        tags=("test",),
        title="Choice event",
        summary="A synthetic choice event.",
        options=options,
    )


def option(option_id: str = "chosen") -> EventOption:
    return EventOption(
        option_id=option_id,
        label=option_id.title(),
        summary=f"Synthetic {option_id} option.",
        short_term_value=0.2 if option_id == "chosen" else 0.0,
    )


def occurrence(
    *,
    event_id: str = "choice_event",
    options: tuple[EventOption, ...] = (EventOption(
        option_id="chosen",
        label="Chosen",
        summary="Chosen option.",
    ),),
) -> EventOccurrence:
    return EventOccurrence(
        event_id=event_id,
        version="1",
        week=1,
        category="test",
        effective_weight=1.0,
        title="Choice event",
        summary="A synthetic choice event.",
        tags=("test",),
        options=options,
    )


def decision_record(
    *,
    decision_id: str = "decision_test",
    event_id: str = "choice_event",
    chosen_option_id: str = "chosen",
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        agent_id="maya",
        week=1,
        source_event_id=event_id,
        source_event_version="1",
        time_pressure=0.0,
        available_option_ids=(chosen_option_id,),
        unavailable_option_ids=(),
        chosen_option_id=chosen_option_id,
        evaluations=(evaluation(chosen_option_id),),
        strongest_positive_factors=("short_term_value",),
        strongest_negative_factors=(),
    )


def evaluation(option_id: str) -> OptionEvaluation:
    return OptionEvaluation(
        option_id=option_id,
        available=True,
        unavailable_reason="",
        deterministic_score=1.0,
        controlled_noise=0.0,
        final_score=1.0,
        components=(
            DecisionScoreComponent(
                name="short_term_value",
                signal=1.0,
                weight=1.0,
                contribution=1.0,
            ),
        ),
    )


def consequence_definition(
    *,
    event_id: str = "choice_event",
    option_id: str = "chosen",
    effects: tuple[StateEffectDefinition, ...] = (
        StateEffectDefinition(path="mental.stress", delta=1.0),
    ),
    outcomes: tuple[OutcomeDefinition, ...] = (),
) -> OptionConsequenceDefinition:
    return OptionConsequenceDefinition(
        event_id=event_id,
        event_version="1",
        option_id=option_id,
        effects=effects,
        outcomes=outcomes,
    )


def consequence_raw(*, event_id: str = "choice_event", option_id: str = "chosen") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_version": "1",
        "option_id": option_id,
        "effects": [
            {
                "path": "mental.stress",
                "delta": 1.0,
            }
        ],
    }


def event_catalog_with_options() -> EventCatalog:
    return EventCatalog((event_definition(),), event_probability=1.0)


def event_catalog_text() -> str:
    return """
[event_settings]
max_events_per_week = 1
event_probability = 1.0

[[events]]
event_id = "choice_event"
version = "1"
category = "test"
base_weight = 1.0
cooldown_weeks = 0
tags = ["test"]
title = "Choice event"
summary = "A deterministic event."

[[events.options]]
option_id = "chosen"
label = "Chosen"
summary = "Chosen option."
estimated_cost = "0.00"
time_cost_hours = 0.0
energy_cost = 0.0
short_term_value = 0.3
future_value = 0.0
perceived_risk = 0.0
uncertainty = 0.0
social_value = 0.0
social_pressure = 0.0
autonomy_value = 0.0
learning_value = 0.0
health_value = 0.0
comfort_value = 0.0
goal_tags = []
""".strip()


class RuntimeSeedTransition:
    def __init__(self, runtime: ConsequenceRuntimeState) -> None:
        self._runtime = runtime

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        return WeeklyTransitionResult(agent_state=state, consequence_runtime=self._runtime)
