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
    EventCondition,
    EventDefinition,
    EventEngine,
    EventEngineTransition,
    EventOccurrence,
    EventOption,
    load_event_catalog,
    parse_event_catalog,
)
from lifesim.weekly import WeeklyContext

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")
STARTER_CATALOG = Path("configs/events/starter.toml")


def make_config(*, duration_weeks: int = 1, seed: int = 42) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(
            name="decision-test",
            seed=seed,
            duration_weeks=duration_weeks,
        ),
        city=CityConfig(name="Veyra"),
    )


def test_event_option_toml_loading_and_decimal_serialization() -> None:
    catalog = load_event_catalog(STARTER_CATALOG)
    option = catalog.definitions[0].options[1]

    assert option.option_id == "pay_for_faster_transport"
    assert option.estimated_cost == Decimal("14.00")
    assert option.to_dict()["estimated_cost"] == "14.00"
    assert catalog.definitions[0].time_pressure == 0.55


def test_malformed_option_validation() -> None:
    with pytest.raises(TypeError, match="estimated_cost"):
        option(estimated_cost="18.00")

    with pytest.raises(TypeError, match="energy_cost"):
        option(energy_cost=True)

    with pytest.raises(ValueError, match="short_term_value"):
        option(short_term_value=1.2)

    with pytest.raises(TypeError, match="goal_tags"):
        option(goal_tags="social")

    with pytest.raises(TypeError, match="estimated_cost"):
        parse_event_catalog(
            {
                "event_settings": {"event_probability": 1.0},
                "events": [
                    {
                        "event_id": "bad_option",
                        "version": "1",
                        "category": "test",
                        "base_weight": 1.0,
                        "tags": ["test"],
                        "title": "Bad option",
                        "summary": "Bad option data.",
                        "options": [
                            {
                                **option().to_dict(),
                                "estimated_cost": 18.0,
                            }
                        ],
                    }
                ],
            }
        )


def test_duplicate_option_ids_fail_fast_for_python_and_toml() -> None:
    duplicate_options = (option(option_id="same"), option(option_id="same"))

    with pytest.raises(ValueError, match="option_id"):
        EventDefinition(
            event_id="duplicate_definition_options",
            version="1",
            category="test",
            base_weight=1.0,
            conditions=(),
            weight_modifiers=(),
            cooldown_weeks=0,
            tags=("test",),
            title="Duplicate options",
            summary="Duplicate option ids should fail.",
            options=duplicate_options,
        )

    with pytest.raises(ValueError, match="option_id"):
        occurrence(options=duplicate_options)

    with pytest.raises(ValueError, match="option_id"):
        parse_event_catalog(
            {
                "events": [
                    {
                        "event_id": "duplicate_toml_options",
                        "version": "1",
                        "category": "test",
                        "base_weight": 1.0,
                        "tags": ["test"],
                        "title": "Duplicate TOML options",
                        "summary": "Duplicate option ids should fail while loading.",
                        "options": [
                            option(option_id="same").to_dict(),
                            option(option_id="same").to_dict(),
                        ],
                    }
                ],
            }
        )


def test_option_evaluation_invariants_reject_contradictory_audit_records() -> None:
    component = DecisionScoreComponent(
        name="short_term_value",
        signal=1.0,
        weight=1.0,
        contribution=1.0,
    )

    with pytest.raises(ValueError, match="Available options"):
        OptionEvaluation(
            option_id="bad_available",
            available=True,
            unavailable_reason="",
            deterministic_score=1.0,
            controlled_noise=None,
            final_score=1.0,
            components=(component,),
        )

    with pytest.raises(ValueError, match="final_score"):
        OptionEvaluation(
            option_id="bad_sum",
            available=True,
            unavailable_reason="",
            deterministic_score=1.0,
            controlled_noise=0.2,
            final_score=1.0,
            components=(component,),
        )

    with pytest.raises(ValueError, match="Unavailable options"):
        OptionEvaluation(
            option_id="bad_unavailable",
            available=False,
            unavailable_reason="availability_conditions",
            deterministic_score=1.0,
            controlled_noise=None,
            final_score=None,
            components=(),
        )

    with pytest.raises(ValueError, match="score components"):
        OptionEvaluation(
            option_id="bad_components",
            available=False,
            unavailable_reason="availability_conditions",
            deterministic_score=None,
            controlled_noise=None,
            final_score=None,
            components=(component,),
        )


def test_decision_record_and_history_invariants_reject_ambiguous_records() -> None:
    available = evaluation("available")
    unavailable = evaluation("unavailable", available=False)

    with pytest.raises(ValueError, match="unique"):
        decision_record(
            available_option_ids=("available", "available"),
            evaluations=(available,),
        )

    with pytest.raises(ValueError, match="disjoint"):
        decision_record(
            available_option_ids=("available",),
            unavailable_option_ids=("available",),
            evaluations=(available,),
        )

    with pytest.raises(ValueError, match="match listed"):
        decision_record(
            available_option_ids=("available",),
            evaluations=(available, unavailable),
        )

    with pytest.raises(ValueError, match="listed as unavailable"):
        decision_record(
            available_option_ids=("unavailable",),
            evaluations=(unavailable,),
        )

    with pytest.raises(ValueError, match="unique"):
        decision_record(
            available_option_ids=("available",),
            evaluations=(available, available),
        )

    with pytest.raises(ValueError, match="belong"):
        decision_record(
            available_option_ids=("available",),
            unavailable_option_ids=("unavailable",),
            chosen_option_id="unavailable",
            evaluations=(available, unavailable),
        )

    with pytest.raises(ValueError, match="chosen_option_id"):
        decision_record(
            available_option_ids=(),
            unavailable_option_ids=("unavailable",),
            chosen_option_id="unavailable",
            evaluations=(unavailable,),
        )

    with pytest.raises(ValueError, match="chosen_option_id"):
        decision_record(
            available_option_ids=("available",),
            chosen_option_id=None,
            evaluations=(available,),
        )

    with pytest.raises(ValueError, match="decision_id"):
        DecisionHistory((decision_record(), decision_record()))


def test_option_availability_conditions_and_exclusion_from_choice() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event = occurrence(
        options=(
            option(
                option_id="available",
                short_term_value=0.1,
            ),
            option(
                option_id="later",
                short_term_value=1.0,
                availability_conditions=(EventCondition("week_gte", value=3),),
            ),
        )
    )

    record = DecisionEngine().decide_event(maya, context(week=1), event)

    assert record.chosen_option_id == "available"
    assert record.available_option_ids == ("available",)
    assert record.unavailable_option_ids == ("later",)
    assert record.evaluations[1].unavailable_reason == "availability_conditions"


def test_frugal_and_non_frugal_agents_can_prefer_differently() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    frugal = replace(base, personality=replace(base.personality, frugality=1.0))
    relaxed = replace(base, personality=replace(base.personality, frugality=0.0))
    event = occurrence(
        options=(
            option(option_id="free", short_term_value=0.1, future_value=0.1),
            option(
                option_id="paid",
                estimated_cost=Decimal("1300.00"),
                short_term_value=0.8,
                future_value=0.7,
            ),
        )
    )

    assert DecisionEngine().decide_event(frugal, context(seed=4), event).chosen_option_id == "free"
    assert DecisionEngine().decide_event(relaxed, context(seed=4), event).chosen_option_id == "paid"


def test_risk_tolerant_and_risk_averse_agents_can_prefer_differently() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    averse = replace(base, personality=replace(base.personality, risk_tolerance=0.0))
    tolerant = replace(base, personality=replace(base.personality, risk_tolerance=1.0))
    event = occurrence(
        options=(
            option(option_id="safe", future_value=0.1, perceived_risk=0.0),
            option(option_id="risky", future_value=0.5, perceived_risk=0.95),
        )
    )

    assert DecisionEngine().decide_event(averse, context(seed=8), event).chosen_option_id == "safe"
    assert DecisionEngine().decide_event(tolerant, context(seed=8), event).chosen_option_id == "risky"


def test_low_energy_changes_option_evaluation() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    low_energy = replace(base, health=replace(base.health, energy=15.0))
    high_energy = replace(base, health=replace(base.health, energy=95.0))
    event = occurrence(
        options=(
            option(option_id="rest", comfort_value=0.5, health_value=0.3),
            option(option_id="effort", future_value=0.7, energy_cost=65.0),
        )
    )

    assert DecisionEngine().decide_event(low_energy, context(seed=2), event).chosen_option_id == "rest"
    assert DecisionEngine().decide_event(high_energy, context(seed=2), event).chosen_option_id == "effort"


def test_stress_and_time_pressure_change_deliberation_and_noise() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    calm = replace(base, mental=replace(base.mental, stress=5.0, mental_load=5.0))
    stressed = replace(base, mental=replace(base.mental, stress=95.0, mental_load=95.0))
    calm_event = occurrence(time_pressure=0.0, options=(option(option_id="choice"),))
    urgent_event = occurrence(time_pressure=1.0, options=(option(option_id="choice"),))

    calm_eval = DecisionEngine().decide_event(calm, context(seed=5), calm_event).evaluations[0]
    urgent_eval = DecisionEngine().decide_event(stressed, context(seed=5), urgent_event).evaluations[0]

    calm_components = {component.name: component for component in calm_eval.components}
    urgent_components = {component.name: component for component in urgent_eval.components}
    assert urgent_components["short_term_value"].weight > calm_components["short_term_value"].weight
    assert urgent_components["future_value"].weight < calm_components["future_value"].weight
    assert abs(urgent_eval.controlled_noise) > abs(calm_eval.controlled_noise)


def test_goal_tags_and_priority_influence_score() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event = occurrence(
        options=(
            option(option_id="untagged", future_value=0.1),
            option(option_id="aligned", goal_tags=("finance", "stability")),
        )
    )

    record = DecisionEngine().decide_event(maya, context(seed=7), event)
    aligned = next(evaluation for evaluation in record.evaluations if evaluation.option_id == "aligned")
    component = next(component for component in aligned.components if component.name == "goal_alignment")

    assert record.chosen_option_id == "aligned"
    assert component.contribution > 0


def test_controlled_noise_is_deterministic() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event = occurrence(options=(option(option_id="a"), option(option_id="b")))
    engine = DecisionEngine()

    first = engine.decide_event(maya, context(seed=99), event)
    second = engine.decide_event(maya, context(seed=99), event)

    assert first.to_dict() == second.to_dict()


def test_decision_event_week_must_match_context_week() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    with pytest.raises(ValueError, match="event week"):
        DecisionEngine().decide_event(
            maya,
            context(week=2),
            occurrence(event_id="wrong_week", options=(option(option_id="stay"),)),
        )


def test_noise_cannot_reverse_clearly_superior_option() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event = occurrence(
        time_pressure=1.0,
        options=(
            option(option_id="clearly_superior", short_term_value=0.8, future_value=0.8),
            option(option_id="clearly_inferior", short_term_value=0.0, future_value=0.0),
        ),
    )

    record = DecisionEngine().decide_event(maya, context(seed=39), event)
    by_id = {evaluation.option_id: evaluation for evaluation in record.evaluations}
    deterministic_gap = (
        by_id["clearly_superior"].deterministic_score
        - by_id["clearly_inferior"].deterministic_score
    )

    assert deterministic_gap > 0.24
    assert record.chosen_option_id == "clearly_superior"


def test_repeated_runs_and_decision_history_are_isolated() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(
        make_config(duration_weeks=2, seed=12),
        transitions=(
            EventEngineTransition(EventEngine(catalog_with_options())),
            DecisionEngineTransition(DecisionEngine()),
        ),
    )

    first = engine.run(initial_agent=maya)
    second = engine.run(initial_agent=maya)

    assert first.to_dict() == second.to_dict()
    assert first.decision_history is not second.decision_history
    assert len(first.decision_history.records) == 2
    assert len(second.decision_history.records) == 2


def test_generic_non_maya_agent_can_decide() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    generic = replace(
        maya,
        identity=IdentityState(
            agent_id="alex",
            display_name="Alex",
            age_years=24,
            pronouns="they/them",
            life_stage="young_adult",
            origin_city="Dublin",
            current_city="Veyra",
            background="Reusable decision test agent.",
        ),
    )

    record = DecisionEngine().decide_event(generic, context(seed=1), occurrence())

    assert record.agent_id == "alex"
    assert record.chosen_option_id == "stay"


def test_events_without_options_produce_no_decision() -> None:
    result = DecisionEngine().decide_events(
        load_agent_state(MAYA_SCENARIO),
        context(seed=1),
        (occurrence(options=()),),
        DecisionHistory(),
    )

    assert result.records == ()
    assert result.history.records == ()


def test_multiple_events_in_one_week_produce_separate_decisions() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    events = (
        occurrence(event_id="first"),
        occurrence(event_id="second"),
    )

    result = DecisionEngine().decide_events(maya, context(seed=2), events, DecisionHistory())

    assert [record.source_event_id for record in result.records] == ["first", "second"]
    assert len(result.history.records) == 2


def test_later_weekly_transition_can_inspect_same_week_decisions() -> None:
    seen: list[tuple[int, tuple[str, ...]]] = []
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(
        make_config(duration_weeks=1, seed=12),
        transitions=(
            EventEngineTransition(EventEngine(catalog_with_options())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionProbeTransition(seen),
        ),
    )

    result = engine.run(initial_agent=maya)

    assert seen == [(1, ("go",))]
    assert result.states[1].decisions[0].chosen_option_id == "go"


def test_decision_engine_does_not_mutate_agent_state() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    before = maya.to_dict()

    DecisionEngine().decide_event(maya, context(seed=1), occurrence())

    assert maya.to_dict() == before


def test_decision_serialization_is_deterministic_and_auditable() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    record = DecisionEngine().decide_event(
        maya,
        context(seed=3),
        occurrence(options=(option(option_id="stay"), option(option_id="go", social_value=0.4))),
    )
    serialized = record.to_dict()

    assert serialized["decision_id"].startswith("decision_")
    assert serialized["chosen_option_id"] == record.chosen_option_id
    assert serialized["evaluations"][0]["components"][0]["name"] == "short_term_value"
    assert serialized["strongest_positive_factors"]
    assert serialized == DecisionEngine().decide_event(
        maya,
        context(seed=3),
        occurrence(options=(option(option_id="stay"), option(option_id="go", social_value=0.4))),
    ).to_dict()


def test_decision_noise_does_not_alter_m3_event_timeline() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = catalog_with_options(event_probability=1.0)
    events_only = LifeSimEngine(
        make_config(duration_weeks=3, seed=21),
        transitions=(EventEngineTransition(EventEngine(catalog)),),
    ).run(initial_agent=maya)
    with_decisions = LifeSimEngine(
        make_config(duration_weeks=3, seed=21),
        transitions=(
            EventEngineTransition(EventEngine(catalog)),
            DecisionEngineTransition(DecisionEngine()),
        ),
    ).run(initial_agent=maya)

    assert [state.to_dict()["events"] for state in events_only.states[1:]] == [
        state.to_dict()["events"] for state in with_decisions.states[1:]
    ]
    assert events_only.event_history.to_dict() == with_decisions.event_history.to_dict()


def test_demo_cli_outputs_decisions_and_history_as_json(tmp_path: Path) -> None:
    catalog_path = tmp_path / "decision_cli.toml"
    consequence_path = tmp_path / "empty_consequences.toml"
    catalog_path.write_text(
        """
[event_settings]
max_events_per_week = 1
event_probability = 1.0

[[events]]
event_id = "cli_decision_event"
version = "1"
category = "test"
base_weight = 1.0
cooldown_weeks = 0
tags = ["test"]
title = "CLI decision event"
summary = "A deterministic CLI event with options."
time_pressure = 0.2

[[events.options]]
option_id = "stay"
label = "Stay"
summary = "Stay put."
estimated_cost = "0.00"
time_cost_hours = 1.0
energy_cost = 1.0
short_term_value = 0.2
future_value = 0.0
perceived_risk = 0.0
uncertainty = 0.0
social_value = 0.0
social_pressure = 0.0
autonomy_value = 0.2
learning_value = 0.0
health_value = 0.0
comfort_value = 0.3
goal_tags = ["health"]
""".strip(),
        encoding="utf-8",
    )
    consequence_path.write_text("consequences = []", encoding="utf-8")
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
            str(catalog_path),
            "--consequence-catalog",
            str(consequence_path),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    output = json.loads(completed.stdout)

    assert output["states"][1]["decisions"][0]["chosen_option_id"] == "stay"
    assert output["decision_history"]["records"][0]["source_event_id"] == "cli_decision_event"


def context(*, week: int = 1, seed: int = 1) -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=make_config(seed=seed),
        rng=__import__("random").Random(seed),
    )


def occurrence(
    *,
    event_id: str = "choice_event",
    time_pressure: float = 0.0,
    options: tuple[EventOption, ...] | None = None,
) -> EventOccurrence:
    return EventOccurrence(
        event_id=event_id,
        version="1",
        week=1,
        category="test",
        effective_weight=1.0,
        title="Choice event",
        summary="A synthetic decision event.",
        tags=("test",),
        time_pressure=time_pressure,
        options=options if options is not None else (option(option_id="stay"), option(option_id="go")),
    )


def option(
    *,
    option_id: str = "option",
    label: str = "Option",
    summary: str = "A synthetic option.",
    availability_conditions: tuple[EventCondition, ...] = (),
    estimated_cost: Any = Decimal("0.00"),
    time_cost_hours: Any = 1.0,
    energy_cost: Any = 1.0,
    short_term_value: Any = 0.0,
    future_value: Any = 0.0,
    perceived_risk: Any = 0.0,
    uncertainty: Any = 0.0,
    social_value: Any = 0.0,
    social_pressure: Any = 0.0,
    autonomy_value: Any = 0.0,
    learning_value: Any = 0.0,
    health_value: Any = 0.0,
    comfort_value: Any = 0.0,
    goal_tags: Any = (),
) -> EventOption:
    return EventOption(
        option_id=option_id,
        label=label,
        summary=summary,
        availability_conditions=availability_conditions,
        estimated_cost=estimated_cost,
        time_cost_hours=time_cost_hours,
        energy_cost=energy_cost,
        short_term_value=short_term_value,
        future_value=future_value,
        perceived_risk=perceived_risk,
        uncertainty=uncertainty,
        social_value=social_value,
        social_pressure=social_pressure,
        autonomy_value=autonomy_value,
        learning_value=learning_value,
        health_value=health_value,
        comfort_value=comfort_value,
        goal_tags=goal_tags,
    )


def evaluation(option_id: str, *, available: bool = True) -> OptionEvaluation:
    if not available:
        return OptionEvaluation(
            option_id=option_id,
            available=False,
            unavailable_reason="availability_conditions",
            deterministic_score=None,
            controlled_noise=None,
            final_score=None,
            components=(),
        )
    return OptionEvaluation(
        option_id=option_id,
        available=True,
        unavailable_reason="",
        deterministic_score=1.0,
        controlled_noise=0.1,
        final_score=1.1,
        components=(
            DecisionScoreComponent(
                name="short_term_value",
                signal=1.0,
                weight=1.0,
                contribution=1.0,
            ),
        ),
    )


def decision_record(
    *,
    available_option_ids: tuple[str, ...] = ("available",),
    unavailable_option_ids: tuple[str, ...] = (),
    chosen_option_id: str | None = "available",
    evaluations: tuple[OptionEvaluation, ...] | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id="decision_test",
        agent_id="maya",
        week=1,
        source_event_id="event",
        source_event_version="1",
        time_pressure=0.0,
        available_option_ids=available_option_ids,
        unavailable_option_ids=unavailable_option_ids,
        chosen_option_id=chosen_option_id,
        evaluations=evaluations if evaluations is not None else (evaluation("available"),),
        strongest_positive_factors=("short_term_value",),
        strongest_negative_factors=(),
    )


class DecisionProbeTransition:
    def __init__(self, seen: list[tuple[int, tuple[str, ...]]]) -> None:
        self._seen = seen

    def apply(self, state: AgentState, context: WeeklyContext) -> AgentState:
        self._seen.append(
            (
                context.week,
                tuple(decision.chosen_option_id for decision in context.decisions),
            )
        )
        return state


def catalog_with_options(*, event_probability: float = 1.0) -> EventCatalog:
    return EventCatalog(
        (
            EventDefinition(
                event_id="decision_event",
                version="1",
                category="test",
                base_weight=1.0,
                conditions=(),
                weight_modifiers=(),
                cooldown_weeks=0,
                tags=("test",),
                title="Decision event",
                summary="A synthetic decision event.",
                options=(option(option_id="stay"), option(option_id="go", future_value=0.2)),
            ),
        ),
        event_probability=event_probability,
    )
