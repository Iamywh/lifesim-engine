from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import IdentityState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.engine import LifeSimEngine
from lifesim.events import (
    EventCatalog,
    EventCondition,
    EventDefinition,
    EventEngine,
    EventEngineTransition,
    EventHistory,
    WeightModifier,
    load_event_catalog,
    parse_event_catalog,
)
from lifesim.weekly import WeeklyContext

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")
STARTER_CATALOG = Path("configs/events/starter.toml")


def make_config(*, duration_weeks: int = 4, seed: int = 42) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(
            name="event-test",
            seed=seed,
            duration_weeks=duration_weeks,
        ),
        city=CityConfig(name="Veyra"),
    )


def test_event_catalog_loading() -> None:
    catalog = load_event_catalog(STARTER_CATALOG)

    assert catalog.max_events_per_week == 1
    assert catalog.event_probability == 0.45
    assert len(catalog.definitions) == 5
    assert catalog.definitions[0].event_id == "minor_transit_disruption"


def test_invalid_event_configuration_fails_clearly() -> None:
    with pytest.raises(TypeError, match="events"):
        parse_event_catalog({"event_settings": {"event_probability": 0.2}})

    with pytest.raises(ValueError, match="Unsupported"):
        EventCondition(condition_type="python_eval", path="mental.stress", value=10)

    event = event_definition(
        event_id="bad_path",
        conditions=(EventCondition("numeric_gte", path="missing.value", value=1),),
    )
    with pytest.raises(ValueError, match="Invalid event condition path"):
        EventEngine(EventCatalog((event,), event_probability=1.0)).select_events(
            load_agent_state(MAYA_SCENARIO),
            context(seed=1),
            EventHistory(),
        )


def test_eligibility_from_agent_state_and_ineligible_events_excluded() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    eligible = event_definition(
        event_id="high_stress",
        conditions=(EventCondition("numeric_gte", path="mental.stress", value=50.0),),
    )
    ineligible = event_definition(
        event_id="already_connected",
        conditions=(EventCondition("numeric_gte", path="social.city_familiarity", value=90.0),),
    )

    result = EventEngine(EventCatalog((eligible, ineligible), event_probability=1.0)).select_events(
        maya,
        context(seed=3),
        EventHistory(),
    )

    trace_by_id = {candidate.event_id: candidate for candidate in result.trace.candidates}
    assert trace_by_id["high_stress"].eligible is True
    assert trace_by_id["already_connected"].eligible is False
    assert result.occurrences[0].event_id == "high_stress"


def test_weight_modifiers_change_effective_weight() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event = event_definition(
        event_id="lonely_invitation",
        base_weight=2.0,
        modifiers=(
            WeightModifier(
                condition=EventCondition("numeric_gte", path="mental.loneliness", value=45.0),
                multiplier=1.5,
            ),
        ),
    )

    result = EventEngine(EventCatalog((event,), event_probability=1.0)).select_events(
        maya,
        context(seed=1),
        EventHistory(),
    )

    assert result.trace.candidates[0].effective_weight == 3.0
    assert result.occurrences[0].effective_weight == 3.0


def test_weighted_deterministic_selection_and_seed_variation() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = EventCatalog(
        (
            event_definition(event_id="a", base_weight=1.0),
            event_definition(event_id="b", base_weight=1.0),
        ),
        event_probability=1.0,
    )

    first = EventEngine(catalog).select_events(maya, context(seed=1), EventHistory())
    repeated = EventEngine(catalog).select_events(maya, context(seed=1), EventHistory())
    different = EventEngine(catalog).select_events(maya, context(seed=4), EventHistory())

    assert first.occurrences[0].event_id == repeated.occurrences[0].event_id
    assert first.occurrences[0].event_id != different.occurrences[0].event_id


def test_legitimate_no_event_weeks_have_trace() -> None:
    result = EventEngine(
        EventCatalog((event_definition(event_id="maybe"),), event_probability=0.0)
    ).select_events(
        load_agent_state(MAYA_SCENARIO),
        context(seed=1),
        EventHistory(),
    )

    assert result.occurrences == ()
    assert result.trace.no_event is True
    assert result.trace.selected_event_ids == ()


def test_cooldown_enforcement_and_expiry() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = EventEngine(
        EventCatalog(
            (event_definition(event_id="cooldown", cooldown_weeks=1),),
            event_probability=1.0,
        )
    )
    history = EventHistory()

    week_one = engine.select_events(maya, context(week=1, seed=1), history)
    week_two = engine.select_events(maya, context(week=2, seed=1), week_one.history)
    week_three = engine.select_events(maya, context(week=3, seed=1), week_two.history)

    assert [event.event_id for event in week_one.occurrences] == ["cooldown"]
    assert week_two.occurrences == ()
    assert week_two.trace.candidates[0].reason == "cooldown"
    assert [event.event_id for event in week_three.occurrences] == ["cooldown"]


def test_repeated_run_determinism_and_event_history_isolation() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = EventCatalog((event_definition(event_id="repeatable"),), event_probability=1.0)
    engine = LifeSimEngine(
        make_config(duration_weeks=2, seed=7),
        transitions=(EventEngineTransition(EventEngine(catalog)),),
    )

    first = engine.run(initial_agent=maya)
    second = engine.run(initial_agent=maya)

    assert first.to_dict() == second.to_dict()
    assert first.event_history is not second.event_history
    assert len(first.event_history.occurrences) == 2
    assert len(second.event_history.occurrences) == 2


def test_different_seeds_can_produce_different_event_timelines() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = EventCatalog(
        (
            event_definition(event_id="a", base_weight=1.0),
            event_definition(event_id="b", base_weight=1.0),
        ),
        event_probability=1.0,
    )

    first = LifeSimEngine(
        make_config(duration_weeks=1, seed=1),
        transitions=(EventEngineTransition(EventEngine(catalog)),),
    ).run(initial_agent=maya)
    second = LifeSimEngine(
        make_config(duration_weeks=1, seed=4),
        transitions=(EventEngineTransition(EventEngine(catalog)),),
    ).run(initial_agent=maya)

    assert first.states[1].events[0].event_id != second.states[1].events[0].event_id


def test_event_catalog_definitions_retain_no_run_specific_state() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = EventCatalog((event_definition(event_id="stable_definition"),), event_probability=1.0)
    before = catalog.definitions[0].to_dict()

    LifeSimEngine(
        make_config(duration_weeks=2),
        transitions=(EventEngineTransition(EventEngine(catalog)),),
    ).run(initial_agent=maya)

    assert catalog.definitions[0].to_dict() == before


def test_generic_non_maya_agent_events() -> None:
    generic_agent = replace(
        load_agent_state(MAYA_SCENARIO),
        identity=IdentityState(
            agent_id="alex",
            display_name="Alex",
            age_years=24,
            pronouns="they/them",
            life_stage="young_adult",
            origin_city="Dublin",
            current_city="Veyra",
            background="Generic event test agent.",
        ),
    )
    catalog = EventCatalog((event_definition(event_id="generic"),), event_probability=1.0)

    result = LifeSimEngine(
        make_config(duration_weeks=1),
        transitions=(EventEngineTransition(EventEngine(catalog)),),
    ).run(initial_agent=generic_agent)

    assert result.states[1].events[0].event_id == "generic"
    assert result.summaries[1].agent_id == "alex"


def test_event_engine_does_not_mutate_agent_state() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    before = maya.to_dict()

    LifeSimEngine(
        make_config(duration_weeks=2),
        transitions=(EventEngineTransition(EventEngine(load_event_catalog(STARTER_CATALOG))),),
    ).run(initial_agent=maya)

    assert maya.to_dict() == before


def test_event_occurrences_serialize_deterministically() -> None:
    result = LifeSimEngine(
        make_config(duration_weeks=1),
        transitions=(
            EventEngineTransition(
                EventEngine(EventCatalog((event_definition(event_id="serial"),), event_probability=1.0))
            ),
        ),
    ).run(initial_agent=load_agent_state(MAYA_SCENARIO))

    assert result.states[1].events[0].to_dict() == {
        "event_id": "serial",
        "version": "1",
        "week": 1,
        "category": "test",
        "effective_weight": 1.0,
        "title": "Test event",
        "summary": "A deterministic test event.",
        "tags": ["test"],
    }
    assert result.to_dict()["event_history"]["occurrences"][0]["event_id"] == "serial"


def test_integration_with_multiple_m2_weeks_records_events_and_traces() -> None:
    result = LifeSimEngine(
        make_config(duration_weeks=3, seed=11),
        transitions=(EventEngineTransition(EventEngine(load_event_catalog(STARTER_CATALOG))),),
    ).run(initial_agent=load_agent_state(MAYA_SCENARIO))

    assert [state.week for state in result.states] == [0, 1, 2, 3]
    assert all(state.agent_state is not None for state in result.states)
    assert all(state.event_traces for state in result.states[1:])
    assert result.event_history is not None
    assert result.to_dict()["states"][1]["event_traces"][0]["week"] == 1


def context(*, week: int = 1, seed: int = 1) -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=make_config(seed=seed),
        rng=__import__("random").Random(seed),
    )


def event_definition(
    *,
    event_id: str,
    base_weight: float = 1.0,
    conditions: tuple[EventCondition, ...] = (),
    modifiers: tuple[WeightModifier, ...] = (),
    cooldown_weeks: int = 0,
) -> EventDefinition:
    return EventDefinition(
        event_id=event_id,
        version="1",
        category="test",
        base_weight=base_weight,
        conditions=conditions,
        weight_modifiers=modifiers,
        cooldown_weeks=cooldown_weeks,
        tags=("test",),
        title="Test event",
        summary="A deterministic test event.",
    )
