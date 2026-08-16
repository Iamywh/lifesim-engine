from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import IdentityState, RoutineState, SocialConnection, SocialState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.decisions.engine import DecisionEngine, DecisionEngineTransition
from lifesim.engine import LifeSimEngine
from lifesim.events.engine import EventEngine, EventEngineTransition
from lifesim.events.model import EventCatalog, EventDefinition, EventOption
from lifesim.passive.catalog import load_routine_catalog
from lifesim.rng import create_rng
from lifesim.social.catalog import load_social_catalog, parse_social_catalog
from lifesim.social.engine import (
    SOCIAL_EVENT_ID,
    SocialEngine,
    SocialExecutionTransition,
    SocialMaintenanceTransition,
    SocialPlanningTransition,
)
from lifesim.social.model import (
    SocialCatalog,
    SocialContactDefinition,
    SocialHistory,
    SocialInteractionOutcomeAudit,
    SocialInteractionRecord,
    SocialOutcomeProbability,
    SocialRuntimeState,
)
from lifesim.weekly import WeeklyContext

ROOT = Path(__file__).resolve().parents[1]
MAYA_SCENARIO = ROOT / "configs" / "scenarios" / "maya_start.toml"
ROUTINES = ROOT / "configs" / "routines" / "starter.toml"
SOCIAL = ROOT / "configs" / "social" / "starter.toml"


def test_social_connection_defaults_are_backward_compatible() -> None:
    connection = SocialConnection(name="Legacy", relationship="friend", closeness=41.0)

    assert connection.connection_id == "Legacy"
    assert connection.trust == 41.0
    assert connection.strain == 0.0
    assert connection.last_interaction_week == 0


def test_social_state_requires_unique_connection_ids() -> None:
    first = SocialConnection("Lina", "flatmate", 28.0, connection_id="lina")
    second = SocialConnection("Lina again", "flatmate", 32.0, connection_id="lina")

    with pytest.raises(ValueError, match="unique"):
        SocialState(
            support_network_strength=20.0,
            city_familiarity=10.0,
            connections=(first, second),
        )


def test_maya_scenario_and_social_catalog_load_stable_contacts() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = load_social_catalog(SOCIAL)
    by_id = {connection.connection_id: connection for connection in maya.social.connections}

    assert by_id["lina"].closeness == 28.0
    assert by_id["tomas"].closeness == 63.0
    assert catalog.contact("lina").context == "existing"
    assert catalog.contact("noor").context == "education"


def test_maintenance_is_zero_rng_and_mutates_only_social_state() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    maya = replace(
        load_agent_state(MAYA_SCENARIO),
        social=replace(
            base.social,
            connections=tuple(
                replace(connection, last_interaction_week=0, strain=20.0)
                for connection in base.social.connections
            ),
        ),
    )
    engine = social_engine()
    ctx = context(week=12)
    before_rng = ctx.rng.getstate()
    next_state, runtime, record = engine.maintain(maya, ctx, SocialRuntimeState())

    assert ctx.rng.getstate() == before_rng
    assert record.changes
    assert next_state.social != maya.social
    assert replace(next_state, social=maya.social) == maya
    assert runtime.history.maintenance_records == (record,)


def test_planning_uses_routine_exposure_for_new_encounter_probability() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    social_week = replace(maya, routine=RoutineState(current_profile_id="social_week"))
    austerity = replace(maya, routine=RoutineState(current_profile_id="austerity_home_week"))
    engine = social_engine()

    _, _, social_record = engine.plan(social_week, context(seed=9), SocialRuntimeState())
    _, _, austerity_record = engine.plan(austerity, context(seed=9), SocialRuntimeState())

    assert social_record.encounter.probability > austerity_record.encounter.probability


def test_planning_creates_at_most_one_focal_social_event_with_bounded_options() -> None:
    maya = replace(load_agent_state(MAYA_SCENARIO), routine=RoutineState(current_profile_id="social_week"))
    runtime, event, record = social_engine().plan(maya, context(seed=3), SocialRuntimeState())

    assert event is not None
    assert event.event_id == SOCIAL_EVENT_ID
    assert event.category == "social_relationship"
    assert "keep_social_light" in record.option_ids
    assert sum(option.option_id.startswith(("connect:", "seek_support:")) for option in event.options) <= 2
    assert sum(option.option_id.startswith("engage:") for option in event.options) <= 1
    assert runtime.planned_option_ids == tuple(option.option_id for option in event.options)


def test_no_opportunity_week_is_valid_and_has_no_event() -> None:
    state = replace(load_agent_state(MAYA_SCENARIO), social=replace(load_agent_state(MAYA_SCENARIO).social, connections=()))
    catalog = SocialCatalog(
        contacts=(
            SocialContactDefinition(
                "hidden",
                "Hidden",
                "hidden",
                "unavailable_context",
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ),
        ),
        base_new_encounter_probability=0.0,
    )
    engine = SocialEngine(catalog, routine_catalog=load_routine_catalog(ROUTINES))
    runtime, event, record = engine.plan(state, context(), SocialRuntimeState())
    next_state, runtime, execution = engine.execute(state, context(), runtime)

    assert event is None
    assert record.option_ids == ()
    assert execution.interaction_type == "no_opportunity"
    assert next_state == state
    assert runtime.history.interaction_records == (execution,)


def test_known_social_execution_mutates_only_allowed_state_surfaces() -> None:
    state, runtime, event = planned_social_state(seed=4)
    option_id = next(option.option_id for option in event.options if option.option_id.startswith("connect:"))
    decision = decide(state, event, option_id)
    next_state, runtime, record = social_engine().execute(
        state,
        context(seed=4, events=(event,), decisions=(decision,)),
        runtime,
    )

    assert record.relationship_changes
    assert {effect.path for effect in record.state_effects} <= {
        "health.energy",
        "mental.stress",
        "mental.mood",
        "mental.loneliness",
        "needs.belonging",
    }
    assert next_state.identity == state.identity
    assert next_state.financial == state.financial
    assert next_state.education == state.education
    assert next_state.skills == state.skills
    assert next_state.memory == state.memory
    assert next_state.employment == state.employment
    assert runtime.processed_decision_ids == (decision.decision_id,)


def test_new_encounter_can_create_at_most_one_weak_non_duplicate_connection() -> None:
    state = replace(
        load_agent_state(MAYA_SCENARIO),
        social=SocialState(support_network_strength=0.0, city_familiarity=100.0, connections=()),
        routine=RoutineState(current_profile_id="social_week"),
    )
    catalog = parse_social_catalog(
        {
            "social_settings": {"base_new_encounter_probability": 1.0},
            "contacts": [
                {
                    "contact_id": "new_friend",
                    "name": "New Friend",
                    "relationship": "classmate",
                    "context": "education",
                    "base_availability": 0.5,
                    "proximity": 0.9,
                    "responsiveness": 0.9,
                    "volatility": 0.1,
                    "supportiveness": 0.4,
                    "neglect_resistance": 0.5,
                    "initial_closeness": 16.0,
                    "initial_trust": 15.0,
                }
            ],
        }
    )
    engine = SocialEngine(catalog, routine_catalog=load_routine_catalog(ROUTINES))
    runtime, event, _ = engine.plan(state, context(seed=14), SocialRuntimeState())
    assert event is not None
    option_id = next(option.option_id for option in event.options if option.option_id.startswith("engage:"))
    decision = decide(state, event, option_id)
    next_state, _, record = engine.execute(state, context(seed=14, events=(event,), decisions=(decision,)), runtime)

    assert len(next_state.social.connections) <= 1
    assert len({connection.connection_id for connection in next_state.social.connections}) == len(next_state.social.connections)
    if next_state.social.connections:
        assert next_state.social.connections[0].closeness <= 25.0
    assert record.interaction_type == "engage"


def test_runtime_rejects_extra_processed_weeks_and_decision_ids() -> None:
    record = SocialInteractionRecord(
        week=2,
        decision_id="d1",
        option_id="keep_social_light",
        contact_id="",
        interaction_type="keep_social_light",
        outcome=SocialInteractionOutcomeAudit(
            probabilities=(SocialOutcomeProbability("light", 1.0),),
            roll=0.0,
            selected_outcome_id="light",
        ),
    )
    history = SocialHistory(interaction_records=(record,))

    with pytest.raises(ValueError, match="execution weeks"):
        SocialRuntimeState(history=history, processed_execution_weeks=(2, 3), processed_decision_ids=("d1",))
    with pytest.raises(ValueError, match="decision ids"):
        SocialRuntimeState(history=history, processed_execution_weeks=(2,), processed_decision_ids=("d1", "d2"))


def test_repeated_runs_are_identical_and_m3_event_rng_is_unchanged() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    base_engine = LifeSimEngine(
        config(seed=22),
        transitions=(EventEngineTransition(EventEngine(event_catalog())),),
    )
    social_pipeline = (
        SocialMaintenanceTransition(social_engine()),
        SocialPlanningTransition(social_engine()),
        EventEngineTransition(EventEngine(event_catalog())),
        DecisionEngineTransition(DecisionEngine()),
        SocialExecutionTransition(social_engine()),
    )
    with_social_engine = LifeSimEngine(config(seed=22), transitions=social_pipeline)

    base = base_engine.run(initial_agent=maya)
    with_social = with_social_engine.run(initial_agent=maya)
    repeated = with_social_engine.run(initial_agent=maya)

    assert with_social.to_dict() == repeated.to_dict()
    assert base.states[1].event_traces[0].to_dict() == with_social.states[1].event_traces[0].to_dict()


def test_protected_identity_presentation_does_not_change_social_probabilities() -> None:
    maya = replace(load_agent_state(MAYA_SCENARIO), routine=RoutineState(current_profile_id="social_week"))
    altered_identity = IdentityState(
        agent_id=maya.identity.agent_id,
        display_name=maya.identity.display_name,
        age_years=maya.identity.age_years,
        pronouns="they/them",
        life_stage=maya.identity.life_stage,
        origin_city="Different Origin",
        current_city=maya.identity.current_city,
        background="Different presentation text.",
    )
    other = replace(maya, identity=altered_identity)
    engine = social_engine()

    _, event_a, record_a = engine.plan(maya, context(seed=31), SocialRuntimeState())
    _, event_b, record_b = engine.plan(other, context(seed=31), SocialRuntimeState())

    assert record_a.to_dict() == record_b.to_dict()
    assert event_a.to_dict() == event_b.to_dict()


def test_cli_demo_outputs_social_json() -> None:
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
            "--employment-catalog",
            "configs/employment/starter.toml",
            "--development-catalog",
            "configs/development/starter.toml",
            "--social-catalog",
            "configs/social/starter.toml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["social_history"]["maintenance_records"]
    assert payload["social_history"]["planning_records"]
    assert payload["states"][1]["social_records"]
    assert "social" in payload["states"][-1]["agent"]


def planned_social_state(*, seed: int):
    state = replace(load_agent_state(MAYA_SCENARIO), routine=RoutineState(current_profile_id="social_week"))
    engine = social_engine()
    runtime, event, _ = engine.plan(state, context(seed=seed), SocialRuntimeState())
    assert event is not None
    return state, runtime, event


def decide(state, event, option_id):
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    return replace(decision, chosen_option_id=option_id)


def social_engine() -> SocialEngine:
    return SocialEngine(load_social_catalog(SOCIAL), routine_catalog=load_routine_catalog(ROUTINES))


def context(*, week: int = 1, seed: int = 7, **overrides) -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=config(seed=seed),
        rng=create_rng(seed),
        **overrides,
    )


def config(*, duration_weeks: int = 1, seed: int = 7) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(name="social-test", seed=seed, duration_weeks=duration_weeks),
        city=CityConfig(name="Test City"),
    )


def event_catalog() -> EventCatalog:
    return EventCatalog(
        definitions=(
            EventDefinition(
                event_id="rng_probe_event",
                version="1",
                category="probe",
                base_weight=1.0,
                conditions=(),
                weight_modifiers=(),
                cooldown_weeks=0,
                tags=("probe",),
                title="Probe",
                summary="Probe event.",
                options=(EventOption("observe", "Observe", "Observe."),),
            ),
        ),
        event_probability=1.0,
    )
