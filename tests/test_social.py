from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import (
    EmploymentState,
    IdentityState,
    RoutineState,
    SocialConnection,
    SocialState,
)
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.consequences.catalog import load_consequence_catalog
from lifesim.consequences.engine import ConsequenceEngine, DecisionConsequenceTransition
from lifesim.decisions.engine import DecisionEngine, DecisionEngineTransition
from lifesim.employment.catalog import load_employment_catalog
from lifesim.employment.engine import EmploymentMarketEngine, EmploymentMarketTransition
from lifesim.engine import LifeSimEngine
from lifesim.events.catalog import load_event_catalog
from lifesim.events.engine import EventEngine, EventEngineTransition
from lifesim.events.model import EventCatalog, EventDefinition, EventOccurrence, EventOption
from lifesim.passive.catalog import load_routine_catalog
from lifesim.passive.engine import PassiveCashflowEngine, PassiveCashflowTransition
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
EVENTS = ROOT / "configs" / "events" / "starter.toml"
CONSEQUENCES = ROOT / "configs" / "consequences" / "starter.toml"
EMPLOYMENT = ROOT / "configs" / "employment" / "starter.toml"


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
    assert sum(option.option_id.startswith(("connect:", "seek_support:")) for option in event.options) <= 4
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
                    "employment",
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


def test_social_execution_rejects_fresh_runtime_without_planning() -> None:
    with pytest.raises(ValueError, match="planning"):
        social_engine().execute(
            load_agent_state(MAYA_SCENARIO),
            context(),
            SocialRuntimeState(),
        )


def test_social_execution_requires_real_decision_when_planning_created_event() -> None:
    state, runtime, event = planned_social_state(seed=4)

    with pytest.raises(ValueError, match="decision"):
        social_engine().execute(state, context(seed=4, events=(event,)), runtime)


def test_social_execution_rejects_tampered_runtime_with_cleared_planned_fields() -> None:
    state, runtime, event = planned_social_state(seed=4)
    tampered = SocialRuntimeState(
        history=runtime.history,
        processed_planning_weeks=runtime.processed_planning_weeks,
    )
    decision = decide(state, event, next(option.option_id for option in event.options))

    with pytest.raises(ValueError, match="planned fields"):
        social_engine().execute(
            state,
            context(seed=4, events=(event,), decisions=(decision,)),
            tampered,
        )


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


def test_relationship_quality_is_visible_to_m4_score_components() -> None:
    state = state_with_connections(
        SocialConnection("Warm", "friend", 84.0, connection_id="warm", trust=82.0, strain=2.0),
        SocialConnection("Strained", "friend", 84.0, connection_id="strained", trust=8.0, strain=74.0),
    )
    event = EventOccurrence(
        event_id=SOCIAL_EVENT_ID,
        version="1",
        week=1,
        category="social_relationship",
        effective_weight=1.0,
        title="Social",
        summary="Social.",
        tags=("social",),
        options=(
            contact_option("warm", social_value=0.82, future_value=0.58, comfort_value=0.5, risk=0.05, uncertainty=0.08),
            contact_option("strained", social_value=0.22, future_value=0.22, comfort_value=-0.1, risk=0.65, uncertainty=0.72),
        ),
    )
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    evaluations = evaluations_by_option(decision)

    warm = evaluations["connect:warm"]
    strained = evaluations["connect:strained"]
    assert component(warm, "social_value").contribution > component(strained, "social_value").contribution
    assert component(warm, "perceived_risk").contribution > component(strained, "perceived_risk").contribution
    assert component(warm, "uncertainty").contribution > component(strained, "uncertainty").contribution
    assert warm.deterministic_score > strained.deterministic_score


def test_relationship_change_affects_later_social_option_score() -> None:
    state, runtime, event = planned_social_state(seed=4)
    option_id = next(option.option_id for option in event.options if option.option_id.startswith("connect:lina"))
    before_option = next(option for option in event.options if option.option_id == option_id)
    decision = decide(state, event, option_id)
    next_state, _, _ = social_engine().execute(
        state,
        context(seed=4, events=(event,), decisions=(decision,)),
        runtime,
    )
    later_runtime, later_event, _ = plan_until_option(next_state, "connect:lina", start_seed=1, week=2)
    after_option = next(option for option in later_event.options if option.option_id == "connect:lina")

    assert later_runtime.planned_week == 2
    assert after_option.to_dict() != before_option.to_dict()


def test_support_choice_coexists_with_connect_and_is_not_forced() -> None:
    pressured = replace(
        load_agent_state(MAYA_SCENARIO),
        mental=replace(load_agent_state(MAYA_SCENARIO).mental, stress=82.0, loneliness=70.0),
        needs=replace(load_agent_state(MAYA_SCENARIO).needs, belonging=30.0),
        routine=RoutineState(current_profile_id="social_week"),
    )
    runtime, event, _ = plan_until_option(pressured, "seek_support:tomas", start_seed=1)
    option_ids = {option.option_id for option in event.options}

    assert "connect:tomas" in option_ids
    assert "seek_support:tomas" in option_ids
    connect_decision = decide(pressured, event, "connect:tomas")
    next_state, _, record = social_engine().execute(
        pressured,
        context(events=(event,), decisions=(connect_decision,)),
        runtime,
    )
    assert next_state.social != pressured.social
    assert record.interaction_type == "connect"
    assert record.outcome.selected_outcome_id in {"warm", "neutral", "friction"}


def test_low_need_exposes_connect_without_support() -> None:
    low_need = replace(
        load_agent_state(MAYA_SCENARIO),
        mental=replace(load_agent_state(MAYA_SCENARIO).mental, stress=20.0, loneliness=20.0),
        needs=replace(load_agent_state(MAYA_SCENARIO).needs, belonging=80.0),
        routine=RoutineState(current_profile_id="social_week"),
    )
    _, event, _ = plan_until_option(low_need, "connect:tomas", start_seed=1)
    option_ids = {option.option_id for option in event.options}

    assert "connect:tomas" in option_ids
    assert "seek_support:tomas" not in option_ids


def test_support_outcome_runs_only_for_seek_support_selection() -> None:
    pressured = replace(
        load_agent_state(MAYA_SCENARIO),
        mental=replace(load_agent_state(MAYA_SCENARIO).mental, stress=82.0, loneliness=70.0),
        needs=replace(load_agent_state(MAYA_SCENARIO).needs, belonging=30.0),
        routine=RoutineState(current_profile_id="social_week"),
    )
    runtime, event, _ = plan_until_option(pressured, "seek_support:tomas", start_seed=1)
    decision = decide(pressured, event, "seek_support:tomas")
    _, _, record = social_engine().execute(
        pressured,
        context(events=(event,), decisions=(decision,)),
        runtime,
    )

    assert record.interaction_type == "seek_support"
    assert record.outcome.selected_outcome_id in {"supportive", "limited", "unavailable"}


def test_trust_zero_does_not_fall_back_to_closeness() -> None:
    zero_trust = SocialConnection("Zero", "friend", 80.0, connection_id="zero", trust=0.0)
    high_trust = SocialConnection("High", "friend", 80.0, connection_id="high", trust=80.0)
    state = state_with_connections(zero_trust, high_trust)
    engine = social_engine()
    zero_next, _, zero_record = engine.maintain(state_with_connections(zero_trust), context(week=52), SocialRuntimeState())
    high_next, _, high_record = engine.maintain(state_with_connections(high_trust), context(week=52), SocialRuntimeState())
    event = EventOccurrence(
        event_id=SOCIAL_EVENT_ID,
        version="1",
        week=1,
        category="social_relationship",
        effective_weight=1.0,
        title="Social",
        summary="Social.",
        tags=("social",),
        options=(
            contact_option("zero", social_value=0.25, future_value=0.2, comfort_value=0.1, risk=0.2, uncertainty=0.5),
            contact_option("high", social_value=0.75, future_value=0.5, comfort_value=0.45, risk=0.08, uncertainty=0.12),
        ),
    )
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)

    assert zero_next.social.connections[0].trust < zero_trust.trust + 0.000001
    assert high_next.social.connections[0].trust < high_trust.trust
    assert zero_record.support_network.target < high_record.support_network.target
    assert evaluations_by_option(decision)["connect:zero"].deterministic_score < evaluations_by_option(decision)["connect:high"].deterministic_score


def test_different_agent_id_gets_independent_social_rolls() -> None:
    state = replace(load_agent_state(MAYA_SCENARIO), routine=RoutineState(current_profile_id="social_week"))
    other = replace(state, identity=replace(state.identity, agent_id="maya-other"))

    _, _, record_a = social_engine().plan(state, context(seed=31), SocialRuntimeState())
    _, _, record_b = social_engine().plan(other, context(seed=31), SocialRuntimeState())

    assert [audit.roll for audit in record_a.availability] != [audit.roll for audit in record_b.availability]
    assert record_a.encounter.roll != record_b.encounter.roll


def test_m10_does_not_perturb_m3_m4_m5_m7_or_m8_timelines() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event_catalog = load_event_catalog(EVENTS)
    consequence_catalog = load_consequence_catalog(CONSEQUENCES, event_catalog=event_catalog)
    employment_catalog = load_employment_catalog(EMPLOYMENT)
    base_transitions = (
        PassiveCashflowTransition(PassiveCashflowEngine()),
        EmploymentMarketTransition(EmploymentMarketEngine(employment_catalog)),
        EventEngineTransition(EventEngine(event_catalog)),
        DecisionEngineTransition(DecisionEngine()),
        DecisionConsequenceTransition(ConsequenceEngine(consequence_catalog)),
    )
    m10_transitions = (
        PassiveCashflowTransition(PassiveCashflowEngine()),
        SocialMaintenanceTransition(social_engine()),
        SocialPlanningTransition(social_engine()),
        EmploymentMarketTransition(EmploymentMarketEngine(employment_catalog)),
        EventEngineTransition(EventEngine(event_catalog)),
        DecisionEngineTransition(DecisionEngine()),
        DecisionConsequenceTransition(ConsequenceEngine(consequence_catalog)),
        SocialExecutionTransition(social_engine()),
    )

    base = LifeSimEngine(config(seed=19), transitions=base_transitions).run(initial_agent=maya)
    with_m10 = LifeSimEngine(config(seed=19), transitions=m10_transitions).run(initial_agent=maya)
    base_state = base.states[1]
    m10_state = with_m10.states[1]

    assert base_state.passive_records[0].to_dict() == m10_state.passive_records[0].to_dict()
    assert base_state.employment_records[0].to_dict() == m10_state.employment_records[0].to_dict()
    assert base_state.event_traces[0].to_dict() == m10_state.event_traces[0].to_dict()
    base_decision = next(decision for decision in base_state.decisions if decision.source_event_id != SOCIAL_EVENT_ID)
    m10_decision = next(decision for decision in m10_state.decisions if decision.source_event_id == base_decision.source_event_id)
    assert base_decision.to_dict() == m10_decision.to_dict()
    assert [record.to_dict() for record in base_state.consequences] == [
        record.to_dict() for record in m10_state.consequences
    ]


def test_known_contact_surfacing_is_weighted_without_replacement_and_audited() -> None:
    state = state_with_connections(
        SocialConnection("Strong", "friend", 90.0, connection_id="strong", trust=90.0),
        SocialConnection("Medium", "friend", 45.0, connection_id="medium", trust=45.0),
        SocialConnection("Weak", "friend", 5.0, connection_id="weak", trust=5.0),
    )
    catalog = SocialCatalog(
        contacts=tuple(
            SocialContactDefinition(
                contact_id=connection.connection_id,
                name=connection.name,
                relationship=connection.relationship,
                context="existing",
                base_availability=0.95,
                proximity=0.95,
                responsiveness=0.5,
                volatility=0.1,
                supportiveness=0.4,
                neglect_resistance=0.4,
            )
            for connection in state.social.connections
        ),
        base_new_encounter_probability=0.0,
        max_known_options=2,
    )
    engine = SocialEngine(catalog, routine_catalog=load_routine_catalog(ROUTINES))
    surfaced = set()
    for seed in range(1, 80):
        _, event, record = engine.plan(state, context(seed=seed), SocialRuntimeState())
        surfaced.update(candidate.connection_id for candidate in record.known_selection_candidates if candidate.surfaced)
        if event is not None:
            assert len({draw.selected_connection_id for draw in record.known_selection_draws}) == len(record.known_selection_draws)
            assert len(record.known_selection_draws) <= 2
            assert all(draw.total_weight > 0.0 for draw in record.known_selection_draws)

    assert "weak" in surfaced


def test_encounter_weight_drives_selection_audit_not_responsiveness_or_supportiveness() -> None:
    state = replace(
        load_agent_state(MAYA_SCENARIO),
        social=SocialState(support_network_strength=0.0, city_familiarity=100.0, connections=()),
        routine=RoutineState(current_profile_id="social_week"),
    )
    catalog = SocialCatalog(
        contacts=(
            SocialContactDefinition("likely", "Likely", "neighbor", "general", 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, encounter_weight=9.0),
            SocialContactDefinition("unlikely", "Unlikely", "neighbor", "general", 0.1, 0.9, 0.95, 0.1, 0.95, 0.4, encounter_weight=1.0),
        ),
        base_new_encounter_probability=1.0,
    )
    engine = SocialEngine(catalog, routine_catalog=load_routine_catalog(ROUTINES))
    counts = {"likely": 0, "unlikely": 0}
    sample_record = None
    for seed in range(1, 80):
        _, _, record = engine.plan(state, context(seed=seed), SocialRuntimeState())
        counts[record.encounter.selected_contact_id] += 1
        sample_record = record

    assert counts["likely"] > counts["unlikely"]
    assert sample_record.encounter.total_weight == 10.0
    assert sample_record.encounter.selection_roll is not None
    assert {item.contact_id: item.encounter_weight for item in sample_record.encounter.candidate_weights} == {
        "likely": 9.0,
        "unlikely": 1.0,
    }


def test_encounter_context_validation_and_semantics() -> None:
    with pytest.raises(ValueError, match="context"):
        SocialContactDefinition("bad", "Bad", "bad", "typo", 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    searching = replace(
        load_agent_state(MAYA_SCENARIO),
        social=SocialState(support_network_strength=0.0, city_familiarity=100.0, connections=()),
        routine=RoutineState(current_profile_id="social_week"),
    )
    catalog = SocialCatalog(
        contacts=(
            SocialContactDefinition("work", "Work", "coworker", "employment", 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            SocialContactDefinition("class", "Class", "classmate", "education", 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        ),
        base_new_encounter_probability=1.0,
    )
    engine = SocialEngine(catalog, routine_catalog=load_routine_catalog(ROUTINES))
    _, _, record = engine.plan(searching, context(seed=1), SocialRuntimeState())
    employed = replace(
        searching,
        employment=EmploymentState(
            status="employed",
            role_title="Role",
            employer="Employer",
            weekly_hours=10.0,
            job_search_intensity=0.0,
            source_job_id="job",
            source_job_version="1",
            contract_id="c",
            contract_type="part_time",
            hourly_rate=Decimal("12.00"),
            stability=0.5,
            start_week=1,
        ),
    )
    _, _, employed_record = engine.plan(employed, context(seed=1), SocialRuntimeState())

    assert record.encounter.eligible_contact_ids == ("class",)
    assert set(employed_record.encounter.eligible_contact_ids) == {"class", "work"}


def test_neglect_drift_saturates_and_never_deletes_relationships() -> None:
    weak = SocialConnection("Weak", "friend", 30.0, connection_id="weak", trust=25.0, strain=30.0)
    strong = SocialConnection("Strong", "friend_from_home", 80.0, connection_id="strong", trust=78.0, strain=30.0)
    catalog = SocialCatalog(
        contacts=(
            SocialContactDefinition("weak", "Weak", "friend", "existing", 0.3, 0.3, 0.4, 0.1, 0.3, 0.1),
            SocialContactDefinition("strong", "Strong", "friend_from_home", "existing", 0.3, 0.3, 0.4, 0.1, 0.8, 0.9, remote_contact=True),
        )
    )
    engine = SocialEngine(catalog, routine_catalog=load_routine_catalog(ROUTINES))
    weak_10, _, _ = engine.maintain(state_with_connections(weak), context(week=10), SocialRuntimeState())
    weak_156, _, record_156 = engine.maintain(state_with_connections(weak), context(week=156), SocialRuntimeState())
    strong_156, _, _ = engine.maintain(state_with_connections(strong), context(week=156), SocialRuntimeState())

    assert weak.closeness - weak_156.social.connections[0].closeness < 0.6
    assert weak_10.social.connections[0].closeness > weak_156.social.connections[0].closeness
    assert strong.closeness - strong_156.social.connections[0].closeness < weak.closeness - weak_156.social.connections[0].closeness
    assert weak_156.social.connections
    assert record_156.changes[0].strain_after < record_156.changes[0].strain_before


def test_support_network_rewards_quality_not_address_book_size() -> None:
    strong = state_with_connections(SocialConnection("Strong", "friend", 92.0, connection_id="strong", trust=90.0))
    weak_connections = tuple(
        SocialConnection(f"Weak {index}", "acquaintance", 2.0, connection_id=f"weak-{index}", trust=1.0, strain=80.0)
        for index in range(10)
    )
    weak = state_with_connections(*weak_connections)
    meaningful = state_with_connections(
        SocialConnection("A", "friend", 86.0, connection_id="a", trust=84.0),
        SocialConnection("B", "friend", 84.0, connection_id="b", trust=82.0),
        SocialConnection("C", "friend", 82.0, connection_id="c", trust=80.0),
        SocialConnection("D", "friend", 80.0, connection_id="d", trust=78.0),
        SocialConnection("E", "friend", 78.0, connection_id="e", trust=76.0),
    )
    strong_target = social_engine().maintain(strong, context(), SocialRuntimeState())[2].support_network.target
    weak_target = social_engine().maintain(weak, context(), SocialRuntimeState())[2].support_network.target
    meaningful_target = social_engine().maintain(meaningful, context(), SocialRuntimeState())[2].support_network.target

    assert strong_target > weak_target
    assert weak_target < 15.0
    assert meaningful_target > strong_target


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


def plan_until_option(state, option_id: str, *, start_seed: int, week: int = 1):
    engine = social_engine()
    for seed in range(start_seed, start_seed + 120):
        runtime, event, record = engine.plan(state, context(seed=seed, week=week), SocialRuntimeState())
        if event is not None and option_id in {option.option_id for option in event.options}:
            return runtime, event, record
    raise AssertionError(f"Could not surface social option {option_id}.")


def decide(state, event, option_id):
    decision = DecisionEngine().decide_event(state, context(events=(event,)), event)
    return replace(decision, chosen_option_id=option_id)


def state_with_connections(*connections: SocialConnection):
    base = load_agent_state(MAYA_SCENARIO)
    return replace(
        base,
        social=SocialState(
            support_network_strength=base.social.support_network_strength,
            city_familiarity=base.social.city_familiarity,
            connections=connections,
        ),
        routine=RoutineState(current_profile_id="social_week"),
    )


def contact_option(
    contact_id: str,
    *,
    social_value: float,
    future_value: float,
    comfort_value: float,
    risk: float,
    uncertainty: float,
) -> EventOption:
    return EventOption(
        option_id=f"connect:{contact_id}",
        label=f"Connect {contact_id}",
        summary="Connect.",
        social_value=social_value,
        future_value=future_value,
        comfort_value=comfort_value,
        perceived_risk=risk,
        uncertainty=uncertainty,
        time_cost_hours=2.0,
        energy_cost=6.0,
        requires_full_estimated_cost=False,
    )


def evaluations_by_option(decision):
    return {evaluation.option_id: evaluation for evaluation in decision.evaluations}


def component(evaluation, name: str):
    return next(item for item in evaluation.components if item.name == name)


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
