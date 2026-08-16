from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from lifesim.agents.state import (
    AgentState,
    SocialConnection,
)
from lifesim.decisions.model import DecisionRecord
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.passive.model import RoutineCatalog
from lifesim.rng import derive_stable_seed
from lifesim.social.model import (
    RelationshipChange,
    SocialAvailabilityAudit,
    SocialCatalog,
    SocialContactDefinition,
    SocialEncounterAudit,
    SocialEncounterCandidateWeight,
    SocialHistory,
    SocialInteractionOutcomeAudit,
    SocialInteractionRecord,
    SocialKnownSelectionCandidate,
    SocialKnownSelectionDraw,
    SocialMaintenanceChange,
    SocialMaintenanceRecord,
    SocialOutcomeProbability,
    SocialPlanningRecord,
    SocialRuntimeState,
    SocialStateEffect,
    SocialSupportNetworkAudit,
)
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

SOCIAL_EVENT_ID = "weekly_social_focus"
SOCIAL_EVENT_VERSION = "1"
MIN_AVAILABILITY_PROBABILITY = 0.02
MAX_AVAILABILITY_PROBABILITY = 0.95


class SocialEngine:
    def __init__(
        self,
        catalog: SocialCatalog,
        *,
        routine_catalog: RoutineCatalog,
    ) -> None:
        self._catalog = catalog
        self._routine_catalog = routine_catalog

    @property
    def catalog(self) -> SocialCatalog:
        return self._catalog

    def maintain(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: SocialRuntimeState,
    ) -> tuple[AgentState, SocialRuntimeState, SocialMaintenanceRecord]:
        if context.week in runtime.processed_maintenance_weeks:
            raise ValueError(f"Social maintenance already processed for week {context.week}.")
        connections: list[SocialConnection] = []
        changes: list[SocialMaintenanceChange] = []
        for connection in state.social.connections:
            definition = _definition_or_default(self._catalog, connection)
            gap = max(0, context.week - connection.last_interaction_week)
            closeness_loss, trust_loss = _neglect_losses(connection, definition, gap)
            strain_cooling = min(connection.strain, 1.0 + definition.neglect_resistance * 0.8)
            next_connection = replace(
                connection,
                closeness=_clamp(connection.closeness - closeness_loss),
                trust=_clamp(_trust(connection) - trust_loss),
                strain=_clamp(connection.strain - strain_cooling),
            )
            if next_connection != connection:
                changes.append(
                    SocialMaintenanceChange(
                        connection_id=connection.connection_id,
                        closeness_before=connection.closeness,
                        closeness_after=next_connection.closeness,
                        trust_before=_trust(connection),
                        trust_after=_trust(next_connection),
                        strain_before=connection.strain,
                        strain_after=next_connection.strain,
                        reason="neglect_drift_and_strain_cooling",
                    )
                )
            connections.append(next_connection)
        target, contributor_ids, contributor_scores = _support_network_target(tuple(connections), self._catalog)
        support_after = _clamp(state.social.support_network_strength + (target - state.social.support_network_strength) * 0.18)
        support_audit = SocialSupportNetworkAudit(
            before=state.social.support_network_strength,
            target=target,
            after=support_after,
            connection_count=len(connections),
            meaningful_contributor_ids=contributor_ids,
            meaningful_contributor_scores=contributor_scores,
        )
        social = replace(
            state.social,
            support_network_strength=support_after,
            connections=tuple(connections),
        )
        next_state = replace(state, social=social)
        record = SocialMaintenanceRecord(
            week=context.week,
            changes=tuple(changes),
            support_network=support_audit,
        )
        runtime = replace(
            runtime,
            history=runtime.history.record_maintenance(record),
            processed_maintenance_weeks=runtime.processed_maintenance_weeks + (context.week,),
        )
        return next_state, runtime, record

    def plan(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: SocialRuntimeState,
    ) -> tuple[SocialRuntimeState, EventOccurrence | None, SocialPlanningRecord]:
        if context.week in runtime.processed_planning_weeks:
            raise ValueError(f"Social planning already processed for week {context.week}.")
        routine_profile_id, routine_social_contact = _planned_routine(context, state, self._routine_catalog)
        availability = tuple(
            _availability_audit(
                context,
                state,
                connection,
                _definition_or_default(self._catalog, connection),
                routine_social_contact,
            )
            for connection in state.social.connections
        )
        available_by_id = {audit.connection_id for audit in availability if audit.available}
        selected_connections, selection_candidates, selection_draws = _select_known_connections(
            context,
            state,
            tuple(
                connection
                for connection in state.social.connections
                if connection.connection_id in available_by_id
            ),
            self._catalog,
        )
        known_options = tuple(
            option
            for connection in selected_connections
            for option in _known_options(
                state,
                connection,
                _definition_or_default(self._catalog, connection),
            )
        )
        encounter = _encounter_audit(context, state, self._catalog, routine_social_contact, runtime.history)
        encounter_options: tuple[EventOption, ...] = ()
        if encounter.triggered and encounter.selected_contact_id:
            definition = self._catalog.contact(encounter.selected_contact_id)
            encounter_options = (_encounter_option(definition),)
        options = known_options + encounter_options
        if options:
            options = options + (_keep_light_option(),)
            occurrence: EventOccurrence | None = EventOccurrence(
                event_id=SOCIAL_EVENT_ID,
                version=SOCIAL_EVENT_VERSION,
                week=context.week,
                category="social_relationship",
                effective_weight=1.0,
                title="Weekly social focus",
                summary="Choose a small focal social interaction for this week.",
                tags=("social", "relationships"),
                time_pressure=0.15,
                options=options,
            )
            planned_event_id = occurrence.event_id
            planned_event_version = occurrence.version
            option_ids = tuple(option.option_id for option in options)
            planned_week = context.week
        else:
            occurrence = None
            planned_event_id = ""
            planned_event_version = ""
            option_ids = ()
            planned_week = None
        record = SocialPlanningRecord(
            week=context.week,
            routine_profile_id=routine_profile_id,
            routine_social_contact=routine_social_contact,
            event_id=planned_event_id,
            event_version=planned_event_version,
            option_ids=option_ids,
            availability=availability,
            encounter=encounter,
            known_selection_candidates=selection_candidates,
            known_selection_draws=selection_draws,
        )
        runtime = replace(
            runtime,
            history=runtime.history.record_planning(record),
            planned_event_id=planned_event_id,
            planned_event_version=planned_event_version,
            planned_option_ids=option_ids,
            planned_week=planned_week,
            processed_planning_weeks=runtime.processed_planning_weeks + (context.week,),
        )
        return runtime, occurrence, record

    def execute(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: SocialRuntimeState,
    ) -> tuple[AgentState, SocialRuntimeState, SocialInteractionRecord]:
        if context.week in runtime.processed_execution_weeks:
            raise ValueError(f"Social execution already processed for week {context.week}.")
        planning_record = _planning_record_for_execution(runtime, context.week)
        if not planning_record.event_id and not planning_record.event_version and not planning_record.option_ids:
            if runtime.planned_week is not None or runtime.planned_event_id or runtime.planned_event_version or runtime.planned_option_ids:
                raise ValueError("Expected cleared social planned fields for a no-opportunity planning record.")
            record = SocialInteractionRecord(
                week=context.week,
                decision_id="",
                option_id="",
                contact_id="",
                interaction_type="no_opportunity",
                outcome=None,
            )
            runtime = replace(
                runtime,
                history=runtime.history.record_interaction(record),
                processed_execution_weeks=runtime.processed_execution_weeks + (context.week,),
            )
            return state, runtime, record
        if not planning_record.event_id or not planning_record.event_version or not planning_record.option_ids:
            raise ValueError("Expected social planning record event fields to be all present or all empty.")
        if (
            runtime.planned_week is None
            or not runtime.planned_event_id
            or not runtime.planned_event_version
            or not runtime.planned_option_ids
        ):
            raise ValueError("Expected social runtime planned fields to match the planning record.")
        if runtime.planned_week != context.week:
            raise ValueError("Expected planned social week to match execution week.")
        if (
            runtime.planned_event_id != planning_record.event_id
            or runtime.planned_event_version != planning_record.event_version
            or runtime.planned_option_ids != planning_record.option_ids
        ):
            raise ValueError("Expected social runtime planned fields to match the planning record.")
        decision, event = _find_social_decision(state, context, runtime)
        if decision.decision_id in runtime.processed_decision_ids:
            raise ValueError(f"Social decision '{decision.decision_id}' already processed.")
        option_id = decision.chosen_option_id
        if option_id is None:
            raise ValueError("Expected social decision to choose an option.")
        if option_id == "keep_social_light":
            outcome = _fixed_outcome("light")
            record = SocialInteractionRecord(
                week=context.week,
                decision_id=decision.decision_id,
                option_id=option_id,
                contact_id="",
                interaction_type="keep_social_light",
                outcome=outcome,
            )
            next_state = state
        elif option_id.startswith("connect:"):
            contact_id = option_id.removeprefix("connect:")
            next_state, record = self._execute_known(
                state,
                context,
                decision,
                contact_id,
                interaction_type="connect",
            )
        elif option_id.startswith("seek_support:"):
            contact_id = option_id.removeprefix("seek_support:")
            next_state, record = self._execute_known(
                state,
                context,
                decision,
                contact_id,
                interaction_type="seek_support",
            )
        elif option_id.startswith("engage:"):
            contact_id = option_id.removeprefix("engage:")
            next_state, record = self._execute_encounter(state, context, decision, contact_id)
        else:
            raise ValueError(f"Unexpected social option id '{option_id}'.")
        _ = event
        runtime = replace(
            runtime,
            history=runtime.history.record_interaction(record),
            planned_event_id="",
            planned_event_version="",
            planned_option_ids=(),
            planned_week=None,
            processed_execution_weeks=runtime.processed_execution_weeks + (context.week,),
            processed_decision_ids=runtime.processed_decision_ids + (decision.decision_id,),
        )
        return next_state, runtime, record

    def _execute_known(
        self,
        state: AgentState,
        context: WeeklyContext,
        decision: DecisionRecord,
        contact_id: str,
        *,
        interaction_type: str,
    ) -> tuple[AgentState, SocialInteractionRecord]:
        connection = _connection(state, contact_id)
        definition = _definition_or_default(self._catalog, connection)
        outcome = _known_outcome(context, state, connection, definition, interaction_type)
        next_state, relationship_changes, state_effects = _apply_known_outcome(
            state,
            context.week,
            contact_id,
            interaction_type,
            outcome.selected_outcome_id,
        )
        record = SocialInteractionRecord(
            week=context.week,
            decision_id=decision.decision_id,
            option_id=decision.chosen_option_id or "",
            contact_id=contact_id,
            interaction_type=interaction_type,
            outcome=outcome,
            relationship_changes=relationship_changes,
            state_effects=state_effects,
        )
        return next_state, record

    def _execute_encounter(
        self,
        state: AgentState,
        context: WeeklyContext,
        decision: DecisionRecord,
        contact_id: str,
    ) -> tuple[AgentState, SocialInteractionRecord]:
        if any(connection.connection_id == contact_id for connection in state.social.connections):
            raise ValueError("Expected new social encounter not to duplicate an existing connection.")
        definition = self._catalog.contact(contact_id)
        outcome = _encounter_outcome(context, state, definition)
        next_state = state
        relationship_changes: tuple[RelationshipChange, ...] = ()
        if outcome.selected_outcome_id in {"promising", "neutral"}:
            closeness = definition.initial_closeness + (5.0 if outcome.selected_outcome_id == "promising" else 0.0)
            trust = definition.initial_trust + (4.0 if outcome.selected_outcome_id == "promising" else 0.0)
            new_connection = SocialConnection(
                connection_id=definition.contact_id,
                name=definition.name,
                relationship=definition.relationship,
                closeness=_clamp(closeness),
                trust=_clamp(trust),
                strain=0.0,
                last_interaction_week=context.week,
            )
            next_state = replace(
                state,
                social=replace(
                    state.social,
                    connections=state.social.connections + (new_connection,),
                ),
            )
            relationship_changes = (
                RelationshipChange(contact_id, "closeness", 0.0, new_connection.closeness),
                RelationshipChange(contact_id, "trust", 0.0, _trust(new_connection)),
                RelationshipChange(contact_id, "last_interaction_week", 0.0, float(context.week)),
            )
        state_effects = _interaction_state_effects(next_state, "engage", outcome.selected_outcome_id)
        next_state = _apply_state_effects(next_state, state_effects)
        record = SocialInteractionRecord(
            week=context.week,
            decision_id=decision.decision_id,
            option_id=decision.chosen_option_id or "",
            contact_id=contact_id,
            interaction_type="engage",
            outcome=outcome,
            relationship_changes=relationship_changes,
            state_effects=state_effects,
        )
        return next_state, record


@dataclass(frozen=True, slots=True)
class SocialMaintenanceTransition:
    engine: SocialEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.maintain(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            social_records=(record,),
            social_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class SocialPlanningTransition:
    engine: SocialEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        runtime, occurrence, record = self.engine.plan(state, context, runtime)
        events = () if occurrence is None else (occurrence,)
        return WeeklyTransitionResult(
            agent_state=state,
            events=events,
            social_records=(record,),
            social_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class SocialExecutionTransition:
    engine: SocialEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.execute(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            social_records=(record,),
            social_runtime=runtime,
        )


def _runtime(context: WeeklyContext) -> SocialRuntimeState:
    runtime = context.social_runtime
    if runtime is None:
        return SocialRuntimeState()
    if not isinstance(runtime, SocialRuntimeState):
        raise TypeError("Expected WeeklyContext.social_runtime to contain SocialRuntimeState.")
    return runtime


def _planning_record_for_execution(
    runtime: SocialRuntimeState,
    week: int,
) -> SocialPlanningRecord:
    if week not in runtime.processed_planning_weeks:
        raise ValueError("Expected social planning to be processed before social execution.")
    records = tuple(record for record in runtime.history.planning_records if record.week == week)
    if len(records) != 1:
        raise ValueError("Expected exactly one social planning record before social execution.")
    return records[0]


def _planned_routine(
    context: WeeklyContext,
    state: AgentState,
    catalog: RoutineCatalog,
) -> tuple[str, float]:
    profile_id = state.routine.current_profile_id
    runtime = context.passive_runtime
    planned_profile_id = getattr(runtime, "planned_routine_profile_id", "")
    planned_week = getattr(runtime, "planned_routine_week", None)
    if planned_profile_id and planned_week == context.week:
        profile_id = planned_profile_id
    try:
        profile = catalog.get(profile_id)
    except KeyError:
        return profile_id, 0.25
    return profile.profile_id, profile.social_contact


def _availability_audit(
    context: WeeklyContext,
    state: AgentState,
    connection: SocialConnection,
    definition: SocialContactDefinition,
    routine_social_contact: float,
) -> SocialAvailabilityAudit:
    routine_factor = 0.2 + routine_social_contact * (0.45 if definition.remote_contact else 0.75)
    probability = (
        definition.base_availability * 0.55
        + definition.proximity * routine_factor
        + definition.responsiveness * 0.18
        - connection.strain / 100.0 * 0.18
        - state.mental.stress / 100.0 * 0.04
    )
    probability = _bounded_probability(probability)
    rng = _rng(context, state, "social-known-availability", connection.connection_id)
    roll = rng.random()
    return SocialAvailabilityAudit(
        connection_id=connection.connection_id,
        probability=probability,
        roll=roll,
        available=roll <= probability,
    )


def _neglect_losses(
    connection: SocialConnection,
    definition: SocialContactDefinition,
    gap: int,
) -> tuple[float, float]:
    if gap <= 4:
        return 0.0, 0.0
    trust = _trust(connection)
    strength = (connection.closeness + trust) / 200.0
    vulnerability = 0.55 + (1.0 - strength) * 0.35
    if definition.remote_contact:
        vulnerability *= 0.7
    closeness_pressure = 1.0 - math.exp(-(gap - 4.0) / 26.0)
    trust_pressure = 0.0 if gap <= 8 else 1.0 - math.exp(-(gap - 8.0) / 40.0)
    resistance = 1.0 - definition.neglect_resistance
    closeness_loss = 0.55 * closeness_pressure * resistance * vulnerability
    trust_loss = 0.28 * trust_pressure * resistance * vulnerability
    return round(closeness_loss, 6), round(trust_loss, 6)


def _select_known_connections(
    context: WeeklyContext,
    state: AgentState,
    connections: tuple[SocialConnection, ...],
    catalog: SocialCatalog,
) -> tuple[
    tuple[SocialConnection, ...],
    tuple[SocialKnownSelectionCandidate, ...],
    tuple[SocialKnownSelectionDraw, ...],
]:
    remaining = sorted(connections, key=lambda item: item.connection_id)
    weights_by_id: dict[str, float] = {}
    for connection in connections:
        weights_by_id[connection.connection_id] = _known_selection_weight(
            connection,
            _definition_or_default(catalog, connection),
            context.week,
        )
    selected: list[SocialConnection] = []
    draws: list[SocialKnownSelectionDraw] = []
    rng = _rng(context, state, "social-known-selection")
    for slot in range(min(catalog.max_known_options, len(remaining))):
        total_weight = sum(weights_by_id[connection.connection_id] for connection in remaining)
        if total_weight <= 0.0:
            break
        roll = rng.random() * total_weight
        cursor = 0.0
        for index, connection in enumerate(remaining):
            cursor += weights_by_id[connection.connection_id]
            if roll <= cursor:
                selected.append(connection)
                draws.append(
                    SocialKnownSelectionDraw(
                        slot=slot,
                        roll=roll,
                        total_weight=total_weight,
                        selected_connection_id=connection.connection_id,
                    )
                )
                remaining.pop(index)
                break
    slot_by_id = {connection.connection_id: slot for slot, connection in enumerate(selected)}
    candidates = tuple(
        SocialKnownSelectionCandidate(
            connection_id=connection.connection_id,
            weight=weights_by_id[connection.connection_id],
            surfaced=connection.connection_id in slot_by_id,
            slot=slot_by_id.get(connection.connection_id, -1),
        )
        for connection in sorted(connections, key=lambda item: item.connection_id)
    )
    return tuple(selected), candidates, tuple(draws)


def _known_selection_weight(
    connection: SocialConnection,
    definition: SocialContactDefinition,
    week: int,
) -> float:
    gap = max(0, week - connection.last_interaction_week)
    recency_pressure = min(0.3, max(0.0, (gap - 3.0) / 80.0))
    weight = (
        0.08
        + connection.closeness / 95.0
        + _trust(connection) / 125.0
        + definition.supportiveness * 0.2
        + recency_pressure
        - connection.strain / 115.0
    )
    return round(max(0.02, weight), 12)


def _known_options(
    state: AgentState,
    connection: SocialConnection,
    definition: SocialContactDefinition,
) -> tuple[EventOption, ...]:
    options = [_known_option(state, connection, definition, action="connect")]
    need_support = (
        state.mental.stress >= 58.0
        or state.mental.loneliness >= 58.0
        or state.needs.belonging <= 42.0
    )
    if need_support and _trust(connection) >= 45.0 and connection.closeness >= 35.0 and connection.strain <= 45.0:
        options.append(_known_option(state, connection, definition, action="seek_support"))
    return tuple(options)


def _known_option(
    state: AgentState,
    connection: SocialConnection,
    definition: SocialContactDefinition,
    *,
    action: str,
) -> EventOption:
    quality = _relationship_quality(connection)
    trust = _trust(connection)
    strain = connection.strain
    remote_time_factor = 0.65 if definition.remote_contact else 1.0
    relationship_value = _clamp01(
        quality * 0.72
        + definition.supportiveness * 0.14
        + (0.08 if connection.relationship in {"friend_from_home", "flatmate"} else 0.0)
    )
    risk = _clamp01(0.06 + strain / 130.0 + definition.volatility * 0.18 - trust / 500.0)
    uncertainty = _clamp01(0.08 + strain / 140.0 + (100.0 - trust) / 260.0 - connection.closeness / 420.0)
    comfort = _bounded_signal(-0.12 + quality * 0.75 - strain / 150.0)
    if action == "seek_support":
        option_id = f"seek_support:{connection.connection_id}"
        label = f"Seek support from {connection.name}"
        summary = "Ask for emotional or practical non-financial support."
        time_hours = 2.0 * remote_time_factor
        energy_cost = 8.0 * remote_time_factor + strain / 30.0
        social_value = _bounded_signal(relationship_value + 0.12)
        future_value = _bounded_signal(relationship_value * 0.55 + definition.supportiveness * 0.22)
        social_pressure = 0.18
    elif action == "connect":
        option_id = f"connect:{connection.connection_id}"
        label = f"Connect with {connection.name}"
        summary = "Spend modest time maintaining an existing relationship."
        time_hours = 2.0 * remote_time_factor
        energy_cost = 6.5 * remote_time_factor + strain / 40.0
        social_value = _bounded_signal(relationship_value)
        future_value = _bounded_signal(relationship_value * 0.5 + connection.closeness / 300.0)
        social_pressure = 0.1
    else:
        raise ValueError(f"Unexpected social action '{action}'.")
    return EventOption(
        option_id=option_id,
        label=label,
        summary=summary,
        time_cost_hours=round(time_hours, 6),
        energy_cost=round(energy_cost, 6),
        short_term_value=0.18,
        future_value=future_value,
        perceived_risk=risk,
        uncertainty=uncertainty,
        social_value=social_value,
        social_pressure=social_pressure,
        autonomy_value=0.05,
        health_value=0.05,
        comfort_value=comfort,
        goal_tags=("social", "belonging"),
        requires_full_estimated_cost=False,
    )


def _relationship_quality(connection: SocialConnection) -> float:
    return _clamp01(
        connection.closeness / 100.0 * 0.46
        + _trust(connection) / 100.0 * 0.42
        - connection.strain / 100.0 * 0.3
        + 0.08
    )


def _encounter_audit(
    context: WeeklyContext,
    state: AgentState,
    catalog: SocialCatalog,
    routine_social_contact: float,
    history: SocialHistory,
) -> SocialEncounterAudit:
    known = {connection.connection_id for connection in state.social.connections}
    empty_weights: tuple[SocialEncounterCandidateWeight, ...] = ()
    if len(known) >= catalog.max_network_size:
        return SocialEncounterAudit(0.0, 1.0, False, "", (), empty_weights, 0.0, None)
    surfaced_recently = _recent_encounter_contact_ids(history, context.week, catalog.relisting_cooldown_weeks)
    eligible = tuple(
        contact
        for contact in catalog.contacts
        if contact.contact_id not in known
        and contact.contact_id not in surfaced_recently
        and _context_applicable(state, contact.context)
    )
    candidate_weights = tuple(
        SocialEncounterCandidateWeight(
            contact_id=contact.contact_id,
            encounter_weight=contact.encounter_weight,
        )
        for contact in eligible
    )
    total_weight = sum(weight.encounter_weight for weight in candidate_weights)
    if not eligible:
        return SocialEncounterAudit(0.0, 1.0, False, "", (), empty_weights, 0.0, None)
    city_factor = 0.65 + state.social.city_familiarity / 200.0
    routine_factor = 0.45 + routine_social_contact * 0.85
    probability = _clamp01(catalog.base_new_encounter_probability * city_factor * routine_factor)
    trigger_rng = _rng(context, state, "social-encounter-trigger")
    roll = trigger_rng.random()
    triggered = roll <= probability
    selected = ""
    selection_roll = None
    if triggered:
        selected, selection_roll = _weighted_contact_selection(context, state, eligible)
    return SocialEncounterAudit(
        probability=probability,
        roll=roll,
        triggered=triggered,
        selected_contact_id=selected,
        eligible_contact_ids=tuple(contact.contact_id for contact in eligible),
        candidate_weights=candidate_weights,
        total_weight=total_weight,
        selection_roll=selection_roll,
    )


def _weighted_contact_selection(
    context: WeeklyContext,
    state: AgentState,
    contacts: tuple[SocialContactDefinition, ...],
) -> tuple[str, float]:
    total = sum(contact.encounter_weight for contact in contacts)
    rng = _rng(context, state, "social-encounter-selection")
    roll = rng.random() * total
    cursor = 0.0
    for contact in sorted(contacts, key=lambda item: item.contact_id):
        cursor += contact.encounter_weight
        if roll <= cursor:
            return contact.contact_id, roll
    return max(contacts, key=lambda item: item.contact_id).contact_id, roll


def _encounter_option(definition: SocialContactDefinition) -> EventOption:
    return EventOption(
        option_id=f"engage:{definition.contact_id}",
        label=f"Talk with {definition.name}",
        summary="Make space for a light new acquaintance.",
        time_cost_hours=1.5,
        energy_cost=8.0,
        short_term_value=0.1,
        future_value=0.2,
        perceived_risk=0.22,
        uncertainty=0.35,
        social_value=0.5,
        social_pressure=0.08,
        autonomy_value=0.08,
        learning_value=0.05,
        comfort_value=-0.02,
        goal_tags=("social", "city"),
        requires_full_estimated_cost=False,
    )


def _keep_light_option() -> EventOption:
    return EventOption(
        option_id="keep_social_light",
        label="Keep social light",
        summary="Leave the week socially light without treating it as a failure.",
        time_cost_hours=0.25,
        energy_cost=1.0,
        short_term_value=0.04,
        future_value=0.0,
        perceived_risk=0.04,
        uncertainty=0.02,
        social_value=0.0,
        social_pressure=0.0,
        autonomy_value=0.25,
        comfort_value=0.18,
        goal_tags=("autonomy", "recovery"),
        requires_full_estimated_cost=False,
    )


def _find_social_decision(
    state: AgentState,
    context: WeeklyContext,
    runtime: SocialRuntimeState,
) -> tuple[DecisionRecord, EventOccurrence]:
    matches = tuple(
        decision
        for decision in context.decisions
        if isinstance(decision, DecisionRecord)
        and decision.source_event_id == runtime.planned_event_id
        and decision.source_event_version == runtime.planned_event_version
    )
    if len(matches) != 1:
        raise ValueError("Expected exactly one same-week M4 decision for weekly social focus.")
    decision = matches[0]
    if decision.agent_id != state.identity.agent_id:
        raise ValueError("Social decision does not belong to the current agent.")
    if decision.week != context.week:
        raise ValueError("Expected social decision week to match WeeklyContext.week.")
    event = next(
        (
            occurrence
            for occurrence in context.events
            if isinstance(occurrence, EventOccurrence)
            and occurrence.event_id == decision.source_event_id
            and occurrence.version == decision.source_event_version
        ),
        None,
    )
    if event is None:
        raise ValueError("Expected social decision to reference a same-week event.")
    if event.week != context.week:
        raise ValueError("Expected social event week to match WeeklyContext.week.")
    if event.category != "social_relationship":
        raise ValueError("Expected social decision event category to be social_relationship.")
    option_ids = {option.option_id for option in event.options}
    if set(runtime.planned_option_ids) != option_ids:
        raise ValueError("Expected social event options to match planned options.")
    if decision.chosen_option_id not in option_ids:
        raise ValueError("Expected social decision chosen option to exist on its event.")
    return decision, event


def _known_outcome(
    context: WeeklyContext,
    state: AgentState,
    connection: SocialConnection,
    definition: SocialContactDefinition,
    interaction_type: str,
) -> SocialInteractionOutcomeAudit:
    quality = (
        connection.closeness / 100.0 * 0.34
        + _trust(connection) / 100.0 * 0.3
        + definition.responsiveness * 0.2
        + definition.supportiveness * 0.16
        - connection.strain / 100.0 * 0.28
        - state.mental.stress / 100.0 * 0.08
    )
    quality = _clamp01(quality)
    if interaction_type == "seek_support":
        support = _clamp01(quality + definition.supportiveness * 0.18)
        weights = (
            ("supportive", 0.2 + support * 0.58),
            ("limited", 0.25 + (1.0 - abs(support - 0.5) * 2.0) * 0.25),
            ("unavailable", 0.12 + (1.0 - support) * 0.46),
        )
        namespace = "social-support"
    else:
        friction = _clamp01(0.08 + definition.volatility * 0.22 + connection.strain / 180.0)
        weights = (
            ("warm", 0.22 + quality * 0.55),
            ("neutral", 0.28 + (1.0 - abs(quality - 0.5) * 2.0) * 0.22),
            ("friction", friction),
        )
        namespace = "social-interaction"
    return _roll_outcome(context, state, namespace, connection.connection_id, weights)


def _encounter_outcome(
    context: WeeklyContext,
    state: AgentState,
    definition: SocialContactDefinition,
) -> SocialInteractionOutcomeAudit:
    openness = _clamp01(
        0.25
        + state.personality.curiosity / 220.0
        + state.personality.social_need / 260.0
        + definition.responsiveness * 0.2
        - state.mental.stress / 220.0
    )
    weights = (
        ("promising", 0.16 + openness * 0.44),
        ("neutral", 0.32),
        ("awkward", 0.14 + (1.0 - openness) * 0.28 + definition.volatility * 0.12),
    )
    return _roll_outcome(context, state, "social-interaction", definition.contact_id, weights)


def _roll_outcome(
    context: WeeklyContext,
    state: AgentState,
    namespace: str,
    contact_id: str,
    raw_weights: tuple[tuple[str, float], ...],
) -> SocialInteractionOutcomeAudit:
    total = sum(max(0.0, weight) for _, weight in raw_weights)
    probabilities = tuple(
        SocialOutcomeProbability(outcome_id=outcome_id, probability=max(0.0, weight) / total)
        for outcome_id, weight in raw_weights
    )
    rng = _rng(context, state, namespace, contact_id)
    roll = rng.random()
    cursor = 0.0
    selected = probabilities[-1].outcome_id
    for probability in probabilities:
        cursor += probability.probability
        if roll <= cursor:
            selected = probability.outcome_id
            break
    return SocialInteractionOutcomeAudit(probabilities=probabilities, roll=roll, selected_outcome_id=selected)


def _fixed_outcome(outcome_id: str) -> SocialInteractionOutcomeAudit:
    return SocialInteractionOutcomeAudit(
        probabilities=(SocialOutcomeProbability(outcome_id=outcome_id, probability=1.0),),
        roll=0.0,
        selected_outcome_id=outcome_id,
    )


def _apply_known_outcome(
    state: AgentState,
    week: int,
    contact_id: str,
    interaction_type: str,
    outcome_id: str,
) -> tuple[AgentState, tuple[RelationshipChange, ...], tuple[SocialStateEffect, ...]]:
    deltas = {
        "warm": (4.0, 2.4, -2.0),
        "neutral": (1.2, 0.5, -0.8),
        "friction": (-2.5, -1.5, 4.0),
        "supportive": (3.0, 3.0, -2.2),
        "limited": (0.8, 0.4, -0.6),
        "unavailable": (-1.2, -1.8, 2.5),
    }[outcome_id]
    if interaction_type == "seek_support":
        state_effects = _support_state_effects(state, outcome_id)
    else:
        state_effects = _interaction_state_effects(state, interaction_type, outcome_id)
    connections: list[SocialConnection] = []
    changes: list[RelationshipChange] = []
    for connection in state.social.connections:
        if connection.connection_id != contact_id:
            connections.append(connection)
            continue
        next_connection = replace(
            connection,
            closeness=_clamp(connection.closeness + deltas[0]),
            trust=_clamp(_trust(connection) + deltas[1]),
            strain=_clamp(connection.strain + deltas[2]),
            last_interaction_week=week,
        )
        for field, before, after in (
            ("closeness", connection.closeness, next_connection.closeness),
            ("trust", _trust(connection), _trust(next_connection)),
            ("strain", connection.strain, next_connection.strain),
            ("last_interaction_week", float(connection.last_interaction_week), float(week)),
        ):
            if before != after:
                changes.append(RelationshipChange(contact_id, field, before, after))
        connections.append(next_connection)
    next_state = replace(state, social=replace(state.social, connections=tuple(connections)))
    next_state = _apply_state_effects(next_state, state_effects)
    return next_state, tuple(changes), state_effects


def _interaction_state_effects(
    state: AgentState,
    interaction_type: str,
    outcome_id: str,
) -> tuple[SocialStateEffect, ...]:
    deltas = {
        "warm": {"mental.loneliness": -4.0, "needs.belonging": 3.0, "mental.mood": 2.0, "health.energy": -3.0},
        "neutral": {"mental.loneliness": -1.4, "needs.belonging": 0.8, "health.energy": -2.0},
        "friction": {"mental.stress": 3.0, "mental.mood": -2.5, "health.energy": -3.5},
        "promising": {"mental.loneliness": -3.0, "needs.belonging": 2.5, "mental.mood": 2.0, "health.energy": -4.0},
        "awkward": {"mental.stress": 2.0, "mental.mood": -1.5, "health.energy": -2.5},
    }.get(outcome_id, {"health.energy": -1.0})
    _ = interaction_type
    return _effects_from_deltas(state, deltas)


def _support_state_effects(state: AgentState, outcome_id: str) -> tuple[SocialStateEffect, ...]:
    deltas = {
        "supportive": {"mental.stress": -5.0, "mental.loneliness": -4.0, "needs.belonging": 3.0, "mental.mood": 2.5, "health.energy": -2.0},
        "limited": {"mental.stress": -1.5, "mental.loneliness": -1.0, "needs.belonging": 0.6, "health.energy": -1.5},
        "unavailable": {"mental.stress": 2.5, "mental.loneliness": 2.0, "mental.mood": -1.5, "health.energy": -1.0},
    }[outcome_id]
    return _effects_from_deltas(state, deltas)


def _effects_from_deltas(
    state: AgentState,
    deltas: dict[str, float],
) -> tuple[SocialStateEffect, ...]:
    effects = []
    for path, delta in deltas.items():
        before = _state_value(state, path)
        after = _clamp(before + delta)
        effects.append(SocialStateEffect(path=path, before=before, after=after, clamped=after != before + delta))
    return tuple(effects)


def _apply_state_effects(
    state: AgentState,
    effects: tuple[SocialStateEffect, ...],
) -> AgentState:
    health = state.health
    mental = state.mental
    needs = state.needs
    for effect in effects:
        if effect.path == "health.energy":
            health = replace(health, energy=effect.after)
        elif effect.path == "mental.stress":
            mental = replace(mental, stress=effect.after)
        elif effect.path == "mental.mood":
            mental = replace(mental, mood=effect.after)
        elif effect.path == "mental.loneliness":
            mental = replace(mental, loneliness=effect.after)
        elif effect.path == "needs.belonging":
            needs = replace(needs, belonging=effect.after)
        else:
            raise ValueError(f"Social engine cannot mutate '{effect.path}'.")
    return replace(state, health=health, mental=mental, needs=needs)


def _state_value(state: AgentState, path: str) -> float:
    root, field = path.split(".", 1)
    if root == "health":
        value = getattr(state.health, field)
    elif root == "mental":
        value = getattr(state.mental, field)
    elif root == "needs":
        value = getattr(state.needs, field)
    else:
        raise ValueError(f"Social engine cannot read '{path}'.")
    return float(value)


def _trust(connection: SocialConnection) -> float:
    if connection.trust is None:
        return connection.closeness
    return connection.trust


def _definition_or_default(
    catalog: SocialCatalog,
    connection: SocialConnection,
) -> SocialContactDefinition:
    try:
        return catalog.contact(connection.connection_id)
    except KeyError:
        return SocialContactDefinition(
            contact_id=connection.connection_id,
            name=connection.name,
            relationship=connection.relationship,
            context="existing",
            base_availability=0.35,
            proximity=0.45,
            responsiveness=0.45,
            volatility=0.2,
            supportiveness=0.4,
            neglect_resistance=0.4,
            remote_contact=connection.relationship.endswith("home"),
            initial_closeness=connection.closeness,
            initial_trust=_trust(connection),
        )


def _support_network_target(
    connections: tuple[SocialConnection, ...],
    catalog: SocialCatalog,
) -> tuple[float, tuple[str, ...], tuple[float, ...]]:
    weighted_scores: list[tuple[str, float]] = []
    for connection in connections:
        definition = _definition_or_default(catalog, connection)
        relationship_quality = _relationship_quality(connection)
        quality = max(
            0.0,
            (
                relationship_quality * 74.0
                + definition.supportiveness * 18.0
                - connection.strain * 0.45
            ),
        )
        if relationship_quality >= 0.12:
            weighted_scores.append((connection.connection_id, quality))
    if not weighted_scores:
        return 0.0, (), ()
    top = sorted(weighted_scores, key=lambda item: (-item[1], item[0]))[:5]
    total_weight = sum(1.0 / (index + 1) for index, _ in enumerate(top))
    weighted = sum(score / (index + 1) for index, (_, score) in enumerate(top)) / total_weight
    meaningful_breadth = sum(min(1.0, score / 45.0) for _, score in top)
    breadth_bonus = min(10.0, meaningful_breadth * 2.0)
    target = _clamp(weighted + breadth_bonus)
    return (
        target,
        tuple(connection_id for connection_id, _ in top),
        tuple(round(score, 6) for _, score in top),
    )


def _connection(state: AgentState, contact_id: str) -> SocialConnection:
    for connection in state.social.connections:
        if connection.connection_id == contact_id:
            return connection
    raise ValueError(f"Unknown social connection '{contact_id}'.")


def _context_applicable(state: AgentState, contact_context: str) -> bool:
    if contact_context in {"general", "existing"}:
        return True
    if contact_context == "education":
        return state.education.status == "enrolled"
    if contact_context == "employment":
        return state.employment.status == "employed"
    return False


def _recent_encounter_contact_ids(
    history: SocialHistory,
    current_week: int,
    cooldown_weeks: int,
) -> set[str]:
    if cooldown_weeks <= 0:
        return set()
    recent = set()
    for record in history.planning_records:
        selected = record.encounter.selected_contact_id
        if selected and current_week - record.week <= cooldown_weeks:
            recent.add(selected)
    return recent


def _rng(context: WeeklyContext, state: AgentState, namespace: str, *parts: str) -> random.Random:
    seed = derive_stable_seed(
        namespace,
        str(context.config.simulation.seed),
        context.config.simulation.name,
        state.identity.agent_id,
        str(context.week),
        *parts,
    )
    return random.Random(seed)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, round(float(value), 6)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 12)))


def _bounded_probability(value: float) -> float:
    return max(
        MIN_AVAILABILITY_PROBABILITY,
        min(MAX_AVAILABILITY_PROBABILITY, round(float(value), 12)),
    )


def _bounded_signal(value: float) -> float:
    return max(-1.0, min(1.0, round(float(value), 12)))
