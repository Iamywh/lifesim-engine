from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lifesim.agents.state import SerializableState


@dataclass(frozen=True, slots=True)
class SocialContactDefinition(SerializableState):
    contact_id: str
    name: str
    relationship: str
    context: str
    base_availability: float
    proximity: float
    responsiveness: float
    volatility: float
    supportiveness: float
    neglect_resistance: float
    remote_contact: bool = False
    initial_closeness: float = 18.0
    initial_trust: float = 18.0
    encounter_weight: float = 1.0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("contact_id", "name", "relationship", "context"):
            _require_non_empty(getattr(self, name), name)
        if self.context not in {"existing", "general", "education", "employment"}:
            raise ValueError("Expected social contact context to be one of existing, general, education, employment.")
        for name in (
            "base_availability",
            "proximity",
            "responsiveness",
            "volatility",
            "supportiveness",
            "neglect_resistance",
        ):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0))
        for name in ("initial_closeness", "initial_trust"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=100.0))
        object.__setattr__(
            self,
            "encounter_weight",
            _finite_number(self.encounter_weight, "encounter_weight", minimum=0.000001, maximum=100.0),
        )
        if not isinstance(self.remote_contact, bool):
            raise TypeError("Expected remote_contact to be bool.")
        object.__setattr__(self, "tags", _string_sequence(self.tags, "tags"))


@dataclass(frozen=True, slots=True)
class SocialCatalog(SerializableState):
    contacts: tuple[SocialContactDefinition, ...]
    base_new_encounter_probability: float = 0.18
    max_known_options: int = 2
    max_network_size: int = 12
    relisting_cooldown_weeks: int = 3

    def __post_init__(self) -> None:
        contacts = _typed_tuple(self.contacts, SocialContactDefinition, "contacts")
        if not contacts:
            raise ValueError("Expected social catalog to contain at least one contact.")
        _require_unique(tuple(contact.contact_id for contact in contacts), "contact_id")
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(
            self,
            "base_new_encounter_probability",
            _finite_number(self.base_new_encounter_probability, "base_new_encounter_probability", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(self, "max_known_options", _integer(self.max_known_options, "max_known_options", minimum=0, maximum=5))
        object.__setattr__(self, "max_network_size", _integer(self.max_network_size, "max_network_size", minimum=1, maximum=50))
        object.__setattr__(
            self,
            "relisting_cooldown_weeks",
            _integer(self.relisting_cooldown_weeks, "relisting_cooldown_weeks", minimum=0, maximum=520),
        )

    def contact(self, contact_id: str) -> SocialContactDefinition:
        for contact in self.contacts:
            if contact.contact_id == contact_id:
                return contact
        raise KeyError(f"Unknown social contact '{contact_id}'.")


@dataclass(frozen=True, slots=True)
class SocialMaintenanceChange(SerializableState):
    connection_id: str
    closeness_before: float
    closeness_after: float
    trust_before: float
    trust_after: float
    strain_before: float
    strain_after: float
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty(self.connection_id, "connection_id")
        _require_non_empty(self.reason, "reason")
        for name in (
            "closeness_before",
            "closeness_after",
            "trust_before",
            "trust_after",
            "strain_before",
            "strain_after",
        ):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=100.0))


@dataclass(frozen=True, slots=True)
class SocialSupportNetworkAudit(SerializableState):
    before: float
    target: float
    after: float
    connection_count: int
    meaningful_contributor_ids: tuple[str, ...] = ()
    meaningful_contributor_scores: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("before", "target", "after"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=100.0))
        object.__setattr__(self, "connection_count", _integer(self.connection_count, "connection_count", minimum=0, maximum=1000))
        object.__setattr__(
            self,
            "meaningful_contributor_ids",
            _string_sequence(self.meaningful_contributor_ids, "meaningful_contributor_ids"),
        )
        scores = tuple(
            _finite_number(value, "meaningful_contributor_scores", minimum=0.0, maximum=100.0)
            for value in self.meaningful_contributor_scores
        )
        object.__setattr__(self, "meaningful_contributor_scores", scores)
        if len(self.meaningful_contributor_ids) != len(self.meaningful_contributor_scores):
            raise ValueError("Expected support contributor ids and scores to have matching lengths.")


@dataclass(frozen=True, slots=True)
class SocialMaintenanceRecord(SerializableState):
    week: int
    changes: tuple[SocialMaintenanceChange, ...]
    support_network: SocialSupportNetworkAudit

    def __post_init__(self) -> None:
        object.__setattr__(self, "week", _integer(self.week, "week", minimum=1, maximum=1000000))
        object.__setattr__(self, "changes", _typed_tuple(self.changes, SocialMaintenanceChange, "changes"))
        if not isinstance(self.support_network, SocialSupportNetworkAudit):
            raise TypeError("Expected support_network to be SocialSupportNetworkAudit.")


@dataclass(frozen=True, slots=True)
class SocialAvailabilityAudit(SerializableState):
    connection_id: str
    probability: float
    roll: float
    available: bool

    def __post_init__(self) -> None:
        _require_non_empty(self.connection_id, "connection_id")
        object.__setattr__(self, "probability", _finite_number(self.probability, "probability", minimum=0.0, maximum=1.0))
        object.__setattr__(self, "roll", _finite_number(self.roll, "roll", minimum=0.0, maximum=1.0))
        if not isinstance(self.available, bool):
            raise TypeError("Expected available to be bool.")


@dataclass(frozen=True, slots=True)
class SocialKnownSelectionCandidate(SerializableState):
    connection_id: str
    weight: float
    surfaced: bool
    slot: int = -1

    def __post_init__(self) -> None:
        _require_non_empty(self.connection_id, "connection_id")
        object.__setattr__(self, "weight", _finite_number(self.weight, "weight", minimum=0.0, maximum=1000.0))
        if not isinstance(self.surfaced, bool):
            raise TypeError("Expected surfaced to be bool.")
        object.__setattr__(self, "slot", _integer(self.slot, "slot", minimum=-1, maximum=1000))


@dataclass(frozen=True, slots=True)
class SocialKnownSelectionDraw(SerializableState):
    slot: int
    roll: float
    total_weight: float
    selected_connection_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _integer(self.slot, "slot", minimum=0, maximum=1000))
        object.__setattr__(self, "roll", _finite_number(self.roll, "roll", minimum=0.0, maximum=1000000.0))
        object.__setattr__(self, "total_weight", _finite_number(self.total_weight, "total_weight", minimum=0.000001, maximum=1000000.0))
        _require_non_empty(self.selected_connection_id, "selected_connection_id")


@dataclass(frozen=True, slots=True)
class SocialEncounterCandidateWeight(SerializableState):
    contact_id: str
    encounter_weight: float

    def __post_init__(self) -> None:
        _require_non_empty(self.contact_id, "contact_id")
        object.__setattr__(
            self,
            "encounter_weight",
            _finite_number(self.encounter_weight, "encounter_weight", minimum=0.000001, maximum=100.0),
        )


@dataclass(frozen=True, slots=True)
class SocialEncounterAudit(SerializableState):
    probability: float
    roll: float
    triggered: bool
    selected_contact_id: str = ""
    eligible_contact_ids: tuple[str, ...] = ()
    candidate_weights: tuple[SocialEncounterCandidateWeight, ...] = ()
    total_weight: float = 0.0
    selection_roll: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "probability", _finite_number(self.probability, "probability", minimum=0.0, maximum=1.0))
        object.__setattr__(self, "roll", _finite_number(self.roll, "roll", minimum=0.0, maximum=1.0))
        if not isinstance(self.triggered, bool):
            raise TypeError("Expected triggered to be bool.")
        if not isinstance(self.selected_contact_id, str):
            raise TypeError("Expected selected_contact_id to be a string.")
        object.__setattr__(self, "eligible_contact_ids", _string_sequence(self.eligible_contact_ids, "eligible_contact_ids"))
        object.__setattr__(
            self,
            "candidate_weights",
            _typed_tuple(self.candidate_weights, SocialEncounterCandidateWeight, "candidate_weights"),
        )
        object.__setattr__(self, "total_weight", _finite_number(self.total_weight, "total_weight", minimum=0.0, maximum=1000000.0))
        if self.selection_roll is not None:
            object.__setattr__(self, "selection_roll", _finite_number(self.selection_roll, "selection_roll", minimum=0.0, maximum=1000000.0))
        if self.triggered and self.total_weight <= 0.0:
            raise ValueError("Expected triggered encounter selection to have positive total weight.")
        if not self.triggered and self.selection_roll is not None:
            raise ValueError("Expected no encounter selection roll when trigger did not occur.")
        if self.selected_contact_id and self.selected_contact_id not in self.eligible_contact_ids:
            raise ValueError("Expected selected encounter contact to be eligible.")


@dataclass(frozen=True, slots=True)
class SocialPlanningRecord(SerializableState):
    week: int
    routine_profile_id: str
    routine_social_contact: float
    event_id: str
    event_version: str
    option_ids: tuple[str, ...]
    availability: tuple[SocialAvailabilityAudit, ...]
    encounter: SocialEncounterAudit
    known_selection_candidates: tuple[SocialKnownSelectionCandidate, ...] = ()
    known_selection_draws: tuple[SocialKnownSelectionDraw, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "week", _integer(self.week, "week", minimum=1, maximum=1000000))
        if not isinstance(self.routine_profile_id, str):
            raise TypeError("Expected routine_profile_id to be a string.")
        object.__setattr__(self, "routine_social_contact", _finite_number(self.routine_social_contact, "routine_social_contact", minimum=0.0, maximum=1.0))
        if not isinstance(self.event_id, str) or not isinstance(self.event_version, str):
            raise TypeError("Expected event id/version to be strings.")
        object.__setattr__(self, "option_ids", _string_sequence(self.option_ids, "option_ids"))
        object.__setattr__(self, "availability", _typed_tuple(self.availability, SocialAvailabilityAudit, "availability"))
        if not isinstance(self.encounter, SocialEncounterAudit):
            raise TypeError("Expected encounter to be SocialEncounterAudit.")
        object.__setattr__(
            self,
            "known_selection_candidates",
            _typed_tuple(
                self.known_selection_candidates,
                SocialKnownSelectionCandidate,
                "known_selection_candidates",
            ),
        )
        object.__setattr__(
            self,
            "known_selection_draws",
            _typed_tuple(
                self.known_selection_draws,
                SocialKnownSelectionDraw,
                "known_selection_draws",
            ),
        )


@dataclass(frozen=True, slots=True)
class SocialOutcomeProbability(SerializableState):
    outcome_id: str
    probability: float

    def __post_init__(self) -> None:
        _require_non_empty(self.outcome_id, "outcome_id")
        object.__setattr__(self, "probability", _finite_number(self.probability, "probability", minimum=0.0, maximum=1.0))


@dataclass(frozen=True, slots=True)
class SocialInteractionOutcomeAudit(SerializableState):
    probabilities: tuple[SocialOutcomeProbability, ...]
    roll: float
    selected_outcome_id: str

    def __post_init__(self) -> None:
        probabilities = _typed_tuple(self.probabilities, SocialOutcomeProbability, "probabilities")
        if not probabilities:
            raise ValueError("Expected social outcome probabilities to be non-empty.")
        total = sum(item.probability for item in probabilities)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Expected social outcome probabilities to sum to 1.")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "roll", _finite_number(self.roll, "roll", minimum=0.0, maximum=1.0))
        _require_non_empty(self.selected_outcome_id, "selected_outcome_id")
        if self.selected_outcome_id not in {item.outcome_id for item in probabilities}:
            raise ValueError("Expected selected social outcome to be one of the probabilities.")


@dataclass(frozen=True, slots=True)
class RelationshipChange(SerializableState):
    connection_id: str
    field: str
    before: float
    after: float

    def __post_init__(self) -> None:
        _require_non_empty(self.connection_id, "connection_id")
        if self.field not in {"closeness", "trust", "strain", "last_interaction_week"}:
            raise ValueError("Unexpected relationship change field.")
        for name in ("before", "after"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=1000000.0))


@dataclass(frozen=True, slots=True)
class SocialStateEffect(SerializableState):
    path: str
    before: float
    after: float
    clamped: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.path, "path")
        for name in ("before", "after"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=100.0))
        if not isinstance(self.clamped, bool):
            raise TypeError("Expected clamped to be bool.")


@dataclass(frozen=True, slots=True)
class SocialInteractionRecord(SerializableState):
    week: int
    decision_id: str
    option_id: str
    contact_id: str
    interaction_type: str
    outcome: SocialInteractionOutcomeAudit | None
    relationship_changes: tuple[RelationshipChange, ...] = ()
    state_effects: tuple[SocialStateEffect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "week", _integer(self.week, "week", minimum=1, maximum=1000000))
        for name in ("decision_id", "option_id", "contact_id", "interaction_type"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Expected {name} to be a string.")
        if self.outcome is not None and not isinstance(self.outcome, SocialInteractionOutcomeAudit):
            raise TypeError("Expected outcome to be SocialInteractionOutcomeAudit.")
        object.__setattr__(self, "relationship_changes", _typed_tuple(self.relationship_changes, RelationshipChange, "relationship_changes"))
        object.__setattr__(self, "state_effects", _typed_tuple(self.state_effects, SocialStateEffect, "state_effects"))
        if self.option_id and self.outcome is None:
            raise ValueError("Expected social interaction outcome when an option was chosen.")


@dataclass(frozen=True, slots=True)
class SocialHistory(SerializableState):
    maintenance_records: tuple[SocialMaintenanceRecord, ...] = ()
    planning_records: tuple[SocialPlanningRecord, ...] = ()
    interaction_records: tuple[SocialInteractionRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "maintenance_records", _typed_tuple(self.maintenance_records, SocialMaintenanceRecord, "maintenance_records"))
        object.__setattr__(self, "planning_records", _typed_tuple(self.planning_records, SocialPlanningRecord, "planning_records"))
        object.__setattr__(self, "interaction_records", _typed_tuple(self.interaction_records, SocialInteractionRecord, "interaction_records"))

    def record_maintenance(self, record: SocialMaintenanceRecord) -> SocialHistory:
        return SocialHistory(self.maintenance_records + (record,), self.planning_records, self.interaction_records)

    def record_planning(self, record: SocialPlanningRecord) -> SocialHistory:
        return SocialHistory(self.maintenance_records, self.planning_records + (record,), self.interaction_records)

    def record_interaction(self, record: SocialInteractionRecord) -> SocialHistory:
        return SocialHistory(self.maintenance_records, self.planning_records, self.interaction_records + (record,))


@dataclass(frozen=True, slots=True)
class SocialRuntimeState(SerializableState):
    history: SocialHistory = field(default_factory=SocialHistory)
    processed_maintenance_weeks: tuple[int, ...] = ()
    processed_planning_weeks: tuple[int, ...] = ()
    processed_execution_weeks: tuple[int, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    planned_event_id: str = ""
    planned_event_version: str = ""
    planned_option_ids: tuple[str, ...] = ()
    planned_week: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.history, SocialHistory):
            raise TypeError("Expected social runtime history to be SocialHistory.")
        for name in (
            "processed_maintenance_weeks",
            "processed_planning_weeks",
            "processed_execution_weeks",
        ):
            weeks = _integer_sequence(getattr(self, name), name)
            _require_unique(tuple(str(week) for week in weeks), name)
            object.__setattr__(self, name, weeks)
        object.__setattr__(self, "processed_decision_ids", _string_sequence(self.processed_decision_ids, "processed_decision_ids"))
        _require_unique(self.processed_decision_ids, "processed_decision_ids")
        for name in ("planned_event_id", "planned_event_version"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Expected {name} to be a string.")
        object.__setattr__(self, "planned_option_ids", _string_sequence(self.planned_option_ids, "planned_option_ids"))
        if self.planned_week is not None:
            object.__setattr__(self, "planned_week", _integer(self.planned_week, "planned_week", minimum=1, maximum=1000000))
        has_plan = bool(self.planned_event_id or self.planned_event_version or self.planned_option_ids)
        if has_plan != (self.planned_week is not None):
            raise ValueError("Expected planned social fields to share a planned week.")
        if tuple(record.week for record in self.history.maintenance_records) != self.processed_maintenance_weeks:
            raise ValueError("Expected processed social maintenance weeks to match history.")
        if tuple(record.week for record in self.history.planning_records) != self.processed_planning_weeks:
            raise ValueError("Expected processed social planning weeks to match history.")
        if tuple(record.week for record in self.history.interaction_records) != self.processed_execution_weeks:
            raise ValueError("Expected processed social execution weeks to match history.")
        decision_ids = tuple(record.decision_id for record in self.history.interaction_records if record.decision_id)
        if decision_ids != self.processed_decision_ids:
            raise ValueError("Expected processed social decision ids to match interaction history.")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")


def _finite_number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"Expected '{name}' to be finite and between {minimum} and {maximum}.")
    return result


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected '{name}' to be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"Expected '{name}' to be between {minimum} and {maximum}.")
    return value


def _integer_sequence(values: Any, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        raise TypeError(f"Expected '{name}' to be an integer sequence.")
    return tuple(_integer(value, name, minimum=1, maximum=1000000) for value in values)


def _typed_tuple(values: Any, item_type: type[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, str):
        raise TypeError(f"Expected '{name}' to be a sequence.")
    result = tuple(values)
    for item in result:
        if not isinstance(item, item_type):
            raise TypeError(f"Expected '{name}' to contain {item_type.__name__} values.")
    return result


def _string_sequence(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"Expected '{name}' to be a string sequence.")
    result = tuple(values)
    for value in result:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected '{name}' to contain non-empty strings.")
    return result


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Expected {name} values to be unique.")
