from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from lifesim.agents.state import SerializableState
from lifesim.events.model import EventCatalog, EventCondition

MONEY_PATHS = frozenset(
    {
        "financial.cash",
        "financial.bank_balance",
        "financial.savings",
        "financial.emergency_fund",
    }
)
BOUNDED_FLOAT_PATHS = frozenset(
    {
        "health.physical_health",
        "health.energy",
        "health.mobility",
        "mental.mood",
        "mental.stress",
        "mental.mental_load",
        "mental.recovery_need",
        "mental.loneliness",
        "needs.housing_security",
        "needs.food_security",
        "needs.safety",
        "needs.belonging",
        "needs.autonomy",
        "needs.purpose",
        "education.progress",
        "social.support_network_strength",
        "social.city_familiarity",
    }
)
NONNEGATIVE_FLOAT_PATHS = frozenset({"health.sleep_debt"})
WRITABLE_PATHS = MONEY_PATHS | BOUNDED_FLOAT_PATHS | NONNEGATIVE_FLOAT_PATHS


@dataclass(frozen=True, slots=True)
class StateEffectDefinition(SerializableState):
    path: str
    delta: Decimal | float
    delay_weeks: int = 0
    conditions: tuple[EventCondition, ...] = ()

    def __post_init__(self) -> None:
        if self.path not in WRITABLE_PATHS:
            raise ValueError(f"Unsupported consequence write path '{self.path}'.")
        _integer(self.delay_weeks, "delay_weeks", minimum=0)
        object.__setattr__(
            self,
            "conditions",
            _typed_tuple(self.conditions, EventCondition, "conditions"),
        )
        if self.path in MONEY_PATHS:
            if not isinstance(self.delta, Decimal):
                raise TypeError("Expected monetary effect delta to be Decimal.")
            if not self.delta.is_finite():
                raise ValueError("Expected monetary effect delta to be finite.")
        else:
            object.__setattr__(self, "delta", _finite_number(self.delta, "delta"))


@dataclass(frozen=True, slots=True)
class OutcomeDefinition(SerializableState):
    outcome_id: str
    weight: float
    effects: tuple[StateEffectDefinition, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.outcome_id, "outcome_id")
        object.__setattr__(
            self,
            "weight",
            _finite_number(self.weight, "weight", minimum=0.0, maximum=None),
        )
        if self.weight <= 0:
            raise ValueError("Expected outcome weight to be positive.")
        object.__setattr__(
            self,
            "effects",
            _typed_tuple(self.effects, StateEffectDefinition, "effects"),
        )


@dataclass(frozen=True, slots=True)
class OptionConsequenceDefinition(SerializableState):
    event_id: str
    event_version: str
    option_id: str
    effects: tuple[StateEffectDefinition, ...] = ()
    outcomes: tuple[OutcomeDefinition, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.event_version, "event_version")
        _require_non_empty(self.option_id, "option_id")
        object.__setattr__(
            self,
            "effects",
            _typed_tuple(self.effects, StateEffectDefinition, "effects"),
        )
        outcomes = _typed_tuple(self.outcomes, OutcomeDefinition, "outcomes")
        _require_unique(tuple(outcome.outcome_id for outcome in outcomes), "outcome_id")
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.event_id, self.event_version, self.option_id)


@dataclass(frozen=True, slots=True)
class ConsequenceCatalog(SerializableState):
    definitions: tuple[OptionConsequenceDefinition, ...] = ()

    def __post_init__(self) -> None:
        definitions = _typed_tuple(
            self.definitions,
            OptionConsequenceDefinition,
            "definitions",
        )
        keys = tuple(definition.key for definition in definitions)
        if len(set(keys)) != len(keys):
            raise ValueError("Expected consequence keys to be unique.")
        object.__setattr__(self, "definitions", definitions)

    def find(
        self,
        *,
        event_id: str,
        event_version: str,
        option_id: str,
    ) -> OptionConsequenceDefinition | None:
        key = (event_id, event_version, option_id)
        for definition in self.definitions:
            if definition.key == key:
                return definition
        return None


@dataclass(frozen=True, slots=True)
class ScheduledEffect(SerializableState):
    scheduled_effect_id: str
    source_decision_id: str
    source_consequence_id: str
    source_event_id: str
    source_event_version: str
    chosen_option_id: str
    source_outcome_id: str | None
    created_week: int
    due_week: int
    effect: StateEffectDefinition

    def __post_init__(self) -> None:
        _require_non_empty(self.scheduled_effect_id, "scheduled_effect_id")
        _require_non_empty(self.source_decision_id, "source_decision_id")
        _require_non_empty(self.source_consequence_id, "source_consequence_id")
        _require_non_empty(self.source_event_id, "source_event_id")
        _require_non_empty(self.source_event_version, "source_event_version")
        _require_non_empty(self.chosen_option_id, "chosen_option_id")
        if self.source_outcome_id is not None:
            _require_non_empty(self.source_outcome_id, "source_outcome_id")
        _integer(self.created_week, "created_week", minimum=0)
        _integer(self.due_week, "due_week", minimum=0)
        if self.due_week < self.created_week:
            raise ValueError("Expected due_week to be >= created_week.")
        if not isinstance(self.effect, StateEffectDefinition):
            raise TypeError("Expected scheduled effect to contain StateEffectDefinition.")


@dataclass(frozen=True, slots=True)
class EffectApplication(SerializableState):
    path: str
    requested_delta: Decimal | float
    before: Decimal | float | None
    after: Decimal | float | None
    clamped: bool
    skipped: bool = False
    skip_reason: str = ""
    scheduled_effect_id: str | None = None

    def __post_init__(self) -> None:
        if self.path not in WRITABLE_PATHS:
            raise ValueError(f"Unsupported consequence write path '{self.path}'.")
        if not isinstance(self.clamped, bool):
            raise TypeError("Expected clamped to be bool.")
        if not isinstance(self.skipped, bool):
            raise TypeError("Expected skipped to be bool.")
        if self.skipped and not self.skip_reason:
            raise ValueError("Skipped effect applications must include a skip_reason.")
        if not self.skipped and self.skip_reason:
            raise ValueError("Applied effects must not include a skip_reason.")
        if self.scheduled_effect_id is not None:
            _require_non_empty(self.scheduled_effect_id, "scheduled_effect_id")


@dataclass(frozen=True, slots=True)
class ConsequenceRecord(SerializableState):
    consequence_id: str
    source_decision_id: str
    source_event_id: str
    source_event_version: str
    chosen_option_id: str
    week_resolved: int
    selected_outcome_id: str | None = None
    outcome_roll: float | None = None
    outcome_total_weight: float | None = None
    source_scheduled_effect_id: str | None = None
    effect_applications: tuple[EffectApplication, ...] = ()
    scheduled_effects_created: tuple[ScheduledEffect, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.consequence_id, "consequence_id")
        _require_non_empty(self.source_decision_id, "source_decision_id")
        _require_non_empty(self.source_event_id, "source_event_id")
        _require_non_empty(self.source_event_version, "source_event_version")
        _require_non_empty(self.chosen_option_id, "chosen_option_id")
        _integer(self.week_resolved, "week_resolved", minimum=0)
        if self.selected_outcome_id is not None:
            _require_non_empty(self.selected_outcome_id, "selected_outcome_id")
        for field_name in ("outcome_roll", "outcome_total_weight"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_number(value, field_name))
        if self.source_scheduled_effect_id is not None:
            _require_non_empty(self.source_scheduled_effect_id, "source_scheduled_effect_id")
        object.__setattr__(
            self,
            "effect_applications",
            _typed_tuple(self.effect_applications, EffectApplication, "effect_applications"),
        )
        object.__setattr__(
            self,
            "scheduled_effects_created",
            _typed_tuple(self.scheduled_effects_created, ScheduledEffect, "scheduled_effects_created"),
        )


@dataclass(frozen=True, slots=True)
class ConsequenceHistory:
    records: tuple[ConsequenceRecord, ...] = ()

    def __post_init__(self) -> None:
        records = _typed_tuple(self.records, ConsequenceRecord, "records")
        _require_unique(tuple(record.consequence_id for record in records), "consequence_id")
        object.__setattr__(self, "records", records)

    def record(self, records: tuple[ConsequenceRecord, ...]) -> ConsequenceHistory:
        return ConsequenceHistory(self.records + tuple(records))

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


@dataclass(frozen=True, slots=True)
class ConsequenceRuntimeState:
    history: ConsequenceHistory = field(default_factory=ConsequenceHistory)
    pending_scheduled_effects: tuple[ScheduledEffect, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, ConsequenceHistory):
            raise TypeError("Expected consequence runtime history to be ConsequenceHistory.")
        pending = _typed_tuple(
            self.pending_scheduled_effects,
            ScheduledEffect,
            "pending_scheduled_effects",
        )
        object.__setattr__(self, "pending_scheduled_effects", pending)
        processed = _string_sequence(self.processed_decision_ids, "processed_decision_ids")
        _require_unique(processed, "processed_decision_ids")
        object.__setattr__(self, "processed_decision_ids", processed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.to_dict(),
            "pending_scheduled_effects": [
                effect.to_dict() for effect in self.pending_scheduled_effects
            ],
            "processed_decision_ids": list(self.processed_decision_ids),
        }


def validate_consequence_catalog(
    consequence_catalog: ConsequenceCatalog,
    event_catalog: EventCatalog,
) -> None:
    valid_keys = {
        (event.event_id, event.version, option.option_id)
        for event in event_catalog.definitions
        for option in event.options
    }
    for definition in consequence_catalog.definitions:
        if definition.key not in valid_keys:
            event_id, version, option_id = definition.key
            raise ValueError(
                "Consequence definition references unknown event/version/option "
                f"'{event_id}/{version}/{option_id}'."
            )


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Expected '{name}' to be finite.")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"Expected '{name}' to be >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"Expected '{name}' to be <= {maximum}.")
    return numeric


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected '{name}' to be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Expected '{name}' to be >= {minimum}.")
    return value


def _typed_tuple(values: Any, item_type: type[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, str) or not isinstance(values, list | tuple):
        raise TypeError(f"Expected '{name}' to be a list or tuple.")
    output = tuple(values)
    for item in output:
        if not isinstance(item, item_type):
            raise TypeError(f"Expected '{name}' to contain {item_type.__name__} values.")
    return output


def _string_sequence(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, list | tuple):
        raise TypeError(f"Expected '{name}' to be a list or tuple of strings.")
    strings = tuple(values)
    for item in strings:
        _require_non_empty(item, name)
    return strings


def _require_unique(values: tuple[Any, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Expected '{name}' values to be unique.")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
