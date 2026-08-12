from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from lifesim.agents.state import AgentState, SerializableState
from lifesim.weekly import WeeklyContext


@dataclass(frozen=True, slots=True)
class EventCondition(SerializableState):
    condition_type: str
    path: str = ""
    value: Any = None

    def __post_init__(self) -> None:
        if self.condition_type not in {
            "numeric_gte",
            "numeric_lte",
            "string_equals",
            "collection_empty",
            "collection_non_empty",
            "week_gte",
            "week_lte",
        }:
            raise ValueError(f"Unsupported event condition type '{self.condition_type}'.")
        if self.condition_type.startswith(("numeric", "string", "collection")) and not self.path:
            raise ValueError("Expected condition path for agent-state condition.")

    def evaluate(self, state: AgentState, context: WeeklyContext) -> bool:
        if self.condition_type == "week_gte":
            return context.week >= _number(self.value, "value")
        if self.condition_type == "week_lte":
            return context.week <= _number(self.value, "value")

        actual = _resolve_path(state, self.path)
        if self.condition_type == "numeric_gte":
            return _comparable_number(actual, self.path) >= _comparable_number(self.value, "value")
        if self.condition_type == "numeric_lte":
            return _comparable_number(actual, self.path) <= _comparable_number(self.value, "value")
        if self.condition_type == "string_equals":
            if not isinstance(actual, str):
                raise TypeError(f"Expected '{self.path}' to resolve to a string.")
            return actual == self.value
        if self.condition_type == "collection_empty":
            return len(_collection(actual, self.path)) == 0
        if self.condition_type == "collection_non_empty":
            return len(_collection(actual, self.path)) > 0
        raise ValueError(f"Unsupported event condition type '{self.condition_type}'.")


@dataclass(frozen=True, slots=True)
class WeightModifier(SerializableState):
    condition: EventCondition
    multiplier: float

    def __post_init__(self) -> None:
        if self.multiplier < 0:
            raise ValueError("Expected weight modifier multiplier to be non-negative.")

    def applies(self, state: AgentState, context: WeeklyContext) -> bool:
        return self.condition.evaluate(state, context)


@dataclass(frozen=True, slots=True)
class EventDefinition(SerializableState):
    event_id: str
    version: str
    category: str
    base_weight: float
    conditions: tuple[EventCondition, ...]
    weight_modifiers: tuple[WeightModifier, ...]
    cooldown_weeks: int
    tags: tuple[str, ...]
    title: str
    summary: str

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.version, "version")
        _require_non_empty(self.category, "category")
        _require_non_empty(self.title, "title")
        if self.base_weight < 0:
            raise ValueError("Expected event base_weight to be non-negative.")
        if self.cooldown_weeks < 0:
            raise ValueError("Expected cooldown_weeks to be non-negative.")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "weight_modifiers", tuple(self.weight_modifiers))
        object.__setattr__(self, "tags", tuple(self.tags))

    def is_conditionally_eligible(self, state: AgentState, context: WeeklyContext) -> bool:
        return all(condition.evaluate(state, context) for condition in self.conditions)

    def effective_weight(self, state: AgentState, context: WeeklyContext) -> float:
        weight = self.base_weight
        for modifier in self.weight_modifiers:
            if modifier.applies(state, context):
                weight *= modifier.multiplier
        return round(max(0.0, weight), 12)

    def to_occurrence(self, *, week: int, effective_weight: float) -> EventOccurrence:
        return EventOccurrence(
            event_id=self.event_id,
            version=self.version,
            week=week,
            category=self.category,
            effective_weight=effective_weight,
            title=self.title,
            summary=self.summary,
            tags=self.tags,
        )


@dataclass(frozen=True, slots=True)
class EventOccurrence(SerializableState):
    event_id: str
    version: str
    week: int
    category: str
    effective_weight: float
    title: str
    summary: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        if self.week < 0:
            raise ValueError("Expected occurrence week to be >= 0.")
        if self.effective_weight < 0:
            raise ValueError("Expected occurrence effective_weight to be non-negative.")
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True, slots=True)
class EventHistory:
    occurrences: tuple[EventOccurrence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrences", tuple(self.occurrences))

    def record(self, occurrences: tuple[EventOccurrence, ...]) -> EventHistory:
        return EventHistory(self.occurrences + tuple(occurrences))

    def last_week(self, event_id: str) -> int | None:
        weeks = [occurrence.week for occurrence in self.occurrences if occurrence.event_id == event_id]
        if not weeks:
            return None
        return max(weeks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrences": [occurrence.to_dict() for occurrence in self.occurrences],
        }


@dataclass(frozen=True, slots=True)
class EventCandidateTrace(SerializableState):
    event_id: str
    eligible: bool
    effective_weight: float
    reason: str


@dataclass(frozen=True, slots=True)
class EventSelectionTrace(SerializableState):
    week: int
    trigger_probability: float
    trigger_roll: float
    candidates: tuple[EventCandidateTrace, ...]
    selected_event_ids: tuple[str, ...]

    @property
    def no_event(self) -> bool:
        return len(self.selected_event_ids) == 0

    def to_dict(self) -> dict[str, Any]:
        output = SerializableState.to_dict(self)
        output["no_event"] = self.no_event
        return output


@dataclass(frozen=True, slots=True)
class EventSelectionResult:
    occurrences: tuple[EventOccurrence, ...]
    history: EventHistory
    trace: EventSelectionTrace


@dataclass(frozen=True, slots=True)
class EventCatalog:
    definitions: tuple[EventDefinition, ...]
    max_events_per_week: int = 1
    event_probability: float = 0.35

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", tuple(self.definitions))
        if self.max_events_per_week < 0:
            raise ValueError("Expected max_events_per_week to be non-negative.")
        if not 0 <= self.event_probability <= 1:
            raise ValueError("Expected event_probability to be between 0 and 1.")
        ids = [definition.event_id for definition in self.definitions]
        if len(set(ids)) != len(ids):
            raise ValueError("Expected event_id values to be unique within a catalog.")


def is_on_cooldown(definition: EventDefinition, history: EventHistory, week: int) -> bool:
    last_week = history.last_week(definition.event_id)
    if last_week is None:
        return False
    return week - last_week <= definition.cooldown_weeks


def _resolve_path(state: AgentState, path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"Invalid event condition path '{path}'.")
        current = getattr(current, part)
    return current


def _comparable_number(value: Any, name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float | str):
        return Decimal(str(value))
    raise TypeError(f"Expected '{name}' to resolve to a numeric value.")


def _number(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"Expected '{name}' to be an integer.")
    return value


def _collection(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"Expected '{path}' to resolve to a collection.")
    return tuple(value)


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
