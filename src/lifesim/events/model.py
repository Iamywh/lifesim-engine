from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from lifesim.agents.state import AgentState, SerializableState
from lifesim.weekly import WeeklyContext


@dataclass(frozen=True, slots=True)
class EventCondition(SerializableState):
    condition_type: str
    path: str = ""
    value: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.condition_type, str):
            raise TypeError("Expected event condition type to be a string.")
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
        if self.condition_type.startswith(("numeric", "string", "collection")):
            _validate_path(self.path)
        if self.condition_type == "string_equals" and not isinstance(self.value, str):
            raise TypeError("Expected string_equals condition value to be a string.")
        if self.condition_type in {"week_gte", "week_lte"}:
            _integer(self.value, "value")
        if self.condition_type.startswith("numeric"):
            _comparable_number(self.value, "value")
        if self.condition_type.startswith(("numeric", "string", "collection")) and not self.path:
            raise ValueError("Expected condition path for agent-state condition.")

    def evaluate(self, state: AgentState, context: WeeklyContext) -> bool:
        if self.condition_type == "week_gte":
            return context.week >= _integer(self.value, "value")
        if self.condition_type == "week_lte":
            return context.week <= _integer(self.value, "value")

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
        if not isinstance(self.condition, EventCondition):
            raise TypeError("Expected weight modifier condition to be EventCondition.")
        object.__setattr__(
            self,
            "multiplier",
            _finite_number(
                self.multiplier,
                "multiplier",
                minimum=0.0,
                maximum=None,
            ),
        )

    def applies(self, state: AgentState, context: WeeklyContext) -> bool:
        return self.condition.evaluate(state, context)


@dataclass(frozen=True, slots=True)
class EventOption(SerializableState):
    option_id: str
    label: str
    summary: str
    availability_conditions: tuple[EventCondition, ...] = ()
    estimated_cost: Decimal = Decimal("0.00")
    requires_full_estimated_cost: bool = True
    expected_weekly_financial_gain: Decimal = Decimal("0.00")
    ongoing_weekly_time_hours: float = 0.0
    time_cost_hours: float = 0.0
    energy_cost: float = 0.0
    short_term_value: float = 0.0
    future_value: float = 0.0
    perceived_risk: float = 0.0
    uncertainty: float = 0.0
    social_value: float = 0.0
    social_pressure: float = 0.0
    autonomy_value: float = 0.0
    learning_value: float = 0.0
    health_value: float = 0.0
    comfort_value: float = 0.0
    goal_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.option_id, "option_id")
        _require_non_empty(self.label, "label")
        _require_non_empty(self.summary, "summary")
        object.__setattr__(
            self,
            "availability_conditions",
            _typed_tuple(self.availability_conditions, EventCondition, "availability_conditions"),
        )
        _money(self.estimated_cost, "estimated_cost")
        if not isinstance(self.requires_full_estimated_cost, bool):
            raise TypeError("Expected requires_full_estimated_cost to be bool.")
        _money(self.expected_weekly_financial_gain, "expected_weekly_financial_gain")
        object.__setattr__(
            self,
            "ongoing_weekly_time_hours",
            _finite_number(
                self.ongoing_weekly_time_hours,
                "ongoing_weekly_time_hours",
                minimum=0.0,
                maximum=168.0,
            ),
        )
        object.__setattr__(
            self,
            "time_cost_hours",
            _finite_number(self.time_cost_hours, "time_cost_hours", minimum=0.0, maximum=168.0),
        )
        object.__setattr__(
            self,
            "energy_cost",
            _finite_number(self.energy_cost, "energy_cost", minimum=0.0, maximum=100.0),
        )
        for field_name in (
            "short_term_value",
            "future_value",
            "social_value",
            "autonomy_value",
            "learning_value",
            "health_value",
            "comfort_value",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name, minimum=-1.0, maximum=1.0),
            )
        for field_name in ("perceived_risk", "uncertainty", "social_pressure"):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(self, "goal_tags", _string_sequence(self.goal_tags, "goal_tags"))


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
    time_pressure: float = 0.0
    options: tuple[EventOption, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.version, "version")
        _require_non_empty(self.category, "category")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.summary, "summary")
        object.__setattr__(
            self,
            "base_weight",
            _finite_number(self.base_weight, "base_weight", minimum=0.0, maximum=None),
        )
        _integer(self.cooldown_weeks, "cooldown_weeks", minimum=0)
        conditions = _typed_tuple(self.conditions, EventCondition, "conditions")
        modifiers = _typed_tuple(self.weight_modifiers, WeightModifier, "weight_modifiers")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "weight_modifiers", modifiers)
        object.__setattr__(self, "tags", _string_sequence(self.tags, "tags"))
        object.__setattr__(
            self,
            "time_pressure",
            _finite_number(self.time_pressure, "time_pressure", minimum=0.0, maximum=1.0),
        )
        options = _typed_tuple(self.options, EventOption, "options")
        _require_unique_option_ids(options)
        object.__setattr__(self, "options", options)

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
            time_pressure=self.time_pressure,
            options=self.options,
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
    time_pressure: float = 0.0
    options: tuple[EventOption, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.version, "version")
        _require_non_empty(self.category, "category")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.summary, "summary")
        _integer(self.week, "week", minimum=0)
        object.__setattr__(
            self,
            "effective_weight",
            _finite_number(
                self.effective_weight,
                "effective_weight",
                minimum=0.0,
                maximum=None,
            ),
        )
        object.__setattr__(self, "tags", _string_sequence(self.tags, "tags"))
        object.__setattr__(
            self,
            "time_pressure",
            _finite_number(self.time_pressure, "time_pressure", minimum=0.0, maximum=1.0),
        )
        options = _typed_tuple(self.options, EventOption, "options")
        _require_unique_option_ids(options)
        object.__setattr__(self, "options", options)


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

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        if not isinstance(self.eligible, bool):
            raise TypeError("Expected candidate trace eligible to be bool.")
        object.__setattr__(
            self,
            "effective_weight",
            _finite_number(
                self.effective_weight,
                "effective_weight",
                minimum=0.0,
                maximum=None,
            ),
        )
        _require_non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class EventSelectionDraw(SerializableState):
    slot: int
    roll: float
    total_weight: float
    selected_event_id: str

    def __post_init__(self) -> None:
        _integer(self.slot, "slot", minimum=0)
        total_weight = _finite_number(
            self.total_weight,
            "total_weight",
            minimum=0.0,
            maximum=None,
        )
        object.__setattr__(self, "total_weight", total_weight)
        object.__setattr__(
            self,
            "roll",
            _finite_number(self.roll, "roll", minimum=0.0, maximum=total_weight),
        )
        _require_non_empty(self.selected_event_id, "selected_event_id")


@dataclass(frozen=True, slots=True)
class EventSelectionTrace(SerializableState):
    week: int
    trigger_probability: float
    trigger_roll: float
    candidates: tuple[EventCandidateTrace, ...]
    selected_event_ids: tuple[str, ...]
    selection_draws: tuple[EventSelectionDraw, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        object.__setattr__(
            self,
            "trigger_probability",
            _finite_number(
                self.trigger_probability,
                "trigger_probability",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "trigger_roll",
            _finite_number(self.trigger_roll, "trigger_roll", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "candidates",
            _typed_tuple(self.candidates, EventCandidateTrace, "candidates"),
        )
        object.__setattr__(
            self,
            "selected_event_ids",
            _string_sequence(self.selected_event_ids, "selected_event_ids"),
        )
        object.__setattr__(
            self,
            "selection_draws",
            _typed_tuple(self.selection_draws, EventSelectionDraw, "selection_draws"),
        )

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
        object.__setattr__(
            self,
            "definitions",
            _typed_tuple(self.definitions, EventDefinition, "definitions"),
        )
        _integer(self.max_events_per_week, "max_events_per_week", minimum=0)
        object.__setattr__(
            self,
            "event_probability",
            _finite_number(
                self.event_probability,
                "event_probability",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        ids = [definition.event_id for definition in self.definitions]
        if len(set(ids)) != len(ids):
            raise ValueError("Expected event_id values to be unique within a catalog.")


def is_on_cooldown(definition: EventDefinition, history: EventHistory, week: int) -> bool:
    last_week = history.last_week(definition.event_id)
    if last_week is None:
        return False
    return week - last_week <= definition.cooldown_weeks


def _resolve_path(state: AgentState, path: str) -> Any:
    _validate_path(path)
    current: Any = state
    traversed: list[str] = []
    for part in path.split("."):
        if not is_dataclass(current):
            prefix = ".".join(traversed) or "<root>"
            raise ValueError(
                f"Invalid event condition path '{path}': '{prefix}' is not a dataclass state object."
            )
        declared_fields = {field.name for field in fields(current)}
        if part not in declared_fields:
            raise ValueError(f"Invalid event condition path '{path}'.")
        current = getattr(current, part)
        traversed.append(part)
    return current


def _comparable_number(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"Expected '{name}' to resolve to a numeric value.")
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, int | float | str):
        try:
            decimal_value = Decimal(str(value))
        except InvalidOperation as exc:
            raise TypeError(f"Expected '{name}' to resolve to a numeric value.") from exc
    else:
        raise TypeError(f"Expected '{name}' to resolve to a numeric value.")
    if not decimal_value.is_finite():
        raise ValueError(f"Expected '{name}' to resolve to a finite numeric value.")
    return decimal_value


def _money(value: Any, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected monetary value '{name}' to be Decimal.")
    if not value.is_finite():
        raise ValueError(f"Expected monetary value '{name}' to be finite.")
    if value < Decimal(0):
        raise ValueError(f"Expected monetary value '{name}' to be non-negative.")
    return value


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected '{name}' to be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Expected '{name}' to be >= {minimum}.")
    return value


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None,
    maximum: float | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise TypeError(f"Expected '{name}' to be numeric.")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Expected '{name}' to be finite.")
        numeric = float(value)
    else:
        numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Expected '{name}' to be finite.")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"Expected '{name}' to be >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"Expected '{name}' to be <= {maximum}.")
    return numeric


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


def _require_unique_option_ids(options: tuple[EventOption, ...]) -> None:
    option_ids = [option.option_id for option in options]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("Expected option_id values to be unique within an event.")


def _validate_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("Expected condition path for agent-state condition.")
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid event condition path '{path}'.")
    for part in parts:
        if part.startswith("_") or "__" in part:
            raise ValueError(f"Unsafe event condition path '{path}'.")


def _collection(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"Expected '{path}' to resolve to a collection.")
    return tuple(value)


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
