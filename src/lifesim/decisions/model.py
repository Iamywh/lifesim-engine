from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lifesim.agents.state import SerializableState


@dataclass(frozen=True, slots=True)
class DecisionScoreComponent(SerializableState):
    name: str
    signal: float
    weight: float
    contribution: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        object.__setattr__(self, "signal", _finite_number(self.signal, "signal"))
        object.__setattr__(self, "weight", _finite_number(self.weight, "weight"))
        object.__setattr__(
            self,
            "contribution",
            _finite_number(self.contribution, "contribution"),
        )


@dataclass(frozen=True, slots=True)
class OptionEvaluation(SerializableState):
    option_id: str
    available: bool
    unavailable_reason: str
    deterministic_score: float | None
    controlled_noise: float | None
    final_score: float | None
    components: tuple[DecisionScoreComponent, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.option_id, "option_id")
        if not isinstance(self.available, bool):
            raise TypeError("Expected option evaluation available to be bool.")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("Expected unavailable_reason to be a string.")
        if self.available and self.unavailable_reason:
            raise ValueError("Available options must not include an unavailable reason.")
        if not self.available and not self.unavailable_reason:
            raise ValueError("Unavailable options must include a reason.")
        for field_name in ("deterministic_score", "controlled_noise", "final_score"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_number(value, field_name))
        object.__setattr__(self, "components", tuple(self.components))
        for component in self.components:
            if not isinstance(component, DecisionScoreComponent):
                raise TypeError("Expected components to contain DecisionScoreComponent values.")


@dataclass(frozen=True, slots=True)
class DecisionRecord(SerializableState):
    decision_id: str
    agent_id: str
    week: int
    source_event_id: str
    source_event_version: str
    time_pressure: float
    available_option_ids: tuple[str, ...]
    unavailable_option_ids: tuple[str, ...]
    chosen_option_id: str | None
    evaluations: tuple[OptionEvaluation, ...]
    strongest_positive_factors: tuple[str, ...]
    strongest_negative_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.decision_id, "decision_id")
        _require_non_empty(self.agent_id, "agent_id")
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.source_event_id, "source_event_id")
        _require_non_empty(self.source_event_version, "source_event_version")
        object.__setattr__(
            self,
            "time_pressure",
            _finite_number(self.time_pressure, "time_pressure", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "available_option_ids",
            _string_sequence(self.available_option_ids, "available_option_ids"),
        )
        object.__setattr__(
            self,
            "unavailable_option_ids",
            _string_sequence(self.unavailable_option_ids, "unavailable_option_ids"),
        )
        if self.chosen_option_id is not None:
            _require_non_empty(self.chosen_option_id, "chosen_option_id")
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        for evaluation in self.evaluations:
            if not isinstance(evaluation, OptionEvaluation):
                raise TypeError("Expected evaluations to contain OptionEvaluation values.")
        object.__setattr__(
            self,
            "strongest_positive_factors",
            _string_sequence(self.strongest_positive_factors, "strongest_positive_factors"),
        )
        object.__setattr__(
            self,
            "strongest_negative_factors",
            _string_sequence(self.strongest_negative_factors, "strongest_negative_factors"),
        )


DecisionTrace = DecisionRecord


@dataclass(frozen=True, slots=True)
class DecisionHistory:
    records: tuple[DecisionRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        for record in self.records:
            if not isinstance(record, DecisionRecord):
                raise TypeError("Expected decision history records to contain DecisionRecord values.")

    def record(self, records: tuple[DecisionRecord, ...]) -> DecisionHistory:
        return DecisionHistory(self.records + tuple(records))

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class DecisionSelectionResult:
    records: tuple[DecisionRecord, ...]
    history: DecisionHistory

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))


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


def _string_sequence(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, list | tuple):
        raise TypeError(f"Expected '{name}' to be a list or tuple of strings.")
    strings = tuple(values)
    for item in strings:
        _require_non_empty(item, name)
    return strings


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
