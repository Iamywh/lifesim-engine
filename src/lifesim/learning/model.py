from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lifesim.agents.state import SerializableState

MEMORY_DECAY_PER_WEEK = 0.94


@dataclass(frozen=True, slots=True)
class ExperienceEvaluation(SerializableState):
    valence: float
    salience: float
    affected_domains: tuple[str, ...]
    strongest_positive_effects: tuple[str, ...] = ()
    strongest_negative_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "valence",
            _finite_number(self.valence, "valence", minimum=-1.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "salience",
            _finite_number(self.salience, "salience", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "affected_domains",
            _string_sequence(self.affected_domains, "affected_domains"),
        )
        object.__setattr__(
            self,
            "strongest_positive_effects",
            _string_sequence(self.strongest_positive_effects, "strongest_positive_effects"),
        )
        object.__setattr__(
            self,
            "strongest_negative_effects",
            _string_sequence(self.strongest_negative_effects, "strongest_negative_effects"),
        )


@dataclass(frozen=True, slots=True)
class MemoryUpdate(SerializableState):
    update_type: str
    memory_kind: str
    memory_id: str
    before_strength: float | None
    after_strength: float
    before_valence: float | None
    after_valence: float
    before_exposure_count: int
    after_exposure_count: int
    source_consequence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.update_type, "update_type")
        _require_non_empty(self.memory_kind, "memory_kind")
        _require_non_empty(self.memory_id, "memory_id")
        if self.before_strength is not None:
            object.__setattr__(
                self,
                "before_strength",
                _finite_number(
                    self.before_strength,
                    "before_strength",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "after_strength",
            _finite_number(self.after_strength, "after_strength", minimum=0.0, maximum=1.0),
        )
        if self.before_valence is not None:
            object.__setattr__(
                self,
                "before_valence",
                _finite_number(self.before_valence, "before_valence", minimum=-1.0, maximum=1.0),
            )
        object.__setattr__(
            self,
            "after_valence",
            _finite_number(self.after_valence, "after_valence", minimum=-1.0, maximum=1.0),
        )
        _integer(self.before_exposure_count, "before_exposure_count", minimum=0)
        _integer(self.after_exposure_count, "after_exposure_count", minimum=0)
        if self.after_exposure_count < self.before_exposure_count:
            raise ValueError("Expected after_exposure_count to be >= before_exposure_count.")
        object.__setattr__(
            self,
            "source_consequence_ids",
            _string_sequence(self.source_consequence_ids, "source_consequence_ids"),
        )


@dataclass(frozen=True, slots=True)
class LearningRecord(SerializableState):
    consequence_id: str
    source_decision_id: str
    source_event_id: str
    source_event_version: str
    source_option_id: str
    week_learned: int
    evaluation: ExperienceEvaluation
    updates: tuple[MemoryUpdate, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.consequence_id, "consequence_id")
        _require_non_empty(self.source_decision_id, "source_decision_id")
        _require_non_empty(self.source_event_id, "source_event_id")
        _require_non_empty(self.source_event_version, "source_event_version")
        _require_non_empty(self.source_option_id, "source_option_id")
        _integer(self.week_learned, "week_learned", minimum=0)
        if not isinstance(self.evaluation, ExperienceEvaluation):
            raise TypeError("Expected evaluation to be ExperienceEvaluation.")
        object.__setattr__(self, "updates", tuple(self.updates))
        for update in self.updates:
            if not isinstance(update, MemoryUpdate):
                raise TypeError("Expected updates to contain MemoryUpdate values.")


@dataclass(frozen=True, slots=True)
class LearningHistory:
    records: tuple[LearningRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        for record in self.records:
            if not isinstance(record, LearningRecord):
                raise TypeError("Expected learning history records to contain LearningRecord values.")

    def record(self, records: tuple[LearningRecord, ...]) -> LearningHistory:
        return LearningHistory(self.records + tuple(records))

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


@dataclass(frozen=True, slots=True)
class LearningRuntimeState:
    history: LearningHistory = field(default_factory=LearningHistory)
    processed_consequence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, LearningHistory):
            raise TypeError("Expected learning runtime history to be LearningHistory.")
        processed = _string_sequence(self.processed_consequence_ids, "processed_consequence_ids")
        _require_unique(processed, "processed_consequence_ids")
        object.__setattr__(self, "processed_consequence_ids", processed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.to_dict(),
            "processed_consequence_ids": list(self.processed_consequence_ids),
        }


def effective_memory_strength(strength: float, last_reinforced_week: int, week: int) -> float:
    _integer(last_reinforced_week, "last_reinforced_week", minimum=0)
    _integer(week, "week", minimum=0)
    canonical_strength = _finite_number(strength, "strength", minimum=0.0, maximum=1.0)
    elapsed = max(0, week - last_reinforced_week)
    return round(canonical_strength * (MEMORY_DECAY_PER_WEEK**elapsed), 12)


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


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Expected '{name}' values to be unique.")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
