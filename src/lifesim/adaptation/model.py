from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lifesim.agents.state import SerializableState

PERSONALITY_TRAITS = (
    "risk_tolerance",
    "impulsivity",
    "discipline",
    "frugality",
    "social_need",
    "independence",
    "resilience",
    "curiosity",
    "confidence",
    "patience",
    "conscientiousness",
    "adaptability",
)


@dataclass(frozen=True, slots=True)
class AdaptationSettings(SerializableState):
    max_weekly_trait_delta: float = 0.0015
    trait_adaptation_rate: float = 0.018
    max_trait_anchor_shift: float = 0.12
    evidence_decay_rate: float = 0.92
    routine_stability_gain: float = 1.2
    routine_stability_switch_loss: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "max_weekly_trait_delta",
            "trait_adaptation_rate",
            "max_trait_anchor_shift",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, 0.25))
        object.__setattr__(self, "evidence_decay_rate", _finite(self.evidence_decay_rate, "evidence_decay_rate", 0.0, 1.0))
        for name in ("routine_stability_gain", "routine_stability_switch_loss"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, 10.0))


@dataclass(frozen=True, slots=True)
class HabitDefinition(SerializableState):
    habit_id: str
    name: str
    cadence: str
    behavior_tags: tuple[str, ...]
    formation_rate: float
    reinforcement_rate: float
    nonuse_decay_rate: float
    formation_threshold: float
    minimum_reinforcing_weeks: int
    grace_weeks: int = 2

    def __post_init__(self) -> None:
        for name in ("habit_id", "name", "cadence"):
            _non_empty(getattr(self, name), name)
        object.__setattr__(self, "behavior_tags", _strings(self.behavior_tags, "behavior_tags"))
        if not self.behavior_tags:
            raise ValueError("Expected habit definition behavior_tags to be non-empty.")
        for name in ("formation_rate", "reinforcement_rate", "nonuse_decay_rate"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, 25.0))
        object.__setattr__(self, "formation_threshold", _finite(self.formation_threshold, "formation_threshold", 0.0, 100.0))
        object.__setattr__(self, "minimum_reinforcing_weeks", _integer(self.minimum_reinforcing_weeks, "minimum_reinforcing_weeks", 2, 520))
        object.__setattr__(self, "grace_weeks", _integer(self.grace_weeks, "grace_weeks", 0, 520))


@dataclass(frozen=True, slots=True)
class PersonalityAnchor(SerializableState):
    risk_tolerance: float
    impulsivity: float
    discipline: float
    frugality: float
    social_need: float
    independence: float
    resilience: float
    curiosity: float
    confidence: float
    patience: float
    conscientiousness: float
    adaptability: float

    def __post_init__(self) -> None:
        for trait in PERSONALITY_TRAITS:
            object.__setattr__(self, trait, _finite(getattr(self, trait), trait, 0.0, 1.0))

    def value(self, trait: str) -> float:
        if trait not in PERSONALITY_TRAITS:
            raise ValueError(f"Unknown personality trait '{trait}'.")
        return getattr(self, trait)


@dataclass(frozen=True, slots=True)
class TraitEvidenceMapping(SerializableState):
    evidence_type: str
    evidence_key: str
    trait: str
    coefficient: float

    def __post_init__(self) -> None:
        if self.evidence_type not in {"behavior_tag", "choice_metric", "experienced_outcome"}:
            raise ValueError("Unexpected adaptation evidence_type.")
        _non_empty(self.evidence_key, "evidence_key")
        if self.trait not in PERSONALITY_TRAITS:
            raise ValueError(f"Unknown personality trait '{self.trait}'.")
        object.__setattr__(self, "coefficient", _finite(self.coefficient, "coefficient", -1.0, 1.0))
        if self.coefficient == 0.0:
            raise ValueError("Expected non-zero trait evidence coefficient.")


@dataclass(frozen=True, slots=True)
class AdaptationCatalog(SerializableState):
    settings: AdaptationSettings
    habits: tuple[HabitDefinition, ...]
    trait_mappings: tuple[TraitEvidenceMapping, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.settings, AdaptationSettings):
            raise TypeError("Expected settings to be AdaptationSettings.")
        object.__setattr__(self, "habits", _typed(self.habits, HabitDefinition, "habits"))
        ids = tuple(habit.habit_id for habit in self.habits)
        _unique(ids, "habit_id")
        object.__setattr__(self, "trait_mappings", _typed(self.trait_mappings, TraitEvidenceMapping, "trait_mappings"))

    def habit(self, habit_id: str) -> HabitDefinition:
        for definition in self.habits:
            if definition.habit_id == habit_id:
                return definition
        raise KeyError(f"Unknown adaptation habit '{habit_id}'.")


@dataclass(frozen=True, slots=True)
class BehaviorEvidenceRecord(SerializableState):
    week: int
    decision_id: str
    source_event_id: str
    source_event_version: str
    source_option_id: str
    source_system: str
    behavior_tags: tuple[str, ...]
    executed: bool
    source_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "week", _integer(self.week, "week", 1, 1000000))
        for name in ("decision_id", "source_event_id", "source_event_version", "source_option_id", "source_system"):
            _non_empty(getattr(self, name), name)
        object.__setattr__(self, "behavior_tags", _strings(self.behavior_tags, "behavior_tags"))
        if not self.behavior_tags:
            raise ValueError("Expected behavior evidence tags to be non-empty.")
        if not isinstance(self.executed, bool):
            raise TypeError("Expected executed to be bool.")
        if not self.executed:
            raise ValueError("M11 behavior evidence must come from executed voluntary behavior.")
        object.__setattr__(self, "source_record_ids", _strings(self.source_record_ids, "source_record_ids"))


@dataclass(frozen=True, slots=True)
class HabitCandidateState(SerializableState):
    habit_id: str
    latent_strength: float = 0.0
    reinforcing_weeks: tuple[int, ...] = ()
    last_evidence_week: int = 0

    def __post_init__(self) -> None:
        _non_empty(self.habit_id, "habit_id")
        object.__setattr__(self, "latent_strength", _finite(self.latent_strength, "latent_strength", 0.0, 100.0))
        object.__setattr__(self, "reinforcing_weeks", tuple(_integer(week, "reinforcing_weeks", 1, 1000000) for week in self.reinforcing_weeks))
        _unique(tuple(str(week) for week in self.reinforcing_weeks), "reinforcing_weeks")
        object.__setattr__(self, "last_evidence_week", _integer(self.last_evidence_week, "last_evidence_week", 0, 1000000))


@dataclass(frozen=True, slots=True)
class HabitCandidateChange(SerializableState):
    habit_id: str
    before: float
    after: float
    reinforcing_weeks: int
    reason: str

    def __post_init__(self) -> None:
        _non_empty(self.habit_id, "habit_id")
        object.__setattr__(self, "before", _finite(self.before, "before", 0.0, 100.0))
        object.__setattr__(self, "after", _finite(self.after, "after", 0.0, 100.0))
        object.__setattr__(self, "reinforcing_weeks", _integer(self.reinforcing_weeks, "reinforcing_weeks", 0, 1000000))
        _non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class HabitStrengthChange(SerializableState):
    habit_id: str
    before: float
    after: float
    reason: str
    evidence_decision_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.habit_id, "habit_id")
        object.__setattr__(self, "before", _finite(self.before, "before", 0.0, 100.0))
        object.__setattr__(self, "after", _finite(self.after, "after", 0.0, 100.0))
        _non_empty(self.reason, "reason")
        object.__setattr__(self, "evidence_decision_ids", _strings(self.evidence_decision_ids, "evidence_decision_ids"))


@dataclass(frozen=True, slots=True)
class RoutineStabilityRecord(SerializableState):
    before: float
    after: float
    profile_id: str
    previous_profile_id: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", _finite(self.before, "before", 0.0, 100.0))
        object.__setattr__(self, "after", _finite(self.after, "after", 0.0, 100.0))
        _non_empty(self.profile_id, "profile_id")
        if not isinstance(self.previous_profile_id, str):
            raise TypeError("Expected previous_profile_id to be a string.")
        _non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class TraitEvidenceRecord(SerializableState):
    week: int
    source_id: str
    source_type: str
    source_family: str
    evidence_type: str
    evidence_key: str
    trait: str
    signal: float
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "week", _integer(self.week, "week", 1, 1000000))
        for name in ("source_id", "source_type", "source_family", "evidence_type", "evidence_key"):
            _non_empty(getattr(self, name), name)
        if self.trait not in PERSONALITY_TRAITS:
            raise ValueError(f"Unknown personality trait '{self.trait}'.")
        object.__setattr__(self, "signal", _finite(self.signal, "signal", -1.0, 1.0))
        object.__setattr__(self, "weight", _finite(self.weight, "weight", 0.0, 100.0))


@dataclass(frozen=True, slots=True)
class TraitEvidenceAccumulator(SerializableState):
    trait: str
    signed_evidence: float = 0.0
    evidence_weight: float = 0.0
    distinct_weeks: tuple[int, ...] = ()
    source_families: tuple[str, ...] = ()
    last_evidence_week: int = 0
    last_updated_week: int = 0

    def __post_init__(self) -> None:
        if self.trait not in PERSONALITY_TRAITS:
            raise ValueError(f"Unknown personality trait '{self.trait}'.")
        object.__setattr__(self, "signed_evidence", _finite(self.signed_evidence, "signed_evidence", -100000.0, 100000.0))
        object.__setattr__(self, "evidence_weight", _finite(self.evidence_weight, "evidence_weight", 0.0, 100000.0))
        object.__setattr__(self, "distinct_weeks", tuple(_integer(week, "distinct_weeks", 1, 1000000) for week in self.distinct_weeks))
        _unique(tuple(str(week) for week in self.distinct_weeks), "distinct_weeks")
        object.__setattr__(self, "source_families", _strings(self.source_families, "source_families"))
        object.__setattr__(self, "last_evidence_week", _integer(self.last_evidence_week, "last_evidence_week", 0, 1000000))
        object.__setattr__(self, "last_updated_week", _integer(self.last_updated_week, "last_updated_week", 0, 1000000))
        if self.last_evidence_week and self.last_updated_week and self.last_updated_week < self.last_evidence_week:
            raise ValueError("Expected last_updated_week to be >= last_evidence_week.")


@dataclass(frozen=True, slots=True)
class PersonalityTraitChange(SerializableState):
    trait: str
    anchor: float
    before: float
    target: float
    confidence: float
    delta: float
    after: float
    capped: bool
    reason: str

    def __post_init__(self) -> None:
        if self.trait not in PERSONALITY_TRAITS:
            raise ValueError(f"Unknown personality trait '{self.trait}'.")
        for name in ("anchor", "before", "target", "confidence", "after"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, 1.0))
        object.__setattr__(self, "delta", _finite(self.delta, "delta", -1.0, 1.0))
        if not isinstance(self.capped, bool):
            raise TypeError("Expected capped to be bool.")
        _non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class AdaptationWeekRecord(SerializableState):
    week: int
    behavior_evidence: tuple[BehaviorEvidenceRecord, ...] = ()
    processed_experience_ids: tuple[str, ...] = ()
    habit_candidate_changes: tuple[HabitCandidateChange, ...] = ()
    habit_strength_changes: tuple[HabitStrengthChange, ...] = ()
    routine_stability: RoutineStabilityRecord | None = None
    trait_evidence: tuple[TraitEvidenceRecord, ...] = ()
    personality_changes: tuple[PersonalityTraitChange, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "week", _integer(self.week, "week", 1, 1000000))
        object.__setattr__(self, "behavior_evidence", _typed(self.behavior_evidence, BehaviorEvidenceRecord, "behavior_evidence"))
        _unique(tuple(record.decision_id for record in self.behavior_evidence), "behavior evidence decision_id")
        object.__setattr__(self, "processed_experience_ids", _strings(self.processed_experience_ids, "processed_experience_ids"))
        _unique(self.processed_experience_ids, "processed_experience_ids")
        object.__setattr__(self, "habit_candidate_changes", _typed(self.habit_candidate_changes, HabitCandidateChange, "habit_candidate_changes"))
        object.__setattr__(self, "habit_strength_changes", _typed(self.habit_strength_changes, HabitStrengthChange, "habit_strength_changes"))
        if self.routine_stability is not None and not isinstance(self.routine_stability, RoutineStabilityRecord):
            raise TypeError("Expected routine_stability to be RoutineStabilityRecord.")
        object.__setattr__(self, "trait_evidence", _typed(self.trait_evidence, TraitEvidenceRecord, "trait_evidence"))
        object.__setattr__(self, "personality_changes", _typed(self.personality_changes, PersonalityTraitChange, "personality_changes"))


@dataclass(frozen=True, slots=True)
class AdaptationHistory(SerializableState):
    records: tuple[AdaptationWeekRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", _typed(self.records, AdaptationWeekRecord, "records"))
        _unique(tuple(str(record.week) for record in self.records), "adaptation week")

    def record(self, record: AdaptationWeekRecord) -> AdaptationHistory:
        return AdaptationHistory(self.records + (record,))


@dataclass(frozen=True, slots=True)
class AdaptationRuntimeState(SerializableState):
    history: AdaptationHistory = field(default_factory=AdaptationHistory)
    personality_anchor: PersonalityAnchor | dict[str, float] | None = None
    habit_candidates: tuple[HabitCandidateState, ...] = ()
    trait_accumulators: tuple[TraitEvidenceAccumulator, ...] = ()
    processed_weeks: tuple[int, ...] = ()
    processed_behavior_decision_ids: tuple[str, ...] = ()
    processed_experience_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, AdaptationHistory):
            raise TypeError("Expected adaptation history to be AdaptationHistory.")
        if self.personality_anchor is not None:
            if isinstance(self.personality_anchor, PersonalityAnchor):
                anchor = self.personality_anchor
            elif isinstance(self.personality_anchor, dict):
                values = {}
                for trait in PERSONALITY_TRAITS:
                    if trait not in self.personality_anchor:
                        raise ValueError(f"Missing personality anchor trait '{trait}'.")
                    values[trait] = self.personality_anchor[trait]
                anchor = PersonalityAnchor(**values)
            else:
                raise TypeError("Expected personality_anchor to be PersonalityAnchor.")
            object.__setattr__(self, "personality_anchor", anchor)
        object.__setattr__(self, "habit_candidates", _typed(self.habit_candidates, HabitCandidateState, "habit_candidates"))
        _unique(tuple(candidate.habit_id for candidate in self.habit_candidates), "habit candidate id")
        object.__setattr__(self, "trait_accumulators", _typed(self.trait_accumulators, TraitEvidenceAccumulator, "trait_accumulators"))
        _unique(tuple(accumulator.trait for accumulator in self.trait_accumulators), "trait accumulator")
        weeks = tuple(_integer(week, "processed_weeks", 1, 1000000) for week in self.processed_weeks)
        _unique(tuple(str(week) for week in weeks), "processed_weeks")
        object.__setattr__(self, "processed_weeks", weeks)
        object.__setattr__(self, "processed_behavior_decision_ids", _strings(self.processed_behavior_decision_ids, "processed_behavior_decision_ids"))
        _unique(self.processed_behavior_decision_ids, "processed_behavior_decision_ids")
        object.__setattr__(self, "processed_experience_ids", _strings(self.processed_experience_ids, "processed_experience_ids"))
        _unique(self.processed_experience_ids, "processed_experience_ids")
        if tuple(record.week for record in self.history.records) != self.processed_weeks:
            raise ValueError("Expected processed adaptation weeks to match history.")
        expected_behavior_ids = tuple(
            dict.fromkeys(
                evidence.decision_id
                for record in self.history.records
                for evidence in record.behavior_evidence
            )
        )
        if self.processed_behavior_decision_ids != expected_behavior_ids:
            raise ValueError("Expected processed behavior ids to match adaptation history evidence.")
        expected_experience_ids = tuple(
            experience_id
            for record in self.history.records
            for experience_id in record.processed_experience_ids
        )
        if self.processed_experience_ids != expected_experience_ids:
            raise ValueError("Expected processed experience ids to match adaptation history.")


def _finite(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"Expected '{name}' to be finite and between {minimum} and {maximum}.")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected '{name}' to be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"Expected '{name}' to be between {minimum} and {maximum}.")
    return value


def _non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")


def _strings(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"Expected '{name}' to be a string sequence.")
    result = tuple(values)
    for value in result:
        _non_empty(value, name)
    return result


def _typed(values: Any, item_type: type[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, str):
        raise TypeError(f"Expected '{name}' to be a sequence.")
    result = tuple(values)
    for value in result:
        if not isinstance(value, item_type):
            raise TypeError(f"Expected '{name}' to contain {item_type.__name__}.")
    return result


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Expected {name} values to be unique.")
