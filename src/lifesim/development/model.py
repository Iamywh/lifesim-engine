from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lifesim.agents.state import SerializableState


@dataclass(frozen=True, slots=True)
class SkillDefinition(SerializableState):
    skill_name: str
    category: str
    learning_rate: float
    practice_xp_per_hour: float
    work_xp_per_hour: float

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        _require_non_empty(self.category, "category")
        for name in ("learning_rate", "practice_xp_per_hour", "work_xp_per_hour"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.000001, maximum=20.0),
            )


@dataclass(frozen=True, slots=True)
class CurriculumSkill(SerializableState):
    skill_name: str
    weight: float

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        object.__setattr__(
            self,
            "weight",
            _finite_number(self.weight, "weight", minimum=0.000001, maximum=100.0),
        )


@dataclass(frozen=True, slots=True)
class EducationProgramDefinition(SerializableState):
    program: str
    progress_per_full_study_week: float
    curriculum: tuple[CurriculumSkill, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.program, "program")
        object.__setattr__(
            self,
            "progress_per_full_study_week",
            _finite_number(
                self.progress_per_full_study_week,
                "progress_per_full_study_week",
                minimum=0.000001,
                maximum=10.0,
            ),
        )
        curriculum = _typed_tuple(self.curriculum, CurriculumSkill, "curriculum")
        if not curriculum:
            raise ValueError("Expected education program curriculum to be non-empty.")
        _require_unique(tuple(item.skill_name for item in curriculum), "curriculum skill")
        object.__setattr__(self, "curriculum", curriculum)


@dataclass(frozen=True, slots=True)
class PracticeAllocation(SerializableState):
    skill_name: str
    hours: float

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        object.__setattr__(
            self,
            "hours",
            _finite_number(self.hours, "hours", minimum=0.0, maximum=80.0),
        )


@dataclass(frozen=True, slots=True)
class DevelopmentProfile(SerializableState):
    profile_id: str
    label: str
    summary: str
    education_hours: float
    practice: tuple[PracticeAllocation, ...] = ()
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
        for name in ("profile_id", "label", "summary"):
            _require_non_empty(getattr(self, name), name)
        object.__setattr__(
            self,
            "education_hours",
            _finite_number(self.education_hours, "education_hours", minimum=0.0, maximum=80.0),
        )
        practice = _typed_tuple(self.practice, PracticeAllocation, "practice")
        _require_unique(tuple(item.skill_name for item in practice), "practice skill")
        object.__setattr__(self, "practice", practice)
        total_hours = self.education_hours + sum(item.hours for item in practice)
        if total_hours > 100.0:
            raise ValueError("Expected weekly development hours to be <= 100.")
        object.__setattr__(
            self,
            "energy_cost",
            _finite_number(self.energy_cost, "energy_cost", minimum=0.0, maximum=100.0),
        )
        for name in (
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
                name,
                _finite_number(getattr(self, name), name, minimum=-1.0, maximum=1.0),
            )
        for name in ("perceived_risk", "uncertainty", "social_pressure"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(self, "goal_tags", _string_sequence(self.goal_tags, "goal_tags"))

    @property
    def total_hours(self) -> float:
        return self.education_hours + sum(item.hours for item in self.practice)


@dataclass(frozen=True, slots=True)
class DevelopmentCatalog(SerializableState):
    skills: tuple[SkillDefinition, ...]
    programs: tuple[EducationProgramDefinition, ...]
    profiles: tuple[DevelopmentProfile, ...]

    def __post_init__(self) -> None:
        skills = _typed_tuple(self.skills, SkillDefinition, "skills")
        programs = _typed_tuple(self.programs, EducationProgramDefinition, "programs")
        profiles = _typed_tuple(self.profiles, DevelopmentProfile, "profiles")
        if not profiles:
            raise ValueError("Expected at least one development profile.")
        skill_names = tuple(skill.skill_name for skill in skills)
        _require_unique(skill_names, "skill_name")
        _require_unique(tuple(program.program for program in programs), "program")
        _require_unique(tuple(profile.profile_id for profile in profiles), "profile_id")
        known_skills = set(skill_names)
        for program in programs:
            for mapping in program.curriculum:
                if mapping.skill_name not in known_skills:
                    raise ValueError(f"Unknown curriculum skill '{mapping.skill_name}'.")
        for profile in profiles:
            for allocation in profile.practice:
                if allocation.skill_name not in known_skills:
                    raise ValueError(f"Unknown practice skill '{allocation.skill_name}'.")
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "programs", programs)
        object.__setattr__(self, "profiles", profiles)

    def skill(self, skill_name: str) -> SkillDefinition:
        for skill in self.skills:
            if skill.skill_name == skill_name:
                return skill
        raise KeyError(f"Unknown skill '{skill_name}'.")

    def program(self, program: str) -> EducationProgramDefinition:
        for definition in self.programs:
            if definition.program == program:
                return definition
        raise KeyError(f"Unknown education program '{program}'.")

    def profile(self, profile_id: str) -> DevelopmentProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"Unknown development profile '{profile_id}'.")


@dataclass(frozen=True, slots=True)
class DevelopmentPlanRecord(SerializableState):
    week: int
    event_id: str
    event_version: str
    available_profile_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.event_version, "event_version")
        object.__setattr__(
            self,
            "available_profile_ids",
            _string_sequence(self.available_profile_ids, "available_profile_ids"),
        )


@dataclass(frozen=True, slots=True)
class DevelopmentEfficiencyAudit(SerializableState):
    energy_factor: float
    stress_factor: float
    mental_load_factor: float
    recovery_factor: float
    workload_factor: float
    combined_workload_hours: float
    study_ratio: float
    effective_study_hours: float
    effective_practice_hours: float
    final_efficiency: float

    def __post_init__(self) -> None:
        for name in (
            "energy_factor",
            "stress_factor",
            "mental_load_factor",
            "recovery_factor",
            "workload_factor",
            "final_efficiency",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        for name in (
            "combined_workload_hours",
            "study_ratio",
            "effective_study_hours",
            "effective_practice_hours",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=200.0),
            )


@dataclass(frozen=True, slots=True)
class SkillExperienceSource(SerializableState):
    skill_name: str
    source_type: str
    source_id: str
    hours: float
    raw_experience: float
    effective_experience: float

    def __post_init__(self) -> None:
        for name in ("skill_name", "source_type", "source_id"):
            _require_non_empty(getattr(self, name), name)
        for name in ("hours", "raw_experience", "effective_experience"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=10000.0),
            )


@dataclass(frozen=True, slots=True)
class SkillDevelopmentRecord(SerializableState):
    skill_name: str
    category: str
    level_before: float
    level_after: float
    level_delta: float
    experience_before: float
    experience_after: float
    raw_experience_gain: float
    effective_experience_gain: float
    sources: tuple[SkillExperienceSource, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        _require_non_empty(self.category, "category")
        for name in ("level_before", "level_after"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=100.0),
            )
        for name in (
            "level_delta",
            "experience_before",
            "experience_after",
            "raw_experience_gain",
            "effective_experience_gain",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=10000.0),
            )
        if not math.isclose(self.level_after - self.level_before, self.level_delta, abs_tol=1e-12):
            raise ValueError("Expected skill level_delta to equal level_after - level_before.")
        if not math.isclose(
            self.experience_after - self.experience_before,
            self.raw_experience_gain,
            abs_tol=1e-12,
        ):
            raise ValueError("Expected skill experience gain to equal experience_after - experience_before.")
        object.__setattr__(
            self,
            "sources",
            _typed_tuple(self.sources, SkillExperienceSource, "sources"),
        )
        if not self.sources:
            raise ValueError("Expected skill development record to include source audit.")


@dataclass(frozen=True, slots=True)
class EducationProgressRecord(SerializableState):
    week: int
    program: str
    status_before: str
    status_after: str
    year_before: int
    year_after: int
    progress_before: float
    progress_after: float
    progress_delta: float
    planned_study_hours: float
    effective_study_hours: float
    completed: bool

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        for name in ("program", "status_before", "status_after"):
            _require_non_empty(getattr(self, name), name)
        _integer(self.year_before, "year_before", minimum=0)
        _integer(self.year_after, "year_after", minimum=0)
        for name in ("progress_before", "progress_after"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=100.0),
            )
        for name in ("progress_delta", "planned_study_hours", "effective_study_hours"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=200.0),
            )
        if self.progress_after < self.progress_before:
            raise ValueError("Expected education progress to be non-decreasing.")
        if not math.isclose(self.progress_after - self.progress_before, self.progress_delta, abs_tol=1e-12):
            raise ValueError("Expected education progress_delta to equal progress_after - progress_before.")
        if not isinstance(self.completed, bool):
            raise TypeError("Expected completed to be bool.")


@dataclass(frozen=True, slots=True)
class DevelopmentEffectApplication(SerializableState):
    path: str
    before: float
    after: float
    delta: float
    clamped: bool
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty(self.path, "path")
        for name in ("before", "after", "delta"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=-1000.0, maximum=1000.0),
            )
        if not math.isclose(self.after - self.before, self.delta, abs_tol=1e-12):
            raise ValueError("Expected development effect delta to equal after - before.")
        if not isinstance(self.clamped, bool):
            raise TypeError("Expected clamped to be bool.")
        _require_non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class DevelopmentWeekRecord(SerializableState):
    week: int
    profile_id: str
    decision_id: str
    education_hours: float
    practice: tuple[PracticeAllocation, ...]
    efficiency: DevelopmentEfficiencyAudit
    education_progress: EducationProgressRecord | None
    skill_developments: tuple[SkillDevelopmentRecord, ...]
    effects: tuple[DevelopmentEffectApplication, ...]
    consumed_work_record_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.profile_id, "profile_id")
        _require_non_empty(self.decision_id, "decision_id")
        object.__setattr__(
            self,
            "education_hours",
            _finite_number(self.education_hours, "education_hours", minimum=0.0, maximum=80.0),
        )
        object.__setattr__(self, "practice", _typed_tuple(self.practice, PracticeAllocation, "practice"))
        if not isinstance(self.efficiency, DevelopmentEfficiencyAudit):
            raise TypeError("Expected efficiency to be DevelopmentEfficiencyAudit.")
        if self.education_progress is not None and not isinstance(
            self.education_progress,
            EducationProgressRecord,
        ):
            raise TypeError("Expected education_progress to be EducationProgressRecord.")
        object.__setattr__(
            self,
            "skill_developments",
            _typed_tuple(self.skill_developments, SkillDevelopmentRecord, "skill_developments"),
        )
        _require_unique(tuple(record.skill_name for record in self.skill_developments), "skill development")
        object.__setattr__(
            self,
            "effects",
            _typed_tuple(self.effects, DevelopmentEffectApplication, "effects"),
        )
        object.__setattr__(
            self,
            "consumed_work_record_keys",
            _string_sequence(self.consumed_work_record_keys, "consumed_work_record_keys"),
        )


@dataclass(frozen=True, slots=True)
class DevelopmentHistory:
    plan_records: tuple[DevelopmentPlanRecord, ...] = ()
    week_records: tuple[DevelopmentWeekRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_records",
            _typed_tuple(self.plan_records, DevelopmentPlanRecord, "plan_records"),
        )
        object.__setattr__(
            self,
            "week_records",
            _typed_tuple(self.week_records, DevelopmentWeekRecord, "week_records"),
        )
        _require_unique(tuple(str(record.week) for record in self.plan_records), "development plan week")
        _require_unique(tuple(str(record.week) for record in self.week_records), "development execution week")

    def record_plan(self, record: DevelopmentPlanRecord) -> DevelopmentHistory:
        return DevelopmentHistory(self.plan_records + (record,), self.week_records)

    def record_week(self, record: DevelopmentWeekRecord) -> DevelopmentHistory:
        return DevelopmentHistory(self.plan_records, self.week_records + (record,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_records": [record.to_dict() for record in self.plan_records],
            "week_records": [record.to_dict() for record in self.week_records],
        }


@dataclass(frozen=True, slots=True)
class DevelopmentRuntimeState:
    history: DevelopmentHistory = field(default_factory=DevelopmentHistory)
    planned_event_id: str = ""
    planned_event_version: str = ""
    planned_profile_ids: tuple[str, ...] = ()
    planned_week: int | None = None
    processed_planning_weeks: tuple[int, ...] = ()
    processed_execution_weeks: tuple[int, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    processed_work_record_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, DevelopmentHistory):
            raise TypeError("Expected development history to be DevelopmentHistory.")
        if not isinstance(self.planned_event_id, str):
            raise TypeError("Expected planned_event_id to be a string.")
        if not isinstance(self.planned_event_version, str):
            raise TypeError("Expected planned_event_version to be a string.")
        planned_profile_ids = _string_sequence(self.planned_profile_ids, "planned_profile_ids")
        _require_unique(planned_profile_ids, "planned_profile_ids")
        object.__setattr__(self, "planned_profile_ids", planned_profile_ids)
        if self.planned_week is not None:
            _integer(self.planned_week, "planned_week", minimum=1)
        for name in ("processed_planning_weeks", "processed_execution_weeks"):
            weeks = _integer_sequence(getattr(self, name), name)
            _require_unique(tuple(str(week) for week in weeks), name)
            object.__setattr__(self, name, weeks)
        decision_ids = _string_sequence(self.processed_decision_ids, "processed_decision_ids")
        _require_unique(decision_ids, "processed_decision_ids")
        object.__setattr__(self, "processed_decision_ids", decision_ids)
        work_keys = _string_sequence(self.processed_work_record_keys, "processed_work_record_keys")
        _require_unique(work_keys, "processed_work_record_keys")
        object.__setattr__(self, "processed_work_record_keys", work_keys)
        has_planned_event = bool(self.planned_event_id or self.planned_event_version or planned_profile_ids)
        if has_planned_event:
            _require_non_empty(self.planned_event_id, "planned_event_id")
            _require_non_empty(self.planned_event_version, "planned_event_version")
            if self.planned_week is None:
                raise ValueError("Expected planned development week when planned event is set.")
        if self.planned_week is not None and not has_planned_event:
            raise ValueError("Expected planned development event when planned week is set.")
        plan_weeks = tuple(record.week for record in self.history.plan_records)
        execution_weeks = tuple(record.week for record in self.history.week_records)
        if set(plan_weeks) != set(self.processed_planning_weeks):
            raise ValueError("Expected processed development planning weeks to match plan history.")
        if set(execution_weeks) != set(self.processed_execution_weeks):
            raise ValueError("Expected processed development execution weeks to match execution history.")
        history_decisions = tuple(record.decision_id for record in self.history.week_records)
        if history_decisions != decision_ids:
            raise ValueError("Expected processed decision ids to exactly match development history decisions.")
        consumed = tuple(
            key
            for record in self.history.week_records
            for key in record.consumed_work_record_keys
        )
        if consumed != work_keys:
            raise ValueError("Expected processed work record keys to exactly match consumed history keys.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.to_dict(),
            "planned_event_id": self.planned_event_id,
            "planned_event_version": self.planned_event_version,
            "planned_profile_ids": list(self.planned_profile_ids),
            "planned_week": self.planned_week,
            "processed_planning_weeks": list(self.processed_planning_weeks),
            "processed_execution_weeks": list(self.processed_execution_weeks),
            "processed_decision_ids": list(self.processed_decision_ids),
            "processed_work_record_keys": list(self.processed_work_record_keys),
        }


def _finite_number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Expected '{name}' to be finite.")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"Expected '{name}' to be between {minimum} and {maximum}.")
    return numeric


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected '{name}' to be an integer.")
    if value < minimum:
        raise ValueError(f"Expected '{name}' to be >= {minimum}.")
    return value


def _integer_sequence(values: Any, name: str) -> tuple[int, ...]:
    if isinstance(values, str) or not isinstance(values, list | tuple):
        raise TypeError(f"Expected '{name}' to be a list or tuple of integers.")
    return tuple(_integer(value, name, minimum=1) for value in values)


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


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Expected '{name}' values to be unique.")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
