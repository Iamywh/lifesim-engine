from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from lifesim.agents.state import SerializableState

ACTIVE_APPLICATION_STATUSES = {
    "DISCOVERED",
    "SUBMITTED",
    "INTERVIEW_INVITED",
    "INTERVIEW_ATTENDED",
    "OFFER_AVAILABLE",
    "ACCEPTED",
}


@dataclass(frozen=True, slots=True)
class SkillRequirement(SerializableState):
    skill_name: str
    desired_level: float
    weight: float

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        object.__setattr__(
            self,
            "desired_level",
            _finite_number(self.desired_level, "desired_level", minimum=0.0, maximum=100.0),
        )
        object.__setattr__(
            self,
            "weight",
            _finite_number(self.weight, "weight", minimum=0.0, maximum=None),
        )


@dataclass(frozen=True, slots=True)
class JobDefinition(SerializableState):
    job_id: str
    version: str
    role_title: str
    employer: str
    sector: str
    tags: tuple[str, ...]
    contract_type: str
    weekly_hours: float
    hourly_rate: Decimal
    stability: float
    fixed_term_weeks: int
    physical_demand: float
    mental_demand: float
    social_demand: float
    base_discovery_weight: float
    base_interview_probability: float
    base_offer_probability: float
    skill_requirements: tuple[SkillRequirement, ...] = ()

    def __post_init__(self) -> None:
        for name in ("job_id", "version", "role_title", "employer", "sector", "contract_type"):
            _require_non_empty(getattr(self, name), name)
        object.__setattr__(self, "tags", _string_sequence(self.tags, "tags"))
        object.__setattr__(
            self,
            "weekly_hours",
            _finite_number(self.weekly_hours, "weekly_hours", minimum=1.0, maximum=80.0),
        )
        _require_money(self.hourly_rate, "hourly_rate")
        for name in ("stability", "physical_demand", "mental_demand", "social_demand"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        _integer(self.fixed_term_weeks, "fixed_term_weeks", minimum=0)
        object.__setattr__(
            self,
            "base_discovery_weight",
            _finite_number(
                self.base_discovery_weight,
                "base_discovery_weight",
                minimum=0.0,
                maximum=None,
            ),
        )
        for name in ("base_interview_probability", "base_offer_probability"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        requirements = _typed_tuple(
            self.skill_requirements,
            SkillRequirement,
            "skill_requirements",
        )
        _require_unique(tuple(item.skill_name for item in requirements), "skill requirement")
        object.__setattr__(self, "skill_requirements", requirements)

    @property
    def key(self) -> tuple[str, str]:
        return (self.job_id, self.version)


@dataclass(frozen=True, slots=True)
class EmploymentCatalog(SerializableState):
    jobs: tuple[JobDefinition, ...]
    max_discoveries_per_week: int = 2
    relisting_cooldown_weeks: int = 3

    def __post_init__(self) -> None:
        jobs = _typed_tuple(self.jobs, JobDefinition, "jobs")
        _require_unique(tuple(f"{job.job_id}:{job.version}" for job in jobs), "job key")
        object.__setattr__(self, "jobs", jobs)
        _integer(self.max_discoveries_per_week, "max_discoveries_per_week", minimum=0)
        _integer(self.relisting_cooldown_weeks, "relisting_cooldown_weeks", minimum=0)

    def find(self, job_id: str, version: str) -> JobDefinition | None:
        for job in self.jobs:
            if job.job_id == job_id and job.version == version:
                return job
        return None


@dataclass(frozen=True, slots=True)
class CandidateFitTrace(SerializableState):
    job_id: str
    job_version: str
    skill_score: float
    city_familiarity_modifier: float
    education_modifier: float
    personality_modifier: float
    stress_modifier: float
    final_fit: float

    def __post_init__(self) -> None:
        _require_non_empty(self.job_id, "job_id")
        _require_non_empty(self.job_version, "job_version")
        object.__setattr__(self, "skill_score", _finite_unit(self.skill_score, "skill_score"))
        for name in (
            "city_familiarity_modifier",
            "education_modifier",
            "personality_modifier",
            "stress_modifier",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=-1.0, maximum=1.0),
            )
        object.__setattr__(self, "final_fit", _finite_unit(self.final_fit, "final_fit"))


@dataclass(frozen=True, slots=True)
class ProbabilityAudit(SerializableState):
    stage: str
    application_id: str
    job_id: str
    job_version: str
    base_probability: float
    fit_adjustment: float
    contextual_modifier: float
    final_probability: float
    roll: float
    result: str
    candidate_fit: CandidateFitTrace

    def __post_init__(self) -> None:
        for name in ("stage", "application_id", "job_id", "job_version", "result"):
            _require_non_empty(getattr(self, name), name)
        for name in ("base_probability", "final_probability", "roll"):
            object.__setattr__(self, name, _finite_unit(getattr(self, name), name))
        for name in ("fit_adjustment", "contextual_modifier"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=-1.0, maximum=1.0),
            )
        if not isinstance(self.candidate_fit, CandidateFitTrace):
            raise TypeError("Expected candidate_fit to be CandidateFitTrace.")


@dataclass(frozen=True, slots=True)
class MarketCandidateTrace(SerializableState):
    job_id: str
    job_version: str
    eligible: bool
    weight: float
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty(self.job_id, "job_id")
        _require_non_empty(self.job_version, "job_version")
        if not isinstance(self.eligible, bool):
            raise TypeError("Expected eligible to be bool.")
        object.__setattr__(
            self,
            "weight",
            _finite_number(self.weight, "weight", minimum=0.0, maximum=None),
        )
        _require_non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class EmploymentMarketRecord(SerializableState):
    week: int
    candidates: tuple[MarketCandidateTrace, ...]
    discovered_job_keys: tuple[str, ...]
    event_ids_created: tuple[str, ...]

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        object.__setattr__(
            self,
            "candidates",
            _typed_tuple(self.candidates, MarketCandidateTrace, "candidates"),
        )
        object.__setattr__(
            self,
            "discovered_job_keys",
            _string_sequence(self.discovered_job_keys, "discovered_job_keys"),
        )
        object.__setattr__(
            self,
            "event_ids_created",
            _string_sequence(self.event_ids_created, "event_ids_created"),
        )


@dataclass(frozen=True, slots=True)
class JobApplication(SerializableState):
    application_id: str
    job_id: str
    job_version: str
    status: str
    created_week: int
    updated_week: int
    response_due_week: int = 0
    offer_due_week: int = 0
    source_decision_id: str = ""

    def __post_init__(self) -> None:
        for name in ("application_id", "job_id", "job_version", "status"):
            _require_non_empty(getattr(self, name), name)
        _integer(self.created_week, "created_week", minimum=0)
        _integer(self.updated_week, "updated_week", minimum=0)
        _integer(self.response_due_week, "response_due_week", minimum=0)
        _integer(self.offer_due_week, "offer_due_week", minimum=0)
        if not isinstance(self.source_decision_id, str):
            raise TypeError("Expected source_decision_id to be a string.")


@dataclass(frozen=True, slots=True)
class ApplicationStageRecord(SerializableState):
    application_id: str
    week: int
    stage: str
    status_after: str
    job_id: str
    job_version: str
    decision_id: str = ""
    detail: str = ""
    probability_audit: ProbabilityAudit | None = None

    def __post_init__(self) -> None:
        for name in ("application_id", "stage", "status_after", "job_id", "job_version"):
            _require_non_empty(getattr(self, name), name)
        _integer(self.week, "week", minimum=0)
        for name in ("decision_id", "detail"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Expected {name} to be a string.")
        if self.probability_audit is not None and not isinstance(
            self.probability_audit,
            ProbabilityAudit,
        ):
            raise TypeError("Expected probability_audit to be ProbabilityAudit.")


@dataclass(frozen=True, slots=True)
class ScheduledEmploymentStart(SerializableState):
    contract_id: str
    application_id: str
    job_id: str
    job_version: str
    role_title: str
    employer: str
    contract_type: str
    weekly_hours: float
    hourly_rate: Decimal
    stability: float
    physical_demand: float
    mental_demand: float
    social_demand: float
    start_week: int
    end_week_exclusive: int

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "application_id",
            "job_id",
            "job_version",
            "role_title",
            "employer",
            "contract_type",
        ):
            _require_non_empty(getattr(self, name), name)
        object.__setattr__(
            self,
            "weekly_hours",
            _finite_number(self.weekly_hours, "weekly_hours", minimum=1.0, maximum=80.0),
        )
        _require_money(self.hourly_rate, "hourly_rate")
        for name in ("stability", "physical_demand", "mental_demand", "social_demand"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        _integer(self.start_week, "start_week", minimum=1)
        _integer(self.end_week_exclusive, "end_week_exclusive", minimum=0)
        if self.end_week_exclusive and self.end_week_exclusive <= self.start_week:
            raise ValueError("Expected end_week_exclusive to be after start_week.")


@dataclass(frozen=True, slots=True)
class EmploymentBoundaryRecord(SerializableState):
    week: int
    action: str
    contract_id: str
    role_title: str = ""
    employer: str = ""
    income_stream_id: str = ""
    weekly_wage: Decimal = Decimal("0.00")

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.action, "action")
        _require_non_empty(self.contract_id, "contract_id")
        for name in ("role_title", "employer", "income_stream_id"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Expected {name} to be a string.")
        _require_money(self.weekly_wage, "weekly_wage")


@dataclass(frozen=True, slots=True)
class EmploymentEffectApplication(SerializableState):
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
        if not math.isclose(self.after - self.before, self.delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Expected employment effect delta to equal after - before.")
        if not isinstance(self.clamped, bool):
            raise TypeError("Expected clamped to be bool.")
        _require_non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class EmploymentWorkWeekRecord(SerializableState):
    week: int
    contract_id: str
    weekly_hours: float
    weekly_wage: Decimal
    tenure_weeks_after: int
    effects: tuple[EmploymentEffectApplication, ...]

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.contract_id, "contract_id")
        object.__setattr__(
            self,
            "weekly_hours",
            _finite_number(self.weekly_hours, "weekly_hours", minimum=0.0, maximum=168.0),
        )
        _require_money(self.weekly_wage, "weekly_wage")
        _integer(self.tenure_weeks_after, "tenure_weeks_after", minimum=0)
        object.__setattr__(
            self,
            "effects",
            _typed_tuple(self.effects, EmploymentEffectApplication, "effects"),
        )


@dataclass(frozen=True, slots=True)
class EmploymentHistory:
    market_records: tuple[EmploymentMarketRecord, ...] = ()
    application_records: tuple[ApplicationStageRecord, ...] = ()
    boundary_records: tuple[EmploymentBoundaryRecord, ...] = ()
    work_records: tuple[EmploymentWorkWeekRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "market_records",
            _typed_tuple(self.market_records, EmploymentMarketRecord, "market_records"),
        )
        object.__setattr__(
            self,
            "application_records",
            _typed_tuple(
                self.application_records,
                ApplicationStageRecord,
                "application_records",
            ),
        )
        object.__setattr__(
            self,
            "boundary_records",
            _typed_tuple(self.boundary_records, EmploymentBoundaryRecord, "boundary_records"),
        )
        object.__setattr__(
            self,
            "work_records",
            _typed_tuple(self.work_records, EmploymentWorkWeekRecord, "work_records"),
        )
        _require_unique(tuple(str(record.week) for record in self.market_records), "market week")
        _require_unique(tuple(str(record.week) for record in self.work_records), "work week")

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_records": [record.to_dict() for record in self.market_records],
            "application_records": [record.to_dict() for record in self.application_records],
            "boundary_records": [record.to_dict() for record in self.boundary_records],
            "work_records": [record.to_dict() for record in self.work_records],
        }


@dataclass(frozen=True, slots=True)
class EmploymentRuntimeState:
    history: EmploymentHistory = field(default_factory=EmploymentHistory)
    applications: tuple[JobApplication, ...] = ()
    scheduled_starts: tuple[ScheduledEmploymentStart, ...] = ()
    processed_market_weeks: tuple[int, ...] = ()
    processed_decision_ids: tuple[str, ...] = ()
    processed_boundary_weeks: tuple[int, ...] = ()
    processed_work_weeks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, EmploymentHistory):
            raise TypeError("Expected employment history to be EmploymentHistory.")
        object.__setattr__(
            self,
            "applications",
            _typed_tuple(self.applications, JobApplication, "applications"),
        )
        object.__setattr__(
            self,
            "scheduled_starts",
            _typed_tuple(self.scheduled_starts, ScheduledEmploymentStart, "scheduled_starts"),
        )
        _require_unique(
            tuple(application.application_id for application in self.applications),
            "application_id",
        )
        _require_unique(
            tuple(start.contract_id for start in self.scheduled_starts),
            "scheduled contract_id",
        )
        for name in ("processed_market_weeks", "processed_boundary_weeks", "processed_work_weeks"):
            weeks = _integer_sequence(getattr(self, name), name)
            _require_unique(tuple(str(week) for week in weeks), name)
            object.__setattr__(self, name, weeks)
        decision_ids = _string_sequence(self.processed_decision_ids, "processed_decision_ids")
        _require_unique(decision_ids, "processed_decision_ids")
        object.__setattr__(self, "processed_decision_ids", decision_ids)
        active_keys = tuple(
            f"{application.job_id}:{application.job_version}"
            for application in self.applications
            if application.status in ACTIVE_APPLICATION_STATUSES
        )
        _require_unique(active_keys, "active job application")
        if len([start for start in self.scheduled_starts if start.start_week > 0]) > 1:
            raise ValueError("Expected at most one scheduled employment start.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.to_dict(),
            "applications": [application.to_dict() for application in self.applications],
            "scheduled_starts": [start.to_dict() for start in self.scheduled_starts],
            "processed_market_weeks": list(self.processed_market_weeks),
            "processed_decision_ids": list(self.processed_decision_ids),
            "processed_boundary_weeks": list(self.processed_boundary_weeks),
            "processed_work_weeks": list(self.processed_work_weeks),
        }


def _finite_unit(value: Any, name: str) -> float:
    return _finite_number(value, name, minimum=0.0, maximum=1.0)


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise TypeError(f"Expected '{name}' to be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Expected '{name}' to be finite.")
    if numeric < minimum:
        raise ValueError(f"Expected '{name}' to be >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"Expected '{name}' to be <= {maximum}.")
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


def _require_money(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected monetary value '{name}' to be Decimal.")
    if not value.is_finite():
        raise ValueError(f"Expected monetary value '{name}' to be finite.")
    if value < Decimal(0):
        raise ValueError(f"Expected monetary value '{name}' to be non-negative.")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")
