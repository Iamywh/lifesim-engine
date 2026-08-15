from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from lifesim.agents.state import SerializableState
from lifesim.decisions.model import DecisionRecord


@dataclass(frozen=True, slots=True)
class RoutineProfile(SerializableState):
    profile_id: str
    label: str
    summary: str
    estimated_cost: Decimal
    time_cost_hours: float
    energy_cost: float
    short_term_value: float
    future_value: float
    perceived_risk: float
    uncertainty: float
    social_value: float
    social_pressure: float
    autonomy_value: float
    learning_value: float
    health_value: float
    comfort_value: float
    goal_tags: tuple[str, ...]
    food_budget: Decimal
    minimum_food_budget: Decimal
    transport_budget: Decimal
    discretionary_budget: Decimal
    social_contact: float
    physical_activity: float
    recovery_intensity: float

    def __post_init__(self) -> None:
        _require_non_empty(self.profile_id, "profile_id")
        _require_non_empty(self.label, "label")
        _require_non_empty(self.summary, "summary")
        for name in (
            "estimated_cost",
            "food_budget",
            "minimum_food_budget",
            "transport_budget",
            "discretionary_budget",
        ):
            _require_money(getattr(self, name), name)
        if self.minimum_food_budget > self.food_budget:
            raise ValueError("Expected minimum_food_budget to be <= food_budget.")
        object.__setattr__(self, "time_cost_hours", _finite_number(self.time_cost_hours, "time_cost_hours", minimum=0.0, maximum=168.0))
        object.__setattr__(self, "energy_cost", _finite_number(self.energy_cost, "energy_cost", minimum=0.0, maximum=100.0))
        for name in (
            "short_term_value",
            "future_value",
            "social_value",
            "autonomy_value",
            "learning_value",
            "health_value",
            "comfort_value",
        ):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=-1.0, maximum=1.0))
        for name in ("perceived_risk", "uncertainty", "social_pressure"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0))
        for name in ("social_contact", "physical_activity", "recovery_intensity"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, minimum=0.0, maximum=1.0))
        object.__setattr__(self, "goal_tags", _string_sequence(self.goal_tags, "goal_tags"))


@dataclass(frozen=True, slots=True)
class RoutineCatalog(SerializableState):
    profiles: tuple[RoutineProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        for profile in self.profiles:
            if not isinstance(profile, RoutineProfile):
                raise TypeError("Expected routine catalog profiles to contain RoutineProfile values.")
        ids = tuple(profile.profile_id for profile in self.profiles)
        _require_unique(ids, "profile_id")

    def get(self, profile_id: str) -> RoutineProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"Unknown routine profile '{profile_id}'.")


@dataclass(frozen=True, slots=True)
class FundingTransfer(SerializableState):
    source: str
    amount: Decimal

    def __post_init__(self) -> None:
        _require_non_empty(self.source, "source")
        _require_money(self.amount, "amount")


@dataclass(frozen=True, slots=True)
class CashflowEntry(SerializableState):
    entry_id: str
    kind: str
    name: str
    amount_due: Decimal
    amount_paid: Decimal
    cadence: str
    due_date: str
    paid: bool
    funding: tuple[FundingTransfer, ...] = ()
    reliability: float | None = None
    roll: float | None = None
    arrear_balance_after: Decimal | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.entry_id, "entry_id")
        _require_non_empty(self.kind, "kind")
        _require_non_empty(self.name, "name")
        _require_money(self.amount_due, "amount_due")
        _require_money(self.amount_paid, "amount_paid")
        _require_non_empty(self.cadence, "cadence")
        _require_non_empty(self.due_date, "due_date")
        if not isinstance(self.paid, bool):
            raise TypeError("Expected paid to be bool.")
        object.__setattr__(self, "funding", tuple(self.funding))
        for transfer in self.funding:
            if not isinstance(transfer, FundingTransfer):
                raise TypeError("Expected funding to contain FundingTransfer values.")
        if self.reliability is not None:
            object.__setattr__(self, "reliability", _finite_number(self.reliability, "reliability", minimum=0.0, maximum=1.0))
        if self.roll is not None:
            object.__setattr__(self, "roll", _finite_number(self.roll, "roll", minimum=0.0, maximum=1.0))
        if self.arrear_balance_after is not None:
            _require_money(self.arrear_balance_after, "arrear_balance_after")
        if self.funding:
            total = sum((transfer.amount for transfer in self.funding), Decimal("0.00"))
            if total != self.amount_paid:
                raise ValueError("Expected funding transfer total to match amount_paid.")
        if self.kind != "income":
            if self.amount_paid > self.amount_due:
                raise ValueError("Expected outflow amount_paid to be <= amount_due.")
            if self.paid != (self.amount_paid == self.amount_due):
                raise ValueError("Expected outflow paid flag to reflect full payment.")


@dataclass(frozen=True, slots=True)
class CashflowRecord(SerializableState):
    week: int
    week_start: str
    week_end: str
    entries: tuple[CashflowEntry, ...]
    effects: tuple[RoutineEffectApplication, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.week_start, "week_start")
        _require_non_empty(self.week_end, "week_end")
        object.__setattr__(self, "entries", tuple(self.entries))
        for entry in self.entries:
            if not isinstance(entry, CashflowEntry):
                raise TypeError("Expected cashflow record entries to contain CashflowEntry values.")
        _require_unique(tuple(entry.entry_id for entry in self.entries), "cashflow entry_id")
        object.__setattr__(self, "effects", tuple(self.effects))
        for effect in self.effects:
            if not isinstance(effect, RoutineEffectApplication):
                raise TypeError("Expected cashflow effects to contain RoutineEffectApplication values.")


@dataclass(frozen=True, slots=True)
class RoutineEffectApplication(SerializableState):
    path: str
    before: Decimal | float
    after: Decimal | float
    delta: Decimal | float
    clamped: bool
    reason: str
    source: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.path, "path")
        _require_numeric(self.before, "before")
        _require_numeric(self.after, "after")
        _require_numeric(self.delta, "delta")
        if not isinstance(self.clamped, bool):
            raise TypeError("Expected clamped to be bool.")
        _require_non_empty(self.reason, "reason")
        if not isinstance(self.source, str):
            raise TypeError("Expected source to be a string.")
        if isinstance(self.before, Decimal) or isinstance(self.after, Decimal) or isinstance(self.delta, Decimal):
            if not all(isinstance(value, Decimal) for value in (self.before, self.after, self.delta)):
                raise TypeError("Expected Decimal effect values to be consistently Decimal.")
            if self.after - self.before != self.delta:
                raise ValueError("Expected effect delta to equal after - before.")
        elif not math.isclose(self.after - self.before, self.delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Expected effect delta to equal after - before.")
        if self.clamped and "." not in self.path:
            raise ValueError("Expected clamped effects to target a bounded state path.")


@dataclass(frozen=True, slots=True)
class RoutineWeekRecord(SerializableState):
    week: int
    profile_id: str
    previous_profile_id: str
    weeks_in_current_profile: int
    low_social_streak: int
    decision_id: str
    spending: tuple[CashflowEntry, ...]
    effects: tuple[RoutineEffectApplication, ...]

    def __post_init__(self) -> None:
        _integer(self.week, "week", minimum=0)
        _require_non_empty(self.profile_id, "profile_id")
        if not isinstance(self.previous_profile_id, str):
            raise TypeError("Expected previous_profile_id to be a string.")
        _integer(self.weeks_in_current_profile, "weeks_in_current_profile", minimum=0)
        _integer(self.low_social_streak, "low_social_streak", minimum=0)
        _require_non_empty(self.decision_id, "decision_id")
        object.__setattr__(self, "spending", tuple(self.spending))
        for entry in self.spending:
            if not isinstance(entry, CashflowEntry):
                raise TypeError("Expected routine spending to contain CashflowEntry values.")
        _require_unique(tuple(entry.entry_id for entry in self.spending), "routine spending entry_id")
        object.__setattr__(self, "effects", tuple(self.effects))
        for effect in self.effects:
            if not isinstance(effect, RoutineEffectApplication):
                raise TypeError("Expected routine effects to contain RoutineEffectApplication values.")


@dataclass(frozen=True, slots=True)
class PassiveLifeHistory:
    cashflow_records: tuple[CashflowRecord, ...] = ()
    routine_records: tuple[RoutineWeekRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cashflow_records", tuple(self.cashflow_records))
        object.__setattr__(self, "routine_records", tuple(self.routine_records))
        for record in self.cashflow_records:
            if not isinstance(record, CashflowRecord):
                raise TypeError("Expected cashflow history to contain CashflowRecord values.")
        for record in self.routine_records:
            if not isinstance(record, RoutineWeekRecord):
                raise TypeError("Expected routine history to contain RoutineWeekRecord values.")
        _require_unique(tuple(str(record.week) for record in self.cashflow_records), "cashflow week")
        _require_unique(tuple(str(record.week) for record in self.routine_records), "routine week")

    def record_cashflow(self, record: CashflowRecord) -> PassiveLifeHistory:
        return PassiveLifeHistory(self.cashflow_records + (record,), self.routine_records)

    def record_routine(self, record: RoutineWeekRecord) -> PassiveLifeHistory:
        return PassiveLifeHistory(self.cashflow_records, self.routine_records + (record,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cashflow_records": [record.to_dict() for record in self.cashflow_records],
            "routine_records": [record.to_dict() for record in self.routine_records],
        }


@dataclass(frozen=True, slots=True)
class PassiveLifeRuntimeState:
    history: PassiveLifeHistory = field(default_factory=PassiveLifeHistory)
    planned_routine_profile_id: str = ""
    planned_routine_decision: DecisionRecord | None = None
    planned_routine_week: int | None = None
    processed_cashflow_weeks: tuple[int, ...] = ()
    processed_routine_planning_weeks: tuple[int, ...] = ()
    processed_routine_execution_weeks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, PassiveLifeHistory):
            raise TypeError("Expected passive runtime history to be PassiveLifeHistory.")
        if not isinstance(self.planned_routine_profile_id, str):
            raise TypeError("Expected planned_routine_profile_id to be a string.")
        if self.planned_routine_decision is not None and not isinstance(self.planned_routine_decision, DecisionRecord):
            raise TypeError("Expected planned_routine_decision to be DecisionRecord.")
        if self.planned_routine_week is not None:
            _integer(self.planned_routine_week, "planned_routine_week", minimum=1)
        for name in (
            "processed_cashflow_weeks",
            "processed_routine_planning_weeks",
            "processed_routine_execution_weeks",
        ):
            weeks = _integer_sequence(getattr(self, name), name)
            _require_unique(tuple(str(week) for week in weeks), name)
            object.__setattr__(self, name, weeks)
        if self.planned_routine_decision is None and self.planned_routine_profile_id:
            raise ValueError("Expected planned routine decision when planned profile id is set.")
        if self.planned_routine_decision is not None and not self.planned_routine_profile_id:
            raise ValueError("Expected planned profile id when planned routine decision is set.")
        if (self.planned_routine_decision is None) != (self.planned_routine_week is None):
            raise ValueError("Expected planned routine week to match planned routine decision presence.")
        history_cashflow_weeks = tuple(record.week for record in self.history.cashflow_records)
        history_routine_weeks = tuple(record.week for record in self.history.routine_records)
        if set(history_cashflow_weeks) - set(self.processed_cashflow_weeks):
            raise ValueError("Expected cashflow history weeks to be marked processed.")
        if set(history_routine_weeks) - set(self.processed_routine_execution_weeks):
            raise ValueError("Expected routine history weeks to be marked executed.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.to_dict(),
            "planned_routine_profile_id": self.planned_routine_profile_id,
            "planned_routine_decision": (
                self.planned_routine_decision.to_dict()
                if self.planned_routine_decision is not None
                else None
            ),
            "planned_routine_week": self.planned_routine_week,
            "processed_cashflow_weeks": list(self.processed_cashflow_weeks),
            "processed_routine_planning_weeks": list(self.processed_routine_planning_weeks),
            "processed_routine_execution_weeks": list(self.processed_routine_execution_weeks),
        }


def _require_money(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected '{name}' to be Decimal.")
    if not value.is_finite() or value < Decimal(0):
        raise ValueError(f"Expected '{name}' to be finite and non-negative.")


def _finite_number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Expected '{name}' to be finite.")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"Expected '{name}' to be between {minimum} and {maximum}.")
    return numeric


def _require_numeric(value: Decimal | float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise TypeError(f"Expected '{name}' to be numeric.")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Expected '{name}' to be finite.")
    elif not math.isfinite(float(value)):
        raise ValueError(f"Expected '{name}' to be finite.")


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
