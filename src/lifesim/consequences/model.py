from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from lifesim.agents.state import SerializableState
from lifesim.events.model import EventCatalog, EventCondition
from lifesim.finance import MANDATORY_FUNDING_ORDER, MONEY_ACCOUNTS, SettlementTransfer

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
CHARGE_SHORTFALL_POLICIES = frozenset({"require_full", "arrear"})


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
class FinancialChargeDefinition(SerializableState):
    amount: Decimal
    category: str
    delay_weeks: int = 0
    shortfall_policy: str = "require_full"
    funding_order: tuple[str, ...] = MANDATORY_FUNDING_ORDER
    conditions: tuple[EventCondition, ...] = ()

    def __post_init__(self) -> None:
        _money(self.amount, "amount")
        _require_non_empty(self.category, "category")
        _integer(self.delay_weeks, "delay_weeks", minimum=0)
        if self.shortfall_policy not in CHARGE_SHORTFALL_POLICIES:
            raise ValueError("Expected shortfall_policy to be 'require_full' or 'arrear'.")
        funding_order = _string_sequence(self.funding_order, "funding_order")
        if not funding_order:
            raise ValueError("Expected funding_order to contain at least one account.")
        for account in funding_order:
            if account not in MONEY_ACCOUNTS:
                raise ValueError(f"Unsupported funding account '{account}'.")
        _require_unique(funding_order, "funding_order")
        object.__setattr__(self, "funding_order", funding_order)
        object.__setattr__(
            self,
            "conditions",
            _typed_tuple(self.conditions, EventCondition, "conditions"),
        )


@dataclass(frozen=True, slots=True)
class OutcomeDefinition(SerializableState):
    outcome_id: str
    weight: float
    effects: tuple[StateEffectDefinition, ...] = ()
    financial_charges: tuple[FinancialChargeDefinition, ...] = ()

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
        object.__setattr__(
            self,
            "financial_charges",
            _typed_tuple(self.financial_charges, FinancialChargeDefinition, "financial_charges"),
        )


@dataclass(frozen=True, slots=True)
class OptionConsequenceDefinition(SerializableState):
    event_id: str
    event_version: str
    option_id: str
    effects: tuple[StateEffectDefinition, ...] = ()
    outcomes: tuple[OutcomeDefinition, ...] = ()
    financial_charges: tuple[FinancialChargeDefinition, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.event_version, "event_version")
        _require_non_empty(self.option_id, "option_id")
        object.__setattr__(
            self,
            "effects",
            _typed_tuple(self.effects, StateEffectDefinition, "effects"),
        )
        object.__setattr__(
            self,
            "financial_charges",
            _typed_tuple(self.financial_charges, FinancialChargeDefinition, "financial_charges"),
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
        if not isinstance(self.effect, StateEffectDefinition):
            raise TypeError("Expected scheduled effect to contain StateEffectDefinition.")
        if self.effect.delay_weeks <= 0:
            raise ValueError("Scheduled effects must contain an effect with delay_weeks > 0.")
        if self.due_week != self.created_week + self.effect.delay_weeks:
            raise ValueError("Expected due_week to equal created_week + effect.delay_weeks.")


@dataclass(frozen=True, slots=True)
class ScheduledFinancialCharge(SerializableState):
    scheduled_charge_id: str
    source_decision_id: str
    source_consequence_id: str
    source_event_id: str
    source_event_version: str
    chosen_option_id: str
    source_outcome_id: str | None
    created_week: int
    due_week: int
    charge: FinancialChargeDefinition

    def __post_init__(self) -> None:
        _require_non_empty(self.scheduled_charge_id, "scheduled_charge_id")
        _require_non_empty(self.source_decision_id, "source_decision_id")
        _require_non_empty(self.source_consequence_id, "source_consequence_id")
        _require_non_empty(self.source_event_id, "source_event_id")
        _require_non_empty(self.source_event_version, "source_event_version")
        _require_non_empty(self.chosen_option_id, "chosen_option_id")
        if self.source_outcome_id is not None:
            _require_non_empty(self.source_outcome_id, "source_outcome_id")
        _integer(self.created_week, "created_week", minimum=0)
        _integer(self.due_week, "due_week", minimum=0)
        if not isinstance(self.charge, FinancialChargeDefinition):
            raise TypeError("Expected scheduled charge to contain FinancialChargeDefinition.")
        if self.charge.delay_weeks <= 0:
            raise ValueError("Scheduled financial charges must contain a charge with delay_weeks > 0.")
        if self.due_week != self.created_week + self.charge.delay_weeks:
            raise ValueError("Expected due_week to equal created_week + charge.delay_weeks.")


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
        if self.clamped and self.path not in BOUNDED_FLOAT_PATHS:
            raise ValueError("Clamping is only valid for bounded float consequence paths.")
        self._validate_values()
        if self.scheduled_effect_id is not None:
            _require_non_empty(self.scheduled_effect_id, "scheduled_effect_id")

    def _validate_values(self) -> None:
        if self.path in MONEY_PATHS:
            if not isinstance(self.requested_delta, Decimal) or not self.requested_delta.is_finite():
                raise TypeError("Expected monetary requested_delta to be finite Decimal.")
        else:
            object.__setattr__(
                self,
                "requested_delta",
                _finite_number(self.requested_delta, "requested_delta"),
            )

        if self.skipped:
            if self.before is not None or self.after is not None:
                raise ValueError("Skipped effect applications must not include before/after values.")
            if self.clamped:
                raise ValueError("Skipped effect applications must not be clamped.")
            if not self.skip_reason:
                raise ValueError("Skipped effect applications must include a skip_reason.")
            return

        if self.skip_reason:
            raise ValueError("Applied effects must not include a skip_reason.")
        if self.before is None or self.after is None:
            raise ValueError("Applied effects must include before and after values.")
        if self.path in MONEY_PATHS:
            if not isinstance(self.before, Decimal) or not self.before.is_finite():
                raise TypeError("Expected monetary before value to be finite Decimal.")
            if not isinstance(self.after, Decimal) or not self.after.is_finite():
                raise TypeError("Expected monetary after value to be finite Decimal.")
            return
        object.__setattr__(self, "before", _finite_number(self.before, "before"))
        object.__setattr__(self, "after", _finite_number(self.after, "after"))


@dataclass(frozen=True, slots=True)
class FinancialChargeApplication(SerializableState):
    amount_due: Decimal
    amount_paid: Decimal
    amount_unpaid: Decimal
    category: str
    shortfall_policy: str
    funding_order: tuple[str, ...]
    funding: tuple[SettlementTransfer, ...] = ()
    fully_paid: bool = False
    arrear_created: bool = False
    arrear_obligation_id: str = ""
    arrear_balance_after: Decimal | None = None
    skipped: bool = False
    skip_reason: str = ""
    scheduled_charge_id: str | None = None

    def __post_init__(self) -> None:
        _money(self.amount_due, "amount_due")
        _money(self.amount_paid, "amount_paid")
        _money(self.amount_unpaid, "amount_unpaid")
        _require_non_empty(self.category, "category")
        if self.shortfall_policy not in CHARGE_SHORTFALL_POLICIES:
            raise ValueError("Expected shortfall_policy to be 'require_full' or 'arrear'.")
        object.__setattr__(self, "funding_order", _string_sequence(self.funding_order, "funding_order"))
        object.__setattr__(
            self,
            "funding",
            _typed_tuple(self.funding, SettlementTransfer, "funding"),
        )
        total_funding = sum((transfer.amount for transfer in self.funding), Decimal("0.00"))
        if total_funding != self.amount_paid:
            raise ValueError("Expected funding transfer total to match amount_paid.")
        if not isinstance(self.fully_paid, bool):
            raise TypeError("Expected fully_paid to be bool.")
        if self.fully_paid != (self.amount_unpaid == Decimal("0.00")):
            raise ValueError("Expected fully_paid to reflect amount_unpaid.")
        if not isinstance(self.arrear_created, bool):
            raise TypeError("Expected arrear_created to be bool.")
        if self.arrear_created:
            _require_non_empty(self.arrear_obligation_id, "arrear_obligation_id")
            if self.arrear_balance_after is None:
                raise ValueError("Expected arrear_balance_after when arrear_created is true.")
        elif self.arrear_obligation_id:
            raise ValueError("Expected arrear_obligation_id only when arrear_created is true.")
        if self.arrear_balance_after is not None:
            _money(self.arrear_balance_after, "arrear_balance_after")
        if not isinstance(self.skipped, bool):
            raise TypeError("Expected skipped to be bool.")
        if self.skipped:
            if self.amount_paid != Decimal("0.00") or self.amount_unpaid != Decimal("0.00"):
                raise ValueError("Skipped charges must not include paid or unpaid amounts.")
            _require_non_empty(self.skip_reason, "skip_reason")
        elif self.skip_reason:
            raise ValueError("Applied charges must not include a skip_reason.")
        elif self.amount_paid + self.amount_unpaid != self.amount_due:
            raise ValueError("Expected amount_paid + amount_unpaid to equal amount_due.")
        if self.scheduled_charge_id is not None:
            _require_non_empty(self.scheduled_charge_id, "scheduled_charge_id")


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
    source_scheduled_charge_id: str | None = None
    effect_applications: tuple[EffectApplication, ...] = ()
    financial_charge_applications: tuple[FinancialChargeApplication, ...] = ()
    scheduled_effects_created: tuple[ScheduledEffect, ...] = ()
    scheduled_financial_charges_created: tuple[ScheduledFinancialCharge, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.consequence_id, "consequence_id")
        _require_non_empty(self.source_decision_id, "source_decision_id")
        _require_non_empty(self.source_event_id, "source_event_id")
        _require_non_empty(self.source_event_version, "source_event_version")
        _require_non_empty(self.chosen_option_id, "chosen_option_id")
        _integer(self.week_resolved, "week_resolved", minimum=0)
        if self.selected_outcome_id is not None:
            _require_non_empty(self.selected_outcome_id, "selected_outcome_id")
        self._validate_outcome_audit()
        if self.source_scheduled_effect_id is not None:
            _require_non_empty(self.source_scheduled_effect_id, "source_scheduled_effect_id")
        if self.source_scheduled_charge_id is not None:
            _require_non_empty(self.source_scheduled_charge_id, "source_scheduled_charge_id")
        object.__setattr__(
            self,
            "effect_applications",
            _typed_tuple(self.effect_applications, EffectApplication, "effect_applications"),
        )
        object.__setattr__(
            self,
            "financial_charge_applications",
            _typed_tuple(
                self.financial_charge_applications,
                FinancialChargeApplication,
                "financial_charge_applications",
            ),
        )
        object.__setattr__(
            self,
            "scheduled_effects_created",
            _typed_tuple(self.scheduled_effects_created, ScheduledEffect, "scheduled_effects_created"),
        )
        object.__setattr__(
            self,
            "scheduled_financial_charges_created",
            _typed_tuple(
                self.scheduled_financial_charges_created,
                ScheduledFinancialCharge,
                "scheduled_financial_charges_created",
            ),
        )

    def _validate_outcome_audit(self) -> None:
        has_roll = self.outcome_roll is not None
        has_total = self.outcome_total_weight is not None
        if has_roll != has_total:
            raise ValueError("outcome_roll and outcome_total_weight must both be present or absent.")
        if not has_roll:
            return
        if self.selected_outcome_id is None:
            raise ValueError("selected_outcome_id is required when outcome roll audit is present.")
        roll = _finite_number(self.outcome_roll, "outcome_roll", minimum=0.0)
        total = _finite_number(self.outcome_total_weight, "outcome_total_weight", minimum=0.0)
        if total <= 0.0:
            raise ValueError("Expected outcome_total_weight to be > 0.")
        if roll > total:
            raise ValueError("Expected outcome_roll to be <= outcome_total_weight.")
        object.__setattr__(self, "outcome_roll", roll)
        object.__setattr__(self, "outcome_total_weight", total)


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
    pending_scheduled_financial_charges: tuple[ScheduledFinancialCharge, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.history, ConsequenceHistory):
            raise TypeError("Expected consequence runtime history to be ConsequenceHistory.")
        pending = _typed_tuple(
            self.pending_scheduled_effects,
            ScheduledEffect,
            "pending_scheduled_effects",
        )
        _require_unique(
            tuple(effect.scheduled_effect_id for effect in pending),
            "scheduled_effect_id",
        )
        object.__setattr__(self, "pending_scheduled_effects", pending)
        pending_charges = _typed_tuple(
            self.pending_scheduled_financial_charges,
            ScheduledFinancialCharge,
            "pending_scheduled_financial_charges",
        )
        _require_unique(
            tuple(charge.scheduled_charge_id for charge in pending_charges),
            "scheduled_charge_id",
        )
        object.__setattr__(self, "pending_scheduled_financial_charges", pending_charges)
        processed = _string_sequence(self.processed_decision_ids, "processed_decision_ids")
        _require_unique(processed, "processed_decision_ids")
        object.__setattr__(self, "processed_decision_ids", processed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history.to_dict(),
            "pending_scheduled_effects": [
                effect.to_dict() for effect in self.pending_scheduled_effects
            ],
            "pending_scheduled_financial_charges": [
                charge.to_dict() for charge in self.pending_scheduled_financial_charges
            ],
            "processed_decision_ids": list(self.processed_decision_ids),
        }


def validate_consequence_catalog(
    consequence_catalog: ConsequenceCatalog,
    event_catalog: EventCatalog,
) -> None:
    valid_options = {
        (event.event_id, event.version, option.option_id): option
        for event in event_catalog.definitions
        for option in event.options
    }
    for definition in consequence_catalog.definitions:
        option = valid_options.get(definition.key)
        if option is None:
            event_id, version, option_id = definition.key
            raise ValueError(
                "Consequence definition references unknown event/version/option "
                f"'{event_id}/{version}/{option_id}'."
            )
        if option.requires_full_estimated_cost:
            immediate_charge_total = sum(
                (
                    charge.amount
                    for charge in definition.financial_charges
                    if charge.delay_weeks == 0
                ),
                Decimal("0.00"),
            )
            if immediate_charge_total > option.estimated_cost:
                raise ValueError(
                    "Immediate financial charges must not exceed estimated_cost "
                    f"for '{definition.event_id}/{definition.event_version}/{definition.option_id}'."
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


def _money(value: Any, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected '{name}' to be Decimal.")
    if not value.is_finite() or value < Decimal("0.00"):
        raise ValueError(f"Expected '{name}' to be finite and non-negative.")
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
