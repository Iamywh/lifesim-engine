from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from lifesim.agents.state import Arrear, FinancialState, SerializableState

CENT = Decimal("0.01")
MANDATORY_FUNDING_ORDER = ("bank_balance", "cash", "savings", "emergency_fund")
OPTIONAL_FUNDING_ORDER = ("bank_balance", "cash")
MONEY_ACCOUNTS = frozenset(MANDATORY_FUNDING_ORDER)


@dataclass(frozen=True, slots=True)
class SettlementTransfer(SerializableState):
    source: str
    amount: Decimal

    def __post_init__(self) -> None:
        _require_account(self.source)
        _require_money(self.amount, "amount")


@dataclass(frozen=True, slots=True)
class SettlementResult(SerializableState):
    financial: FinancialState
    amount_due: Decimal
    amount_paid: Decimal
    amount_unpaid: Decimal
    funding_order: tuple[str, ...]
    transfers: tuple[SettlementTransfer, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.financial, FinancialState):
            raise TypeError("Expected financial to be FinancialState.")
        _require_money(self.amount_due, "amount_due")
        _require_money(self.amount_paid, "amount_paid")
        _require_money(self.amount_unpaid, "amount_unpaid")
        object.__setattr__(self, "funding_order", _funding_order(self.funding_order))
        object.__setattr__(self, "transfers", tuple(self.transfers))
        for transfer in self.transfers:
            if not isinstance(transfer, SettlementTransfer):
                raise TypeError("Expected transfers to contain SettlementTransfer values.")
        if self.amount_due != self.amount_paid + self.amount_unpaid:
            raise ValueError("Expected settlement amount_due to equal amount_paid + amount_unpaid.")
        if sum((transfer.amount for transfer in self.transfers), Decimal("0.00")) != self.amount_paid:
            raise ValueError("Expected settlement transfers to sum to amount_paid.")

    @property
    def fully_paid(self) -> bool:
        return self.amount_unpaid == Decimal("0.00")


def money(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError("Expected Decimal money value.")
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def total_liquid(financial: FinancialState) -> Decimal:
    return money(sum((getattr(financial, account) for account in MANDATORY_FUNDING_ORDER), Decimal("0.00")))


def settle_liquid_amount(
    financial: FinancialState,
    amount: Decimal,
    funding_order: tuple[str, ...] = MANDATORY_FUNDING_ORDER,
) -> SettlementResult:
    if not isinstance(financial, FinancialState):
        raise TypeError("Expected financial to be FinancialState.")
    amount = money(amount)
    order = _funding_order(funding_order)
    values = {account: getattr(financial, account) for account in order}
    remaining = amount
    paid = Decimal("0.00")
    transfers: list[SettlementTransfer] = []
    for account in order:
        if remaining <= Decimal("0.00"):
            break
        available = values[account]
        used = min(available, remaining)
        if used <= Decimal("0.00"):
            continue
        values[account] = money(available - used)
        remaining = money(remaining - used)
        paid = money(paid + used)
        transfers.append(SettlementTransfer(account, used))
    return SettlementResult(
        financial=replace(financial, **values),
        amount_due=amount,
        amount_paid=paid,
        amount_unpaid=remaining,
        funding_order=order,
        transfers=tuple(transfers),
    )


def upsert_arrear(
    financial: FinancialState,
    *,
    obligation_id: str,
    category: str,
    unpaid: Decimal,
    week: int,
) -> FinancialState:
    _require_non_empty(obligation_id, "obligation_id")
    _require_non_empty(category, "category")
    unpaid = money(unpaid)
    if unpaid <= Decimal("0.00"):
        return financial
    output: list[Arrear] = []
    found = False
    for arrear in financial.arrears:
        if arrear.obligation_id == obligation_id:
            output.append(
                replace(
                    arrear,
                    balance=money(arrear.balance + unpaid),
                    last_updated_week=week,
                    missed_occurrences=arrear.missed_occurrences + 1,
                )
            )
            found = True
        else:
            output.append(arrear)
    if not found:
        output.append(
            Arrear(
                obligation_id=obligation_id,
                category=category,
                balance=unpaid,
                first_missed_week=week,
                last_updated_week=week,
                missed_occurrences=1,
            )
        )
    return replace(financial, arrears=tuple(output))


def find_arrear(financial: FinancialState, obligation_id: str) -> Arrear | None:
    for arrear in financial.arrears:
        if arrear.obligation_id == obligation_id:
            return arrear
    return None


def _funding_order(values: Any) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, list | tuple):
        raise TypeError("Expected funding_order to be a list or tuple.")
    order = tuple(values)
    if not order:
        raise ValueError("Expected funding_order to be non-empty.")
    for account in order:
        _require_account(account)
    if len(set(order)) != len(order):
        raise ValueError("Expected funding_order accounts to be unique.")
    return order


def _require_account(value: str) -> None:
    if value not in MONEY_ACCOUNTS:
        raise ValueError(f"Unsupported funding account '{value}'.")


def _require_money(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected {name} to be Decimal.")
    if not value.is_finite() or value < Decimal("0.00"):
        raise ValueError(f"Expected {name} to be finite and non-negative.")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected {name} to be a non-empty string.")
