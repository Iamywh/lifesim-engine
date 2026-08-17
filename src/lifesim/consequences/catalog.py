from __future__ import annotations

import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lifesim.consequences.model import (
    ConsequenceCatalog,
    FinancialChargeDefinition,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    StateEffectDefinition,
    validate_consequence_catalog,
)
from lifesim.events.catalog import _parse_condition
from lifesim.events.model import EventCatalog
from lifesim.finance import MANDATORY_FUNDING_ORDER


def load_consequence_catalog(
    path: str | Path,
    *,
    event_catalog: EventCatalog | None = None,
) -> ConsequenceCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_consequence_catalog(raw, event_catalog=event_catalog)


def parse_consequence_catalog(
    raw: dict[str, Any],
    *,
    event_catalog: EventCatalog | None = None,
) -> ConsequenceCatalog:
    definitions = raw.get("consequences")
    if definitions is None:
        definitions = []
    if not isinstance(definitions, list):
        raise TypeError("Expected consequence catalog to contain a 'consequences' list.")
    catalog = ConsequenceCatalog(
        definitions=tuple(_parse_consequence(definition) for definition in definitions)
    )
    if event_catalog is not None:
        validate_consequence_catalog(catalog, event_catalog)
    return catalog


def _parse_consequence(raw: dict[str, Any]) -> OptionConsequenceDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each consequence to be a table.")
    return OptionConsequenceDefinition(
        event_id=raw["event_id"],
        event_version=raw["event_version"],
        option_id=raw["option_id"],
        effects=tuple(_parse_effect(effect) for effect in raw.get("effects", [])),
        outcomes=tuple(_parse_outcome(outcome) for outcome in raw.get("outcomes", [])),
        financial_charges=tuple(
            _parse_financial_charge(charge)
            for charge in raw.get("financial_charges", [])
        ),
    )


def _parse_outcome(raw: dict[str, Any]) -> OutcomeDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each outcome to be a table.")
    return OutcomeDefinition(
        outcome_id=raw["outcome_id"],
        weight=raw["weight"],
        effects=tuple(_parse_effect(effect) for effect in raw.get("effects", [])),
        financial_charges=tuple(
            _parse_financial_charge(charge)
            for charge in raw.get("financial_charges", [])
        ),
    )


def _parse_effect(raw: dict[str, Any]) -> StateEffectDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each state effect to be a table.")
    path = raw["path"]
    return StateEffectDefinition(
        path=path,
        delta=_parse_delta(raw["delta"], path),
        delay_weeks=raw.get("delay_weeks", 0),
        conditions=tuple(_parse_condition(condition) for condition in raw.get("conditions", [])),
    )


def _parse_delta(value: Any, path: str) -> Decimal | float:
    if path.startswith("financial."):
        if not isinstance(value, str):
            raise TypeError("Expected monetary consequence delta to be a string.")
        try:
            return Decimal(value)
        except InvalidOperation as error:
            raise ValueError("Expected monetary consequence delta to be a decimal string.") from error
    return value


def _parse_financial_charge(raw: dict[str, Any]) -> FinancialChargeDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each financial charge to be a table.")
    amount = raw["amount"]
    if not isinstance(amount, str):
        raise TypeError("Expected financial charge amount to be a string.")
    try:
        parsed_amount = Decimal(amount)
    except InvalidOperation as error:
        raise ValueError("Expected financial charge amount to be a decimal string.") from error
    return FinancialChargeDefinition(
        amount=parsed_amount,
        category=raw["category"],
        delay_weeks=raw.get("delay_weeks", 0),
        shortfall_policy=raw.get("shortfall_policy", "require_full"),
        funding_order=tuple(raw.get("funding_order", MANDATORY_FUNDING_ORDER)),
        conditions=tuple(_parse_condition(condition) for condition in raw.get("conditions", [])),
    )
