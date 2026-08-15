from __future__ import annotations

import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lifesim.passive.model import RoutineCatalog, RoutineProfile


def load_routine_catalog(path: str | Path) -> RoutineCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_routine_catalog(raw)


def parse_routine_catalog(raw: dict[str, Any]) -> RoutineCatalog:
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        raise TypeError("Expected routine catalog to contain a 'profiles' list.")
    return RoutineCatalog(tuple(_parse_profile(profile) for profile in profiles))


def _parse_profile(raw: dict[str, Any]) -> RoutineProfile:
    if not isinstance(raw, dict):
        raise TypeError("Expected each routine profile to be a table.")
    return RoutineProfile(
        profile_id=_required(raw, "profile_id"),
        label=_required(raw, "label"),
        summary=_required(raw, "summary"),
        estimated_cost=_money(raw, "estimated_cost"),
        time_cost_hours=raw.get("time_cost_hours", 0.0),
        energy_cost=raw.get("energy_cost", 0.0),
        short_term_value=raw.get("short_term_value", 0.0),
        future_value=raw.get("future_value", 0.0),
        perceived_risk=raw.get("perceived_risk", 0.0),
        uncertainty=raw.get("uncertainty", 0.0),
        social_value=raw.get("social_value", 0.0),
        social_pressure=raw.get("social_pressure", 0.0),
        autonomy_value=raw.get("autonomy_value", 0.0),
        learning_value=raw.get("learning_value", 0.0),
        health_value=raw.get("health_value", 0.0),
        comfort_value=raw.get("comfort_value", 0.0),
        goal_tags=tuple(raw.get("goal_tags", ())),
        food_budget=_money(raw, "food_budget"),
        minimum_food_budget=_money(raw, "minimum_food_budget", default=raw.get("food_budget")),
        transport_budget=_money(raw, "transport_budget"),
        discretionary_budget=_money(raw, "discretionary_budget"),
        social_contact=raw.get("social_contact", 0.0),
        physical_activity=raw.get("physical_activity", 0.0),
        recovery_intensity=raw.get("recovery_intensity", 0.0),
    )


def _required(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected routine profile '{key}' to be a non-empty string.")
    return value


def _money(raw: dict[str, Any], key: str, *, default: Any = None) -> Decimal:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"Expected routine monetary value '{key}' to be a string.")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Expected routine monetary value '{key}' to be a decimal string.") from error
