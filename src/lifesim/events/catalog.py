from __future__ import annotations

import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lifesim.events.model import (
    EventCatalog,
    EventCondition,
    EventDefinition,
    EventOption,
    WeightModifier,
)


def load_event_catalog(path: str | Path) -> EventCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_event_catalog(raw)


def parse_event_catalog(raw: dict[str, Any]) -> EventCatalog:
    settings = raw.get("event_settings", {})
    if not isinstance(settings, dict):
        raise TypeError("Expected event catalog 'event_settings' to be a table.")
    events = raw.get("events")
    if not isinstance(events, list):
        raise TypeError("Expected event catalog to contain an 'events' list.")

    return EventCatalog(
        definitions=tuple(_parse_event(event) for event in events),
        max_events_per_week=settings.get("max_events_per_week", 1),
        event_probability=settings.get("event_probability", 0.35),
    )


def _parse_event(raw: dict[str, Any]) -> EventDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each event to be a table.")
    return EventDefinition(
        event_id=raw["event_id"],
        version=raw["version"],
        category=raw["category"],
        base_weight=raw["base_weight"],
        conditions=tuple(_parse_condition(item) for item in raw.get("conditions", [])),
        weight_modifiers=tuple(
            _parse_weight_modifier(item) for item in raw.get("weight_modifiers", [])
        ),
        cooldown_weeks=raw.get("cooldown_weeks", 0),
        tags=raw.get("tags", []),
        title=raw["title"],
        summary=raw["summary"],
        time_pressure=raw.get("time_pressure", 0.0),
        options=tuple(_parse_option(item) for item in raw.get("options", [])),
    )


def _parse_condition(raw: dict[str, Any]) -> EventCondition:
    if not isinstance(raw, dict):
        raise TypeError("Expected event condition to be a table.")
    return EventCondition(
        condition_type=raw["type"],
        path=raw.get("path", ""),
        value=raw.get("value"),
    )


def _parse_weight_modifier(raw: dict[str, Any]) -> WeightModifier:
    if not isinstance(raw, dict):
        raise TypeError("Expected weight modifier to be a table.")
    return WeightModifier(
        condition=EventCondition(
            condition_type=raw["type"],
            path=raw.get("path", ""),
            value=raw.get("value"),
        ),
        multiplier=raw["multiplier"],
    )


def _parse_option(raw: dict[str, Any]) -> EventOption:
    if not isinstance(raw, dict):
        raise TypeError("Expected event option to be a table.")
    return EventOption(
        option_id=raw["option_id"],
        label=raw["label"],
        summary=raw["summary"],
        availability_conditions=tuple(
            _parse_condition(item) for item in raw.get("availability_conditions", [])
        ),
        estimated_cost=_money(raw.get("estimated_cost", "0.00"), "estimated_cost"),
        requires_full_estimated_cost=raw.get("requires_full_estimated_cost", True),
        expected_weekly_financial_gain=_money(
            raw.get("expected_weekly_financial_gain", "0.00"),
            "expected_weekly_financial_gain",
        ),
        ongoing_weekly_time_hours=raw.get("ongoing_weekly_time_hours", 0.0),
        time_cost_hours=raw["time_cost_hours"],
        energy_cost=raw["energy_cost"],
        short_term_value=raw["short_term_value"],
        future_value=raw["future_value"],
        perceived_risk=raw["perceived_risk"],
        uncertainty=raw["uncertainty"],
        social_value=raw["social_value"],
        social_pressure=raw["social_pressure"],
        autonomy_value=raw["autonomy_value"],
        learning_value=raw["learning_value"],
        health_value=raw["health_value"],
        comfort_value=raw["comfort_value"],
        goal_tags=raw.get("goal_tags", []),
    )


def _money(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"Expected monetary value '{name}' to be a string.")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Expected monetary value '{name}' to be a decimal string.") from error
