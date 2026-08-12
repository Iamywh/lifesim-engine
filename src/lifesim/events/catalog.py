from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from lifesim.events.model import (
    EventCatalog,
    EventCondition,
    EventDefinition,
    WeightModifier,
)


def load_event_catalog(path: str | Path) -> EventCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_event_catalog(raw)


def parse_event_catalog(raw: dict[str, Any]) -> EventCatalog:
    settings = raw.get("event_settings", {})
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
        tags=tuple(raw.get("tags", [])),
        title=raw["title"],
        summary=raw["summary"],
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
