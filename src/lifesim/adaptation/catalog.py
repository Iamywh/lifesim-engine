from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from lifesim.adaptation.model import (
    AdaptationCatalog,
    AdaptationSettings,
    HabitDefinition,
    TraitEvidenceMapping,
)


def load_adaptation_catalog(path: str | Path) -> AdaptationCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_adaptation_catalog(raw)


def parse_adaptation_catalog(raw: dict[str, Any]) -> AdaptationCatalog:
    settings = raw.get("settings", {})
    if not isinstance(settings, dict):
        raise TypeError("Expected adaptation settings to be a table.")
    habits = raw.get("habits")
    if not isinstance(habits, list):
        raise TypeError("Expected adaptation catalog to contain a habits list.")
    mappings = raw.get("trait_mappings", [])
    if not isinstance(mappings, list):
        raise TypeError("Expected trait_mappings to be a list.")
    return AdaptationCatalog(
        settings=AdaptationSettings(**settings),
        habits=tuple(_habit(item) for item in habits),
        trait_mappings=tuple(_mapping(item) for item in mappings),
    )


def _habit(raw: dict[str, Any]) -> HabitDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected habit definitions to be tables.")
    return HabitDefinition(
        habit_id=_required(raw, "habit_id"),
        name=_required(raw, "name"),
        cadence=_required(raw, "cadence"),
        behavior_tags=tuple(raw.get("behavior_tags", ())),
        formation_rate=raw.get("formation_rate", 2.5),
        reinforcement_rate=raw.get("reinforcement_rate", 1.2),
        nonuse_decay_rate=raw.get("nonuse_decay_rate", 0.25),
        formation_threshold=raw.get("formation_threshold", 9.0),
        minimum_reinforcing_weeks=raw.get("minimum_reinforcing_weeks", 3),
        grace_weeks=raw.get("grace_weeks", 2),
        contradiction_tags=tuple(raw.get("contradiction_tags", ())),
    )


def _mapping(raw: dict[str, Any]) -> TraitEvidenceMapping:
    if not isinstance(raw, dict):
        raise TypeError("Expected trait mappings to be tables.")
    return TraitEvidenceMapping(
        evidence_type=_required(raw, "evidence_type"),
        evidence_key=_required(raw, "evidence_key"),
        trait=_required(raw, "trait"),
        coefficient=raw.get("coefficient"),
    )


def _required(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected adaptation '{key}' to be a non-empty string.")
    return value
