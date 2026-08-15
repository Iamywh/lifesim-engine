from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    name: str
    seed: int
    duration_weeks: int
    start_date: date = date(2026, 1, 5)


@dataclass(frozen=True, slots=True)
class CityConfig:
    name: str


@dataclass(frozen=True, slots=True)
class LifeSimConfig:
    simulation: SimulationConfig
    city: CityConfig


def load_config(path: str | Path) -> LifeSimConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> LifeSimConfig:
    simulation = raw.get("simulation", {})
    city = raw.get("city", {})

    return LifeSimConfig(
        simulation=SimulationConfig(
            name=_require_str(simulation, "name"),
            seed=_require_int(simulation, "seed", minimum=0),
            duration_weeks=_require_int(simulation, "duration_weeks", minimum=0),
            start_date=_optional_date(simulation, "start_date", date(2026, 1, 5)),
        ),
        city=CityConfig(
            name=_require_str(city, "name"),
        ),
    )


def _require_int(section: dict[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = section.get(key)
    if not isinstance(value, int):
        raise TypeError(f"Expected integer config value for '{key}'.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Expected config value '{key}' to be >= {minimum}.")
    return value


def _require_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string config value for '{key}'.")
    return value


def _optional_date(section: dict[str, Any], key: str, default: date) -> date:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"Expected ISO date string config value for '{key}'.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Expected ISO date string config value for '{key}'.") from error
