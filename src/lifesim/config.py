from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    name: str
    seed: int
    steps: int


@dataclass(frozen=True, slots=True)
class WorldConfig:
    initial_population: int


@dataclass(frozen=True, slots=True)
class LifeSimConfig:
    simulation: SimulationConfig
    world: WorldConfig


def load_config(path: str | Path) -> LifeSimConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> LifeSimConfig:
    simulation = raw.get("simulation", {})
    world = raw.get("world", {})

    return LifeSimConfig(
        simulation=SimulationConfig(
            name=_require_str(simulation, "name"),
            seed=_require_int(simulation, "seed", minimum=0),
            steps=_require_int(simulation, "steps", minimum=0),
        ),
        world=WorldConfig(
            initial_population=_require_int(world, "initial_population", minimum=0),
        ),
    )


def _require_int(section: dict[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = section.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer config value for '{key}'.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Expected config value '{key}' to be >= {minimum}.")
    return value


def _require_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string config value for '{key}'.")
    return value
