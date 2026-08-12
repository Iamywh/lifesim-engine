from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lifesim.config import LifeSimConfig
from lifesim.rng import create_rng


@dataclass(frozen=True, slots=True)
class SimulationState:
    week: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    name: str
    seed: int
    city_name: str
    states: tuple[SimulationState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "city_name": self.city_name,
            "states": [state.to_dict() for state in self.states],
        }


class LifeSimEngine:
    def __init__(self, config: LifeSimConfig) -> None:
        self._config = config

    def run(self) -> SimulationResult:
        self._rng = create_rng(self._config.simulation.seed)
        states = []

        for week in range(self._config.simulation.duration_weeks + 1):
            states.append(
                SimulationState(
                    week=week,
                )
            )

        return SimulationResult(
            name=self._config.simulation.name,
            seed=self._config.simulation.seed,
            city_name=self._config.city.name,
            states=tuple(states),
        )
