from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lifesim.config import LifeSimConfig
from lifesim.rng import create_rng


@dataclass(frozen=True, slots=True)
class SimulationState:
    step: int
    population: int
    entropy_marker: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "population": self.population,
            "entropy_marker": self.entropy_marker,
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    name: str
    seed: int
    states: tuple[SimulationState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "states": [state.to_dict() for state in self.states],
        }


class LifeSimEngine:
    def __init__(self, config: LifeSimConfig) -> None:
        self._config = config
        self._rng = create_rng(config.simulation.seed)

    def run(self) -> SimulationResult:
        states = []
        population = self._config.world.initial_population

        for step in range(self._config.simulation.steps + 1):
            states.append(
                SimulationState(
                    step=step,
                    population=population,
                    entropy_marker=self._rng.random(),
                )
            )

        return SimulationResult(
            name=self._config.simulation.name,
            seed=self._config.simulation.seed,
            states=tuple(states),
        )
