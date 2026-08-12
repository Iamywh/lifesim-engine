from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lifesim.agents.state import AgentState
from lifesim.config import LifeSimConfig
from lifesim.rng import create_rng
from lifesim.weekly import WeeklyContext, WeeklyPipeline, WeeklySummary, WeeklyTransition


@dataclass(frozen=True, slots=True)
class SimulationState:
    week: int
    agent_state: AgentState | None = None

    def __post_init__(self) -> None:
        if self.week < 0:
            raise ValueError("Expected simulation state week to be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        output = {
            "week": self.week,
        }
        if self.agent_state is not None:
            output["agent"] = self.agent_state.to_dict()
        return output


@dataclass(frozen=True, slots=True)
class SimulationResult:
    name: str
    seed: int
    city_name: str
    states: tuple[SimulationState, ...]
    summaries: tuple[WeeklySummary, ...] = ()

    def __post_init__(self) -> None:
        weeks = tuple(state.week for state in self.states)
        if weeks and weeks[0] != 0:
            raise ValueError("Expected first simulation week to be 0.")
        if weeks != tuple(range(len(weeks))):
            raise ValueError("Expected simulation weeks to be sequential.")

    def to_dict(self) -> dict[str, Any]:
        output = {
            "name": self.name,
            "seed": self.seed,
            "city_name": self.city_name,
            "states": [state.to_dict() for state in self.states],
        }
        if self.summaries:
            output["summaries"] = [summary.to_dict() for summary in self.summaries]
        return output


class LifeSimEngine:
    def __init__(
        self,
        config: LifeSimConfig,
        transitions: Sequence[WeeklyTransition] = (),
    ) -> None:
        self._config = config
        self._pipeline = WeeklyPipeline(transitions)

    def run(self, initial_agent: AgentState | None = None) -> SimulationResult:
        rng = create_rng(self._config.simulation.seed)
        states: list[SimulationState] = [
            SimulationState(
                week=0,
                agent_state=initial_agent,
            )
        ]
        summaries: list[WeeklySummary] = []

        if initial_agent is None:
            for week in range(1, self._config.simulation.duration_weeks + 1):
                states.append(SimulationState(week=week))
        else:
            summaries.append(
                WeeklySummary(
                    week=0,
                    agent_id=initial_agent.identity.agent_id,
                    state_changed=False,
                )
            )
            previous_agent = initial_agent
            for week in range(1, self._config.simulation.duration_weeks + 1):
                context = WeeklyContext(
                    week=week,
                    config=self._config,
                    rng=rng,
                )
                next_agent = self._pipeline.advance(previous_agent, context)
                states.append(
                    SimulationState(
                        week=week,
                        agent_state=next_agent,
                    )
                )
                summaries.append(
                    WeeklySummary(
                        week=week,
                        agent_id=next_agent.identity.agent_id,
                        state_changed=next_agent != previous_agent,
                    )
                )
                previous_agent = next_agent

        return SimulationResult(
            name=self._config.simulation.name,
            seed=self._config.simulation.seed,
            city_name=self._config.city.name,
            states=tuple(states),
            summaries=tuple(summaries),
        )
