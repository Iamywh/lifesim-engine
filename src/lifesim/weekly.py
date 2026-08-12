from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from random import Random
from typing import Any, Protocol

from lifesim.agents.state import AgentState
from lifesim.config import LifeSimConfig


@dataclass(frozen=True, slots=True)
class WeeklyContext:
    week: int
    config: LifeSimConfig
    rng: Random

    def __post_init__(self) -> None:
        if self.week < 1:
            raise ValueError("Expected weekly transition context week to be >= 1.")


class WeeklyTransition(Protocol):
    def apply(self, state: AgentState, context: WeeklyContext) -> AgentState:
        """Return the next immutable agent state for the supplied week.

        Implementations must not retain run-specific mutable state between
        runs. Persistent simulation state belongs in AgentState or in explicit
        run context objects so repeated LifeSimEngine.run() calls remain
        deterministic.
        """


@dataclass(frozen=True, slots=True)
class WeeklySummary:
    week: int
    agent_id: str
    state_changed: bool

    def __post_init__(self) -> None:
        if self.week < 0:
            raise ValueError("Expected summary week to be >= 0.")
        if not self.agent_id:
            raise ValueError("Expected summary agent_id to be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "agent_id": self.agent_id,
            "state_changed": self.state_changed,
        }


class WeeklyPipeline:
    def __init__(self, transitions: Sequence[WeeklyTransition] = ()) -> None:
        self._transitions = tuple(transitions)

    @property
    def transitions(self) -> tuple[WeeklyTransition, ...]:
        return self._transitions

    def advance(self, state: AgentState, context: WeeklyContext) -> AgentState:
        next_state = state

        for transition in self._transitions:
            candidate = transition.apply(next_state, context)
            if not isinstance(candidate, AgentState):
                raise TypeError("Expected weekly transition to return AgentState.")
            if candidate.identity.agent_id != state.identity.agent_id:
                raise ValueError("Weekly transitions must preserve agent identity in M2.")
            next_state = candidate

        if next_state is state:
            return replace(state)
        return next_state
