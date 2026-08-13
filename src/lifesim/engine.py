from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lifesim.agents.state import AgentState
from lifesim.config import LifeSimConfig
from lifesim.events.model import EventHistory, EventOccurrence, EventSelectionTrace
from lifesim.rng import create_rng
from lifesim.weekly import WeeklyContext, WeeklyPipeline, WeeklySummary, WeeklyTransition


@dataclass(frozen=True, slots=True)
class SimulationState:
    week: int
    agent_state: AgentState | None = None
    events: tuple[EventOccurrence, ...] = ()
    event_traces: tuple[EventSelectionTrace, ...] = ()
    decisions: tuple[Any, ...] = ()
    consequences: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.week < 0:
            raise ValueError("Expected simulation state week to be >= 0.")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "event_traces", tuple(self.event_traces))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "consequences", tuple(self.consequences))

    def to_dict(self) -> dict[str, Any]:
        output = {
            "week": self.week,
        }
        if self.agent_state is not None:
            output["agent"] = self.agent_state.to_dict()
            output["events"] = [event.to_dict() for event in self.events]
            output["decisions"] = [decision.to_dict() for decision in self.decisions]
            output["consequences"] = [
                consequence.to_dict() for consequence in self.consequences
            ]
        if self.event_traces:
            output["event_traces"] = [trace.to_dict() for trace in self.event_traces]
        return output


@dataclass(frozen=True, slots=True)
class SimulationResult:
    name: str
    seed: int
    city_name: str
    states: tuple[SimulationState, ...]
    summaries: tuple[WeeklySummary, ...] = ()
    event_history: EventHistory | None = None
    decision_history: Any | None = None
    consequence_history: Any | None = None
    pending_scheduled_effects: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        weeks = tuple(state.week for state in self.states)
        if weeks and weeks[0] != 0:
            raise ValueError("Expected first simulation week to be 0.")
        if weeks != tuple(range(len(weeks))):
            raise ValueError("Expected simulation weeks to be sequential.")
        object.__setattr__(
            self,
            "pending_scheduled_effects",
            tuple(self.pending_scheduled_effects),
        )

    def to_dict(self) -> dict[str, Any]:
        output = {
            "name": self.name,
            "seed": self.seed,
            "city_name": self.city_name,
            "states": [state.to_dict() for state in self.states],
        }
        if self.summaries:
            output["summaries"] = [summary.to_dict() for summary in self.summaries]
        if self.event_history is not None:
            output["event_history"] = self.event_history.to_dict()
        if self.decision_history is not None:
            output["decision_history"] = self.decision_history.to_dict()
        if self.consequence_history is not None:
            output["consequence_history"] = self.consequence_history.to_dict()
        if self.pending_scheduled_effects:
            output["pending_scheduled_effects"] = [
                effect.to_dict() for effect in self.pending_scheduled_effects
            ]
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
        event_history = EventHistory() if initial_agent is not None else None
        decision_history = None
        consequence_runtime = None

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
                    event_history=event_history,
                    decision_history=decision_history,
                    consequence_runtime=consequence_runtime,
                )
                transition_result = self._pipeline.advance(previous_agent, context)
                next_agent = transition_result.agent_state
                if transition_result.event_history is not None:
                    event_history = transition_result.event_history
                if transition_result.decision_history is not None:
                    decision_history = transition_result.decision_history
                if transition_result.consequence_runtime is not None:
                    consequence_runtime = transition_result.consequence_runtime
                states.append(
                    SimulationState(
                        week=week,
                        agent_state=next_agent,
                        events=transition_result.events,
                        event_traces=transition_result.event_traces,
                        decisions=transition_result.decisions,
                        consequences=transition_result.consequences,
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

        consequence_history = None
        pending_scheduled_effects = ()
        if consequence_runtime is not None:
            consequence_history = consequence_runtime.history
            pending_scheduled_effects = consequence_runtime.pending_scheduled_effects

        return SimulationResult(
            name=self._config.simulation.name,
            seed=self._config.simulation.seed,
            city_name=self._config.city.name,
            states=tuple(states),
            summaries=tuple(summaries),
            event_history=event_history,
            decision_history=decision_history,
            consequence_history=consequence_history,
            pending_scheduled_effects=pending_scheduled_effects,
        )
