from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from random import Random
from typing import Any, Protocol

from lifesim.agents.state import AgentState
from lifesim.config import LifeSimConfig


@dataclass(frozen=True, slots=True)
class WeeklyContext:
    week: int
    config: LifeSimConfig
    rng: Random
    events: tuple[Any, ...] = ()
    decisions: tuple[Any, ...] = ()
    consequences: tuple[Any, ...] = ()
    learning_records: tuple[Any, ...] = ()
    passive_records: tuple[Any, ...] = ()
    employment_records: tuple[Any, ...] = ()
    event_history: Any | None = None
    decision_history: Any | None = None
    consequence_runtime: Any | None = None
    learning_runtime: Any | None = None
    passive_runtime: Any | None = None
    employment_runtime: Any | None = None

    def __post_init__(self) -> None:
        if self.week < 1:
            raise ValueError("Expected weekly transition context week to be >= 1.")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "consequences", tuple(self.consequences))
        object.__setattr__(self, "learning_records", tuple(self.learning_records))
        object.__setattr__(self, "passive_records", tuple(self.passive_records))
        object.__setattr__(self, "employment_records", tuple(self.employment_records))

    @property
    def week_start(self) -> date:
        return self.config.simulation.start_date + timedelta(weeks=self.week - 1)

    @property
    def week_end(self) -> date:
        return self.week_start + timedelta(days=6)


class WeeklyTransition(Protocol):
    def apply(
        self,
        state: AgentState,
        context: WeeklyContext,
    ) -> AgentState | WeeklyTransitionResult:
        """Return the next immutable agent state for the supplied week.

        Implementations must not retain run-specific mutable state between
        runs. Persistent simulation state belongs in AgentState or in explicit
        run context objects so repeated LifeSimEngine.run() calls remain
        deterministic.
        """


@dataclass(frozen=True, slots=True)
class WeeklyTransitionResult:
    agent_state: AgentState
    events: tuple[Any, ...] = ()
    event_traces: tuple[Any, ...] = ()
    decisions: tuple[Any, ...] = ()
    consequences: tuple[Any, ...] = ()
    learning_records: tuple[Any, ...] = ()
    passive_records: tuple[Any, ...] = ()
    employment_records: tuple[Any, ...] = ()
    event_history: Any | None = None
    decision_history: Any | None = None
    consequence_runtime: Any | None = None
    learning_runtime: Any | None = None
    passive_runtime: Any | None = None
    employment_runtime: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "event_traces", tuple(self.event_traces))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "consequences", tuple(self.consequences))
        object.__setattr__(self, "learning_records", tuple(self.learning_records))
        object.__setattr__(self, "passive_records", tuple(self.passive_records))
        object.__setattr__(self, "employment_records", tuple(self.employment_records))


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

    def advance(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        next_state = state
        events: list[Any] = []
        event_traces: list[Any] = []
        decisions: list[Any] = []
        consequences: list[Any] = []
        learning_records: list[Any] = []
        passive_records: list[Any] = []
        employment_records: list[Any] = []
        event_history = context.event_history
        decision_history = context.decision_history
        consequence_runtime = context.consequence_runtime
        learning_runtime = context.learning_runtime
        passive_runtime = context.passive_runtime
        employment_runtime = context.employment_runtime

        for transition in self._transitions:
            candidate = transition.apply(next_state, context)
            if isinstance(candidate, WeeklyTransitionResult):
                result = candidate
            else:
                result = WeeklyTransitionResult(agent_state=candidate)

            candidate_state = result.agent_state
            if not isinstance(candidate_state, AgentState):
                raise TypeError("Expected weekly transition to return AgentState.")
            if candidate_state.identity.agent_id != state.identity.agent_id:
                raise ValueError("Weekly transitions must preserve agent identity in M2.")
            next_state = candidate_state
            events.extend(result.events)
            event_traces.extend(result.event_traces)
            decisions.extend(result.decisions)
            consequences.extend(result.consequences)
            learning_records.extend(result.learning_records)
            passive_records.extend(result.passive_records)
            employment_records.extend(result.employment_records)
            if result.events:
                context = replace(context, events=tuple(events))
            if result.decisions:
                context = replace(context, decisions=tuple(decisions))
            if result.consequences:
                context = replace(context, consequences=tuple(consequences))
            if result.learning_records:
                context = replace(context, learning_records=tuple(learning_records))
            if result.passive_records:
                context = replace(context, passive_records=tuple(passive_records))
            if result.employment_records:
                context = replace(context, employment_records=tuple(employment_records))
            if result.event_history is not None:
                event_history = result.event_history
                context = replace(context, event_history=event_history)
            if result.decision_history is not None:
                decision_history = result.decision_history
                context = replace(context, decision_history=decision_history)
            if result.consequence_runtime is not None:
                consequence_runtime = result.consequence_runtime
                context = replace(context, consequence_runtime=consequence_runtime)
            if result.learning_runtime is not None:
                learning_runtime = result.learning_runtime
                context = replace(context, learning_runtime=learning_runtime)
            if result.passive_runtime is not None:
                passive_runtime = result.passive_runtime
                context = replace(context, passive_runtime=passive_runtime)
            if result.employment_runtime is not None:
                employment_runtime = result.employment_runtime
                context = replace(context, employment_runtime=employment_runtime)

        if next_state is state:
            next_state = replace(state)
        return WeeklyTransitionResult(
            agent_state=next_state,
            events=tuple(events),
            event_traces=tuple(event_traces),
            decisions=tuple(decisions),
            consequences=tuple(consequences),
            learning_records=tuple(learning_records),
            passive_records=tuple(passive_records),
            employment_records=tuple(employment_records),
            event_history=event_history,
            decision_history=decision_history,
            consequence_runtime=consequence_runtime,
            learning_runtime=learning_runtime,
            passive_runtime=passive_runtime,
            employment_runtime=employment_runtime,
        )
