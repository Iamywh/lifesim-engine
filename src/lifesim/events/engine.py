from __future__ import annotations

from dataclasses import dataclass

from lifesim.agents.state import AgentState
from lifesim.events.model import (
    EventCandidateTrace,
    EventCatalog,
    EventDefinition,
    EventHistory,
    EventOccurrence,
    EventSelectionDraw,
    EventSelectionResult,
    EventSelectionTrace,
    is_on_cooldown,
)
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult


class EventEngine:
    def __init__(self, catalog: EventCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> EventCatalog:
        return self._catalog

    def select_events(
        self,
        state: AgentState,
        context: WeeklyContext,
        history: EventHistory,
    ) -> EventSelectionResult:
        candidates: list[tuple[EventDefinition, float]] = []
        traces: list[EventCandidateTrace] = []

        for definition in self._catalog.definitions:
            if is_on_cooldown(definition, history, context.week):
                traces.append(_candidate_trace(definition, False, 0.0, "cooldown"))
                continue
            if not definition.is_conditionally_eligible(state, context):
                traces.append(_candidate_trace(definition, False, 0.0, "conditions"))
                continue
            weight = definition.effective_weight(state, context)
            if weight <= 0:
                traces.append(_candidate_trace(definition, False, weight, "zero_weight"))
                continue
            candidates.append((definition, weight))
            traces.append(_candidate_trace(definition, True, weight, "eligible"))

        trigger_roll = context.rng.random()
        occurrences: tuple[EventOccurrence, ...] = ()
        selection_draws: tuple[EventSelectionDraw, ...] = ()

        if candidates and trigger_roll < self._catalog.event_probability:
            occurrences, selection_draws = self._select_weighted(candidates, context)

        next_history = history.record(occurrences)
        trace = EventSelectionTrace(
            week=context.week,
            trigger_probability=self._catalog.event_probability,
            trigger_roll=trigger_roll,
            candidates=tuple(traces),
            selected_event_ids=tuple(occurrence.event_id for occurrence in occurrences),
            selection_draws=selection_draws,
        )
        return EventSelectionResult(
            occurrences=occurrences,
            history=next_history,
            trace=trace,
        )

    def _select_weighted(
        self,
        candidates: list[tuple[EventDefinition, float]],
        context: WeeklyContext,
    ) -> tuple[tuple[EventOccurrence, ...], tuple[EventSelectionDraw, ...]]:
        remaining = list(candidates)
        selected: list[EventOccurrence] = []
        draws: list[EventSelectionDraw] = []

        for slot in range(min(self._catalog.max_events_per_week, len(remaining))):
            total_weight = sum(weight for _, weight in remaining)
            if total_weight <= 0:
                break
            roll = context.rng.random() * total_weight
            cumulative = 0.0
            for index, (definition, weight) in enumerate(remaining):
                cumulative += weight
                if roll <= cumulative:
                    selected.append(
                        definition.to_occurrence(
                            week=context.week,
                            effective_weight=weight,
                        )
                    )
                    draws.append(
                        EventSelectionDraw(
                            slot=slot,
                            roll=roll,
                            total_weight=total_weight,
                            selected_event_id=definition.event_id,
                        )
                    )
                    remaining.pop(index)
                    break

        return tuple(selected), tuple(draws)


@dataclass(frozen=True, slots=True)
class EventEngineTransition:
    engine: EventEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        history = context.event_history
        if history is None:
            history = EventHistory()
        if not isinstance(history, EventHistory):
            raise TypeError("Expected WeeklyContext.event_history to contain EventHistory.")

        result = self.engine.select_events(state, context, history)
        return WeeklyTransitionResult(
            agent_state=state,
            events=result.occurrences,
            event_traces=(result.trace,),
            event_history=result.history,
        )


def _candidate_trace(
    definition: EventDefinition,
    eligible: bool,
    effective_weight: float,
    reason: str,
) -> EventCandidateTrace:
    return EventCandidateTrace(
        event_id=definition.event_id,
        eligible=eligible,
        effective_weight=effective_weight,
        reason=reason,
    )
