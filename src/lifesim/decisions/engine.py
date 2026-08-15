from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from random import Random

from lifesim.agents.state import AgentState, GoalItem
from lifesim.decisions.model import (
    DecisionHistory,
    DecisionRecord,
    DecisionScoreComponent,
    DecisionSelectionResult,
    OptionEvaluation,
)
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.learning.retrieval import retrieve_memory_signal
from lifesim.rng import derive_stable_seed
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult


class DecisionEngine:
    def decide_events(
        self,
        state: AgentState,
        context: WeeklyContext,
        events: tuple[EventOccurrence, ...],
        history: DecisionHistory,
    ) -> DecisionSelectionResult:
        decided_keys = {
            (decision.week, decision.source_event_id, decision.source_event_version)
            for decision in history.records
        } | {
            (decision.week, decision.source_event_id, decision.source_event_version)
            for decision in context.decisions
            if isinstance(decision, DecisionRecord)
        }
        records = tuple(
            self.decide_event(state, context, event)
            for event in events
            if event.options
            and (context.week, event.event_id, event.version) not in decided_keys
        )
        return DecisionSelectionResult(records=records, history=history.record(records))

    def decide_event(
        self,
        state: AgentState,
        context: WeeklyContext,
        event: EventOccurrence,
    ) -> DecisionRecord:
        if event.week != context.week:
            raise ValueError("Expected event week to match WeeklyContext.week.")
        evaluations = tuple(
            self._evaluate_option(state, context, event, option)
            for option in event.options
        )
        available = tuple(evaluation for evaluation in evaluations if evaluation.available)
        chosen = _choose(available)
        chosen_components = chosen.components if chosen is not None else ()

        return DecisionRecord(
            decision_id=_stable_id(
                "decision",
                str(context.config.simulation.seed),
                state.identity.agent_id,
                str(context.week),
                event.event_id,
                event.version,
            ),
            agent_id=state.identity.agent_id,
            week=context.week,
            source_event_id=event.event_id,
            source_event_version=event.version,
            time_pressure=event.time_pressure,
            available_option_ids=tuple(evaluation.option_id for evaluation in available),
            unavailable_option_ids=tuple(
                evaluation.option_id for evaluation in evaluations if not evaluation.available
            ),
            chosen_option_id=chosen.option_id if chosen is not None else None,
            evaluations=evaluations,
            strongest_positive_factors=_strongest(chosen_components, positive=True),
            strongest_negative_factors=_strongest(chosen_components, positive=False),
        )

    def _evaluate_option(
        self,
        state: AgentState,
        context: WeeklyContext,
        event: EventOccurrence,
        option: EventOption,
    ) -> OptionEvaluation:
        unavailable_reason = _unavailable_reason(state, context, option)
        if unavailable_reason:
            return OptionEvaluation(
                option_id=option.option_id,
                available=False,
                unavailable_reason=unavailable_reason,
                deterministic_score=None,
                controlled_noise=None,
                final_score=None,
                components=(),
            )

        memory = retrieve_memory_signal(state, context.week, event, option)
        components = _score_components(state, context, event, option, memory.signal)
        deterministic_score = round(sum(component.contribution for component in components), 12)
        noise = _decision_noise(state, context, event, option)
        return OptionEvaluation(
            option_id=option.option_id,
            available=True,
            unavailable_reason="",
            deterministic_score=deterministic_score,
            controlled_noise=noise,
            final_score=round(deterministic_score + noise, 12),
            components=components,
            memory_evidence=memory.evidence,
        )


@dataclass(frozen=True, slots=True)
class DecisionEngineTransition:
    engine: DecisionEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        history = context.decision_history
        if history is None:
            history = DecisionHistory()
        if not isinstance(history, DecisionHistory):
            raise TypeError("Expected WeeklyContext.decision_history to contain DecisionHistory.")

        result = self.engine.decide_events(state, context, context.events, history)
        return WeeklyTransitionResult(
            agent_state=state,
            decisions=result.records,
            decision_history=result.history,
        )


def _unavailable_reason(
    state: AgentState,
    context: WeeklyContext,
    option: EventOption,
) -> str:
    for condition in option.availability_conditions:
        if not condition.evaluate(state, context):
            return "availability_conditions"
    if (
        option.requires_full_estimated_cost
        and option.estimated_cost > _liquid_resources(state)
        and option.estimated_cost > Decimal(0)
    ):
        return "insufficient_liquid_resources"
    return ""


def _score_components(
    state: AgentState,
    context: WeeklyContext,
    event: EventOccurrence,
    option: EventOption,
    memory_signal: float,
) -> tuple[DecisionScoreComponent, ...]:
    personality = state.personality
    mental = state.mental
    health = state.health
    needs = state.needs
    time_pressure = event.time_pressure

    cost_ratio = _cost_ratio(option.estimated_cost, _liquid_resources(state))
    energy_ratio = option.energy_cost / 100.0
    time_ratio = option.time_cost_hours / 168.0
    goal_signal = _goal_alignment_signal(state, option)

    specs = (
        (
            "short_term_value",
            option.short_term_value,
            1.0 + personality.impulsivity * 0.7 + mental.stress / 100.0 * 0.3 + time_pressure * 0.4,
        ),
        (
            "future_value",
            option.future_value,
            max(
                0.2,
                0.8
                + personality.patience * 0.6
                + personality.discipline * 0.4
                + personality.conscientiousness * 0.4
                - time_pressure * 0.3,
            ),
        ),
        ("financial_cost", -cost_ratio, 0.5 + personality.frugality * 2.0),
        (
            "energy_cost",
            -energy_ratio,
            0.5 + (1.0 - health.energy / 100.0) * 1.5 + mental.recovery_need / 100.0 * 0.5,
        ),
        ("time_cost", -time_ratio, 0.4 + personality.conscientiousness * 0.5),
        ("perceived_risk", -option.perceived_risk, 1.2 - personality.risk_tolerance * 0.9),
        (
            "social_value",
            option.social_value,
            0.5
            + personality.social_need * 0.8
            + mental.loneliness / 100.0 * 0.5
            + (1.0 - needs.belonging / 100.0) * 0.4,
        ),
        (
            "autonomy_value",
            option.autonomy_value,
            0.5 + personality.independence * 0.8 + (1.0 - needs.autonomy / 100.0) * 0.3,
        ),
        ("learning_value", option.learning_value, 0.4 + personality.curiosity * 0.8),
        ("health_value", option.health_value, 0.5 + (1.0 - health.physical_health / 100.0) * 0.3),
        (
            "comfort_value",
            option.comfort_value,
            0.4 + mental.recovery_need / 100.0 * 0.7 + (1.0 - health.energy / 100.0) * 0.5,
        ),
        (
            "goal_alignment",
            goal_signal,
            0.25 + personality.discipline * 0.35 + personality.conscientiousness * 0.25,
        ),
        (
            "uncertainty",
            -option.uncertainty,
            max(
                0.05,
                0.4 + time_pressure * 0.3 - personality.confidence * 0.25 - personality.adaptability * 0.25,
            ),
        ),
        (
            "social_pressure",
            option.social_pressure,
            0.2 + personality.social_need * 0.3 - personality.independence * 0.4,
        ),
        (
            "memory_experience",
            memory_signal,
            0.45 + personality.conscientiousness * 0.15 + min(0.15, context.week * 0.005),
        ),
    )
    return tuple(
        DecisionScoreComponent(
            name=name,
            signal=round(signal, 12),
            weight=round(weight, 12),
            contribution=round(signal * weight, 12),
        )
        for name, signal, weight in specs
    )


def _decision_noise(
    state: AgentState,
    context: WeeklyContext,
    event: EventOccurrence,
    option: EventOption,
) -> float:
    personality = state.personality
    mental = state.mental
    magnitude = (
        0.03
        + personality.impulsivity * 0.05
        + mental.stress / 100.0 * 0.04
        + mental.mental_load / 100.0 * 0.03
        + event.time_pressure * 0.05
        - personality.conscientiousness * 0.04
        - personality.patience * 0.03
    )
    magnitude = max(0.0, min(0.12, magnitude))
    rng = Random(
        derive_stable_seed(
            "decision-noise",
            str(context.config.simulation.seed),
            state.identity.agent_id,
            str(context.week),
            event.event_id,
            event.version,
            option.option_id,
        )
    )
    return round((rng.random() * 2.0 - 1.0) * magnitude, 12)


def _choose(evaluations: tuple[OptionEvaluation, ...]) -> OptionEvaluation | None:
    if not evaluations:
        return None
    return max(evaluations, key=lambda evaluation: (evaluation.final_score, evaluation.option_id))


def _stable_id(*parts: str) -> str:
    return f"decision_{hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16]}"


def _liquid_resources(state: AgentState) -> Decimal:
    financial = state.financial
    return financial.cash + financial.bank_balance + financial.savings + financial.emergency_fund


def _cost_ratio(cost: Decimal, liquid_resources: Decimal) -> float:
    """Normalize perceived cost against liquid resources, capping outliers.

    Monetary state stays exact in Decimal. Only this derived ratio enters the
    float utility domain. If liquid resources are zero, any positive perceived
    cost receives the maximum normalized pressure.
    """
    if cost == Decimal(0):
        return 0.0
    if liquid_resources <= Decimal(0):
        return 1.5
    return min(1.5, float(cost / liquid_resources))


def _goal_alignment_signal(state: AgentState, option: EventOption) -> float:
    option_tags = set(option.goal_tags)
    if not option_tags:
        return 0.0
    goals = (
        state.goals.short_term
        + state.goals.medium_term
        + state.goals.long_term
    )
    score = sum(goal.priority / 5.0 for goal in goals if _goal_matches(goal, option_tags))
    return min(1.0, score / 3.0)


def _goal_matches(goal: GoalItem, option_tags: set[str]) -> bool:
    return bool(set(goal.tags) & option_tags)


def _strongest(
    components: tuple[DecisionScoreComponent, ...],
    *,
    positive: bool,
) -> tuple[str, ...]:
    if positive:
        matches = [component for component in components if component.contribution > 0]
        ordered = sorted(matches, key=lambda component: component.contribution, reverse=True)
    else:
        matches = [component for component in components if component.contribution < 0]
        ordered = sorted(matches, key=lambda component: component.contribution)
    return tuple(component.name for component in ordered[:3])
