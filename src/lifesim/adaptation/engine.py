from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lifesim.adaptation.model import (
    PERSONALITY_TRAITS,
    AdaptationCatalog,
    AdaptationRuntimeState,
    AdaptationWeekRecord,
    BehaviorEvidenceRecord,
    HabitCandidateChange,
    HabitCandidateState,
    HabitStrengthChange,
    PersonalityTraitChange,
    RoutineStabilityRecord,
    TraitEvidenceAccumulator,
    TraitEvidenceRecord,
)
from lifesim.agents.state import AgentState, Habit, HabitsState
from lifesim.decisions.model import DecisionRecord
from lifesim.development.model import DevelopmentWeekRecord
from lifesim.employment.model import ApplicationStageRecord
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.learning.model import LearningRecord
from lifesim.passive.model import RoutineWeekRecord
from lifesim.social.model import SocialInteractionRecord
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult


class AdaptationEngine:
    def __init__(self, catalog: AdaptationCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> AdaptationCatalog:
        return self._catalog

    def adapt(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: AdaptationRuntimeState,
    ) -> tuple[AgentState, AdaptationRuntimeState, AdaptationWeekRecord]:
        if context.week in runtime.processed_weeks:
            raise ValueError(f"Adaptation already processed for week {context.week}.")
        anchor = runtime.personality_anchor or _personality_anchor(state)
        decisions = _decisions_by_id(context.decisions)
        events = _events_by_key(context.events)
        processed_behavior = list(runtime.processed_behavior_decision_ids)
        behavior = _behavior_evidence(context, decisions, events, set(processed_behavior))
        processed_behavior.extend(record.decision_id for record in behavior)
        existing_habits = {habit.habit_id: habit for habit in state.habits.items}
        candidate_changes, candidates, new_habits, habit_changes = _update_habits(
            state,
            context,
            self._catalog,
            runtime.habit_candidates,
            behavior,
            existing_habits,
            events,
        )
        routine_stability, routine_after = _routine_stability(
            state,
            context,
            self._catalog,
        )
        trait_evidence = _trait_evidence(
            context,
            self._catalog,
            behavior,
            decisions,
            events,
            set(runtime.processed_experience_ids),
        )
        accumulators = _update_accumulators(
            self._catalog,
            runtime.trait_accumulators,
            trait_evidence,
            context.week,
        )
        personality, personality_changes = _update_personality(
            state,
            self._catalog,
            anchor,
            accumulators,
        )
        habits = tuple(_merge_habits(state.habits.items, new_habits, habit_changes, context.week))
        next_state = replace(
            state,
            personality=personality,
            habits=HabitsState(routine_stability=routine_after, items=habits),
        )
        record = AdaptationWeekRecord(
            week=context.week,
            behavior_evidence=behavior,
            habit_candidate_changes=candidate_changes,
            habit_strength_changes=habit_changes,
            routine_stability=routine_stability,
            trait_evidence=trait_evidence,
            personality_changes=personality_changes,
        )
        processed_experience = tuple(
            record.consequence_id
            for record in context.learning_records
            if isinstance(record, LearningRecord)
            and record.consequence_id not in runtime.processed_experience_ids
        )
        runtime = replace(
            runtime,
            history=runtime.history.record(record),
            personality_anchor=anchor,
            habit_candidates=candidates,
            trait_accumulators=accumulators,
            processed_weeks=runtime.processed_weeks + (context.week,),
            processed_behavior_decision_ids=tuple(processed_behavior),
            processed_experience_ids=runtime.processed_experience_ids + processed_experience,
        )
        return next_state, runtime, record


@dataclass(frozen=True, slots=True)
class AdaptationTransition:
    engine: AdaptationEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.adapt(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            adaptation_records=(record,),
            adaptation_runtime=runtime,
        )


def _runtime(context: WeeklyContext) -> AdaptationRuntimeState:
    runtime = context.adaptation_runtime
    if runtime is None:
        return AdaptationRuntimeState()
    if not isinstance(runtime, AdaptationRuntimeState):
        raise TypeError("Expected WeeklyContext.adaptation_runtime to contain AdaptationRuntimeState.")
    return runtime


def _behavior_evidence(
    context: WeeklyContext,
    decisions: dict[str, DecisionRecord],
    events: dict[tuple[str, str], EventOccurrence],
    processed_behavior: set[str],
) -> tuple[BehaviorEvidenceRecord, ...]:
    records: list[BehaviorEvidenceRecord] = []
    seen = set(processed_behavior)
    for source_system, record in _executed_records(context):
        decision_id = getattr(record, "decision_id", "")
        if not decision_id or decision_id in seen:
            continue
        decision = decisions.get(decision_id)
        if decision is None or decision.chosen_option_id is None:
            raise ValueError("Executed behavior record must reference a real M4 decision.")
        if decision.week != context.week:
            raise ValueError("Executed behavior decision must belong to the current adaptation week.")
        event = events.get((decision.source_event_id, decision.source_event_version))
        if event is None:
            raise ValueError("Executed behavior decision must reference a same-week event.")
        if event.week != context.week:
            raise ValueError("Executed behavior event must belong to the current adaptation week.")
        option = _option(event, decision.chosen_option_id)
        if not option.behavior_tags:
            continue
        seen.add(decision_id)
        records.append(
            BehaviorEvidenceRecord(
                week=context.week,
                decision_id=decision_id,
                source_event_id=decision.source_event_id,
                source_event_version=decision.source_event_version,
                source_option_id=decision.chosen_option_id,
                source_system=source_system,
                behavior_tags=option.behavior_tags,
                executed=True,
                source_record_ids=(_source_record_id(source_system, record),),
            )
        )
    return tuple(records)


def _executed_records(context: WeeklyContext) -> tuple[tuple[str, Any], ...]:
    output = []
    output.extend(("routine", record) for record in context.passive_records if isinstance(record, RoutineWeekRecord))
    output.extend(("development", record) for record in context.development_records if isinstance(record, DevelopmentWeekRecord))
    output.extend(("social", record) for record in context.social_records if isinstance(record, SocialInteractionRecord) and record.decision_id)
    output.extend(
        ("employment", record)
        for record in context.employment_records
        if isinstance(record, ApplicationStageRecord) and record.decision_id
    )
    return tuple(output)


def _update_habits(
    state: AgentState,
    context: WeeklyContext,
    catalog: AdaptationCatalog,
    candidates: tuple[HabitCandidateState, ...],
    evidence: tuple[BehaviorEvidenceRecord, ...],
    existing_habits: dict[str, Habit],
    events: dict[tuple[str, str], EventOccurrence],
) -> tuple[
    tuple[HabitCandidateChange, ...],
    tuple[HabitCandidateState, ...],
    tuple[Habit, ...],
    tuple[HabitStrengthChange, ...],
]:
    evidence_by_habit = {
        definition.habit_id: tuple(
            record
            for record in evidence
            if set(record.behavior_tags).intersection(definition.behavior_tags)
        )
        for definition in catalog.habits
    }
    candidates_by_id = {candidate.habit_id: candidate for candidate in candidates}
    changes: list[HabitCandidateChange] = []
    next_candidates: list[HabitCandidateState] = []
    new_habits: list[Habit] = []
    habit_changes: list[HabitStrengthChange] = []
    offered_tags = _offered_behavior_tags(events)
    for definition in catalog.habits:
        matches = evidence_by_habit[definition.habit_id]
        candidate = candidates_by_id.get(definition.habit_id, HabitCandidateState(definition.habit_id))
        before = candidate.latent_strength
        reinforcing_weeks = candidate.reinforcing_weeks
        if matches and context.week not in reinforcing_weeks:
            gain = definition.formation_rate * (1.0 - before / 100.0)
            after = _clamp100(before + gain)
            reinforcing_weeks = reinforcing_weeks + (context.week,)
            candidate = HabitCandidateState(definition.habit_id, after, reinforcing_weeks, context.week)
            changes.append(HabitCandidateChange(definition.habit_id, before, after, len(reinforcing_weeks), "executed_behavior"))
        elif not matches and candidate.last_evidence_week and context.week - candidate.last_evidence_week > definition.grace_weeks:
            after = _clamp100(before - definition.nonuse_decay_rate * 0.35)
            candidate = replace(candidate, latent_strength=after)
            if after != before:
                changes.append(HabitCandidateChange(definition.habit_id, before, after, len(reinforcing_weeks), "latent_inactivity"))
        next_candidates.append(candidate)
        existing = existing_habits.get(definition.habit_id)
        if existing is None and len(reinforcing_weeks) >= definition.minimum_reinforcing_weeks and candidate.latent_strength >= definition.formation_threshold:
            strength = _clamp100(max(5.0, min(15.0, candidate.latent_strength * 0.75)))
            new_habits.append(
                Habit(
                    habit_id=definition.habit_id,
                    name=definition.name,
                    cadence=definition.cadence,
                    strength=strength,
                    behavior_tags=definition.behavior_tags,
                    formed_week=context.week,
                    last_reinforced_week=context.week,
                )
            )
            habit_changes.append(HabitStrengthChange(definition.habit_id, 0.0, strength, "formed", tuple(record.decision_id for record in matches)))
        elif existing is not None:
            if matches:
                after = _clamp100(existing.strength + definition.reinforcement_rate * (1.0 - existing.strength / 100.0))
                if after != existing.strength:
                    habit_changes.append(HabitStrengthChange(definition.habit_id, existing.strength, after, "reinforced", tuple(record.decision_id for record in matches)))
            elif set(definition.behavior_tags).intersection(offered_tags) and context.week - existing.last_reinforced_week > definition.grace_weeks:
                after = _clamp100(existing.strength - definition.nonuse_decay_rate)
                if after != existing.strength:
                    habit_changes.append(HabitStrengthChange(definition.habit_id, existing.strength, after, "observable_nonuse"))
    return tuple(changes), tuple(next_candidates), tuple(new_habits), tuple(habit_changes)


def _routine_stability(
    state: AgentState,
    context: WeeklyContext,
    catalog: AdaptationCatalog,
) -> tuple[RoutineStabilityRecord | None, float]:
    records = tuple(record for record in context.passive_records if isinstance(record, RoutineWeekRecord))
    if not records:
        return None, state.habits.routine_stability
    record = records[-1]
    before = state.habits.routine_stability
    if record.weeks_in_current_profile > 1:
        after = _clamp100(before + catalog.settings.routine_stability_gain * (1.0 - before / 100.0))
        reason = "same_profile_repeated"
    elif record.previous_profile_id:
        after = _clamp100(before - catalog.settings.routine_stability_switch_loss)
        reason = "profile_switched"
    else:
        after = before
        reason = "first_observed_profile"
    return RoutineStabilityRecord(before, after, record.profile_id, record.previous_profile_id, reason), after


def _trait_evidence(
    context: WeeklyContext,
    catalog: AdaptationCatalog,
    behavior: tuple[BehaviorEvidenceRecord, ...],
    decisions: dict[str, DecisionRecord],
    events: dict[tuple[str, str], EventOccurrence],
    processed_experience: set[str],
) -> tuple[TraitEvidenceRecord, ...]:
    records: list[TraitEvidenceRecord] = []
    for evidence in behavior:
        for tag in evidence.behavior_tags:
            records.extend(_mapped_records(context.week, catalog, "behavior_tag", tag, evidence.decision_id, evidence.source_system, 0.35, 1.0))
        decision = decisions[evidence.decision_id]
        event = events[(decision.source_event_id, decision.source_event_version)]
        records.extend(_choice_metric_evidence(context.week, catalog, decision, event))
    for record in context.learning_records:
        if not isinstance(record, LearningRecord) or record.consequence_id in processed_experience:
            continue
        evaluation = record.evaluation
        if evaluation.salience < 0.35:
            continue
        key = "positive_salient" if evaluation.valence > 0 else "negative_salient"
        records.extend(_mapped_records(context.week, catalog, "experienced_outcome", key, record.consequence_id, "learning", abs(evaluation.valence), evaluation.salience))
    return tuple(records)


def _choice_metric_evidence(
    week: int,
    catalog: AdaptationCatalog,
    decision: DecisionRecord,
    event: EventOccurrence,
) -> tuple[TraitEvidenceRecord, ...]:
    available = [option for option in event.options if option.option_id in decision.available_option_ids]
    chosen = next((option for option in available if option.option_id == decision.chosen_option_id), None)
    if chosen is None or len(available) < 2:
        return ()
    metrics = {
        "risk_preference": (chosen.perceived_risk, [option.perceived_risk for option in available]),
        "future_orientation": (chosen.future_value, [option.future_value for option in available]),
        "short_term_orientation": (chosen.short_term_value, [option.short_term_value for option in available]),
        "cost_restraint": (-float(chosen.estimated_cost), [-float(option.estimated_cost) for option in available]),
        "learning_orientation": (chosen.learning_value, [option.learning_value for option in available]),
        "social_orientation": (chosen.social_value, [option.social_value for option in available]),
        "autonomy_orientation": (chosen.autonomy_value, [option.autonomy_value for option in available]),
        "goal_alignment": (_option_evaluation_signal(decision, chosen.option_id, "goal_alignment"), [_option_evaluation_signal(decision, option.option_id, "goal_alignment") for option in available]),
    }
    records: list[TraitEvidenceRecord] = []
    for key, (chosen_value, values) in metrics.items():
        variance = max(values) - min(values)
        if variance < 0.05:
            continue
        mean = sum(values) / len(values)
        signal = max(-1.0, min(1.0, (chosen_value - mean) / max(variance, 0.000001)))
        if abs(signal) < 0.05:
            continue
        records.extend(_mapped_records(week, catalog, "choice_metric", key, decision.decision_id, "decision", signal, min(1.0, variance)))
    return tuple(records)


def _mapped_records(
    week: int,
    catalog: AdaptationCatalog,
    evidence_type: str,
    key: str,
    source_id: str,
    source_type: str,
    signal: float,
    weight: float,
) -> tuple[TraitEvidenceRecord, ...]:
    return tuple(
        TraitEvidenceRecord(
            week=week,
            source_id=source_id,
            source_type=source_type,
            evidence_type=evidence_type,
            evidence_key=key,
            trait=mapping.trait,
            signal=max(-1.0, min(1.0, signal * (1.0 if mapping.coefficient > 0 else -1.0))),
            weight=abs(mapping.coefficient) * weight,
        )
        for mapping in catalog.trait_mappings
        if mapping.evidence_type == evidence_type and mapping.evidence_key == key
    )


def _update_accumulators(
    catalog: AdaptationCatalog,
    accumulators: tuple[TraitEvidenceAccumulator, ...],
    evidence: tuple[TraitEvidenceRecord, ...],
    week: int,
) -> tuple[TraitEvidenceAccumulator, ...]:
    by_trait = {item.trait: item for item in accumulators}
    grouped = {trait: [] for trait in PERSONALITY_TRAITS}
    for record in evidence:
        grouped[record.trait].append(record)
    output = []
    for trait in PERSONALITY_TRAITS:
        current = by_trait.get(trait, TraitEvidenceAccumulator(trait))
        gap = max(0, week - current.last_evidence_week) if current.last_evidence_week else 0
        decay = catalog.settings.evidence_decay_rate ** gap
        signed = current.signed_evidence * decay
        weight = current.evidence_weight * decay
        weeks = set(current.distinct_weeks)
        source_types = set(current.source_types)
        for record in grouped[trait]:
            signed += record.signal * record.weight
            weight += record.weight
            weeks.add(record.week)
            source_types.add(record.source_type)
        last = week if grouped[trait] else current.last_evidence_week
        output.append(TraitEvidenceAccumulator(trait, signed, weight, tuple(sorted(weeks)), tuple(sorted(source_types)), last))
    return tuple(output)


def _update_personality(
    state: AgentState,
    catalog: AdaptationCatalog,
    anchor: dict[str, float],
    accumulators: tuple[TraitEvidenceAccumulator, ...],
):
    values = state.personality.to_dict()
    changes: list[PersonalityTraitChange] = []
    for accumulator in accumulators:
        before = values[accumulator.trait]
        if accumulator.evidence_weight <= 0.0 or len(accumulator.distinct_weeks) < 3:
            continue
        signal = max(-1.0, min(1.0, accumulator.signed_evidence / accumulator.evidence_weight))
        diversity = min(1.0, len(accumulator.source_types) / 3.0)
        confidence = min(1.0, accumulator.evidence_weight / 4.0) * min(1.0, len(accumulator.distinct_weeks) / 8.0) * (0.55 + diversity * 0.45)
        target = max(0.0, min(1.0, anchor[accumulator.trait] + signal * catalog.settings.max_trait_anchor_shift))
        raw_delta = (target - before) * catalog.settings.trait_adaptation_rate * confidence
        cap = catalog.settings.max_weekly_trait_delta
        delta = max(-cap, min(cap, raw_delta))
        after = max(0.0, min(1.0, round(before + delta, 12)))
        values[accumulator.trait] = after
        changes.append(
            PersonalityTraitChange(
                trait=accumulator.trait,
                anchor=anchor[accumulator.trait],
                before=before,
                target=target,
                confidence=confidence,
                delta=after - before,
                after=after,
                capped=abs(raw_delta) > cap,
                reason="accumulated_evidence" if after != before else "below_movement_precision",
            )
        )
    return type(state.personality)(**values), tuple(changes)


def _merge_habits(
    existing: tuple[Habit, ...],
    new_habits: tuple[Habit, ...],
    changes: tuple[HabitStrengthChange, ...],
    week: int,
) -> tuple[Habit, ...]:
    change_by_id = {change.habit_id: change for change in changes}
    result = []
    for habit in existing:
        change = change_by_id.get(habit.habit_id)
        if change is None or change.reason == "formed":
            result.append(habit)
        else:
            last_reinforced = week if change.reason == "reinforced" else habit.last_reinforced_week
            result.append(
                replace(
                    habit,
                    strength=change.after,
                    last_reinforced_week=last_reinforced,
                )
            )
    existing_ids = {habit.habit_id for habit in existing}
    result.extend(habit for habit in new_habits if habit.habit_id not in existing_ids)
    return tuple(result)


def _personality_anchor(state: AgentState) -> dict[str, float]:
    values = state.personality.to_dict()
    return {trait: values[trait] for trait in PERSONALITY_TRAITS}


def _decisions_by_id(decisions: tuple[Any, ...]) -> dict[str, DecisionRecord]:
    records = tuple(decision for decision in decisions if isinstance(decision, DecisionRecord))
    ids = tuple(decision.decision_id for decision in records)
    if len(ids) != len(set(ids)):
        raise ValueError("Expected adaptation context decisions to have unique decision_id values.")
    return {decision.decision_id: decision for decision in records}


def _events_by_key(events: tuple[Any, ...]) -> dict[tuple[str, str], EventOccurrence]:
    records = tuple(event for event in events if isinstance(event, EventOccurrence))
    keys = tuple((event.event_id, event.version) for event in records)
    if len(keys) != len(set(keys)):
        raise ValueError("Expected adaptation context events to have unique event/version keys.")
    return {(event.event_id, event.version): event for event in records}


def _option(event: EventOccurrence, option_id: str) -> EventOption:
    for option in event.options:
        if option.option_id == option_id:
            return option
    raise ValueError("Executed decision selected an option not present on its event.")


def _source_record_id(source_system: str, record: Any) -> str:
    return f"{source_system}:{getattr(record, 'week', 0)}:{getattr(record, 'decision_id', '')}:{getattr(record, 'profile_id', getattr(record, 'stage', getattr(record, 'interaction_type', 'record')))}"


def _offered_behavior_tags(events: dict[tuple[str, str], EventOccurrence]) -> set[str]:
    return {tag for event in events.values() for option in event.options for tag in option.behavior_tags}


def _option_evaluation_signal(decision: DecisionRecord, option_id: str, component_name: str) -> float:
    evaluation = next(item for item in decision.evaluations if item.option_id == option_id)
    component = next((item for item in evaluation.components if item.name == component_name), None)
    return 0.0 if component is None else component.signal


def _clamp100(value: float) -> float:
    return max(0.0, min(100.0, round(value, 6)))
