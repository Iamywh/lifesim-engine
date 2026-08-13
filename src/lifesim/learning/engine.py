from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from lifesim.agents.state import (
    AgentState,
    EpisodicMemory,
    LessonMemory,
    MemoryState,
    MistakeMemory,
    SuccessfulPatternMemory,
)
from lifesim.consequences.model import ConsequenceRecord, EffectApplication
from lifesim.decisions.model import DecisionHistory, DecisionRecord
from lifesim.events.model import EventHistory, EventOccurrence
from lifesim.learning.model import (
    ExperienceEvaluation,
    LearningRecord,
    LearningRuntimeState,
    MemoryUpdate,
    effective_memory_strength,
)
from lifesim.rng import derive_stable_seed
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

EPISODE_SALIENCE_THRESHOLD = 0.05
PATTERN_MIN_DISTINCT_EPISODES = 2
PATTERN_VALENCE_THRESHOLD = 0.2

POSITIVE_INCREASE_PATHS = frozenset(
    {
        "financial.cash",
        "financial.bank_balance",
        "financial.savings",
        "financial.emergency_fund",
        "health.physical_health",
        "health.energy",
        "health.mobility",
        "mental.mood",
        "needs.housing_security",
        "needs.food_security",
        "needs.safety",
        "needs.belonging",
        "needs.autonomy",
        "needs.purpose",
        "education.progress",
        "social.support_network_strength",
        "social.city_familiarity",
    }
)
NEGATIVE_INCREASE_PATHS = frozenset(
    {
        "health.sleep_debt",
        "mental.stress",
        "mental.mental_load",
        "mental.recovery_need",
        "mental.loneliness",
    }
)
DOMAIN_BY_PATH_PREFIX = {
    "financial.": "financial",
    "health.": "physical",
    "mental.": "mental",
    "needs.belonging": "social",
    "education.": "education",
    "social.": "social",
}


class LearningEngine:
    """Turns experienced consequences into psychological memory.

    This engine reads objective `ConsequenceRecord` values and returns a new
    `AgentState` whose only changed section is `memory`.
    """

    def learn_from_consequences(
        self,
        state: AgentState,
        context: WeeklyContext,
        records: tuple[ConsequenceRecord, ...],
        runtime: LearningRuntimeState,
    ) -> tuple[AgentState, LearningRuntimeState, tuple[LearningRecord, ...]]:
        memory = state.memory
        processed = list(runtime.processed_consequence_ids)
        learning_records: list[LearningRecord] = []

        for record in records:
            if record.week_resolved != context.week:
                raise ValueError("Expected consequence record week_resolved to match WeeklyContext.week.")
            if record.consequence_id in processed:
                continue
            evaluation = evaluate_experience(record)
            if evaluation.salience < EPISODE_SALIENCE_THRESHOLD:
                learning_record = LearningRecord(
                    consequence_id=record.consequence_id,
                    source_decision_id=record.source_decision_id,
                    source_event_id=record.source_event_id,
                    source_event_version=record.source_event_version,
                    source_option_id=record.chosen_option_id,
                    week_learned=context.week,
                    evaluation=evaluation,
                    updates=(),
                )
                processed.append(record.consequence_id)
                learning_records.append(learning_record)
                continue

            memory, updates = _apply_memory_update(memory, context, record, evaluation)
            learning_record = LearningRecord(
                consequence_id=record.consequence_id,
                source_decision_id=record.source_decision_id,
                source_event_id=record.source_event_id,
                source_event_version=record.source_event_version,
                source_option_id=record.chosen_option_id,
                week_learned=context.week,
                evaluation=evaluation,
                updates=updates,
            )
            processed.append(record.consequence_id)
            learning_records.append(learning_record)

        next_state = replace(state, memory=memory)
        _assert_only_memory_changed(state, next_state)
        history = runtime.history.record(tuple(learning_records))
        next_runtime = LearningRuntimeState(
            history=history,
            processed_consequence_ids=tuple(processed),
        )
        return next_state, next_runtime, tuple(learning_records)


@dataclass(frozen=True, slots=True)
class LearningTransition:
    engine: LearningEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, next_runtime, records = self.engine.learn_from_consequences(
            state,
            context,
            _consequence_records(context.consequences),
            runtime,
        )
        return WeeklyTransitionResult(
            agent_state=next_state,
            learning_records=records,
            learning_runtime=next_runtime,
        )


def evaluate_experience(record: ConsequenceRecord) -> ExperienceEvaluation:
    scored: list[tuple[str, str, float]] = []
    for application in record.effect_applications:
        if application.skipped:
            continue
        score = _score_application(application)
        if score == 0.0:
            continue
        scored.append((_domain(application.path), application.path, score))

    if not scored:
        return ExperienceEvaluation(
            valence=0.0,
            salience=0.0,
            affected_domains=(),
            strongest_positive_effects=(),
            strongest_negative_effects=(),
        )

    total_magnitude = sum(abs(score) for _, _, score in scored)
    valence = _clamp(sum(score for _, _, score in scored) / total_magnitude, -1.0, 1.0)
    domains = tuple(sorted({domain for domain, _, _ in scored}))
    salience = _clamp(total_magnitude / 2.0 + len(domains) * 0.05, 0.0, 1.0)
    positive = tuple(
        path
        for _, path, score in sorted(
            (item for item in scored if item[2] > 0.0),
            key=lambda item: item[2],
            reverse=True,
        )[:3]
    )
    negative = tuple(
        path
        for _, path, score in sorted(
            (item for item in scored if item[2] < 0.0),
            key=lambda item: item[2],
        )[:3]
    )
    return ExperienceEvaluation(
        valence=round(valence, 12),
        salience=round(salience, 12),
        affected_domains=domains,
        strongest_positive_effects=positive,
        strongest_negative_effects=negative,
    )


def _apply_memory_update(
    memory: MemoryState,
    context: WeeklyContext,
    record: ConsequenceRecord,
    evaluation: ExperienceEvaluation,
) -> tuple[MemoryState, tuple[MemoryUpdate, ...]]:
    tags = _event_tags(context.events, record)
    if not tags:
        tags = _historical_event_tags(context, record)
    episode, episode_update = _upsert_episode(memory, context.week, record, evaluation, tags)
    episodes = _replace_episode(memory.episodic_memories, episode)
    memory = replace(memory, episodic_memories=episodes)
    pattern_memory, pattern_updates = _upsert_patterns(memory, context.week, record)
    return pattern_memory, (episode_update,) + pattern_updates


def _upsert_episode(
    memory: MemoryState,
    week: int,
    record: ConsequenceRecord,
    evaluation: ExperienceEvaluation,
    tags: tuple[str, ...],
) -> tuple[EpisodicMemory, MemoryUpdate]:
    existing = _find_episode(memory, record.source_decision_id)
    source_ids = _source_ids(
        existing.source_consequence_ids if existing is not None else (),
        record.consequence_id,
    )
    if existing is None:
        memory_id = _memory_id("episode", record.source_decision_id)
        episode = EpisodicMemory(
            summary=f"Experienced {record.source_event_id}/{record.chosen_option_id}.",
            week=week,
            emotional_weight=round(evaluation.salience * 100.0, 12),
            memory_id=memory_id,
            source_decision_id=record.source_decision_id,
            source_event_id=record.source_event_id,
            source_event_version=record.source_event_version,
            source_option_id=record.chosen_option_id,
            first_week=week,
            last_reinforced_week=week,
            exposure_count=1,
            salience=evaluation.salience,
            strength=evaluation.salience,
            valence=evaluation.valence,
            tags=tags,
            affected_domains=evaluation.affected_domains,
            source_consequence_ids=source_ids,
        )
        return episode, MemoryUpdate(
            update_type="created",
            memory_kind="episodic",
            memory_id=episode.memory_id,
            before_strength=None,
            after_strength=episode.strength,
            before_valence=None,
            after_valence=episode.valence,
            before_exposure_count=0,
            after_exposure_count=episode.exposure_count,
            source_consequence_ids=source_ids,
        )

    before_strength = effective_memory_strength(existing.strength, existing.last_reinforced_week, week)
    after_strength = _clamp(before_strength + evaluation.salience * (1.0 - before_strength) * 0.75, 0.0, 1.0)
    before_count = existing.exposure_count
    after_count = before_count + 1
    after_valence = (
        (existing.valence * before_count) + evaluation.valence
    ) / after_count
    episode = replace(
        existing,
        emotional_weight=round(max(existing.emotional_weight, evaluation.salience * 100.0), 12),
        last_reinforced_week=week,
        exposure_count=after_count,
        salience=max(existing.salience, evaluation.salience),
        strength=round(after_strength, 12),
        valence=round(after_valence, 12),
        tags=_source_ids(existing.tags, *tags),
        affected_domains=_source_ids(existing.affected_domains, *evaluation.affected_domains),
        source_consequence_ids=source_ids,
    )
    return episode, MemoryUpdate(
        update_type="reinforced",
        memory_kind="episodic",
        memory_id=episode.memory_id,
        before_strength=before_strength,
        after_strength=episode.strength,
        before_valence=existing.valence,
        after_valence=episode.valence,
        before_exposure_count=before_count,
        after_exposure_count=after_count,
        source_consequence_ids=source_ids,
    )


def _upsert_patterns(
    memory: MemoryState,
    week: int,
    record: ConsequenceRecord,
) -> tuple[MemoryState, tuple[MemoryUpdate, ...]]:
    episodes = tuple(
        episode
        for episode in memory.episodic_memories
        if episode.source_event_id == record.source_event_id
        and episode.source_event_version == record.source_event_version
        and episode.source_option_id == record.chosen_option_id
    )
    if len(episodes) < PATTERN_MIN_DISTINCT_EPISODES:
        return memory, ()

    effective_strengths = tuple(
        effective_memory_strength(episode.strength, episode.last_reinforced_week, week)
        for episode in episodes
    )
    weighted_strength = sum(effective_strengths)
    if weighted_strength <= 0.0:
        return memory, ()
    valence = sum(
        episode.valence * effective_strength
        for episode, effective_strength in zip(episodes, effective_strengths, strict=True)
    ) / weighted_strength
    salience = max(episode.salience for episode in episodes)
    strength = _clamp(weighted_strength / len(episodes), 0.0, 1.0)
    if abs(valence) < PATTERN_VALENCE_THRESHOLD:
        return memory, ()

    tags = _source_ids(*(episode.tags for episode in episodes))
    domains = _source_ids(*(episode.affected_domains for episode in episodes))
    source_ids = _source_ids(*(episode.source_consequence_ids for episode in episodes))
    if valence < 0.0:
        memory, updates = _upsert_negative_pattern(
            memory,
            week,
            record,
            valence,
            salience,
            strength,
            tags,
            domains,
            source_ids,
            len(episodes),
        )
    else:
        memory, updates = _upsert_positive_pattern(
            memory,
            week,
            record,
            valence,
            salience,
            strength,
            tags,
            domains,
            source_ids,
            len(episodes),
        )
    return memory, updates


def _upsert_negative_pattern(
    memory: MemoryState,
    week: int,
    record: ConsequenceRecord,
    valence: float,
    salience: float,
    strength: float,
    tags: tuple[str, ...],
    domains: tuple[str, ...],
    source_ids: tuple[str, ...],
    exposure_count: int,
) -> tuple[MemoryState, tuple[MemoryUpdate, ...]]:
    mistake_id = _memory_id("mistake", record.source_event_id, record.source_event_version, record.chosen_option_id)
    lesson_id = _memory_id("lesson", record.source_event_id, record.source_event_version, record.chosen_option_id)
    mistake, mistake_update = _pattern_memory(
        _find_by_id(memory.mistakes, mistake_id),
        MistakeMemory,
        "mistake",
        mistake_id,
        week,
        record,
        valence,
        salience,
        strength,
        tags,
        domains,
        source_ids,
        exposure_count,
    )
    lesson, lesson_update = _pattern_memory(
        _find_by_id(memory.lessons_learned, lesson_id),
        LessonMemory,
        "lesson",
        lesson_id,
        week,
        record,
        valence,
        salience,
        min(1.0, strength * 0.8),
        tags,
        domains,
        source_ids,
        exposure_count,
    )
    return replace(
        memory,
        mistakes=_replace_by_id(memory.mistakes, mistake),
        lessons_learned=_replace_by_id(memory.lessons_learned, lesson),
    ), (mistake_update, lesson_update)


def _upsert_positive_pattern(
    memory: MemoryState,
    week: int,
    record: ConsequenceRecord,
    valence: float,
    salience: float,
    strength: float,
    tags: tuple[str, ...],
    domains: tuple[str, ...],
    source_ids: tuple[str, ...],
    exposure_count: int,
) -> tuple[MemoryState, tuple[MemoryUpdate, ...]]:
    pattern_id = _memory_id("success", record.source_event_id, record.source_event_version, record.chosen_option_id)
    lesson_id = _memory_id("lesson", record.source_event_id, record.source_event_version, record.chosen_option_id)
    pattern, pattern_update = _pattern_memory(
        _find_by_id(memory.successful_patterns, pattern_id),
        SuccessfulPatternMemory,
        "successful_pattern",
        pattern_id,
        week,
        record,
        valence,
        salience,
        strength,
        tags,
        domains,
        source_ids,
        exposure_count,
    )
    lesson, lesson_update = _pattern_memory(
        _find_by_id(memory.lessons_learned, lesson_id),
        LessonMemory,
        "lesson",
        lesson_id,
        week,
        record,
        valence,
        salience,
        min(1.0, strength * 0.8),
        tags,
        domains,
        source_ids,
        exposure_count,
    )
    return replace(
        memory,
        successful_patterns=_replace_by_id(memory.successful_patterns, pattern),
        lessons_learned=_replace_by_id(memory.lessons_learned, lesson),
    ), (pattern_update, lesson_update)


def _pattern_memory(
    existing: LessonMemory | MistakeMemory | SuccessfulPatternMemory | None,
    memory_type: type[LessonMemory | MistakeMemory | SuccessfulPatternMemory],
    kind: str,
    memory_id: str,
    week: int,
    record: ConsequenceRecord,
    valence: float,
    salience: float,
    strength: float,
    tags: tuple[str, ...],
    domains: tuple[str, ...],
    source_ids: tuple[str, ...],
    exposure_count: int,
) -> tuple[LessonMemory | MistakeMemory | SuccessfulPatternMemory, MemoryUpdate]:
    before_strength = (
        None
        if existing is None
        else effective_memory_strength(existing.strength, existing.last_reinforced_week, week)
    )
    before_valence = None if existing is None else existing.valence
    before_count = 0 if existing is None else existing.exposure_count
    kwargs = {
        "memory_id": memory_id,
        "source_event_id": record.source_event_id,
        "source_event_version": record.source_event_version,
        "source_option_id": record.chosen_option_id,
        "first_week": week if existing is None else existing.first_week,
        "last_reinforced_week": week,
        "exposure_count": exposure_count,
        "salience": round(salience, 12),
        "strength": round(strength, 12),
        "valence": round(valence, 12),
        "tags": tags,
        "affected_domains": domains,
        "source_consequence_ids": source_ids,
    }
    if memory_type is LessonMemory:
        memory = LessonMemory(
            lesson=f"{record.source_event_id}/{record.chosen_option_id} tended {'positive' if valence > 0 else 'negative'}.",
            confidence=round(strength * 100.0, 12),
            **kwargs,
        )
    elif memory_type is MistakeMemory:
        memory = MistakeMemory(
            mistake=f"{record.source_event_id}/{record.chosen_option_id} tended negative.",
            lesson_hint="avoid_or_prepare",
            severity=round(abs(valence) * strength * 100.0, 12),
            **kwargs,
        )
    else:
        memory = SuccessfulPatternMemory(
            pattern=f"{record.source_event_id}/{record.chosen_option_id} tended positive.",
            reliability=round(strength * 100.0, 12),
            **kwargs,
        )
    return memory, MemoryUpdate(
        update_type="created" if existing is None else "reinforced",
        memory_kind=kind,
        memory_id=memory.memory_id,
        before_strength=before_strength,
        after_strength=memory.strength,
        before_valence=before_valence,
        after_valence=memory.valence,
        before_exposure_count=before_count,
        after_exposure_count=memory.exposure_count,
        source_consequence_ids=source_ids,
    )


def _replace_episode(
    memories: tuple[EpisodicMemory, ...],
    updated: EpisodicMemory,
) -> tuple[EpisodicMemory, ...]:
    output: list[EpisodicMemory] = []
    replaced_existing = False
    for memory in memories:
        if memory.memory_id == updated.memory_id:
            output.append(updated)
            replaced_existing = True
        else:
            output.append(memory)
    if not replaced_existing:
        output.append(updated)
    return tuple(output)


def _replace_by_id(
    memories: tuple[Any, ...],
    updated: Any,
) -> tuple[Any, ...]:
    output: list[Any] = []
    replaced_existing = False
    for memory in memories:
        if memory.memory_id == updated.memory_id:
            output.append(updated)
            replaced_existing = True
        else:
            output.append(memory)
    if not replaced_existing:
        output.append(updated)
    return tuple(output)


def _find_episode(memory: MemoryState, decision_id: str) -> EpisodicMemory | None:
    for episode in memory.episodic_memories:
        if episode.source_decision_id == decision_id and episode.memory_id:
            return episode
    return None


def _find_by_id(memories: tuple[Any, ...], memory_id: str) -> Any | None:
    for memory in memories:
        if memory.memory_id == memory_id:
            return memory
    return None


def _score_application(application: EffectApplication) -> float:
    delta = _delta(application)
    if delta == 0 or delta == 0.0:
        return 0.0
    if application.path in POSITIVE_INCREASE_PATHS:
        sign = 1.0 if delta > 0 else -1.0
    elif application.path in NEGATIVE_INCREASE_PATHS:
        sign = -1.0 if delta > 0 else 1.0
    else:
        return 0.0
    return round(sign * _normalized_magnitude(application.path, delta, application.before) * _domain_weight(application.path), 12)


def _delta(application: EffectApplication) -> Decimal | float:
    if isinstance(application.before, Decimal) or isinstance(application.after, Decimal):
        if not isinstance(application.before, Decimal) or not isinstance(application.after, Decimal):
            raise TypeError("Expected Decimal before/after for monetary learning effects.")
        return application.after - application.before
    return float(application.after) - float(application.before)


def _normalized_magnitude(path: str, delta: Decimal | float, before: Decimal | float | None) -> float:
    amount = abs(delta)
    if path.startswith("financial."):
        decimal_amount = amount if isinstance(amount, Decimal) else Decimal(str(amount))
        return _clamp(float(decimal_amount / Decimal("125.0")), 0.0, 1.0)
    if path == "health.sleep_debt":
        return _clamp(float(amount) / 5.0, 0.0, 1.0)
    return _clamp(float(amount) / 25.0, 0.0, 1.0)


def _domain_weight(path: str) -> float:
    if path.startswith("financial."):
        return 1.1
    if path in {"mental.stress", "mental.loneliness", "health.energy", "health.sleep_debt"}:
        return 1.0
    if path.startswith("education."):
        return 0.9
    return 0.8


def _domain(path: str) -> str:
    for prefix, domain in DOMAIN_BY_PATH_PREFIX.items():
        if path.startswith(prefix):
            return domain
    return "general"


def _event_tags(events: tuple[Any, ...], record: ConsequenceRecord) -> tuple[str, ...]:
    for event in events:
        if not isinstance(event, EventOccurrence):
            continue
        if event.event_id == record.source_event_id and event.version == record.source_event_version:
            option_tags: tuple[str, ...] = ()
            for option in event.options:
                if option.option_id == record.chosen_option_id:
                    option_tags = option.goal_tags
                    break
            return _source_ids(event.tags, option_tags)
    return ()


def _historical_event_tags(context: WeeklyContext, record: ConsequenceRecord) -> tuple[str, ...]:
    decision = _historical_decision(context.decision_history, record.source_decision_id)
    event_id = decision.source_event_id if decision is not None else record.source_event_id
    event_version = (
        decision.source_event_version
        if decision is not None
        else record.source_event_version
    )
    option_id = decision.chosen_option_id if decision is not None else record.chosen_option_id
    history = context.event_history
    if not isinstance(history, EventHistory):
        return ()
    matching_events = tuple(
        occurrence
        for occurrence in history.occurrences
        if occurrence.event_id == event_id and occurrence.version == event_version
    )
    if decision is not None:
        matching_events = tuple(
            occurrence
            for occurrence in matching_events
            if occurrence.week == decision.week
        )
    if not matching_events:
        return ()
    event = matching_events[-1]
    option_tags: tuple[str, ...] = ()
    for option in event.options:
        if option.option_id == option_id:
            option_tags = option.goal_tags
            break
    return _source_ids(event.tags, option_tags)


def _historical_decision(history: Any, decision_id: str) -> DecisionRecord | None:
    if not isinstance(history, DecisionHistory):
        return None
    for decision in history.records:
        if decision.decision_id == decision_id:
            return decision
    return None


def _consequence_records(values: tuple[Any, ...]) -> tuple[ConsequenceRecord, ...]:
    records = tuple(values)
    for record in records:
        if not isinstance(record, ConsequenceRecord):
            raise TypeError("Expected WeeklyContext.consequences to contain ConsequenceRecord values.")
    return records


def _runtime(context: WeeklyContext) -> LearningRuntimeState:
    runtime = context.learning_runtime
    if runtime is None:
        return LearningRuntimeState()
    if not isinstance(runtime, LearningRuntimeState):
        raise TypeError("Expected WeeklyContext.learning_runtime to contain LearningRuntimeState.")
    return runtime


def _source_ids(*groups: Any) -> tuple[str, ...]:
    output: list[str] = []
    for group in groups:
        if isinstance(group, str):
            values = (group,)
        else:
            values = tuple(group)
        for value in values:
            if value and value not in output:
                output.append(value)
    return tuple(output)


def _memory_id(*parts: str) -> str:
    return f"memory_{derive_stable_seed(*parts):016x}"


def _assert_only_memory_changed(before: AgentState, after: AgentState) -> None:
    before_dict = before.to_dict()
    after_dict = after.to_dict()
    before_dict.pop("memory")
    after_dict.pop("memory")
    if before_dict != after_dict:
        raise ValueError("Learning transitions may only modify AgentState.memory.")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
