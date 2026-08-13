from __future__ import annotations

from dataclasses import dataclass

from lifesim.agents.state import (
    AgentState,
    EpisodicMemory,
    LessonMemory,
    MistakeMemory,
    SuccessfulPatternMemory,
)
from lifesim.decisions.model import DecisionMemoryEvidence
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.learning.model import effective_memory_strength

MEMORY_SIGNAL_LIMIT = 1.0


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    signal: float
    evidence: tuple[DecisionMemoryEvidence, ...]


def retrieve_memory_signal(
    state: AgentState,
    week: int,
    event: EventOccurrence,
    option: EventOption,
) -> MemoryRetrievalResult:
    evidence: list[DecisionMemoryEvidence] = []
    query_tags = set(event.tags) | set(option.goal_tags)
    option_domains = _option_domains(option)

    for memory in state.memory.episodic_memories:
        evidence.extend(_episode_evidence(memory, week, event, option, query_tags, option_domains))
    for memory in state.memory.lessons_learned:
        evidence.extend(_pattern_evidence(memory, week, event, option, query_tags, option_domains, "lesson"))
    for memory in state.memory.mistakes:
        evidence.extend(_pattern_evidence(memory, week, event, option, query_tags, option_domains, "mistake"))
    for memory in state.memory.successful_patterns:
        evidence.extend(
            _pattern_evidence(memory, week, event, option, query_tags, option_domains, "successful_pattern")
        )

    ordered = tuple(
        sorted(
            evidence,
            key=lambda item: (
                _match_rank(item.match_type),
                item.memory_id,
            ),
        )
    )
    signal = _clamp(sum(item.contribution for item in ordered), -MEMORY_SIGNAL_LIMIT, MEMORY_SIGNAL_LIMIT)
    return MemoryRetrievalResult(signal=round(signal, 12), evidence=ordered)


def _episode_evidence(
    memory: EpisodicMemory,
    week: int,
    event: EventOccurrence,
    option: EventOption,
    query_tags: set[str],
    option_domains: set[str],
) -> tuple[DecisionMemoryEvidence, ...]:
    if not memory.memory_id:
        return ()
    if _matches_exact(memory, event, option):
        return (_evidence(memory.memory_id, "exact_episode", memory.strength, memory.last_reinforced_week, week, memory.valence, 1.0),)
    if _matches_broad(memory, query_tags, option_domains):
        return (_evidence(memory.memory_id, "broad_episode", memory.strength, memory.last_reinforced_week, week, memory.valence, 0.35),)
    return ()


def _pattern_evidence(
    memory: LessonMemory | MistakeMemory | SuccessfulPatternMemory,
    week: int,
    event: EventOccurrence,
    option: EventOption,
    query_tags: set[str],
    option_domains: set[str],
    kind: str,
) -> tuple[DecisionMemoryEvidence, ...]:
    if not memory.memory_id:
        return ()
    if _matches_exact(memory, event, option):
        return (_evidence(memory.memory_id, f"exact_{kind}", memory.strength, memory.last_reinforced_week, week, memory.valence, 0.8),)
    if _matches_broad(memory, query_tags, option_domains):
        return (_evidence(memory.memory_id, f"broad_{kind}", memory.strength, memory.last_reinforced_week, week, memory.valence, 0.25),)
    return ()


def _evidence(
    memory_id: str,
    match_type: str,
    strength: float,
    last_reinforced_week: int,
    week: int,
    valence: float,
    multiplier: float,
) -> DecisionMemoryEvidence:
    effective_strength = effective_memory_strength(strength, last_reinforced_week, week)
    return DecisionMemoryEvidence(
        memory_id=memory_id,
        match_type=match_type,
        effective_strength=effective_strength,
        valence=round(valence, 12),
        contribution=round(effective_strength * valence * multiplier, 12),
    )


def _matches_exact(
    memory: EpisodicMemory | LessonMemory | MistakeMemory | SuccessfulPatternMemory,
    event: EventOccurrence,
    option: EventOption,
) -> bool:
    return (
        memory.source_event_id == event.event_id
        and memory.source_event_version == event.version
        and memory.source_option_id == option.option_id
    )


def _matches_broad(
    memory: EpisodicMemory | LessonMemory | MistakeMemory | SuccessfulPatternMemory,
    query_tags: set[str],
    option_domains: set[str],
) -> bool:
    return bool(set(memory.tags) & query_tags or set(memory.affected_domains) & option_domains)


def _option_domains(option: EventOption) -> set[str]:
    domains: set[str] = set()
    if option.estimated_cost:
        domains.add("financial")
    if option.energy_cost:
        domains.add("physical")
    if option.social_value or option.social_pressure:
        domains.add("social")
    if option.learning_value:
        domains.add("education")
    if option.health_value or option.comfort_value:
        domains.add("physical")
    if option.perceived_risk or option.uncertainty:
        domains.add("risk")
    return domains


def _match_rank(match_type: str) -> int:
    return 0 if match_type.startswith("exact_") else 1


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
