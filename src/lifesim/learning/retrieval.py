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


@dataclass(frozen=True, slots=True)
class _RetrievalEvidence:
    evidence: DecisionMemoryEvidence
    lineage_ids: tuple[str, ...]


def retrieve_memory_signal(
    state: AgentState,
    week: int,
    event: EventOccurrence,
    option: EventOption,
) -> MemoryRetrievalResult:
    evidence: list[_RetrievalEvidence] = []
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

    ordered_items = tuple(
        sorted(
            evidence,
            key=lambda item: (
                _match_rank(item.evidence.match_type),
                item.evidence.memory_id,
            ),
        )
    )
    ordered = tuple(item.evidence for item in ordered_items)
    signal = _aggregate_signal(ordered_items)
    return MemoryRetrievalResult(signal=round(signal, 12), evidence=ordered)


def _episode_evidence(
    memory: EpisodicMemory,
    week: int,
    event: EventOccurrence,
    option: EventOption,
    query_tags: set[str],
    option_domains: set[str],
) -> tuple[_RetrievalEvidence, ...]:
    if not memory.memory_id:
        return ()
    if _matches_exact(memory, event, option):
        return (
            _evidence_item(
                memory,
                "exact_episode",
                week,
                1.0,
            ),
        )
    if _matches_broad(memory, query_tags, option_domains):
        return (
            _evidence_item(
                memory,
                "broad_episode",
                week,
                0.35,
            ),
        )
    return ()


def _pattern_evidence(
    memory: LessonMemory | MistakeMemory | SuccessfulPatternMemory,
    week: int,
    event: EventOccurrence,
    option: EventOption,
    query_tags: set[str],
    option_domains: set[str],
    kind: str,
) -> tuple[_RetrievalEvidence, ...]:
    if not memory.memory_id:
        return ()
    if _matches_exact(memory, event, option):
        return (
            _evidence_item(
                memory,
                f"exact_{kind}",
                week,
                0.8,
            ),
        )
    if _matches_broad(memory, query_tags, option_domains):
        return (
            _evidence_item(
                memory,
                f"broad_{kind}",
                week,
                0.25,
            ),
        )
    return ()


def _evidence_item(
    memory: EpisodicMemory | LessonMemory | MistakeMemory | SuccessfulPatternMemory,
    match_type: str,
    week: int,
    multiplier: float,
) -> _RetrievalEvidence:
    effective_strength = effective_memory_strength(memory.strength, memory.last_reinforced_week, week)
    evidence = DecisionMemoryEvidence(
        memory_id=memory.memory_id,
        match_type=match_type,
        effective_strength=effective_strength,
        valence=round(memory.valence, 12),
        contribution=round(effective_strength * memory.valence * multiplier, 12),
    )
    return _RetrievalEvidence(evidence=evidence, lineage_ids=_lineage_id(memory))


def _aggregate_signal(evidence: tuple[_RetrievalEvidence, ...]) -> float:
    by_lineage: dict[str, list[DecisionMemoryEvidence]] = {}
    for item in evidence:
        lineage_ids = item.lineage_ids
        contribution = item.evidence.contribution / len(lineage_ids)
        for lineage_id in lineage_ids:
            by_lineage.setdefault(lineage_id, []).append(
                DecisionMemoryEvidence(
                    memory_id=item.evidence.memory_id,
                    match_type=item.evidence.match_type,
                    effective_strength=item.evidence.effective_strength,
                    valence=item.evidence.valence,
                    contribution=contribution,
                )
            )

    lineage_contributions: list[float] = []
    for lineage_id in sorted(by_lineage):
        values = sorted(
            by_lineage[lineage_id],
            key=lambda item: (abs(item.contribution), item.memory_id),
            reverse=True,
        )
        strongest = values[0].contribution
        reinforcement = sum(item.contribution for item in values[1:]) * 0.2
        lineage_contributions.append(strongest + reinforcement)
    ordered = sorted(lineage_contributions, key=abs, reverse=True)
    signal = sum(contribution * (0.5**index) for index, contribution in enumerate(ordered))
    return _clamp(signal, -MEMORY_SIGNAL_LIMIT, MEMORY_SIGNAL_LIMIT)


def _lineage_id(
    memory: EpisodicMemory | LessonMemory | MistakeMemory | SuccessfulPatternMemory,
) -> tuple[str, ...]:
    if memory.source_consequence_ids:
        return memory.source_consequence_ids
    return (memory.memory_id,)


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
