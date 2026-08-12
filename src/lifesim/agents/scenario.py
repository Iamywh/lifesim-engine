from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from lifesim.agents.state import (
    AgentState,
    Debt,
    EmploymentState,
    EpisodicMemory,
    FinancialState,
    GoalItem,
    GoalsState,
    Habit,
    HabitsState,
    HealthState,
    IdentityState,
    IncomeStream,
    KnowledgeState,
    KnowledgeTopic,
    LanguageAbility,
    LessonMemory,
    MemoryState,
    MentalState,
    MistakeMemory,
    NeedsState,
    PersonalityState,
    RecurringCommitment,
    SkillRating,
    SkillsState,
    SocialConnection,
    SocialState,
    SuccessfulPatternMemory,
)


def load_agent_state(path: str | Path) -> AgentState:
    scenario_path = Path(path)
    with scenario_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_agent_state(raw)


def parse_agent_state(raw: dict[str, Any]) -> AgentState:
    agent = _require_section(raw, "agent")

    return AgentState(
        identity=_parse_identity(_require_section(agent, "identity")),
        financial=_parse_financial(_require_section(agent, "financial")),
        health=HealthState(**_require_section(agent, "health")),
        mental=MentalState(**_require_section(agent, "mental")),
        needs=NeedsState(**_require_section(agent, "needs")),
        personality=PersonalityState(**_require_section(agent, "personality")),
        goals=_parse_goals(_require_section(agent, "goals")),
        skills=_parse_skills(_require_section(agent, "skills")),
        employment=EmploymentState(**_require_section(agent, "employment")),
        social=_parse_social(_require_section(agent, "social")),
        habits=_parse_habits(_require_section(agent, "habits")),
        knowledge=_parse_knowledge(_require_section(agent, "knowledge")),
        memory=_parse_memory(_require_section(agent, "memory")),
    )


def _parse_identity(raw: dict[str, Any]) -> IdentityState:
    return IdentityState(**raw)


def _parse_financial(raw: dict[str, Any]) -> FinancialState:
    return FinancialState(
        currency=raw["currency"],
        cash=raw["cash"],
        bank_balance=raw["bank_balance"],
        savings=raw["savings"],
        emergency_fund=raw["emergency_fund"],
        debts=_items(raw, "debts", Debt),
        income_streams=_items(raw, "income_streams", IncomeStream),
        recurring_commitments=_items(raw, "recurring_commitments", RecurringCommitment),
    )


def _parse_goals(raw: dict[str, Any]) -> GoalsState:
    return GoalsState(
        short_term=_items(raw, "short_term", GoalItem),
        medium_term=_items(raw, "medium_term", GoalItem),
        long_term=_items(raw, "long_term", GoalItem),
    )


def _parse_skills(raw: dict[str, Any]) -> SkillsState:
    return SkillsState(items=_items(raw, "items", SkillRating))


def _parse_social(raw: dict[str, Any]) -> SocialState:
    return SocialState(
        support_network_strength=raw["support_network_strength"],
        city_familiarity=raw["city_familiarity"],
        connections=_items(raw, "connections", SocialConnection),
    )


def _parse_habits(raw: dict[str, Any]) -> HabitsState:
    return HabitsState(
        routine_stability=raw["routine_stability"],
        items=_items(raw, "items", Habit),
    )


def _parse_knowledge(raw: dict[str, Any]) -> KnowledgeState:
    return KnowledgeState(
        topics=_items(raw, "topics", KnowledgeTopic),
        languages=_items(raw, "languages", LanguageAbility),
    )


def _parse_memory(raw: dict[str, Any]) -> MemoryState:
    return MemoryState(
        episodic_memories=_items(raw, "episodic_memories", EpisodicMemory),
        lessons_learned=_items(raw, "lessons_learned", LessonMemory),
        mistakes=_items(raw, "mistakes", MistakeMemory),
        successful_patterns=_items(raw, "successful_patterns", SuccessfulPatternMemory),
    )


def _require_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{key}' scenario section.")
    return value


def _items(raw: dict[str, Any], key: str, item_type: type) -> tuple[Any, ...]:
    values = raw.get(key, [])
    if not isinstance(values, list):
        raise TypeError(f"Expected '{key}' to be a list.")
    return tuple(item_type(**item) for item in values)
