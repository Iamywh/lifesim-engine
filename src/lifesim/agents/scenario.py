from __future__ import annotations

import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lifesim.agents.state import (
    AcuteCondition,
    AgentState,
    Arrear,
    Debt,
    EducationState,
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
    RoutineState,
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
        health=_parse_health(_require_section(agent, "health")),
        mental=MentalState(**_require_section(agent, "mental")),
        needs=NeedsState(**_require_section(agent, "needs")),
        personality=PersonalityState(**_require_section(agent, "personality")),
        education=EducationState(**_require_section(agent, "education")),
        goals=_parse_goals(_require_section(agent, "goals")),
        skills=_parse_skills(_require_section(agent, "skills")),
        employment=EmploymentState(**_require_section(agent, "employment")),
        social=_parse_social(_require_section(agent, "social")),
        habits=_parse_habits(_require_section(agent, "habits")),
        knowledge=_parse_knowledge(_require_section(agent, "knowledge")),
        memory=_parse_memory(_require_section(agent, "memory")),
        routine=RoutineState(**agent.get("routine", {})),
    )


def _parse_identity(raw: dict[str, Any]) -> IdentityState:
    return IdentityState(**raw)


def _parse_financial(raw: dict[str, Any]) -> FinancialState:
    return FinancialState(
        currency=raw["currency"],
        cash=_money(raw["cash"], "cash"),
        bank_balance=_money(raw["bank_balance"], "bank_balance"),
        savings=_money(raw["savings"], "savings"),
        emergency_fund=_money(raw["emergency_fund"], "emergency_fund"),
        debts=tuple(
            Debt(
                name=item["name"],
                balance=_money(item["balance"], "balance"),
                minimum_payment=_money(item["minimum_payment"], "minimum_payment"),
                interest_rate=_money(item["interest_rate"], "interest_rate"),
                payment_cadence=item.get("payment_cadence", "monthly"),
                due_day=item.get("due_day", 1),
                consecutive_missed_payments=item.get("consecutive_missed_payments", 0),
            )
            for item in _raw_items(raw, "debts")
        ),
        income_streams=tuple(
            IncomeStream(
                name=item["name"],
                amount=_money(item["amount"], "amount"),
                cadence=item["cadence"],
                reliability=item["reliability"],
                due_day=item.get("due_day", 1),
            )
            for item in _raw_items(raw, "income_streams")
        ),
        recurring_commitments=tuple(
            RecurringCommitment(
                name=item["name"],
                amount=_money(item["amount"], "amount"),
                cadence=item["cadence"],
                category=item["category"],
                due_day=item.get("due_day", 1),
            )
            for item in _raw_items(raw, "recurring_commitments")
        ),
        arrears=tuple(
            Arrear(
                obligation_id=item["obligation_id"],
                category=item["category"],
                balance=_money(item["balance"], "balance"),
                first_missed_week=item["first_missed_week"],
                last_updated_week=item["last_updated_week"],
                missed_occurrences=item["missed_occurrences"],
            )
            for item in _raw_items(raw, "arrears")
        ),
    )


def _parse_health(raw: dict[str, Any]) -> HealthState:
    return HealthState(
        physical_health=raw["physical_health"],
        energy=raw["energy"],
        sleep_debt=raw["sleep_debt"],
        mobility=raw["mobility"],
        acute_conditions=_items(raw, "acute_conditions", AcuteCondition),
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
    return tuple(item_type(**item) for item in _raw_items(raw, key))


def _raw_items(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = raw.get(key, [])
    if not isinstance(values, list):
        raise TypeError(f"Expected '{key}' to be a list.")
    if not all(isinstance(item, dict) for item in values):
        raise TypeError(f"Expected '{key}' to contain tables.")
    return values


def _money(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"Expected monetary value '{name}' to be a string.")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Expected monetary value '{name}' to be a decimal string.") from error
