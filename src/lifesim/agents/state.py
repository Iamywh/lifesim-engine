from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from typing import Any


class SerializableState:
    def to_dict(self) -> dict[str, Any]:
        return {field.name: _serialize(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class IdentityState(SerializableState):
    agent_id: str
    display_name: str
    age_years: int
    pronouns: str
    life_stage: str
    origin_city: str
    current_city: str
    background: str

    def __post_init__(self) -> None:
        _require_non_empty(self.agent_id, "agent_id")
        _require_non_empty(self.display_name, "display_name")
        _require_non_negative(self.age_years, "age_years")
        _require_non_empty(self.current_city, "current_city")


@dataclass(frozen=True, slots=True)
class Debt(SerializableState):
    name: str
    balance: Decimal
    minimum_payment: Decimal
    interest_rate: Decimal

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_money(self.balance, "balance")
        _require_money(self.minimum_payment, "minimum_payment")
        _require_rate(self.interest_rate, "interest_rate")


@dataclass(frozen=True, slots=True)
class IncomeStream(SerializableState):
    name: str
    amount: Decimal
    cadence: str
    reliability: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_money(self.amount, "amount")
        _require_non_empty(self.cadence, "cadence")
        _require_bounded(self.reliability, "reliability", minimum=0.0, maximum=1.0)


@dataclass(frozen=True, slots=True)
class RecurringCommitment(SerializableState):
    name: str
    amount: Decimal
    cadence: str
    category: str

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_money(self.amount, "amount")
        _require_non_empty(self.cadence, "cadence")
        _require_non_empty(self.category, "category")


@dataclass(frozen=True, slots=True)
class FinancialState(SerializableState):
    currency: str
    cash: Decimal
    bank_balance: Decimal
    savings: Decimal
    emergency_fund: Decimal
    debts: tuple[Debt, ...]
    income_streams: tuple[IncomeStream, ...]
    recurring_commitments: tuple[RecurringCommitment, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.currency, "currency")
        if len(self.currency) != 3:
            raise ValueError("Expected 'currency' to use an ISO-style three-letter code.")
        _require_money(self.cash, "cash")
        _require_money(self.bank_balance, "bank_balance")
        _require_money(self.savings, "savings")
        _require_money(self.emergency_fund, "emergency_fund")
        _freeze_tuple(self, "debts")
        _freeze_tuple(self, "income_streams")
        _freeze_tuple(self, "recurring_commitments")


@dataclass(frozen=True, slots=True)
class AcuteCondition(SerializableState):
    name: str
    severity: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_percent(self.severity, "severity")


@dataclass(frozen=True, slots=True)
class HealthState(SerializableState):
    physical_health: float
    energy: float
    sleep_debt: float
    mobility: float
    acute_conditions: tuple[AcuteCondition, ...]

    def __post_init__(self) -> None:
        _require_percent(self.physical_health, "physical_health")
        _require_percent(self.energy, "energy")
        _require_non_negative(self.sleep_debt, "sleep_debt")
        _require_percent(self.mobility, "mobility")
        _freeze_tuple(self, "acute_conditions")


@dataclass(frozen=True, slots=True)
class MentalState(SerializableState):
    mood: float
    stress: float
    mental_load: float
    recovery_need: float
    loneliness: float

    def __post_init__(self) -> None:
        _require_percent(self.mood, "mood")
        _require_percent(self.stress, "stress")
        _require_percent(self.mental_load, "mental_load")
        _require_percent(self.recovery_need, "recovery_need")
        _require_percent(self.loneliness, "loneliness")


@dataclass(frozen=True, slots=True)
class NeedsState(SerializableState):
    housing_security: float
    food_security: float
    safety: float
    belonging: float
    autonomy: float
    purpose: float

    def __post_init__(self) -> None:
        _require_percent(self.housing_security, "housing_security")
        _require_percent(self.food_security, "food_security")
        _require_percent(self.safety, "safety")
        _require_percent(self.belonging, "belonging")
        _require_percent(self.autonomy, "autonomy")
        _require_percent(self.purpose, "purpose")


@dataclass(frozen=True, slots=True)
class PersonalityState(SerializableState):
    risk_tolerance: float
    impulsivity: float
    discipline: float
    frugality: float
    social_need: float
    independence: float
    resilience: float
    curiosity: float
    confidence: float
    patience: float
    conscientiousness: float
    adaptability: float

    def __post_init__(self) -> None:
        _require_trait(self.risk_tolerance, "risk_tolerance")
        _require_trait(self.impulsivity, "impulsivity")
        _require_trait(self.discipline, "discipline")
        _require_trait(self.frugality, "frugality")
        _require_trait(self.social_need, "social_need")
        _require_trait(self.independence, "independence")
        _require_trait(self.resilience, "resilience")
        _require_trait(self.curiosity, "curiosity")
        _require_trait(self.confidence, "confidence")
        _require_trait(self.patience, "patience")
        _require_trait(self.conscientiousness, "conscientiousness")
        _require_trait(self.adaptability, "adaptability")


@dataclass(frozen=True, slots=True)
class EducationState(SerializableState):
    status: str
    program: str
    current_year: int
    total_years: int
    progress: float
    weekly_study_hours: float

    def __post_init__(self) -> None:
        _require_non_empty(self.status, "status")
        _require_non_negative(self.current_year, "current_year")
        _require_non_negative(self.total_years, "total_years")
        if self.status == "enrolled":
            _require_non_empty(self.program, "program")
            if self.total_years < 1:
                raise ValueError("Expected 'total_years' to be >= 1 when enrolled.")
            if self.current_year < 1:
                raise ValueError("Expected 'current_year' to be >= 1 when enrolled.")
        if self.total_years > 0 and self.current_year > self.total_years:
            raise ValueError("Expected 'current_year' to be <= 'total_years'.")
        _require_percent(self.progress, "progress")
        _require_bounded(
            self.weekly_study_hours,
            "weekly_study_hours",
            minimum=0.0,
            maximum=168.0,
        )


@dataclass(frozen=True, slots=True)
class GoalItem(SerializableState):
    description: str
    priority: int
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.description, "description")
        _require_bounded(self.priority, "priority", minimum=1, maximum=5)
        if isinstance(self.tags, str) or not isinstance(self.tags, list | tuple):
            raise TypeError("Expected 'tags' to be a list or tuple of strings.")
        object.__setattr__(self, "tags", tuple(self.tags))
        for tag in self.tags:
            _require_non_empty(tag, "tags")


@dataclass(frozen=True, slots=True)
class GoalsState(SerializableState):
    short_term: tuple[GoalItem, ...]
    medium_term: tuple[GoalItem, ...]
    long_term: tuple[GoalItem, ...]

    def __post_init__(self) -> None:
        _freeze_tuple(self, "short_term")
        _freeze_tuple(self, "medium_term")
        _freeze_tuple(self, "long_term")


@dataclass(frozen=True, slots=True)
class SkillRating(SerializableState):
    name: str
    category: str
    level: float
    experience: float = 0.0

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.category, "category")
        _require_percent(self.level, "level")
        _require_non_negative(self.experience, "experience")


@dataclass(frozen=True, slots=True)
class SkillsState(SerializableState):
    items: tuple[SkillRating, ...]

    def __post_init__(self) -> None:
        _freeze_tuple(self, "items")


@dataclass(frozen=True, slots=True)
class EmploymentState(SerializableState):
    status: str
    role_title: str
    employer: str
    weekly_hours: float
    job_search_intensity: float

    def __post_init__(self) -> None:
        _require_non_empty(self.status, "status")
        _require_bounded(self.weekly_hours, "weekly_hours", minimum=0.0, maximum=168.0)
        _require_percent(self.job_search_intensity, "job_search_intensity")


@dataclass(frozen=True, slots=True)
class SocialConnection(SerializableState):
    name: str
    relationship: str
    closeness: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.relationship, "relationship")
        _require_percent(self.closeness, "closeness")


@dataclass(frozen=True, slots=True)
class SocialState(SerializableState):
    support_network_strength: float
    city_familiarity: float
    connections: tuple[SocialConnection, ...]

    def __post_init__(self) -> None:
        _require_percent(self.support_network_strength, "support_network_strength")
        _require_percent(self.city_familiarity, "city_familiarity")
        _freeze_tuple(self, "connections")


@dataclass(frozen=True, slots=True)
class Habit(SerializableState):
    name: str
    cadence: str
    strength: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.cadence, "cadence")
        _require_percent(self.strength, "strength")


@dataclass(frozen=True, slots=True)
class HabitsState(SerializableState):
    routine_stability: float
    items: tuple[Habit, ...]

    def __post_init__(self) -> None:
        _require_percent(self.routine_stability, "routine_stability")
        _freeze_tuple(self, "items")


@dataclass(frozen=True, slots=True)
class KnowledgeTopic(SerializableState):
    name: str
    familiarity: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_percent(self.familiarity, "familiarity")


@dataclass(frozen=True, slots=True)
class LanguageAbility(SerializableState):
    language: str
    proficiency: float

    def __post_init__(self) -> None:
        _require_non_empty(self.language, "language")
        _require_percent(self.proficiency, "proficiency")


@dataclass(frozen=True, slots=True)
class KnowledgeState(SerializableState):
    topics: tuple[KnowledgeTopic, ...]
    languages: tuple[LanguageAbility, ...]

    def __post_init__(self) -> None:
        _freeze_tuple(self, "topics")
        _freeze_tuple(self, "languages")


@dataclass(frozen=True, slots=True)
class EpisodicMemory(SerializableState):
    summary: str
    week: int
    emotional_weight: float

    def __post_init__(self) -> None:
        _require_non_empty(self.summary, "summary")
        _require_non_negative(self.week, "week")
        _require_percent(self.emotional_weight, "emotional_weight")


@dataclass(frozen=True, slots=True)
class LessonMemory(SerializableState):
    lesson: str
    confidence: float

    def __post_init__(self) -> None:
        _require_non_empty(self.lesson, "lesson")
        _require_percent(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class MistakeMemory(SerializableState):
    mistake: str
    lesson_hint: str
    severity: float

    def __post_init__(self) -> None:
        _require_non_empty(self.mistake, "mistake")
        _require_percent(self.severity, "severity")


@dataclass(frozen=True, slots=True)
class SuccessfulPatternMemory(SerializableState):
    pattern: str
    reliability: float

    def __post_init__(self) -> None:
        _require_non_empty(self.pattern, "pattern")
        _require_percent(self.reliability, "reliability")


@dataclass(frozen=True, slots=True)
class MemoryState(SerializableState):
    episodic_memories: tuple[EpisodicMemory, ...]
    lessons_learned: tuple[LessonMemory, ...]
    mistakes: tuple[MistakeMemory, ...]
    successful_patterns: tuple[SuccessfulPatternMemory, ...]

    def __post_init__(self) -> None:
        _freeze_tuple(self, "episodic_memories")
        _freeze_tuple(self, "lessons_learned")
        _freeze_tuple(self, "mistakes")
        _freeze_tuple(self, "successful_patterns")


@dataclass(frozen=True, slots=True)
class AgentState(SerializableState):
    identity: IdentityState
    financial: FinancialState
    health: HealthState
    mental: MentalState
    needs: NeedsState
    personality: PersonalityState
    education: EducationState
    goals: GoalsState
    skills: SkillsState
    employment: EmploymentState
    social: SocialState
    habits: HabitsState
    knowledge: KnowledgeState
    memory: MemoryState


PersonState = AgentState


def _serialize(value: Any) -> Any:
    if isinstance(value, SerializableState):
        return value.to_dict()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value


def _freeze_tuple(instance: object, attribute: str) -> None:
    value = getattr(instance, attribute)
    if not isinstance(value, tuple):
        object.__setattr__(instance, attribute, tuple(value))


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{name}' to be a non-empty string.")


def _require_non_negative(value: float, name: str) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    if value < 0:
        raise ValueError(f"Expected '{name}' to be non-negative.")


def _require_money(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected decimal value '{name}' to be Decimal.")
    if not value.is_finite():
        raise ValueError(f"Expected decimal value '{name}' to be finite.")
    if value < Decimal(0):
        raise ValueError(f"Expected decimal value '{name}' to be non-negative.")


def _require_rate(value: Decimal, name: str) -> None:
    _require_money(value, name)
    if value > Decimal(1):
        raise ValueError(f"Expected decimal rate '{name}' to be between 0 and 1.")


def _require_percent(value: float, name: str) -> None:
    _require_bounded(value, name, minimum=0.0, maximum=100.0)


def _require_trait(value: float, name: str) -> None:
    _require_bounded(value, name, minimum=0.0, maximum=1.0)


def _require_bounded(value: float, name: str, *, minimum: float, maximum: float) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    if not minimum <= value <= maximum:
        raise ValueError(f"Expected '{name}' to be between {minimum} and {maximum}.")
