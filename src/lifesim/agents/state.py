from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

type GoalHorizon = Literal["short", "medium", "long"]


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
    balance: float
    minimum_payment: float
    interest_rate: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_negative(self.balance, "balance")
        _require_non_negative(self.minimum_payment, "minimum_payment")
        _require_bounded(self.interest_rate, "interest_rate", minimum=0.0, maximum=1.0)


@dataclass(frozen=True, slots=True)
class IncomeStream(SerializableState):
    name: str
    amount: float
    cadence: str
    reliability: float

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_negative(self.amount, "amount")
        _require_non_empty(self.cadence, "cadence")
        _require_bounded(self.reliability, "reliability", minimum=0.0, maximum=1.0)


@dataclass(frozen=True, slots=True)
class RecurringCommitment(SerializableState):
    name: str
    amount: float
    cadence: str
    category: str

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_negative(self.amount, "amount")
        _require_non_empty(self.cadence, "cadence")
        _require_non_empty(self.category, "category")


@dataclass(frozen=True, slots=True)
class FinancialState(SerializableState):
    currency: str
    cash: float
    bank_balance: float
    savings: float
    emergency_fund: float
    debts: tuple[Debt, ...]
    income_streams: tuple[IncomeStream, ...]
    recurring_commitments: tuple[RecurringCommitment, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.currency, "currency")
        if len(self.currency) != 3:
            raise ValueError("Expected 'currency' to use an ISO-style three-letter code.")
        _require_non_negative(self.cash, "cash")
        _require_non_negative(self.bank_balance, "bank_balance")
        _require_non_negative(self.savings, "savings")
        _require_non_negative(self.emergency_fund, "emergency_fund")
        _freeze_tuple(self, "debts")
        _freeze_tuple(self, "income_streams")
        _freeze_tuple(self, "recurring_commitments")


@dataclass(frozen=True, slots=True)
class HealthState(SerializableState):
    physical_health: float
    energy: float
    sleep_quality: float
    mobility: float

    def __post_init__(self) -> None:
        _require_percent(self.physical_health, "physical_health")
        _require_percent(self.energy, "energy")
        _require_percent(self.sleep_quality, "sleep_quality")
        _require_percent(self.mobility, "mobility")


@dataclass(frozen=True, slots=True)
class MentalState(SerializableState):
    mood: float
    stress: float
    confidence: float
    loneliness: float
    resilience: float

    def __post_init__(self) -> None:
        _require_percent(self.mood, "mood")
        _require_percent(self.stress, "stress")
        _require_percent(self.confidence, "confidence")
        _require_percent(self.loneliness, "loneliness")
        _require_percent(self.resilience, "resilience")


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
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    risk_tolerance: float
    adaptability: float

    def __post_init__(self) -> None:
        _require_trait(self.openness, "openness")
        _require_trait(self.conscientiousness, "conscientiousness")
        _require_trait(self.extraversion, "extraversion")
        _require_trait(self.agreeableness, "agreeableness")
        _require_trait(self.neuroticism, "neuroticism")
        _require_trait(self.risk_tolerance, "risk_tolerance")
        _require_trait(self.adaptability, "adaptability")


@dataclass(frozen=True, slots=True)
class GoalItem(SerializableState):
    description: str
    priority: int

    def __post_init__(self) -> None:
        _require_non_empty(self.description, "description")
        _require_bounded(self.priority, "priority", minimum=1, maximum=5)


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


def _require_percent(value: float, name: str) -> None:
    _require_bounded(value, name, minimum=0.0, maximum=100.0)


def _require_trait(value: float, name: str) -> None:
    _require_bounded(value, name, minimum=0.0, maximum=1.0)


def _require_bounded(value: float, name: str, *, minimum: float, maximum: float) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"Expected '{name}' to be numeric.")
    if not minimum <= value <= maximum:
        raise ValueError(f"Expected '{name}' to be between {minimum} and {maximum}.")
