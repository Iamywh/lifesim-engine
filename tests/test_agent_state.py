from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import pytest

from lifesim.agents.scenario import load_agent_state, parse_agent_state
from lifesim.agents.state import (
    AcuteCondition,
    AgentState,
    Debt,
    EducationState,
    EmploymentState,
    FinancialState,
    GoalItem,
    GoalsState,
    HabitsState,
    HealthState,
    IdentityState,
    KnowledgeState,
    MemoryState,
    MentalState,
    NeedsState,
    PersonalityState,
    PersonState,
    SkillRating,
    SkillsState,
    SocialState,
)
from lifesim.rng import create_rng

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")


def test_maya_scenario_loading_builds_composed_agent_state() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    assert isinstance(maya, AgentState)
    assert PersonState is AgentState
    assert maya.identity.display_name == "Maya"
    assert maya.identity.age_years == 21
    assert maya.identity.current_city == "Veyra"
    assert maya.education.status == "enrolled"
    assert maya.education.program == "Urban Studies BA"
    assert maya.employment.status == "seeking_entry_level_work"
    assert all(0.0 <= skill.level <= 100.0 for skill in maya.skills.items)


def test_composed_agent_state_keeps_sections_distinct_and_immutable() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    assert isinstance(maya.identity, IdentityState)
    assert isinstance(maya.financial, FinancialState)
    assert isinstance(maya.health, HealthState)
    assert isinstance(maya.mental, MentalState)
    assert isinstance(maya.needs, NeedsState)
    assert isinstance(maya.personality, PersonalityState)
    assert isinstance(maya.education, EducationState)
    assert isinstance(maya.goals, GoalsState)
    assert isinstance(maya.skills, SkillsState)
    assert isinstance(maya.employment, EmploymentState)
    assert isinstance(maya.social, SocialState)
    assert isinstance(maya.habits, HabitsState)
    assert isinstance(maya.knowledge, KnowledgeState)
    assert isinstance(maya.memory, MemoryState)

    with pytest.raises(FrozenInstanceError):
        maya.identity.age_years = 22


def test_decimal_monetary_values_serialize_to_exact_checkpoint_strings() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    financial = maya.financial
    serialized = maya.to_dict()["financial"]

    assert financial.cash == Decimal("180.00")
    assert financial.bank_balance == Decimal("980.00")
    assert financial.cash + financial.bank_balance == Decimal("1160.00")
    assert serialized["cash"] == "180.00"
    assert serialized["bank_balance"] == "980.00"
    assert serialized["debts"][0]["balance"] == "450.00"
    assert serialized["debts"][0]["interest_rate"] == "0.00"
    assert serialized["income_streams"][0]["amount"] == "180.00"
    assert serialized["recurring_commitments"][0]["amount"] == "520.00"


def test_agent_state_serializes_to_checkpoint_ready_dictionary() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    serialized = maya.to_dict()

    assert serialized["identity"]["agent_id"] == "maya"
    assert set(serialized["financial"]) == {
        "currency",
        "cash",
        "bank_balance",
        "savings",
        "emergency_fund",
        "debts",
        "income_streams",
        "recurring_commitments",
    }
    assert serialized["education"] == {
        "status": "enrolled",
        "program": "Urban Studies BA",
        "current_year": 2,
        "total_years": 3,
        "progress": 45.0,
        "weekly_study_hours": 18.0,
    }
    assert serialized["goals"]["short_term"][0]["priority"] == 5
    assert serialized["memory"] == {
        "episodic_memories": [],
        "lessons_learned": [],
        "mistakes": [],
        "successful_patterns": [],
    }


def test_validation_rejects_impossible_values() -> None:
    with pytest.raises(ValueError, match="risk_tolerance"):
        PersonalityState(
            risk_tolerance=1.2,
            impulsivity=0.5,
            discipline=0.5,
            frugality=0.5,
            social_need=0.5,
            independence=0.5,
            resilience=0.5,
            curiosity=0.5,
            confidence=0.5,
            patience=0.5,
            conscientiousness=0.5,
            adaptability=0.5,
        )

    with pytest.raises(TypeError, match="Decimal"):
        FinancialState(
            currency="EUR",
            cash=1.0,
            bank_balance=Decimal("0.00"),
            savings=Decimal("0.00"),
            emergency_fund=Decimal("0.00"),
            debts=(),
            income_streams=(),
            recurring_commitments=(),
        )

    with pytest.raises(ValueError, match="cash"):
        FinancialState(
            currency="EUR",
            cash=Decimal("-1.00"),
            bank_balance=Decimal("0.00"),
            savings=Decimal("0.00"),
            emergency_fund=Decimal("0.00"),
            debts=(),
            income_streams=(),
            recurring_commitments=(),
        )

    with pytest.raises(ValueError, match="level"):
        SkillRating(name="impossible skill", category="test", level=101.0)

    with pytest.raises(ValueError, match="age_years"):
        _generic_agent(identity=_identity(display_name="Alex", age_years=-1))


def test_education_state_validation() -> None:
    education = EducationState(
        status="enrolled",
        program="Computer Science",
        current_year=1,
        total_years=4,
        progress=12.5,
        weekly_study_hours=20.0,
    )

    assert education.to_dict()["status"] == "enrolled"

    with pytest.raises(ValueError, match="current_year"):
        EducationState(
            status="enrolled",
            program="Computer Science",
            current_year=5,
            total_years=4,
            progress=12.5,
            weekly_study_hours=20.0,
        )

    with pytest.raises(ValueError, match="program"):
        EducationState(
            status="enrolled",
            program="",
            current_year=1,
            total_years=4,
            progress=12.5,
            weekly_study_hours=20.0,
        )

    with pytest.raises(ValueError, match="total_years"):
        EducationState(
            status="enrolled",
            program="Computer Science",
            current_year=0,
            total_years=0,
            progress=12.5,
            weekly_study_hours=20.0,
        )

    with pytest.raises(ValueError, match="current_year"):
        EducationState(
            status="enrolled",
            program="Computer Science",
            current_year=0,
            total_years=4,
            progress=12.5,
            weekly_study_hours=20.0,
        )

    not_enrolled = EducationState(
        status="not_enrolled",
        program="",
        current_year=0,
        total_years=0,
        progress=0.0,
        weekly_study_hours=0.0,
    )

    assert not_enrolled.to_dict()["program"] == ""


def test_health_supports_sleep_debt_and_acute_conditions() -> None:
    health = HealthState(
        physical_health=75.0,
        energy=60.0,
        sleep_debt=8.5,
        mobility=90.0,
        acute_conditions=(AcuteCondition(name="mild cold", severity=20.0),),
    )

    assert health.to_dict()["sleep_debt"] == 8.5
    assert health.to_dict()["acute_conditions"] == [{"name": "mild cold", "severity": 20.0}]

    with pytest.raises(ValueError, match="sleep_debt"):
        HealthState(
            physical_health=75.0,
            energy=60.0,
            sleep_debt=-1.0,
            mobility=90.0,
            acute_conditions=(),
        )


def test_deterministic_initialization_preserves_m0_rng_foundation() -> None:
    first = load_agent_state(MAYA_SCENARIO)
    second = load_agent_state(MAYA_SCENARIO)
    first_rng = create_rng(42)
    second_rng = create_rng(42)

    assert first.to_dict() == second.to_dict()
    assert [first_rng.random() for _ in range(4)] == [second_rng.random() for _ in range(4)]


def test_generic_agent_reuse_is_independent_of_maya_scenario() -> None:
    agent = _generic_agent(identity=_identity(agent_id="alex", display_name="Alex"))

    assert agent.identity.agent_id == "alex"
    assert agent.identity.display_name != "Maya"
    assert agent.to_dict()["identity"]["current_city"] == "Veyra"
    assert agent.to_dict()["education"]["program"] == "Computer Science"
    assert agent.to_dict()["skills"]["items"][0]["name"] == "customer service"


def test_parse_agent_state_accepts_non_maya_scenario_data() -> None:
    scenario = {
        "agent": _generic_agent(
            identity=_identity(agent_id="sam", display_name="Sam", age_years=29)
        ).to_dict()
    }

    agent = parse_agent_state(scenario)

    assert agent.identity.agent_id == "sam"
    assert agent.identity.age_years == 29
    assert agent.financial.cash == Decimal("100.00")


def _generic_agent(*, identity: IdentityState) -> AgentState:
    return AgentState(
        identity=identity,
        financial=FinancialState(
            currency="EUR",
            cash=Decimal("100.00"),
            bank_balance=Decimal("600.00"),
            savings=Decimal("50.00"),
            emergency_fund=Decimal("0.00"),
            debts=(
                Debt(
                    "small loan",
                    balance=Decimal("100.00"),
                    minimum_payment=Decimal("10.00"),
                    interest_rate=Decimal("0.10"),
                ),
            ),
            income_streams=(),
            recurring_commitments=(),
        ),
        health=HealthState(
            physical_health=75.0,
            energy=65.0,
            sleep_debt=2.0,
            mobility=90.0,
            acute_conditions=(),
        ),
        mental=MentalState(
            mood=60.0,
            stress=35.0,
            mental_load=45.0,
            recovery_need=30.0,
            loneliness=30.0,
        ),
        needs=NeedsState(
            housing_security=50.0,
            food_security=75.0,
            safety=80.0,
            belonging=35.0,
            autonomy=65.0,
            purpose=50.0,
        ),
        personality=PersonalityState(
            risk_tolerance=0.4,
            impulsivity=0.3,
            discipline=0.5,
            frugality=0.6,
            social_need=0.5,
            independence=0.6,
            resilience=0.6,
            curiosity=0.7,
            confidence=0.5,
            patience=0.5,
            conscientiousness=0.5,
            adaptability=0.6,
        ),
        education=EducationState(
            status="enrolled",
            program="Computer Science",
            current_year=1,
            total_years=4,
            progress=15.0,
            weekly_study_hours=16.0,
        ),
        goals=GoalsState(
            short_term=(GoalItem("Find work", priority=5),),
            medium_term=(GoalItem("Build local routines", priority=3),),
            long_term=(GoalItem("Create stability", priority=4),),
        ),
        skills=SkillsState(
            items=(SkillRating(name="customer service", category="work", level=40.0),),
        ),
        employment=EmploymentState(
            status="seeking_work",
            role_title="",
            employer="",
            weekly_hours=0.0,
            job_search_intensity=70.0,
        ),
        social=SocialState(
            support_network_strength=20.0,
            city_familiarity=15.0,
            connections=(),
        ),
        habits=HabitsState(routine_stability=30.0, items=()),
        knowledge=KnowledgeState(topics=(), languages=()),
        memory=MemoryState(
            episodic_memories=(),
            lessons_learned=(),
            mistakes=(),
            successful_patterns=(),
        ),
    )


def _identity(
    *,
    agent_id: str = "generic",
    display_name: str,
    age_years: int = 21,
) -> IdentityState:
    return IdentityState(
        agent_id=agent_id,
        display_name=display_name,
        age_years=age_years,
        pronouns="they/them",
        life_stage="young_adult",
        origin_city="Porto",
        current_city="Veyra",
        background="Reusable test character.",
    )
