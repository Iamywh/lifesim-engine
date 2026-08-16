from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lifesim.adaptation.catalog import load_adaptation_catalog
from lifesim.adaptation.engine import AdaptationEngine, AdaptationTransition
from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import AgentState
from lifesim.config import LifeSimConfig, load_config
from lifesim.consequences.catalog import load_consequence_catalog
from lifesim.consequences.engine import (
    ConsequenceEngine,
    DecisionConsequenceTransition,
    ScheduledConsequenceTransition,
)
from lifesim.decisions.engine import DecisionEngine, DecisionEngineTransition
from lifesim.development.catalog import load_development_catalog
from lifesim.development.engine import (
    DevelopmentEngine,
    DevelopmentExecutionTransition,
    DevelopmentPlanningTransition,
)
from lifesim.employment.catalog import load_employment_catalog
from lifesim.employment.engine import (
    EmploymentBoundaryEngine,
    EmploymentBoundaryTransition,
    EmploymentDecisionTransition,
    EmploymentMarketEngine,
    EmploymentMarketTransition,
    EmploymentProcessEngine,
    EmploymentWorkEngine,
    EmploymentWorkTransition,
)
from lifesim.engine import LifeSimEngine
from lifesim.events.catalog import load_event_catalog
from lifesim.events.engine import EventEngine, EventEngineTransition
from lifesim.learning.engine import LearningEngine, LearningTransition
from lifesim.passive.catalog import load_routine_catalog
from lifesim.passive.engine import (
    ArrearSettlementEngine,
    ArrearSettlementTransition,
    PassiveCashflowEngine,
    PassiveCashflowTransition,
    RoutineEngine,
    RoutineExecutionTransition,
    RoutinePlanningTransition,
)
from lifesim.social.catalog import load_social_catalog
from lifesim.social.engine import (
    SocialEngine,
    SocialExecutionTransition,
    SocialMaintenanceTransition,
    SocialPlanningTransition,
)
from lifesim.weekly import WeeklyTransition


@dataclass(frozen=True, slots=True)
class CanonicalRunSetup:
    config: LifeSimConfig
    initial_agent: AgentState
    engine: LifeSimEngine


def build_canonical_transitions(
    *,
    event_catalog_path: Path = Path("configs/events/starter.toml"),
    consequence_catalog_path: Path = Path("configs/consequences/starter.toml"),
    routine_catalog_path: Path = Path("configs/routines/starter.toml"),
    employment_catalog_path: Path = Path("configs/employment/starter.toml"),
    development_catalog_path: Path = Path("configs/development/starter.toml"),
    social_catalog_path: Path = Path("configs/social/starter.toml"),
    adaptation_catalog_path: Path = Path("configs/adaptation/starter.toml"),
) -> tuple[WeeklyTransition, ...]:
    event_catalog = load_event_catalog(event_catalog_path)
    consequence_catalog = load_consequence_catalog(
        consequence_catalog_path,
        event_catalog=event_catalog,
    )
    consequence_engine = ConsequenceEngine(consequence_catalog)
    learning_engine = LearningEngine()
    routine_catalog = load_routine_catalog(routine_catalog_path)
    routine_engine = RoutineEngine(routine_catalog)
    employment_catalog = load_employment_catalog(employment_catalog_path)
    development_engine = DevelopmentEngine(
        load_development_catalog(development_catalog_path),
        employment_catalog=employment_catalog,
    )
    social_engine = SocialEngine(
        load_social_catalog(social_catalog_path),
        routine_catalog=routine_catalog,
    )
    adaptation_engine = AdaptationEngine(load_adaptation_catalog(adaptation_catalog_path))
    decision_engine = DecisionEngine()
    return (
        ScheduledConsequenceTransition(consequence_engine),
        LearningTransition(learning_engine),
        EmploymentBoundaryTransition(EmploymentBoundaryEngine()),
        PassiveCashflowTransition(PassiveCashflowEngine()),
        SocialMaintenanceTransition(social_engine),
        RoutinePlanningTransition(routine_engine, decision_engine),
        DevelopmentPlanningTransition(development_engine),
        SocialPlanningTransition(social_engine),
        EmploymentMarketTransition(EmploymentMarketEngine(employment_catalog)),
        EventEngineTransition(EventEngine(event_catalog)),
        DecisionEngineTransition(decision_engine),
        DecisionConsequenceTransition(consequence_engine),
        EmploymentDecisionTransition(EmploymentProcessEngine(employment_catalog)),
        LearningTransition(learning_engine),
        EmploymentWorkTransition(EmploymentWorkEngine()),
        DevelopmentExecutionTransition(development_engine),
        SocialExecutionTransition(social_engine),
        RoutineExecutionTransition(routine_engine),
        ArrearSettlementTransition(ArrearSettlementEngine()),
        AdaptationTransition(adaptation_engine),
    )


def build_canonical_engine(
    config: LifeSimConfig,
    **catalog_paths: Path,
) -> LifeSimEngine:
    return LifeSimEngine(
        config,
        transitions=build_canonical_transitions(**catalog_paths),
    )


def load_canonical_run_setup(
    *,
    config_path: Path = Path("configs/default.toml"),
    agent_scenario_path: Path = Path("configs/scenarios/maya_start.toml"),
    **catalog_paths: Path,
) -> CanonicalRunSetup:
    config = load_config(config_path)
    initial_agent = load_agent_state(agent_scenario_path)
    engine = build_canonical_engine(config, **catalog_paths)
    return CanonicalRunSetup(config=config, initial_agent=initial_agent, engine=engine)
