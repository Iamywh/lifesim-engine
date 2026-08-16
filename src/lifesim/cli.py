from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
from lifesim.config import load_config
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
    PassiveCashflowEngine,
    PassiveCashflowTransition,
    RoutineEngine,
    RoutineExecutionTransition,
    RoutinePlanningTransition,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lifesim-demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Path to a TOML configuration file.",
    )
    parser.add_argument(
        "--agent-scenario",
        type=Path,
        help="Optional path to an agent scenario TOML file.",
    )
    parser.add_argument(
        "--event-catalog",
        type=Path,
        default=Path("configs/events/starter.toml"),
        help="Path to an event catalog TOML file used when an agent scenario is supplied.",
    )
    parser.add_argument(
        "--consequence-catalog",
        type=Path,
        default=Path("configs/consequences/starter.toml"),
        help="Path to a consequence catalog TOML file used when an agent scenario is supplied.",
    )
    parser.add_argument(
        "--routine-catalog",
        type=Path,
        default=Path("configs/routines/starter.toml"),
        help="Path to a routine catalog TOML file used when an agent scenario is supplied.",
    )
    parser.add_argument(
        "--employment-catalog",
        type=Path,
        default=Path("configs/employment/starter.toml"),
        help="Path to an employment catalog TOML file used when an agent scenario is supplied.",
    )
    parser.add_argument(
        "--development-catalog",
        type=Path,
        default=Path("configs/development/starter.toml"),
        help="Path to a development catalog TOML file used when an agent scenario is supplied.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    initial_agent = None
    transitions = ()
    if args.agent_scenario is not None:
        initial_agent = load_agent_state(args.agent_scenario)
        event_catalog = load_event_catalog(args.event_catalog)
        consequence_catalog = load_consequence_catalog(
            args.consequence_catalog,
            event_catalog=event_catalog,
        )
        consequence_engine = ConsequenceEngine(consequence_catalog)
        learning_engine = LearningEngine()
        routine_engine = RoutineEngine(load_routine_catalog(args.routine_catalog))
        employment_catalog = load_employment_catalog(args.employment_catalog)
        development_engine = DevelopmentEngine(
            load_development_catalog(args.development_catalog),
            employment_catalog=employment_catalog,
        )
        decision_engine = DecisionEngine()
        transitions = (
            ScheduledConsequenceTransition(consequence_engine),
            LearningTransition(learning_engine),
            EmploymentBoundaryTransition(EmploymentBoundaryEngine()),
            PassiveCashflowTransition(PassiveCashflowEngine()),
            RoutinePlanningTransition(routine_engine, decision_engine),
            DevelopmentPlanningTransition(development_engine),
            EmploymentMarketTransition(EmploymentMarketEngine(employment_catalog)),
            EventEngineTransition(EventEngine(event_catalog)),
            DecisionEngineTransition(decision_engine),
            DecisionConsequenceTransition(consequence_engine),
            EmploymentDecisionTransition(EmploymentProcessEngine(employment_catalog)),
            LearningTransition(learning_engine),
            EmploymentWorkTransition(EmploymentWorkEngine()),
            DevelopmentExecutionTransition(development_engine),
            RoutineExecutionTransition(routine_engine),
        )
    result = LifeSimEngine(config, transitions=transitions).run(initial_agent=initial_agent)
    output = result.to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
