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
from lifesim.engine import LifeSimEngine
from lifesim.events.catalog import load_event_catalog
from lifesim.events.engine import EventEngine, EventEngineTransition


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
        transitions = (
            ScheduledConsequenceTransition(consequence_engine),
            EventEngineTransition(EventEngine(event_catalog)),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(consequence_engine),
        )
    result = LifeSimEngine(config, transitions=transitions).run(initial_agent=initial_agent)
    output = result.to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
