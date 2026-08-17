from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
from lifesim.canonical import build_canonical_transitions
from lifesim.config import load_config
from lifesim.engine import LifeSimEngine


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
    parser.add_argument(
        "--social-catalog",
        type=Path,
        default=Path("configs/social/starter.toml"),
        help="Path to a social relationship catalog TOML file used when an agent scenario is supplied.",
    )
    parser.add_argument(
        "--adaptation-catalog",
        type=Path,
        default=Path("configs/adaptation/starter.toml"),
        help="Path to an adaptation catalog TOML file used when an agent scenario is supplied.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    initial_agent = None
    transitions = ()
    if args.agent_scenario is not None:
        initial_agent = load_agent_state(args.agent_scenario)
        transitions = build_canonical_transitions(
            event_catalog_path=args.event_catalog,
            consequence_catalog_path=args.consequence_catalog,
            routine_catalog_path=args.routine_catalog,
            employment_catalog_path=args.employment_catalog,
            development_catalog_path=args.development_catalog,
            social_catalog_path=args.social_catalog,
            adaptation_catalog_path=args.adaptation_catalog,
        )
    result = LifeSimEngine(config, transitions=transitions).run(initial_agent=initial_agent)
    output = result.to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
