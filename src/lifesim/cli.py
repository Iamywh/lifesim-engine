from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    initial_agent = None
    if args.agent_scenario is not None:
        initial_agent = load_agent_state(args.agent_scenario)
    result = LifeSimEngine(config).run(initial_agent=initial_agent)
    output = result.to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
