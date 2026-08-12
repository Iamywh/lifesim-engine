from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
from lifesim.config import load_config
from lifesim.engine import LifeSimEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LifeSim M0 demo simulation.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    engine = LifeSimEngine(config)
    result = engine.run()
    output = result.to_dict()
    if args.agent_scenario is not None:
        output = {
            "simulation": output,
            "agent": load_agent_state(args.agent_scenario).to_dict(),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
