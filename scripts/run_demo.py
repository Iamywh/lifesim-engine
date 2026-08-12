from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    engine = LifeSimEngine(config)
    result = engine.run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
