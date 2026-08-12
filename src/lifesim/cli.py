from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    result = LifeSimEngine(config).run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
