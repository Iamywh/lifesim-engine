from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifesim.calibration.runner import (
    DEFAULT_CHECKPOINTS,
    HOLDOUT_SEEDS,
    run_calibration,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic LifeSim calibration cohorts.")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical/maya_v1.toml"))
    parser.add_argument(
        "--agent-scenario",
        type=Path,
        default=Path("configs/scenarios/maya_start.toml"),
    )
    parser.add_argument("--duration-weeks", type=int, default=156)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("docs/calibration/m12_report.md"))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = (
        HOLDOUT_SEEDS
        if args.holdout
        else tuple(range(args.seed_start, args.seed_start + args.seed_count))
    )
    label = "holdout" if args.holdout else "calibration"
    result = run_calibration(
        label=label,
        config_path=args.config,
        agent_scenario_path=args.agent_scenario,
        seeds=seeds,
        duration_weeks=args.duration_weeks,
        checkpoints=DEFAULT_CHECKPOINTS,
    )
    holdout = None
    if args.include_holdout and not args.holdout:
        holdout = run_calibration(
            label="holdout",
            config_path=args.config,
            agent_scenario_path=args.agent_scenario,
            seeds=HOLDOUT_SEEDS,
            duration_weeks=args.duration_weeks,
            checkpoints=DEFAULT_CHECKPOINTS,
        )
    write_markdown_report(result, holdout, args.report)
    if args.json:
        output = {"calibration": result.to_dict()}
        if holdout is not None:
            output["holdout"] = holdout.to_dict()
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        warnings = ", ".join(result.warnings) if result.warnings else "none"
        print(
            f"{label}: runs={result.run_count} failures={result.failure_count} "
            f"warnings={warnings}"
        )


if __name__ == "__main__":
    main()
