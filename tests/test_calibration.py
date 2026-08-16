from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
from lifesim.calibration.runner import run_calibration
from lifesim.canonical import build_canonical_engine, build_canonical_transitions
from lifesim.config import load_config

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")


def test_canonical_builder_is_deterministic_and_includes_arrear_settlement() -> None:
    transitions = build_canonical_transitions()
    names = tuple(type(transition).__name__ for transition in transitions)

    assert "ArrearSettlementTransition" in names
    assert names.index("RoutineExecutionTransition") < names.index("ArrearSettlementTransition")
    assert names.index("ArrearSettlementTransition") < names.index("AdaptationTransition")

    config = load_config("configs/default.toml")
    maya = load_agent_state(MAYA_SCENARIO)
    engine = build_canonical_engine(config)

    assert engine.run(initial_agent=maya).to_dict() == engine.run(initial_agent=maya).to_dict()


def test_demo_cli_uses_canonical_engine_for_weekly_maya_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--config",
            "configs/default.toml",
            "--agent-scenario",
            str(MAYA_SCENARIO),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    output = json.loads(completed.stdout)

    assert output["states"][1]["agent"]["identity"]["agent_id"] == "maya"
    assert output["passive_history"]["arrear_settlement_records"]


def test_calibration_runs_fixed_seed_smoke_cohort() -> None:
    result = run_calibration(
        label="smoke",
        seeds=(0, 1),
        duration_weeks=12,
        checkpoints=(12,),
    )

    assert result.run_count == 2
    assert result.failure_count == 0
    assert 12 in result.metrics["checkpoints"]
    assert result.metrics["avg_events_per_run"] >= 0.0
