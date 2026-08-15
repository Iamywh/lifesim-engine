from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import AgentState, IdentityState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.engine import LifeSimEngine, SimulationResult, SimulationState
from lifesim.weekly import WeeklyContext

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")


def make_config(*, duration_weeks: int = 3, seed: int = 42) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(
            name="weekly-test",
            seed=seed,
            duration_weeks=duration_weeks,
        ),
        city=CityConfig(name="Veyra"),
    )


def test_maya_runs_from_week_zero_through_configured_duration() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(make_config(duration_weeks=5)).run(initial_agent=maya)

    assert isinstance(result, SimulationResult)
    assert len(result.states) == 6
    assert [state.week for state in result.states] == [0, 1, 2, 3, 4, 5]
    assert result.states[0].agent_state == maya
    assert result.states[-1].agent_state is not None
    assert result.states[-1].agent_state.identity.agent_id == "maya"
    assert result.summaries[-1].week == 5


def test_zero_week_simulation_records_only_initial_agent_snapshot() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(make_config(duration_weeks=0)).run(initial_agent=maya)

    assert len(result.states) == 1
    assert result.states[0].week == 0
    assert result.states[0].agent_state == maya
    assert result.summaries[0].state_changed is False


def test_repeated_agent_runs_on_same_engine_are_deterministic() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(make_config(duration_weeks=4, seed=99))

    first = engine.run(initial_agent=maya)
    second = engine.run(initial_agent=maya)

    assert first.to_dict() == second.to_dict()


def test_rng_consuming_transition_is_deterministic_across_repeated_engine_runs() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(
        make_config(duration_weeks=4, seed=123),
        transitions=[SeededMoodTransition()],
    )

    first = engine.run(initial_agent=maya)
    second = engine.run(initial_agent=maya)

    assert first.to_dict() == second.to_dict()
    assert first.states[0].agent_state is not None
    assert first.states[1].agent_state is not None
    assert first.states[0].agent_state.mental.mood != first.states[1].agent_state.mental.mood


def test_generic_non_maya_agent_can_run_through_weekly_loop() -> None:
    generic_agent = replace(
        load_agent_state(MAYA_SCENARIO),
        identity=IdentityState(
            agent_id="alex",
            display_name="Alex",
            age_years=24,
            pronouns="they/them",
            life_stage="young_adult",
            origin_city="Dublin",
            current_city="Veyra",
            background="Generic non-Maya test agent.",
        ),
    )

    result = LifeSimEngine(make_config(duration_weeks=2)).run(initial_agent=generic_agent)

    assert [state.agent_state.identity.agent_id for state in result.states if state.agent_state] == [
        "alex",
        "alex",
        "alex",
    ]


def test_historical_snapshots_remain_immutable_across_later_weeks() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(
        make_config(duration_weeks=2),
        transitions=(MoodLiftTransition(amount=1.0),),
    ).run(initial_agent=maya)

    week_zero = result.states[0].agent_state
    week_one = result.states[1].agent_state
    week_two = result.states[2].agent_state

    assert week_zero is maya
    assert week_one is not week_zero
    assert week_two is not week_one
    assert week_zero is not None
    assert week_one is not None
    assert week_two is not None
    assert week_zero.mental.mood == 62.0
    assert week_one.mental.mood == 63.0
    assert week_two.mental.mood == 64.0
    with pytest.raises(FrozenInstanceError):
        week_zero.mental.mood = 10.0


def test_transition_pipeline_runs_in_order_for_each_week() -> None:
    calls: list[str] = []
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(
        make_config(duration_weeks=2),
        transitions=(RecordingTransition("economy", calls), RecordingTransition("summary", calls)),
    )

    engine.run(initial_agent=maya)

    assert calls == ["economy:1", "summary:1", "economy:2", "summary:2"]


def test_pipeline_normalizes_transition_sequence_to_tuple() -> None:
    calls: list[str] = []
    transitions = [RecordingTransition("economy", calls)]
    engine = LifeSimEngine(make_config(duration_weeks=1), transitions=transitions)
    transitions.append(RecordingTransition("late", calls))

    engine.run(initial_agent=load_agent_state(MAYA_SCENARIO))

    assert calls == ["economy:1"]


def test_transition_can_replace_state_without_mutating_previous_snapshot() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(
        make_config(duration_weeks=1),
        transitions=(MoodLiftTransition(amount=5.0),),
    ).run(initial_agent=maya)

    week_zero = result.states[0].agent_state
    week_one = result.states[1].agent_state

    assert week_zero is not None
    assert week_one is not None
    assert week_zero.mental.mood == 62.0
    assert week_one.mental.mood == 67.0
    assert result.summaries[1].state_changed is True


def test_exact_decimal_serialization_survives_multiple_snapshots() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(make_config(duration_weeks=3)).run(initial_agent=maya)
    serialized = result.to_dict()

    for state in serialized["states"]:
        financial = state["agent"]["financial"]
        assert financial["cash"] == "180.00"
        assert financial["debts"][0]["interest_rate"] == "0.00"

    assert maya.financial.cash == Decimal("180.00")


def test_invalid_transition_return_type_is_rejected() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    with pytest.raises(TypeError, match="AgentState"):
        LifeSimEngine(make_config(), transitions=(InvalidTransition(),)).run(initial_agent=maya)


def test_m2_preserves_agent_identity_across_transitions() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    with pytest.raises(ValueError, match="preserve agent identity"):
        LifeSimEngine(make_config(), transitions=(IdentitySwapTransition(),)).run(initial_agent=maya)


def test_m0_no_agent_run_shape_still_works() -> None:
    result = LifeSimEngine(make_config(duration_weeks=2)).run()

    assert [state.to_dict() for state in result.states] == [
        {"week": 0},
        {"week": 1},
        {"week": 2},
    ]
    assert result.summaries == ()


def test_simulation_result_validates_week_sequence() -> None:
    with pytest.raises(ValueError, match="sequential"):
        SimulationResult(
            name="bad",
            seed=1,
            city_name="Veyra",
            states=(SimulationState(week=0), SimulationState(week=2)),
        )


def test_demo_cli_agent_scenario_outputs_weekly_maya_snapshots() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--config",
            "configs/default.toml",
            "--agent-scenario",
            "configs/scenarios/maya_start.toml",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    output = json.loads(completed.stdout)

    assert [state["week"] for state in output["states"]] == [0, 1, 2, 3, 4, 5]
    assert output["states"][0]["agent"]["identity"]["agent_id"] == "maya"
    assert output["states"][-1]["agent"]["financial"]["cash"] == "180.00"
    assert output["states"][-1]["passive_records"]
    assert output["summaries"][-1] == {
        "agent_id": "maya",
        "state_changed": True,
        "week": 5,
    }


class RecordingTransition:
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls

    def apply(self, state: AgentState, context: WeeklyContext) -> AgentState:
        self._calls.append(f"{self._name}:{context.week}")
        return state


class MoodLiftTransition:
    def __init__(self, amount: float) -> None:
        self._amount = amount

    def apply(self, state: AgentState, context: WeeklyContext) -> AgentState:
        return replace(
            state,
            mental=replace(
                state.mental,
                mood=min(100.0, state.mental.mood + self._amount),
            ),
        )


class SeededMoodTransition:
    def apply(self, state: AgentState, context: WeeklyContext) -> AgentState:
        return replace(
            state,
            mental=replace(
                state.mental,
                mood=min(100.0, state.mental.mood + context.rng.random()),
            ),
        )


class InvalidTransition:
    def apply(self, state: AgentState, context: WeeklyContext) -> dict[str, object]:
        return {"state": state, "week": context.week}


class IdentitySwapTransition:
    def apply(self, state: AgentState, context: WeeklyContext) -> AgentState:
        return replace(
            state,
            identity=replace(
                state.identity,
                agent_id=f"{state.identity.agent_id}-{context.week}",
            ),
        )
