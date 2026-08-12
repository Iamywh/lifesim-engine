from __future__ import annotations

from lifesim.config import LifeSimConfig, SimulationConfig, WorldConfig
from lifesim.engine import LifeSimEngine


def make_config(seed: int) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(name="determinism-test", seed=seed, steps=3),
        world=WorldConfig(initial_population=4),
    )


def test_engine_is_deterministic_for_same_seed() -> None:
    first = LifeSimEngine(make_config(seed=99)).run()
    second = LifeSimEngine(make_config(seed=99)).run()

    assert first.to_dict() == second.to_dict()


def test_engine_changes_rng_sequence_for_different_seed() -> None:
    first = LifeSimEngine(make_config(seed=99)).run()
    second = LifeSimEngine(make_config(seed=100)).run()

    assert first.to_dict()["states"] != second.to_dict()["states"]


def test_engine_emits_initial_state_plus_configured_steps() -> None:
    result = LifeSimEngine(make_config(seed=1)).run()

    assert len(result.states) == 4
    assert result.states[0].step == 0
    assert result.states[-1].step == 3
    assert {state.population for state in result.states} == {4}
