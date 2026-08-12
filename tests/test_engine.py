from __future__ import annotations

from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.engine import LifeSimEngine
from lifesim.rng import create_rng


def make_config(seed: int) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(name="determinism-test", seed=seed, duration_weeks=3),
        city=CityConfig(name="Test City"),
    )


def test_engine_is_deterministic_for_same_seed() -> None:
    engine = LifeSimEngine(make_config(seed=99))
    first = engine.run()
    second = engine.run()

    assert first.to_dict() == second.to_dict()


def test_rng_foundation_changes_sequence_for_different_seed() -> None:
    first_rng = create_rng(99)
    second_rng = create_rng(100)

    assert [first_rng.random() for _ in range(3)] != [second_rng.random() for _ in range(3)]


def test_engine_emits_initial_state_plus_configured_weeks() -> None:
    result = LifeSimEngine(make_config(seed=1)).run()

    assert len(result.states) == 4
    assert result.city_name == "Test City"
    assert result.states[0].week == 0
    assert result.states[-1].week == 3
    assert result.to_dict()["states"] == [{"week": 0}, {"week": 1}, {"week": 2}, {"week": 3}]
