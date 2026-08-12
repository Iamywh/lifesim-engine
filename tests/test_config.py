from __future__ import annotations

import pytest

from lifesim.config import LifeSimConfig, parse_config


def test_parse_config_builds_typed_config() -> None:
    config = parse_config(
        {
            "simulation": {"name": "test-run", "seed": 123, "steps": 2},
            "world": {"initial_population": 7},
        }
    )

    assert isinstance(config, LifeSimConfig)
    assert config.simulation.name == "test-run"
    assert config.simulation.seed == 123
    assert config.simulation.steps == 2
    assert config.world.initial_population == 7


def test_parse_config_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="name"):
        parse_config({"simulation": {"seed": 1, "steps": 1}, "world": {"initial_population": 1}})
