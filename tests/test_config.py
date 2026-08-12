from __future__ import annotations

import pytest

from lifesim.config import LifeSimConfig, parse_config


def test_parse_config_builds_typed_config() -> None:
    config = parse_config(
        {
            "simulation": {"name": "test-run", "seed": 123, "duration_weeks": 2},
            "city": {"name": "Test City"},
        }
    )

    assert isinstance(config, LifeSimConfig)
    assert config.simulation.name == "test-run"
    assert config.simulation.seed == 123
    assert config.simulation.duration_weeks == 2
    assert config.city.name == "Test City"


def test_parse_config_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="name"):
        parse_config({"simulation": {"seed": 1, "duration_weeks": 1}, "city": {"name": "Test City"}})
