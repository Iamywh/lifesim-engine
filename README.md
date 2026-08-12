# LifeSim Engine

LifeSim Engine is a deterministic simulation core for life-simulation experiments. M0 establishes the Python package skeleton, configuration loading, reproducible random number generation, and a minimal week-based engine loop that future milestones can expand.

## Requirements

- Python 3.12+
- `pip`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run

```powershell
python scripts/run_demo.py --config configs/default.toml
```

The default configuration defines a simulation name, seed, duration in weeks, and contextual city metadata. Later milestones will add agents within that city context.

## Test

```powershell
pytest
ruff check .
```

## Project Layout

```text
configs/              Example and default configuration files
runs/                 Local simulation outputs; ignored except for .gitkeep
scripts/              Developer and demo entry-point scripts
src/lifesim/          LifeSim Engine package
tests/                Pytest suite
```

## Determinism

Simulation runs are seeded from configuration. Each `run()` resets its RNG from `simulation.seed`, so repeated runs with the same configuration produce the same results. RNG probes stay in tests rather than domain state.
