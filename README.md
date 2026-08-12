# LifeSim Engine

LifeSim Engine is a deterministic simulation core for life-simulation experiments. M0 establishes the Python package skeleton, configuration loading, reproducible random number generation, and a minimal engine loop that future milestones can expand.

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

## Test

```powershell
pytest
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

Simulation runs are seeded from configuration. Use the same `simulation.seed` and deterministic engine inputs to reproduce the same RNG sequence and resulting states.
