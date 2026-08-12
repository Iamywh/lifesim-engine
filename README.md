# LifeSim Engine

LifeSim Engine is a deterministic simulation core for life-simulation experiments. M0 established the Python package skeleton, configuration loading, reproducible random number generation, and a minimal week-based engine loop. M1 adds reusable composed agent state and a Maya starting scenario.

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

To print Maya's initial immutable state snapshot from scenario data:

```powershell
python scripts/run_demo.py --config configs/default.toml --agent-scenario configs/scenarios/maya_start.toml
```

Maya-specific values live in `configs/scenarios/maya_start.toml`; the core engine and agent model remain reusable for future characters.

M1 stores monetary scenario values as quoted decimal strings in TOML, parses them to `Decimal`, and serializes them back to exact strings for future checkpoints and JSON logs. Maya is also represented with state-only education, health, mental, personality, finance, memory, and skill components; no education, employment, decision, event, or personality evolution logic runs yet.

## Test

```powershell
pytest
ruff check .
```

## Project Layout

```text
configs/              Example and default configuration files
configs/scenarios/    Agent scenario files
runs/                 Local simulation outputs; ignored except for .gitkeep
scripts/              Developer and demo entry-point scripts
src/lifesim/          LifeSim Engine package
tests/                Pytest suite
```

## Determinism

Simulation runs are seeded from configuration. Each `run()` resets its RNG from `simulation.seed`, so repeated runs with the same configuration produce the same results. RNG probes stay in tests rather than domain state.
