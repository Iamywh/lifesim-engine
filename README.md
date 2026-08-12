# LifeSim Engine

LifeSim Engine is a deterministic simulation core for life-simulation experiments. M0 established the Python package skeleton and deterministic configuration. M1 added reusable composed agent state and a Maya starting scenario. M2 integrates those pieces with a generic weekly loop that can carry an agent state from week 0 through `duration_weeks`. M3 adds a reusable event engine for deterministic, state-conditioned weekly occurrences. M4 adds a transparent decision engine that chooses among event options without applying consequences.

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

To run Maya through the weekly engine with the starter event catalog and decision options:

```powershell
python scripts/run_demo.py --config configs/default.toml --agent-scenario configs/scenarios/maya_start.toml --event-catalog configs/events/starter.toml
```

Maya-specific values live in `configs/scenarios/maya_start.toml`; the core engine and agent model remain reusable for future characters. Event definitions and perceived decision options live in data under `configs/events/`. When an agent scenario is supplied, each weekly simulation snapshot includes the complete serialized `AgentState` plus that week's event occurrences, event selection traces, decisions, and score traces.

M1 stores monetary scenario values as quoted decimal strings in TOML, parses them to `Decimal`, and serializes them back to exact strings for future checkpoints and JSON logs. Maya is also represented with state-only education, health, mental, personality, finance, memory, and skill components; no education, employment, consequence, or personality evolution logic runs yet.

## Weekly Lifecycle

An agent run starts by seeding a per-run RNG, recording week 0 with the supplied immutable `AgentState`, and then advancing week by week through a small transition pipeline. Transitions receive a `WeeklyContext`, return a new `AgentState` or explicit transition result, and are validated before the next snapshot is recorded. Transitions must not retain run-specific mutable state between runs; durable simulation state belongs in `AgentState` or explicit run context. Later transitions in a week can inspect events produced by earlier transitions. Employment, relationship, memory-learning, consequence, and personality-evolution systems are intentionally left for later milestones.

## Event Engine

M3 events are observations, opportunities, or incidents selected by the run RNG from data-oriented `EventDefinition` records. The event engine evaluates safe condition primitives against `AgentState` and `WeeklyContext`, applies transparent weight modifiers, enforces cooldowns through run-level `EventHistory`, and records deterministic `EventOccurrence` values plus selection traces for auditability. Condition paths can traverse only declared dataclass state fields. Selection traces include trigger rolls, candidates, and weighted draw records with the random roll, total weight, and selected event id for each slot. Events do not mutate `AgentState` and do not choose agent responses or consequences.

## Decision Engine

M4 decisions choose among structured `EventOption` records attached to event occurrences. Option costs are exact quoted `Decimal` values in TOML and serialize back to strings; perceived utility signals are bounded floats. The decision engine filters unavailable options with the same safe condition mechanism used by events, scores available options with inspectable weighted components derived from personality, current state, time pressure, goals, and perceived costs, then adds small deterministic noise from a SHA-256-derived local RNG stream. Decision records and run-level `DecisionHistory` explain the chosen option, all option scores, unavailable options, and strongest positive/negative factors. Decisions do not mutate `AgentState`, create memories, apply costs, or produce consequences.

## Test

```powershell
pytest
ruff check .
```

## Project Layout

```text
configs/              Example and default configuration files
configs/events/       Event catalog files
configs/scenarios/    Agent scenario files
runs/                 Local simulation outputs; ignored except for .gitkeep
scripts/              Developer and demo entry-point scripts
src/lifesim/decisions/ Decision engine package
src/lifesim/          LifeSim Engine package
tests/                Pytest suite
```

## Determinism

Simulation runs are seeded from configuration. Each `run()` resets its event RNG from `simulation.seed`, so repeated runs with the same configuration produce the same results. Decision noise uses deterministic derived seeds and does not consume the event-selection RNG stream. RNG probes stay in tests rather than domain state.
