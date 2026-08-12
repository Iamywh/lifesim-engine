# LifeSim Engine

LifeSim Engine is a deterministic simulation core for life-simulation experiments. M0 established the Python package skeleton and deterministic configuration. M1 added reusable composed agent state and a Maya starting scenario. M2 integrates those pieces with a generic weekly loop that can carry an agent state from week 0 through `duration_weeks`. M3 adds a reusable event engine for deterministic, state-conditioned weekly occurrences. M4 adds a transparent decision engine that chooses among event options. M5 adds a consequence engine that applies actual state changes and delayed effects from chosen decisions.

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

To run Maya through the weekly engine with starter events, decisions, and consequences:

```powershell
python scripts/run_demo.py --config configs/default.toml --agent-scenario configs/scenarios/maya_start.toml --event-catalog configs/events/starter.toml --consequence-catalog configs/consequences/starter.toml
```

Maya-specific values live in `configs/scenarios/maya_start.toml`; the core engine and agent model remain reusable for future characters. Event definitions and perceived decision options live in data under `configs/events/`; real outcome definitions live separately under `configs/consequences/`. When an agent scenario is supplied, each weekly simulation snapshot includes the complete serialized `AgentState` plus that week's event occurrences, event selection traces, decisions, score traces, and consequence records.

M1 stores monetary scenario values as quoted decimal strings in TOML, parses them to `Decimal`, and serializes them back to exact strings for future checkpoints and JSON logs. M5 uses the same exact Decimal handling for monetary consequence deltas. Maya is also represented with state-only education, health, mental, personality, finance, memory, and skill components; no employment, memory-learning, relationship-specific, skill-learning, or personality evolution logic runs yet.

## Weekly Lifecycle

An agent run starts by seeding a per-run RNG, recording week 0 with the supplied immutable `AgentState`, and then advancing week by week through a small transition pipeline. Transitions receive a `WeeklyContext`, return a new `AgentState` or explicit transition result, and are validated before the next snapshot is recorded. Transitions must not retain run-specific mutable state between runs; durable simulation state belongs in `AgentState` or explicit run context. Later transitions in a week can inspect events, decisions, and consequences produced by earlier transitions.

The default M5 causal order is:

```text
scheduled consequences due this week
event selection
decision selection
immediate consequences and newly scheduled delayed effects
```

## Event Engine

M3 events are observations, opportunities, or incidents selected by the run RNG from data-oriented `EventDefinition` records. The event engine evaluates safe condition primitives against `AgentState` and `WeeklyContext`, applies transparent weight modifiers, enforces cooldowns through run-level `EventHistory`, and records deterministic `EventOccurrence` values plus selection traces for auditability. Condition paths can traverse only declared dataclass state fields. Selection traces include trigger rolls, candidates, and weighted draw records with the random roll, total weight, and selected event id for each slot. Events do not mutate `AgentState` and do not choose agent responses or consequences.

## Decision Engine

M4 decisions choose among structured `EventOption` records attached to event occurrences. Option costs are exact quoted `Decimal` values in TOML and serialize back to strings; perceived utility signals are bounded floats. The decision engine filters unavailable options with the same safe condition mechanism used by events, scores available options with inspectable weighted components derived from personality, current state, time pressure, goals, and perceived costs, then adds small deterministic noise from a SHA-256-derived local RNG stream. Decision records and run-level `DecisionHistory` explain the chosen option, all option scores, unavailable options, and strongest positive/negative factors. Decisions do not mutate `AgentState`, create memories, apply costs, or produce consequences.

## Consequence Engine

M5 consequences are actual outcomes keyed by `event_id`, `event_version`, and `option_id` in a separate consequence catalog. The Decision Engine never inspects these real effects. Consequences can apply immediate additive state deltas, select one deterministic weighted actual outcome branch, or schedule delayed effects for future weeks. Writable paths are explicitly allowlisted; identity, personality, memory, skills, employment, goals, and relationship-specific state are excluded from M5.

Consequence application is atomic at the chosen outcome level. Decimal monetary underflow fails clearly without partial mutation. Bounded 0-100 float fields clamp with explicit before/after/clamped trace records, while delayed effects preserve provenance from decision to consequence to scheduled effect to application. Run-level `ConsequenceRuntimeState` resets for every `run()` and serializes consequence history plus pending scheduled effects.

## Test

```powershell
pytest
ruff check .
```

## Project Layout

```text
configs/              Example and default configuration files
configs/consequences/ Consequence catalog files
configs/events/       Event catalog files
configs/scenarios/    Agent scenario files
runs/                 Local simulation outputs; ignored except for .gitkeep
scripts/              Developer and demo entry-point scripts
src/lifesim/consequences/ Consequence engine package
src/lifesim/decisions/ Decision engine package
src/lifesim/          LifeSim Engine package
tests/                Pytest suite
```

## Determinism

Simulation runs are seeded from configuration. Each `run()` resets its event RNG from `simulation.seed`, so repeated runs with the same configuration produce the same results. Decision noise and consequence outcome selection use deterministic SHA-256-derived local seeds and do not consume the event-selection RNG stream. RNG probes stay in tests rather than domain state.
