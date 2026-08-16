# LifeSim Engine

LifeSim Engine is a deterministic simulation core for life-simulation experiments. M0 established the Python package skeleton and deterministic configuration. M1 added reusable composed agent state and a Maya starting scenario. M2 integrates those pieces with a generic weekly loop that can carry an agent state from week 0 through `duration_weeks`. M3 adds a reusable event engine for deterministic, state-conditioned weekly occurrences. M4 adds a transparent decision engine that chooses among event options. M5 adds a consequence engine that applies actual state changes and delayed effects from chosen decisions. M6 adds memory and learning so experienced consequences can influence future decisions. M7 adds passive life and routine state so ordinary weeks can still advance Maya's money, needs, health, and mental load. M8 adds a deterministic employment market, hiring pipeline, contracts, wages, and weekly work effects. M9 adds zero-RNG skills and education development through M4 weekly development choices.

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
python scripts/run_demo.py --config configs/default.toml --agent-scenario configs/scenarios/maya_start.toml --event-catalog configs/events/starter.toml --consequence-catalog configs/consequences/starter.toml --routine-catalog configs/routines/starter.toml --employment-catalog configs/employment/starter.toml --development-catalog configs/development/starter.toml
```

Maya-specific values live in `configs/scenarios/maya_start.toml`; the core engine and agent model remain reusable for future characters. Event definitions and perceived decision options live in data under `configs/events/`; real outcome definitions live separately under `configs/consequences/`; weekly routine profiles live under `configs/routines/`; employment opportunities live under `configs/employment/`; development skills, education programs, and weekly study/practice profiles live under `configs/development/`. When an agent scenario is supplied, each weekly simulation snapshot includes the complete serialized `AgentState` plus that week's event occurrences, event selection traces, decisions, score traces, consequence records, learning records, passive life records, employment records, and development records.

M1 stores monetary scenario values as quoted decimal strings in TOML, parses them to `Decimal`, and serializes them back to exact strings for future checkpoints and JSON logs. M5 uses the same exact Decimal handling for monetary consequence deltas, and M8 uses it for hourly rates and weekly wages. Maya is also represented with state-only education, health, mental, personality, finance, memory, and skill components; no relationship-specific, promotion, performance-review, personality, or habit evolution logic runs yet.

## Weekly Lifecycle

An agent run starts by seeding a per-run RNG, recording week 0 with the supplied immutable `AgentState`, and then advancing week by week through a small transition pipeline. Transitions receive a `WeeklyContext`, return a new `AgentState` or explicit transition result, and are validated before the next snapshot is recorded. Transitions must not retain run-specific mutable state between runs; durable simulation state belongs in `AgentState` or explicit run context. Later transitions in a week can inspect events, decisions, and consequences produced by earlier transitions.

The default M9 causal order is:

```text
scheduled consequences due this week
memory learning from those scheduled consequences
employment boundary starts/ends due this week
passive income, obligations, debt interest, debt payments, and arrears
routine planning through the Decision Engine
development planning through the Decision Engine
employment market discovery and employer stage resolution
event selection
decision selection using current memory
immediate consequences and newly scheduled delayed effects
employment process updates from M4 decisions
memory learning from immediate consequences
weekly work effects
development execution and education/skill progress
routine execution and passive routine effects
```

## Event Engine

M3 events are observations, opportunities, or incidents selected by the run RNG from data-oriented `EventDefinition` records. The event engine evaluates safe condition primitives against `AgentState` and `WeeklyContext`, applies transparent weight modifiers, enforces cooldowns through run-level `EventHistory`, and records deterministic `EventOccurrence` values plus selection traces for auditability. Condition paths can traverse only declared dataclass state fields. Selection traces include trigger rolls, candidates, and weighted draw records with the random roll, total weight, and selected event id for each slot. Events do not mutate `AgentState` and do not choose agent responses or consequences.

## Decision Engine

M4 decisions choose among structured `EventOption` records attached to event occurrences. Option costs and expected recurring weekly financial gains are exact quoted `Decimal` values in TOML and serialize back to strings; perceived utility signals and ongoing weekly time commitments are bounded floats. The decision engine filters unavailable options with the same safe condition mechanism used by events, scores available options with inspectable weighted components derived from personality, current state, time pressure, goals, perceived costs, recurring financial upside, and recurring time load, then adds small deterministic noise from a SHA-256-derived local RNG stream. Decision records and run-level `DecisionHistory` explain the chosen option, all option scores, unavailable options, and strongest positive/negative factors. Decisions do not mutate `AgentState`, create memories, apply costs, or produce consequences.

## Consequence Engine

M5 consequences are actual outcomes keyed by `event_id`, `event_version`, and `option_id` in a separate consequence catalog. The Decision Engine never inspects these real effects. Consequences can apply immediate additive state deltas, select one deterministic weighted actual outcome branch, or schedule delayed effects for future weeks. Writable paths are explicitly allowlisted; identity, personality, memory, skills, employment, goals, and relationship-specific state are excluded from M5.

Consequence application is atomic at the chosen outcome level. Decimal monetary underflow fails clearly without partial mutation. Bounded 0-100 float fields clamp with explicit before/after/clamped trace records, while delayed effects preserve provenance from decision to consequence to scheduled effect to application. Run-level `ConsequenceRuntimeState` resets for every `run()` and serializes consequence history plus pending scheduled effects.

Same-week decision consequences are resolved in the order carried by `WeeklyContext.decisions`; transition code must not reorder them by id or hash. Delayed scheduled effects due in the same week resolve by earliest `due_week`, preserving the existing runtime tuple order for effects with equal due weeks.

## Memory & Learning Engine

M6 keeps objective history, psychological memory, and decision influence separate. `ConsequenceHistory` remains engine truth about what happened. `AgentState.memory` stores what the agent retains from experienced consequences. The Decision Engine then reads structured memories as bounded `memory_experience` score evidence for future options.

Learning is deterministic and state-only. It evaluates actual applied consequence deltas into bounded valence, salience, affected domains, and strongest positive/negative effects. Meaningful experiences create or reinforce episodic memories; delayed effects from the same source decision reinforce the original episode. Repeated exact event+option experiences can form mistakes, lessons, or successful patterns, while single modest outcomes remain limited evidence.

The learning transition mutates only `AgentState.memory`. It never fabricates consequences, changes option availability, reads hidden outcome probabilities, sees unexperienced outcome branches, or alters personality, skills, relationships, employment, goals, finances, health, needs, or education directly. Learning records and decision memory evidence make the chain auditable from consequence to memory update to future decision contribution.

## Passive Life & Routine Engine

M7 models the ordinary mechanics of a week. Passive financial life applies income, rent, recurring commitments, debt interest, debt minimum payments, and arrears using exact `Decimal` money. Obligatory shortfalls are audited and can create or reinforce arrears without allowing balances to go negative.

Routine planning creates a normal, auditable weekly routine decision from data-defined routine profiles, using the existing Decision Engine. Routine execution then applies the selected profile's actual passive effects to bounded health, mental, needs, routine, and city-familiarity state. It does not create events, hidden consequences, memories, skills, employment, relationships, or personality evolution.

Special events remain separate: they are stochastic interruptions or opportunities selected by the Event Engine around ordinary life, not replacements for it.

A week in which nothing exceptional happens is still a week of life: Maya still eats, rests, studies, moves around the city, pays bills when due, and carries the effects of tight resources or recovery-focused choices.

## Employment Engine

M8 separates the market from the agent. The market creates opportunities and outcomes; the agent chooses how to respond. Job discovery, interview invitations, rejections, offers, contract starts, fixed-term endings, and weekly work demands are external employment mechanics. Applying, attending an interview, and accepting an offer are normal M4 decisions represented as synthetic `EventOccurrence` values with visible `EventOption` tradeoffs, including generic recurring pay and time signals rather than employment-specific prose parsing.

Employment is probabilistic and auditable. Search intensity and city familiarity influence whether a market discovery slot triggers; they do not change the relative weighted selection among employers once a slot is triggered. Each discovery slot records trigger probability, trigger roll, whether it triggered, weighted-selection total, weighted roll, and selected job key when applicable. Skill fit, city familiarity, education status, confidence, conscientiousness, and stress can influence employer-side interview and offer probabilities, but skills never guarantee success and weak fit is not automatic failure. Protected or arbitrary identity presentation fields such as pronouns, origin city, and background prose are not used in fit or probability calculations.

The hiring sequence is deliberately paced across weeks: an opening can be discovered, Maya may apply, the employer resolves the application later, an interview invitation can create another M4 decision, a later employer response can create an offer decision, and accepted offers start no earlier than the next week. Employment wages are installed as weekly `IncomeStream` values with `source_type = "employment"` and are paid by the M7 passive cashflow engine; M8 does not maintain a second payroll or cash-balance system.

Active work applies modest deterministic weekly effects to ordinary-life state such as energy, stress, mental load, recovery need, and purpose, then routine execution can help Maya recover or socialize. Fixed-term contracts use `end_week_exclusive`: a contract starting in week 10 for 8 working weeks is active for weeks 10 through 17 and ends at the start of week 18.

## Skills & Education Engine

M9 keeps practical development separate from M6 psychological memory. The guiding rules are: "Practice creates experience; experience can create skill." and "Enrollment creates an opportunity to progress, not automatic progress."

Development profiles are data-defined weekly study/practice options. A `DevelopmentPlanningTransition` creates one `weekly_development` event per week and calls the normal M4 Decision Engine; there is no separate development decision brain. The chosen profile records planned study hours, practice allocations, and the same transparent M4 score audit as other choices.

Development execution is deterministic and zero-RNG. It consumes the same-week M4 development decision exactly once, then converts actual study, direct practice, and completed M8 work records into skill experience. Skill growth aggregates sources first, applies smooth diminishing returns, and never decreases skills or exceeds level 100. Missing skills can be created only when valid catalog-defined experience is genuinely gained.

Education progress depends on actual chosen study hours, nominal weekly study load, current energy, stress, mental load, recovery need, and combined work plus development workload. Progress is gradual, non-decreasing, clamped at 100, and advances academic year from overall program progress. Completing a degree changes only education state and whatever skills were actually developed; it does not create money, employment, personality changes, relationships, or memories.

## Test

```powershell
pytest
ruff check .
```

## Project Layout

```text
configs/              Example and default configuration files
configs/consequences/ Consequence catalog files
configs/development/  Skills, education programs, and weekly development profiles
configs/employment/   Employment job-market catalog files
configs/events/       Event catalog files
configs/routines/     Weekly routine profile catalog files
configs/scenarios/    Agent scenario files
runs/                 Local simulation outputs; ignored except for .gitkeep
scripts/              Developer and demo entry-point scripts
src/lifesim/consequences/ Consequence engine package
src/lifesim/decisions/ Decision engine package
src/lifesim/development/ Skills and education development package
src/lifesim/employment/ Employment market and work engine package
src/lifesim/learning/ Memory and learning engine package
src/lifesim/passive/  Passive life and routine engine package
src/lifesim/          LifeSim Engine package
tests/                Pytest suite
```

## Determinism

Simulation runs are seeded from configuration. Each `run()` resets its event RNG from `simulation.seed`, so repeated runs with the same configuration produce the same results. Decision noise, consequence outcome selection, passive income reliability, employment market discovery, application responses, and interview outcomes use deterministic SHA-256-derived local seeds and do not consume each other's RNG streams. Routine execution, work effects, development execution, skill growth, education progress, memory formation, reinforcement, decay, and retrieval are deterministic and use no RNG. RNG probes stay in tests rather than domain state.
