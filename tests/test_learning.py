from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import (
    AgentState,
    EpisodicMemory,
    MemoryState,
)
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.consequences import (
    ConsequenceCatalog,
    ConsequenceEngine,
    ConsequenceRecord,
    ConsequenceRuntimeState,
    DecisionConsequenceTransition,
    EffectApplication,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    ScheduledConsequenceTransition,
    ScheduledEffect,
    StateEffectDefinition,
)
from lifesim.decisions import DecisionEngine, DecisionEngineTransition
from lifesim.engine import LifeSimEngine
from lifesim.events import (
    EventCatalog,
    EventDefinition,
    EventEngine,
    EventEngineTransition,
    EventOccurrence,
    EventOption,
)
from lifesim.learning import (
    LearningEngine,
    LearningRuntimeState,
    LearningTransition,
    effective_memory_strength,
    evaluate_experience,
    retrieve_memory_signal,
)
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")


def test_backward_compatible_maya_memory_loading() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    assert isinstance(maya.memory, MemoryState)
    assert maya.memory.episodic_memories == ()


def test_consequence_creates_meaningful_episodic_memory() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    next_state, runtime, records = LearningEngine().learn_from_consequences(
        maya,
        context(),
        (consequence_record(applications=(application("mental.stress", 10.0, 57.0, 67.0),)),),
        LearningRuntimeState(),
    )

    episode = next_state.memory.episodic_memories[0]
    assert episode.source_decision_id == "decision_test"
    assert episode.source_event_id == "choice_event"
    assert episode.source_option_id == "chosen"
    assert episode.valence < 0
    assert episode.strength > 0
    assert episode.source_consequence_ids == ("consequence_test",)
    assert records[0].updates[0].memory_id == episode.memory_id
    assert runtime.history.records == records


def test_trivial_noop_consequence_does_not_create_artificial_memory() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    next_state, _, records = LearningEngine().learn_from_consequences(
        maya,
        context(),
        (consequence_record(applications=(application("mental.stress", 0.0, 57.0, 57.0),)),),
        LearningRuntimeState(),
    )

    assert next_state.memory == maya.memory
    assert records[0].evaluation.salience == 0.0
    assert records[0].updates == ()


def test_only_experienced_consequences_are_learned() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    outcome = OptionConsequenceDefinition(
        event_id="choice_event",
        event_version="1",
        option_id="chosen",
        outcomes=(
            # With seed 1 this high-weight branch is selected; the unchosen branch
            # must not leak into memory.
            OutcomeDefinition(
                "experienced",
                100.0,
                effects=(StateEffectDefinition(path="mental.stress", delta=2.0),),
            ),
            OutcomeDefinition(
                "unexperienced",
                1.0,
                effects=(StateEffectDefinition(path="education.progress", delta=25.0),),
            ),
        ),
    )
    engine = LifeSimEngine(
        config(duration_weeks=1, seed=1),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(ConsequenceCatalog((outcome,)))),
            LearningTransition(LearningEngine()),
        ),
    )

    result = engine.run(initial_agent=maya)
    episode = result.states[1].agent_state.memory.episodic_memories[0]

    assert episode.affected_domains == ("mental",)
    assert "education" not in episode.affected_domains
    assert result.states[1].consequences[0].selected_outcome_id == "experienced"


def test_delayed_consequence_reinforces_original_decision_experience() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    learning = LearningEngine()
    state, runtime, _ = learning.learn_from_consequences(
        maya,
        context(week=1),
        (consequence_record(consequence_id="consequence_immediate"),),
        LearningRuntimeState(),
    )
    state, runtime, _ = learning.learn_from_consequences(
        state,
        context(week=2),
        (consequence_record(consequence_id="consequence_delayed"),),
        runtime,
    )

    assert len(state.memory.episodic_memories) == 1
    episode = state.memory.episodic_memories[0]
    assert episode.exposure_count == 2
    assert runtime.processed_consequence_ids == ("consequence_immediate", "consequence_delayed")


def test_same_consequence_record_cannot_be_learned_twice() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    record = consequence_record()
    next_state, runtime, _ = LearningEngine().learn_from_consequences(
        maya,
        context(),
        (record,),
        LearningRuntimeState(),
    )
    again_state, runtime, records = LearningEngine().learn_from_consequences(
        next_state,
        context(),
        (record,),
        runtime,
    )

    assert again_state == next_state
    assert records == ()
    assert runtime.processed_consequence_ids == ("consequence_test",)


def test_learning_runtime_resets_between_runs() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = LifeSimEngine(
        config(duration_weeks=1, seed=1),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(ConsequenceCatalog((consequence_definition(),)))),
            LearningTransition(LearningEngine()),
        ),
    )

    first = engine.run(initial_agent=maya)
    second = engine.run(initial_agent=maya)

    assert first.to_dict() == second.to_dict()
    assert first.learning_history is not second.learning_history


def test_memory_processing_changes_only_agent_memory() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    next_state, _, _ = LearningEngine().learn_from_consequences(
        maya,
        context(),
        (consequence_record(),),
        LearningRuntimeState(),
    )
    before = maya.to_dict()
    after = next_state.to_dict()
    before.pop("memory")
    after.pop("memory")

    assert before == after
    assert next_state.memory != maya.memory


def test_repeated_positive_experiences_form_successful_pattern() -> None:
    state, _ = learn_sequence(
        (
            consequence_record(
                consequence_id="consequence_1",
                decision_id="decision_1",
                applications=(application("health.energy", 12.0, 40.0, 52.0),),
            ),
            consequence_record(
                consequence_id="consequence_2",
                decision_id="decision_2",
                applications=(application("health.energy", 12.0, 42.0, 54.0),),
            ),
        )
    )

    assert state.memory.successful_patterns
    assert state.memory.successful_patterns[0].valence > 0
    assert state.memory.lessons_learned


def test_repeated_negative_experiences_form_mistake_and_negative_lesson() -> None:
    state, _ = learn_sequence(
        (
            consequence_record(consequence_id="consequence_1", decision_id="decision_1"),
            consequence_record(consequence_id="consequence_2", decision_id="decision_2"),
        )
    )

    assert state.memory.mistakes
    assert state.memory.mistakes[0].valence < 0
    assert state.memory.lessons_learned[0].valence < 0


def test_single_modest_bad_outcome_does_not_create_extreme_generalized_avoidance() -> None:
    state, _ = learn_sequence((consequence_record(),))

    assert state.memory.mistakes == ()
    assert state.memory.lessons_learned == ()
    assert abs(state.memory.episodic_memories[0].valence) <= 1.0
    assert state.memory.episodic_memories[0].strength < 0.5


def test_memory_strength_decays_deterministically_and_reinforcement_counteracts_decay() -> None:
    decayed = effective_memory_strength(0.8, last_reinforced_week=1, week=5)
    repeat = effective_memory_strength(0.8, last_reinforced_week=1, week=5)
    first_salience = consequence_record(consequence_id="consequence_1", decision_id="decision_same")
    first_strength = evaluate_experience(first_salience).salience
    state, _ = learn_sequence(
        (
            first_salience,
            consequence_record(consequence_id="consequence_2", decision_id="decision_same"),
        ),
        weeks=(1, 5),
    )

    assert decayed == repeat
    assert state.memory.episodic_memories[0].strength > effective_memory_strength(first_strength, 1, 5)


def test_exact_event_option_memory_retrieves_more_strongly_than_broad_tag_match() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    memory = replace(
        maya.memory,
        episodic_memories=(
            EpisodicMemory(
                "Exact",
                1,
                80.0,
                memory_id="memory_exact",
                source_event_id="choice_event",
                source_event_version="1",
                source_option_id="chosen",
                last_reinforced_week=1,
                strength=0.8,
                valence=-0.5,
                tags=("money",),
                affected_domains=("financial",),
            ),
            EpisodicMemory(
                "Broad",
                1,
                80.0,
                memory_id="memory_broad",
                source_event_id="other_event",
                source_event_version="1",
                source_option_id="other",
                last_reinforced_week=1,
                strength=0.8,
                valence=-0.5,
                tags=("money",),
                affected_domains=("financial",),
            ),
        ),
    )
    state = replace(maya, memory=memory)

    result = retrieve_memory_signal(state, 1, occurrence(tags=("money",)), option(estimated_cost=Decimal("1.00")))

    exact = next(item for item in result.evidence if item.memory_id == "memory_exact")
    broad = next(item for item in result.evidence if item.memory_id == "memory_broad")
    assert abs(exact.contribution) > abs(broad.contribution)


def test_irrelevant_memories_contribute_zero_and_signal_is_bounded() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    memory = replace(
        maya.memory,
        episodic_memories=(
            EpisodicMemory(
                "Irrelevant",
                1,
                90.0,
                memory_id="memory_irrelevant",
                source_event_id="other",
                source_event_version="1",
                source_option_id="other",
                last_reinforced_week=1,
                strength=1.0,
                valence=-1.0,
                tags=("other",),
                affected_domains=("other",),
        ),
        ),
    )
    state = replace(maya, memory=memory)

    irrelevant = retrieve_memory_signal(state, 1, occurrence(), option())
    strong = replace(
        state,
        memory=replace(
            state.memory,
            episodic_memories=tuple(
                EpisodicMemory(
                    f"Exact {index}",
                    1,
                    100.0,
                    memory_id=f"memory_{index}",
                    source_event_id="choice_event",
                    source_event_version="1",
                    source_option_id="chosen",
                    last_reinforced_week=1,
                    strength=1.0,
                    valence=-1.0,
                )
                for index in range(3)
            ),
        ),
    )

    assert irrelevant.signal == 0.0
    assert retrieve_memory_signal(strong, 1, occurrence(), option()).signal == -1.0


def test_future_decision_can_differ_because_of_learned_experience() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    event = occurrence(
        options=(
            option("repeat_bad", short_term_value=0.18),
            option("alternative", short_term_value=0.0),
        )
    )
    baseline = DecisionEngine().decide_event(maya, context(), event).chosen_option_id
    learned_state, _ = learn_sequence(
        (
            consequence_record(
                consequence_id="consequence_1",
                decision_id="decision_1",
                option_id="repeat_bad",
                applications=(application("mental.stress", 20.0, 57.0, 77.0),),
            ),
            consequence_record(
                consequence_id="consequence_2",
                decision_id="decision_2",
                option_id="repeat_bad",
                applications=(application("mental.stress", 20.0, 57.0, 77.0),),
            ),
        )
    )
    learned = DecisionEngine().decide_event(learned_state, context(week=1), event)
    repeat_eval = next(item for item in learned.evaluations if item.option_id == "repeat_bad")

    assert baseline == "repeat_bad"
    assert learned.chosen_option_id == "alternative"
    assert repeat_eval.memory_evidence
    assert any(component.name == "memory_experience" for component in repeat_eval.components)


def test_first_decision_before_any_experience_matches_baseline_and_not_retroactive() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    baseline = LifeSimEngine(
        config(duration_weeks=1, seed=1),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
        ),
    ).run(initial_agent=maya)
    with_learning = LifeSimEngine(
        config(duration_weeks=1, seed=1),
        transitions=(
            LearningTransition(LearningEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(ConsequenceCatalog((consequence_definition(),)))),
            LearningTransition(LearningEngine()),
        ),
    ).run(initial_agent=maya)

    assert baseline.states[1].decisions[0].to_dict() == with_learning.states[1].decisions[0].to_dict()
    assert with_learning.states[1].agent_state.memory.episodic_memories
    assert with_learning.states[1].decisions[0].evaluations[0].memory_evidence == ()


def test_scheduled_consequence_is_learned_before_same_week_decision() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    delayed = ScheduledEffect(
        scheduled_effect_id="scheduled_bad",
        source_decision_id="decision_previous",
        source_consequence_id="consequence_previous",
        source_event_id="choice_event",
        source_event_version="1",
        chosen_option_id="chosen",
        source_outcome_id=None,
        created_week=0,
        due_week=1,
        effect=StateEffectDefinition(path="mental.stress", delta=20.0, delay_weeks=1),
    )
    result = LifeSimEngine(
        config(duration_weeks=1, seed=1),
        transitions=(
            SeedConsequenceRuntime(ConsequenceRuntimeState(pending_scheduled_effects=(delayed,))),
            ScheduledConsequenceTransition(ConsequenceEngine(ConsequenceCatalog())),
            LearningTransition(LearningEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
        ),
    ).run(initial_agent=maya)

    evaluation = result.states[1].decisions[0].evaluations[0]
    assert result.states[1].learning_records[0].source_decision_id == "decision_previous"
    assert evaluation.memory_evidence


def test_mixed_social_outcome_produces_limited_valence() -> None:
    record = consequence_record(
        event_id="social_invitation",
        option_id="accept_invitation",
        applications=(
            application("financial.bank_balance", Decimal("-20.00"), Decimal("980.00"), Decimal("960.00")),
            application("health.energy", -12.0, 64.0, 52.0),
            application("mental.loneliness", -8.0, 48.0, 40.0),
            application("needs.belonging", 6.0, 45.0, 51.0),
        ),
    )

    evaluation = evaluate_experience(record)

    assert -0.4 < evaluation.valence < 0.4
    assert set(evaluation.affected_domains) == {"financial", "mental", "physical", "social"}


def test_generic_agent_repeated_runs_and_prior_timelines_remain_deterministic() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    generic = replace(maya, identity=replace(maya.identity, agent_id="alex", display_name="Alex"))
    transitions = (
        ScheduledConsequenceTransition(ConsequenceEngine(ConsequenceCatalog())),
        LearningTransition(LearningEngine()),
        EventEngineTransition(EventEngine(event_catalog())),
        DecisionEngineTransition(DecisionEngine()),
        DecisionConsequenceTransition(ConsequenceEngine(ConsequenceCatalog())),
        LearningTransition(LearningEngine()),
    )
    with_learning = LifeSimEngine(config(duration_weeks=2, seed=9), transitions=transitions)
    without_learning = LifeSimEngine(
        config(duration_weeks=2, seed=9),
        transitions=(
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(ConsequenceCatalog())),
        ),
    )

    first = with_learning.run(initial_agent=generic)
    second = with_learning.run(initial_agent=generic)
    baseline = without_learning.run(initial_agent=generic)

    assert first.to_dict() == second.to_dict()
    assert [state.to_dict()["events"] for state in first.states[1:]] == [
        state.to_dict()["events"] for state in baseline.states[1:]
    ]
    assert [state.to_dict()["consequences"] for state in first.states[1:]] == [
        state.to_dict()["consequences"] for state in baseline.states[1:]
    ]


def test_cli_demo_exposes_learning_audit_json(tmp_path: Path) -> None:
    events_path = tmp_path / "events.toml"
    consequences_path = tmp_path / "consequences.toml"
    events_path.write_text(event_catalog_text(), encoding="utf-8")
    consequences_path.write_text(
        """
[[consequences]]
event_id = "choice_event"
event_version = "1"
option_id = "chosen"

[[consequences.effects]]
path = "mental.stress"
delta = 6.0
delay_weeks = 0
""".strip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--config",
            "configs/default.toml",
            "--agent-scenario",
            "configs/scenarios/maya_start.toml",
            "--event-catalog",
            str(events_path),
            "--consequence-catalog",
            str(consequences_path),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    output = json.loads(completed.stdout)

    assert output["states"][1]["learning_records"]
    assert output["learning_history"]["records"]


def learn_sequence(
    records: tuple[ConsequenceRecord, ...],
    *,
    weeks: tuple[int, ...] | None = None,
) -> tuple[AgentState, LearningRuntimeState]:
    state = load_agent_state(MAYA_SCENARIO)
    runtime = LearningRuntimeState()
    if weeks is None:
        weeks = tuple(range(1, len(records) + 1))
    for week, record in zip(weeks, records, strict=True):
        state, runtime, _ = LearningEngine().learn_from_consequences(
            state,
            context(week=week),
            (record,),
            runtime,
        )
    return state, runtime


def config(*, duration_weeks: int = 1, seed: int = 1) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(name="learning-test", seed=seed, duration_weeks=duration_weeks),
        city=CityConfig(name="Veyra"),
    )


def context(*, week: int = 1, seed: int = 1) -> WeeklyContext:
    return WeeklyContext(week=week, config=config(seed=seed), rng=__import__("random").Random(seed))


def event_catalog() -> EventCatalog:
    return EventCatalog((event_definition(),), event_probability=1.0)


def event_definition() -> EventDefinition:
    return EventDefinition(
        event_id="choice_event",
        version="1",
        category="test",
        base_weight=1.0,
        conditions=(),
        weight_modifiers=(),
        cooldown_weeks=0,
        tags=("money",),
        title="Choice event",
        summary="A synthetic choice event.",
        options=(option(),),
    )


def occurrence(
    *,
    event_id: str = "choice_event",
    tags: tuple[str, ...] = ("money",),
    options: tuple[EventOption, ...] = (EventOption(
        option_id="chosen",
        label="Chosen",
        summary="Chosen option.",
        short_term_value=0.2,
        goal_tags=("money",),
    ),),
) -> EventOccurrence:
    return EventOccurrence(
        event_id=event_id,
        version="1",
        week=1,
        category="test",
        effective_weight=1.0,
        title="Choice event",
        summary="A synthetic choice event.",
        tags=tags,
        options=options,
    )


def option(
    option_id: str = "chosen",
    *,
    short_term_value: float = 0.2,
    estimated_cost: Decimal = Decimal("0.00"),
) -> EventOption:
    return EventOption(
        option_id=option_id,
        label=option_id.title(),
        summary=f"Synthetic {option_id} option.",
        estimated_cost=estimated_cost,
        short_term_value=short_term_value,
        goal_tags=("money",),
    )


def consequence_definition() -> OptionConsequenceDefinition:
    return OptionConsequenceDefinition(
        event_id="choice_event",
        event_version="1",
        option_id="chosen",
        effects=(StateEffectDefinition(path="mental.stress", delta=6.0),),
    )


def consequence_record(
    *,
    consequence_id: str = "consequence_test",
    decision_id: str = "decision_test",
    event_id: str = "choice_event",
    option_id: str = "chosen",
    applications: tuple[EffectApplication, ...] = (
        EffectApplication(
            path="mental.stress",
            requested_delta=6.0,
            before=57.0,
            after=63.0,
            clamped=False,
        ),
    ),
) -> ConsequenceRecord:
    return ConsequenceRecord(
        consequence_id=consequence_id,
        source_decision_id=decision_id,
        source_event_id=event_id,
        source_event_version="1",
        chosen_option_id=option_id,
        week_resolved=1,
        effect_applications=applications,
    )


def application(
    path: str,
    delta: float | Decimal,
    before: float | Decimal,
    after: float | Decimal,
) -> EffectApplication:
    return EffectApplication(
        path=path,
        requested_delta=delta,
        before=before,
        after=after,
        clamped=False,
    )


def event_catalog_text() -> str:
    return """
[event_settings]
max_events_per_week = 1
event_probability = 1.0

[[events]]
event_id = "choice_event"
version = "1"
category = "test"
base_weight = 1.0
cooldown_weeks = 0
tags = ["money"]
title = "Choice event"
summary = "A deterministic event."

[[events.options]]
option_id = "chosen"
label = "Chosen"
summary = "Chosen option."
estimated_cost = "0.00"
time_cost_hours = 0.0
energy_cost = 0.0
short_term_value = 0.2
future_value = 0.0
perceived_risk = 0.0
uncertainty = 0.0
social_value = 0.0
social_pressure = 0.0
autonomy_value = 0.0
learning_value = 0.0
health_value = 0.0
comfort_value = 0.0
goal_tags = ["money"]
""".strip()


class SeedConsequenceRuntime:
    def __init__(self, runtime: ConsequenceRuntimeState) -> None:
        self._runtime = runtime

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        return WeeklyTransitionResult(agent_state=state, consequence_runtime=self._runtime)
