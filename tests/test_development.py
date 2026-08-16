from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from random import Random

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import EducationState, MentalState, SkillRating, SkillsState
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.consequences import (
    ConsequenceCatalog,
    ConsequenceEngine,
    DecisionConsequenceTransition,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    StateEffectDefinition,
)
from lifesim.decisions import DecisionEngine, DecisionEngineTransition, DecisionHistory
from lifesim.decisions.model import DecisionRecord, OptionEvaluation
from lifesim.development import (
    DEVELOPMENT_EVENT_ID,
    DevelopmentEngine,
    DevelopmentExecutionTransition,
    DevelopmentPlanningTransition,
    DevelopmentRuntimeState,
    load_development_catalog,
    parse_development_catalog,
)
from lifesim.employment import (
    EmploymentCatalog,
    EmploymentMarketEngine,
    EmploymentMarketTransition,
    EmploymentWorkEngine,
    EmploymentWorkTransition,
    EmploymentWorkWeekRecord,
    JobDefinition,
    SkillRequirement,
    candidate_fit,
    load_employment_catalog,
)
from lifesim.engine import LifeSimEngine
from lifesim.events import (
    EventCatalog,
    EventDefinition,
    EventEngine,
    EventEngineTransition,
    EventOption,
)
from lifesim.passive import PassiveCashflowEngine, PassiveCashflowTransition
from lifesim.weekly import WeeklyContext

MAYA_SCENARIO = Path("configs/scenarios/maya_start.toml")
DEVELOPMENT = Path("configs/development/starter.toml")
EMPLOYMENT = Path("configs/employment/starter.toml")


def test_development_catalog_loads_and_validates_data() -> None:
    catalog = load_development_catalog(DEVELOPMENT)

    assert catalog.program("Urban Studies BA").progress_per_full_study_week == 1.6
    assert catalog.profile("reduced_study").education_hours == 8.0
    assert catalog.profile("light_self_development").education_hours == 0.0
    with pytest.raises(ValueError, match="skill_name"):
        parse_development_catalog(catalog_raw(skills=[skill_raw("dup"), skill_raw("dup")]))
    with pytest.raises(ValueError, match="Unknown curriculum skill"):
        parse_development_catalog(catalog_raw(programs=[program_raw(curriculum_skill="missing")]))
    with pytest.raises(ValueError, match="learning_rate"):
        parse_development_catalog(catalog_raw(skills=[skill_raw("bad", learning_rate=0.0)]))
    with pytest.raises(ValueError, match="progress_per_full_study_week"):
        parse_development_catalog(catalog_raw(programs=[program_raw(progress_per_full_study_week=-1.0)]))
    with pytest.raises(ValueError, match="education_hours"):
        parse_development_catalog(catalog_raw(profiles=[profile_raw("bad_hours", education_hours=-1.0)]))
    with pytest.raises(ValueError, match="practice skill"):
        parse_development_catalog(
            catalog_raw(
                profiles=[
                    profile_raw(
                        "duplicate_practice",
                        practice=[
                            {"skill_name": "research", "hours": 1.0},
                            {"skill_name": "research", "hours": 2.0},
                        ],
                    )
                ]
            )
        )


def test_existing_state_is_backward_compatible_and_maya_unchanged() -> None:
    maya = load_agent_state(MAYA_SCENARIO)

    assert maya.education == EducationState(
        status="enrolled",
        program="Urban Studies BA",
        current_year=2,
        total_years=3,
        progress=45.0,
        weekly_study_hours=18.0,
    )
    assert [skill.name for skill in maya.skills.items] == [
        "customer service",
        "basic budgeting",
        "spreadsheets",
        "local language",
    ]
    assert SkillRating("new", "test", 0.0).experience == 0.0
    EducationState("completed", "Urban Studies BA", 3, 3, 100.0, 18.0)


def test_planning_creates_one_weekly_development_opportunity_without_deciding() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = DevelopmentEngine(load_development_catalog(DEVELOPMENT))
    runtime, event = engine.plan(
        maya,
        context(),
        DevelopmentRuntimeState(),
    )

    assert event.event_id == DEVELOPMENT_EVENT_ID
    assert event.category == "development"
    assert len(event.options) == len(load_development_catalog(DEVELOPMENT).profiles)
    assert runtime.history.plan_records[0].available_profile_ids == tuple(option.option_id for option in event.options)
    assert runtime.planned_profile_ids == runtime.history.plan_records[0].available_profile_ids
    assert runtime.planned_event_id == DEVELOPMENT_EVENT_ID
    assert not hasattr(engine, "decide")
    assert not hasattr(runtime, "planned_decision")


def test_normal_decision_engine_makes_weekly_development_choice() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = DevelopmentEngine(load_development_catalog(DEVELOPMENT))
    runtime, event = engine.plan(maya, context(), DevelopmentRuntimeState())
    history = DecisionHistory()
    result = DecisionEngine().decide_events(maya, replace(context(), events=(event,)), (event,), history)
    decision = result.records[0]

    assert isinstance(decision, DecisionRecord)
    assert decision in result.history.records
    assert runtime.planned_profile_ids
    assert "learning_value" in {component.name for component in decision.evaluations[0].components}


def test_execution_requires_real_same_week_m4_decision_and_event() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = DevelopmentEngine(load_development_catalog(DEVELOPMENT))
    runtime, event = engine.plan(maya, context(), DevelopmentRuntimeState())
    decision = decide_development(maya, event)

    with pytest.raises(ValueError, match="same-week M4 decision"):
        engine.execute(maya, replace(context(), events=(event,)), runtime)
    with pytest.raises(ValueError, match="same-week event"):
        engine.execute(maya, replace(context(), decisions=(decision,)), runtime)
    with pytest.raises(ValueError, match="current agent"):
        bad_decision = replace(decision, agent_id="other")
        engine.execute(maya, replace(context(), events=(event,), decisions=(bad_decision,)), runtime)
    with pytest.raises(ValueError, match="match WeeklyContext.week"):
        bad_decision = replace(decision, week=2)
        engine.execute(maya, replace(context(), events=(event,), decisions=(bad_decision,)), runtime)
    with pytest.raises(ValueError, match="chosen option"):
        bad_decision = invalid_chosen_decision(decision, "missing")
        engine.execute(maya, replace(context(), events=(event,), decisions=(bad_decision,)), runtime)
    with pytest.raises(ValueError, match="exactly one"):
        engine.execute(maya, replace(context(), events=(event,), decisions=(decision, decision)), runtime)


def test_zero_more_and_stressed_study_progress_behave_smoothly() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    light = execute_profile(maya, "reduced_study")
    balanced = execute_profile(maya, "balanced_study")
    zero = execute_profile(maya, "zero_study", catalog=zero_catalog())
    stressed = execute_profile(
        replace(
            maya,
            health=replace(maya.health, energy=18.0),
            mental=MentalState(mood=40.0, stress=85.0, mental_load=82.0, recovery_need=78.0, loneliness=30.0),
        ),
        "balanced_study",
    )

    assert zero[2].education_progress is not None
    assert zero[2].education_progress.progress_delta == 0.0
    assert balanced[2].education_progress.progress_delta > light[2].education_progress.progress_delta
    assert 0.0 < stressed[2].education_progress.progress_delta < balanced[2].education_progress.progress_delta


def test_completed_agents_only_see_zero_education_development_profiles() -> None:
    catalog = load_development_catalog(DEVELOPMENT)
    completed = completed_state(load_agent_state(MAYA_SCENARIO))
    enrolled_runtime, enrolled_event = DevelopmentEngine(catalog).plan(
        load_agent_state(MAYA_SCENARIO),
        context(),
        DevelopmentRuntimeState(),
    )
    completed_runtime, completed_event = DevelopmentEngine(catalog).plan(
        completed,
        context(),
        DevelopmentRuntimeState(),
    )

    enrolled_ids = {option.option_id for option in enrolled_event.options}
    completed_ids = {option.option_id for option in completed_event.options}
    assert "balanced_study" in enrolled_ids
    assert "intensive_study" in enrolled_ids
    assert completed_ids == {"light_self_development"}
    assert completed_runtime.history.plan_records[0].available_profile_ids == ("light_self_development",)
    assert enrolled_runtime.history.plan_records[0].available_profile_ids != completed_runtime.history.plan_records[0].available_profile_ids


def test_completed_agents_do_not_continue_formal_study_or_curriculum_xp() -> None:
    completed = completed_state(load_agent_state(MAYA_SCENARIO))
    next_state, _, record = execute_profile(completed, "light_self_development")

    assert record.education_hours == 0.0
    assert record.education_progress is None
    assert next_state.education == completed.education
    assert all(source.source_type != "education" for item in record.skill_developments for source in item.sources)
    assert skill(next_state, "local language").experience > skill(completed, "local language").experience
    assert effect(record, "health.energy").delta > -3.0


def test_unavailable_completed_study_profile_is_rejected_at_execution() -> None:
    catalog = load_development_catalog(DEVELOPMENT)
    completed = completed_state(load_agent_state(MAYA_SCENARIO))
    runtime, filtered_event = DevelopmentEngine(catalog).plan(completed, context(), DevelopmentRuntimeState())
    _, full_event = DevelopmentEngine(catalog).plan(load_agent_state(MAYA_SCENARIO), context(), DevelopmentRuntimeState())
    decision = replace(decide_development(load_agent_state(MAYA_SCENARIO), full_event), chosen_option_id="balanced_study")

    assert "balanced_study" not in {option.option_id for option in filtered_event.options}
    with pytest.raises(ValueError, match="planned profile ids"):
        DevelopmentEngine(catalog).execute(
            completed,
            replace(context(), events=(full_event,), decisions=(decision,)),
            runtime,
        )


def test_education_progress_clamps_advances_year_and_completes_without_rewards() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    almost_done = replace(
        maya,
        education=replace(maya.education, progress=99.8, current_year=3),
    )
    next_state, _, record = execute_profile(almost_done, "intensive_study")

    assert next_state.education.status == "completed"
    assert next_state.education.progress == 100.0
    assert next_state.education.current_year == next_state.education.total_years
    assert record.education_progress.completed is True
    assert next_state.financial == almost_done.financial
    assert next_state.employment == almost_done.employment
    assert next_state.personality == almost_done.personality


def test_curriculum_and_practice_create_missing_skills_and_gradual_growth() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    next_state, _, record = execute_profile(maya, "admin_skill_focus")
    by_name = {skill.name: skill for skill in next_state.skills.items}
    dev_by_name = {item.skill_name: item for item in record.skill_developments}

    assert "research" in by_name
    assert by_name["research"].experience > 0
    assert by_name["spreadsheets"].experience > skill(maya, "spreadsheets").experience
    assert by_name["spreadsheets"].level > skill(maya, "spreadsheets").level
    assert dev_by_name["spreadsheets"].level_delta < 6.0
    assert all(item.level_after <= 100.0 for item in record.skill_developments)


def test_existing_skill_categories_are_preserved_when_experience_changes() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    next_state, _, record = execute_profile(maya, "admin_skill_focus")
    by_name = {skill.name: skill for skill in next_state.skills.items}
    dev_by_name = {item.skill_name: item for item in record.skill_developments}

    assert by_name["spreadsheets"].category == "technical"
    assert dev_by_name["spreadsheets"].category == "technical"
    assert by_name["research"].category == "academic"


def test_higher_level_skill_has_stronger_diminishing_returns() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    low = replace(base, skills=SkillsState(items=(SkillRating("spreadsheets", "work", 20.0, 0.0),)))
    high = replace(base, skills=SkillsState(items=(SkillRating("spreadsheets", "work", 90.0, 0.0),)))

    low_record = skill_record(execute_profile(low, "admin_skill_focus")[2], "spreadsheets")
    high_record = skill_record(execute_profile(high, "admin_skill_focus")[2], "spreadsheets")

    assert low_record.level_delta > high_record.level_delta
    assert high_record.level_after > high_record.level_before


def test_completed_work_creates_relevant_xp_and_employment_without_work_does_not() -> None:
    catalog = dev_catalog()
    employment_catalog = EmploymentCatalog((job("work_job"),))
    employed = employed_state(load_agent_state(MAYA_SCENARIO), job_id="work_job")
    work = EmploymentWorkWeekRecord(
        week=1,
        contract_id=employed.employment.contract_id,
        weekly_hours=20.0,
        weekly_wage=Decimal("200.00"),
        tenure_weeks_after=1,
        effects=(),
    )
    engine = DevelopmentEngine(catalog, employment_catalog=employment_catalog)
    runtime, event = engine.plan(employed, context(), DevelopmentRuntimeState())
    decision = decide_development(employed, event)
    next_state, runtime, record = engine.execute(
        employed,
        replace(context(), events=(event,), decisions=(decision,), employment_records=(work,)),
        runtime,
    )
    no_work = execute_profile(employed, "reduced_study")[0]

    assert skill(next_state, "customer service").experience > skill(no_work, "customer service").experience
    assert any(source.source_type == "work" for item in record.skill_developments for source in item.sources)


def test_missing_employment_catalog_or_job_fails_for_work_xp() -> None:
    employed = employed_state(load_agent_state(MAYA_SCENARIO), job_id="missing")
    work = EmploymentWorkWeekRecord(
        week=1,
        contract_id="contract_dev",
        weekly_hours=20.0,
        weekly_wage=Decimal("200.00"),
        tenure_weeks_after=1,
        effects=(),
    )
    runtime, event = DevelopmentEngine(dev_catalog()).plan(
        employed,
        context(),
        DevelopmentRuntimeState(),
    )
    decision = decide_development(employed, event)

    with pytest.raises(ValueError, match="employment catalog"):
        DevelopmentEngine(dev_catalog()).execute(
            employed,
            replace(context(), events=(event,), decisions=(decision,), employment_records=(work,)),
            runtime,
        )
    with pytest.raises(ValueError, match="active employment job"):
        DevelopmentEngine(dev_catalog(), employment_catalog=EmploymentCatalog((job("other"),))).execute(
            employed,
            replace(context(), events=(event,), decisions=(decision,), employment_records=(work,)),
            runtime,
        )


def test_multiple_xp_sources_aggregate_order_independently() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    employed = employed_state(base, job_id="work_job")
    employment_catalog = EmploymentCatalog((job("work_job"),))
    work = EmploymentWorkWeekRecord(
        week=1,
        contract_id="contract_dev",
        weekly_hours=20.0,
        weekly_wage=Decimal("200.00"),
        tenure_weeks_after=1,
        effects=(),
    )
    skills = [
        skill_raw("research"),
        skill_raw("spreadsheets"),
        skill_raw("basic budgeting"),
        skill_raw("customer service"),
    ]
    forward_profile = parse_development_catalog(
        catalog_raw(
            skills=skills,
            profiles=[
                profile_raw(
                    "admin_skill_focus",
                    education_hours=12.0,
                    practice=[
                        {"skill_name": "spreadsheets", "hours": 5.0},
                        {"skill_name": "basic budgeting", "hours": 2.0},
                    ],
                )
            ],
        )
    )
    reversed_profile = parse_development_catalog(
        catalog_raw(
            skills=skills,
            profiles=[
                profile_raw(
                    "admin_skill_focus",
                    education_hours=12.0,
                    practice=[
                        {"skill_name": "basic budgeting", "hours": 2.0},
                        {"skill_name": "spreadsheets", "hours": 5.0},
                    ],
                )
            ]
        )
    )
    forward = execute_profile(
        employed,
        "admin_skill_focus",
        catalog=forward_profile,
        employment_catalog=employment_catalog,
        employment_records=(work,),
    )[0]
    reverse = execute_profile(
        employed,
        "admin_skill_focus",
        catalog=reversed_profile,
        employment_catalog=employment_catalog,
        employment_records=(work,),
    )[0]

    assert forward.skills.to_dict() == reverse.skills.to_dict()


def test_heavy_workload_costs_more_than_light_without_threshold_cliff() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    light = execute_profile(maya, "reduced_study")
    heavy = execute_profile(employed_state(maya, weekly_hours=40.0), "intensive_study")

    light_energy = effect(light[2], "health.energy")
    heavy_energy = effect(heavy[2], "health.energy")
    assert heavy_energy.delta < light_energy.delta
    assert 0.0 < heavy[2].efficiency.final_efficiency < light[2].efficiency.final_efficiency


def test_existing_employment_hours_increase_generic_m4_time_cost_pressure() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    working = employed_state(maya, weekly_hours=40.0)
    _, event = DevelopmentEngine(dev_catalog()).plan(maya, context(), DevelopmentRuntimeState())

    unemployed_decision = decide_development(maya, event)
    working_decision = decide_development(working, event)
    unemployed = evaluations_by_option(unemployed_decision)
    employed = evaluations_by_option(working_decision)

    unemployed_gap = abs(component(unemployed["intensive_study"], "time_cost").contribution) - abs(
        component(unemployed["reduced_study"], "time_cost").contribution
    )
    employed_gap = abs(component(employed["intensive_study"], "time_cost").contribution) - abs(
        component(employed["reduced_study"], "time_cost").contribution
    )
    assert employed_gap > unemployed_gap
    assert working_decision.chosen_option_id in working_decision.available_option_ids
    assert component(working_decision.evaluations[0], "time_cost").weight > component(
        unemployed_decision.evaluations[0],
        "time_cost",
    ).weight
    assert unemployed_decision.evaluations[0].controlled_noise == decide_development(maya, event).evaluations[
        0
    ].controlled_noise


def test_development_mutation_boundaries() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    next_state, _, _ = execute_profile(maya, "balanced_study")

    assert next_state.identity == maya.identity
    assert next_state.financial == maya.financial
    assert next_state.employment == maya.employment
    assert next_state.personality == maya.personality
    assert next_state.goals == maya.goals
    assert next_state.social == maya.social
    assert next_state.habits == maya.habits
    assert next_state.knowledge == maya.knowledge
    assert next_state.memory == maya.memory
    assert next_state.routine == maya.routine


def test_runtime_exactly_once_and_consistency_guards() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    engine = DevelopmentEngine(dev_catalog())
    runtime, event = engine.plan(maya, context(), DevelopmentRuntimeState())
    decision = decide_development(maya, event)
    with pytest.raises(ValueError, match="planning already processed"):
        engine.plan(maya, context(), runtime)
    _, runtime, _ = engine.execute(maya, replace(context(), events=(event,), decisions=(decision,)), runtime)
    with pytest.raises(ValueError, match="execution already processed"):
        engine.execute(
            maya,
            replace(context(), events=(event,), decisions=(decision,)),
            replace(
                runtime,
                planned_event_id=DEVELOPMENT_EVENT_ID,
                planned_event_version="1",
                planned_profile_ids=(decision.chosen_option_id,),
                planned_week=1,
            ),
        )
    with pytest.raises(ValueError, match="planning weeks"):
        DevelopmentRuntimeState(history=runtime.history, processed_planning_weeks=(1, 2), processed_execution_weeks=(1,))
    with pytest.raises(ValueError, match="execution weeks"):
        DevelopmentRuntimeState(history=runtime.history, processed_planning_weeks=(1,), processed_execution_weeks=(1, 2))
    with pytest.raises(ValueError, match="decision ids"):
        DevelopmentRuntimeState(
            history=runtime.history,
            processed_planning_weeks=(1,),
            processed_execution_weeks=(1,),
            processed_decision_ids=(runtime.processed_decision_ids[0], "orphan"),
        )
    with pytest.raises(ValueError, match="decision ids"):
        DevelopmentRuntimeState(
            history=runtime.history,
            processed_planning_weeks=(1,),
            processed_execution_weeks=(1,),
            processed_decision_ids=(),
        )
    consumed = runtime.processed_work_record_keys
    with pytest.raises(ValueError, match="work record keys"):
        DevelopmentRuntimeState(
            history=runtime.history,
            processed_planning_weeks=(1,),
            processed_execution_weeks=(1,),
            processed_decision_ids=runtime.processed_decision_ids,
            processed_work_record_keys=consumed + ("orphan-work:1",),
        )


def test_repeated_runs_identical_and_m9_does_not_perturb_unrelated_rng() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    base_transitions = (
        PassiveCashflowTransition(PassiveCashflowEngine()),
        EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT))),
        EventEngineTransition(EventEngine(event_catalog())),
        DecisionEngineTransition(DecisionEngine()),
        DecisionConsequenceTransition(ConsequenceEngine(consequence_catalog())),
    )
    m9_transitions = (
        PassiveCashflowTransition(PassiveCashflowEngine()),
        DevelopmentPlanningTransition(DevelopmentEngine(dev_catalog())),
        EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT))),
        EventEngineTransition(EventEngine(event_catalog())),
        DecisionEngineTransition(DecisionEngine()),
        DecisionConsequenceTransition(ConsequenceEngine(consequence_catalog())),
        DevelopmentExecutionTransition(DevelopmentEngine(dev_catalog())),
    )

    base = LifeSimEngine(config(seed=44), transitions=base_transitions).run(initial_agent=maya)
    with_m9_engine = LifeSimEngine(config(seed=44), transitions=m9_transitions)
    with_m9 = with_m9_engine.run(initial_agent=maya)
    repeated = with_m9_engine.run(initial_agent=maya)

    assert with_m9.to_dict() == repeated.to_dict()
    assert base.states[1].passive_records[0].to_dict() == with_m9.states[1].passive_records[0].to_dict()
    assert base.states[1].employment_records[0].to_dict() == with_m9.states[1].employment_records[0].to_dict()
    assert base.states[1].event_traces[0].to_dict() == with_m9.states[1].event_traces[0].to_dict()
    base_choice = next(decision for decision in base.states[1].decisions if decision.source_event_id == "choice_event")
    m9_choice = next(decision for decision in with_m9.states[1].decisions if decision.source_event_id == "choice_event")
    assert base_choice.evaluations[0].controlled_noise == m9_choice.evaluations[0].controlled_noise
    assert base.states[1].consequences[0].to_dict() == with_m9.states[1].consequences[0].to_dict()


def test_integration_work_development_routine_and_candidate_fit_visibility() -> None:
    maya = employed_state(load_agent_state(MAYA_SCENARIO), job_id="work_job", weekly_hours=20.0)
    employment_catalog = EmploymentCatalog((job("work_job"),))
    engine = LifeSimEngine(
        config(duration_weeks=2, seed=12),
        transitions=(
            DevelopmentPlanningTransition(DevelopmentEngine(dev_catalog(), employment_catalog=employment_catalog)),
            DecisionEngineTransition(DecisionEngine()),
            EmploymentWorkTransition(EmploymentWorkEngine()),
            DevelopmentExecutionTransition(DevelopmentEngine(dev_catalog(), employment_catalog=employment_catalog)),
        ),
    )
    result = engine.run(initial_agent=maya)
    next_state = result.states[-1].agent_state

    assert result.development_history is not None
    assert result.development_history.week_records
    assert skill(next_state, "customer service").experience > skill(maya, "customer service").experience
    assert candidate_fit(next_state, job("future")).skill_score >= candidate_fit(maya, job("future")).skill_score
    assert next_state.financial == maya.financial


def test_cli_demo_outputs_development_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--config",
            "configs/default.toml",
            "--agent-scenario",
            "configs/scenarios/maya_start.toml",
            "--event-catalog",
            "configs/events/starter.toml",
            "--consequence-catalog",
            "configs/consequences/starter.toml",
            "--routine-catalog",
            "configs/routines/starter.toml",
            "--employment-catalog",
            "configs/employment/starter.toml",
            "--development-catalog",
            "configs/development/starter.toml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["development_history"]["plan_records"]
    assert payload["development_history"]["week_records"]
    assert payload["states"][1]["development_records"]
    assert "education" in payload["states"][-1]["agent"]


def execute_profile(
    state,
    profile_id: str,
    *,
    catalog=None,
    employment_catalog=None,
    employment_records=(),
):
    catalog = catalog or dev_catalog()
    engine = DevelopmentEngine(catalog, employment_catalog=employment_catalog)
    runtime, event = engine.plan(state, context(), DevelopmentRuntimeState())
    decision = replace(decide_development(state, event), chosen_option_id=profile_id)
    return engine.execute(
        state,
        replace(context(), events=(event,), decisions=(decision,), employment_records=employment_records),
        runtime,
    )


def invalid_chosen_decision(decision: DecisionRecord, option_id: str) -> DecisionRecord:
    evaluation = OptionEvaluation(
        option_id=option_id,
        available=True,
        unavailable_reason="",
        deterministic_score=1.0,
        controlled_noise=0.0,
        final_score=1.0,
        components=decision.evaluations[0].components,
    )
    return replace(
        decision,
        available_option_ids=(option_id,),
        unavailable_option_ids=(),
        chosen_option_id=option_id,
        evaluations=(evaluation,),
    )


def decide_development(state, event) -> DecisionRecord:
    return DecisionEngine().decide_event(state, replace(context(), events=(event,)), event)


def dev_catalog():
    return load_development_catalog(DEVELOPMENT)


def zero_catalog():
    return parse_development_catalog(catalog_raw(profiles=[profile_raw("zero_study", education_hours=0.0)]))


def context(*, week: int = 1, seed: int = 7, **overrides) -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=config(seed=seed),
        rng=Random(seed),
        **overrides,
    )


def config(*, duration_weeks: int = 1, seed: int = 7) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(name="development-test", seed=seed, duration_weeks=duration_weeks),
        city=CityConfig(name="Test City"),
    )


def skill(state, name: str) -> SkillRating:
    return next(item for item in state.skills.items if item.name == name)


def skill_record(record, name: str):
    return next(item for item in record.skill_developments if item.skill_name == name)


def effect(record, path: str):
    return next(item for item in record.effects if item.path == path)


def component(evaluation, name: str):
    return next(item for item in evaluation.components if item.name == name)


def evaluations_by_option(decision: DecisionRecord):
    return {evaluation.option_id: evaluation for evaluation in decision.evaluations}


def completed_state(state):
    return replace(
        state,
        education=EducationState(
            status="completed",
            program=state.education.program,
            current_year=state.education.total_years,
            total_years=state.education.total_years,
            progress=100.0,
            weekly_study_hours=state.education.weekly_study_hours,
        ),
    )


def employed_state(state, *, job_id: str = "work_job", weekly_hours: float = 20.0):
    from lifesim.agents.state import EmploymentState

    return replace(
        state,
        employment=EmploymentState(
            status="employed",
            role_title="Test Role",
            employer="Test Employer",
            weekly_hours=weekly_hours,
            job_search_intensity=0.0,
            source_job_id=job_id,
            source_job_version="1",
            contract_id="contract_dev",
            contract_type="part_time",
            hourly_rate=Decimal("10.00"),
            stability=0.5,
            physical_demand=0.2,
            mental_demand=0.3,
            social_demand=0.4,
            start_week=1,
            tenure_weeks=1,
        ),
    )


def job(job_id: str) -> JobDefinition:
    return JobDefinition(
        job_id=job_id,
        version="1",
        role_title="Test Role",
        employer="Test Employer",
        sector="test",
        tags=("test",),
        contract_type="part_time",
        weekly_hours=20.0,
        hourly_rate=Decimal("10.00"),
        stability=0.5,
        fixed_term_weeks=0,
        physical_demand=0.2,
        mental_demand=0.3,
        social_demand=0.4,
        base_discovery_weight=1.0,
        base_interview_probability=0.35,
        base_offer_probability=0.35,
        skill_requirements=(
            SkillRequirement("customer service", 40.0, 1.0),
            SkillRequirement("spreadsheets", 35.0, 0.6),
        ),
    )


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
        tags=("test",),
        title="Choice event",
        summary="A deterministic event.",
        options=(EventOption("chosen", "Chosen", "Chosen option.", short_term_value=0.2),),
    )


def consequence_catalog() -> ConsequenceCatalog:
    return ConsequenceCatalog(
        (
            OptionConsequenceDefinition(
                event_id="choice_event",
                event_version="1",
                option_id="chosen",
                outcomes=(
                    OutcomeDefinition(
                        "normal",
                        0.5,
                        effects=(StateEffectDefinition(path="mental.stress", delta=1.0),),
                    ),
                    OutcomeDefinition(
                        "hard",
                        0.5,
                        effects=(StateEffectDefinition(path="mental.stress", delta=4.0),),
                    ),
                ),
            ),
        )
    )


def catalog_raw(*, skills=None, programs=None, profiles=None):
    return {
        "skills": skills if skills is not None else [skill_raw("research"), skill_raw("spreadsheets")],
        "education_programs": programs if programs is not None else [program_raw()],
        "profiles": profiles if profiles is not None else [profile_raw("balanced_study")],
    }


def skill_raw(skill_name: str, **overrides):
    raw = {
        "skill_name": skill_name,
        "category": "test",
        "learning_rate": 0.8,
        "practice_xp_per_hour": 1.0,
        "work_xp_per_hour": 1.0,
    }
    raw.update(overrides)
    return raw


def program_raw(*, curriculum_skill: str = "research", progress_per_full_study_week: float = 1.5):
    return {
        "program": "Urban Studies BA",
        "progress_per_full_study_week": progress_per_full_study_week,
        "curriculum": [{"skill_name": curriculum_skill, "weight": 1.0}],
    }


def profile_raw(
    profile_id: str,
    *,
    education_hours: float = 10.0,
    practice=None,
    **overrides,
):
    raw = {
        "profile_id": profile_id,
        "label": profile_id,
        "summary": "Test profile.",
        "education_hours": education_hours,
        "practice": practice if practice is not None else [{"skill_name": "spreadsheets", "hours": 1.0}],
        "energy_cost": 10.0,
        "learning_value": 0.4,
        "future_value": 0.3,
        "comfort_value": -0.1,
        "goal_tags": ["education"],
    }
    raw.update(overrides)
    return raw
