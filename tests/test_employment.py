from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from random import Random

import pytest

from lifesim.agents.scenario import load_agent_state
from lifesim.agents.state import EmploymentState, IncomeStream
from lifesim.config import CityConfig, LifeSimConfig, SimulationConfig
from lifesim.consequences import (
    ConsequenceCatalog,
    ConsequenceEngine,
    DecisionConsequenceTransition,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    StateEffectDefinition,
)
from lifesim.decisions import DecisionEngine, DecisionEngineTransition
from lifesim.decisions.model import DecisionRecord, DecisionScoreComponent, OptionEvaluation
from lifesim.employment import (
    EmploymentBoundaryEngine,
    EmploymentBoundaryTransition,
    EmploymentCatalog,
    EmploymentDecisionTransition,
    EmploymentMarketEngine,
    EmploymentMarketTransition,
    EmploymentProcessEngine,
    EmploymentRuntimeState,
    EmploymentWorkEngine,
    EmploymentWorkTransition,
    JobApplication,
    JobDefinition,
    ScheduledEmploymentStart,
    SkillRequirement,
    candidate_fit,
    load_employment_catalog,
    parse_employment_catalog,
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
EMPLOYMENT = Path("configs/employment/starter.toml")


def test_employment_catalog_loads_and_validates_data() -> None:
    catalog = load_employment_catalog(EMPLOYMENT)

    assert len(catalog.jobs) == 6
    assert catalog.jobs[0].hourly_rate == Decimal("11.80")
    with pytest.raises(ValueError, match="job key"):
        EmploymentCatalog((job("dup"), job("dup")))
    with pytest.raises(ValueError, match="base_interview_probability"):
        job("bad_probability", base_interview_probability=1.5)
    with pytest.raises(TypeError, match="hourly_rate"):
        parse_employment_catalog({"jobs": [job_raw("bad_money", hourly_rate=12.0)]})
    with pytest.raises(ValueError, match="weekly_hours"):
        job("bad_hours", weekly_hours=0.0)
    with pytest.raises(ValueError, match="physical_demand"):
        job("bad_demand", physical_demand=2.0)
    with pytest.raises(ValueError, match="skill requirement"):
        job(
            "duplicate_skill",
            skill_requirements=(
                SkillRequirement("customer service", 30.0, 1.0),
                SkillRequirement("customer service", 40.0, 1.0),
            ),
        )


def test_employment_state_and_income_streams_are_backward_compatible() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    stream = IncomeStream("legacy", Decimal("1.00"), "weekly", 1.0)

    assert maya.employment.status == "seeking_entry_level_work"
    assert maya.employment.contract_id == ""
    assert stream.source_type == "generic"
    assert stream.source_id == ""
    EmploymentState(
        status="seeking_entry_level_work",
        role_title="",
        employer="",
        weekly_hours=0.0,
        job_search_intensity=10.0,
    )
    with pytest.raises(ValueError, match="non-employed weekly_hours"):
        EmploymentState(
            status="seeking_entry_level_work",
            role_title="",
            employer="",
            weekly_hours=10.0,
            job_search_intensity=10.0,
        )


def test_market_discovery_search_intensity_cooldown_and_determinism() -> None:
    catalog = load_employment_catalog(EMPLOYMENT)
    maya = load_agent_state(MAYA_SCENARIO)
    no_search = replace(
        maya,
        employment=replace(maya.employment, job_search_intensity=0.0),
    )

    empty = EmploymentMarketEngine(catalog).advance_market(no_search, context(), EmploymentRuntimeState())
    first = EmploymentMarketEngine(catalog).advance_market(maya, context(seed=9), EmploymentRuntimeState())
    second = EmploymentMarketEngine(catalog).advance_market(maya, context(seed=9), EmploymentRuntimeState())
    week2 = EmploymentMarketEngine(catalog).advance_market(
        maya,
        context(week=2, seed=9),
        first[0],
    )

    assert empty[1] == ()
    assert len(first[1]) <= catalog.max_discoveries_per_week
    assert [event.to_dict() for event in first[1]] == [event.to_dict() for event in second[1]]
    assert not set(first[2].discovered_job_keys) & set(week2[2].discovered_job_keys)


def test_market_transition_does_not_perturb_m3_event_rng() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    base = LifeSimEngine(
        config(seed=44),
        transitions=(EventEngineTransition(EventEngine(event_catalog())),),
    ).run(initial_agent=maya)
    with_employment = LifeSimEngine(
        config(seed=44),
        transitions=(
            EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT))),
            EventEngineTransition(EventEngine(event_catalog())),
        ),
    ).run(initial_agent=maya)

    assert base.states[1].event_traces[0].to_dict() == with_employment.states[1].event_traces[0].to_dict()


def test_opening_decision_uses_m4_and_processes_application_without_extra_brain() -> None:
    from lifesim import employment

    maya = load_agent_state(MAYA_SCENARIO)
    catalog = EmploymentCatalog((job("entry"),))
    event = EmploymentMarketEngine(catalog).advance_market(maya, context(), EmploymentRuntimeState())[1][0]
    decision = DecisionEngine().decide_event(maya, context(events=(event,)), event)
    runtime, records = EmploymentProcessEngine(catalog).process_decisions(
        maya,
        replace(context(), decisions=(decision,)),
        EmploymentRuntimeState(),
    )

    assert not hasattr(employment, "EmploymentDecisionEngine")
    assert decision.evaluations
    assert records[0].decision_id == decision.decision_id
    if decision.chosen_option_id == "apply":
        assert runtime.applications[0].status == "SUBMITTED"
        assert runtime.applications[0].response_due_week == 2
    else:
        assert runtime.applications == ()
        assert records[0].status_after == "SKIPPED"


def test_application_interview_offer_acceptance_and_next_week_start() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    catalog = EmploymentCatalog((job("flow", base_interview_probability=0.95, base_offer_probability=0.95),))
    application = JobApplication(
        application_id="app_flow",
        job_id="flow",
        job_version="1",
        status="SUBMITTED",
        created_week=1,
        updated_week=1,
        response_due_week=2,
    )
    runtime = EmploymentRuntimeState(applications=(application,))
    runtime, events, _ = EmploymentMarketEngine(catalog).advance_market(maya, context(week=2), runtime)
    interview_event = next(event for event in events if event.event_id.startswith("employment_interview:"))
    runtime, _ = EmploymentProcessEngine(catalog).process_decisions(
        maya,
        replace(context(week=2), decisions=(decision(interview_event, "attend_interview"),)),
        runtime,
    )
    runtime, events, _ = EmploymentMarketEngine(catalog).advance_market(maya, context(week=3), runtime)
    offer_event = next(event for event in events if event.event_id.startswith("employment_offer:"))
    runtime, _ = EmploymentProcessEngine(catalog).process_decisions(
        maya,
        replace(context(week=3), decisions=(decision(offer_event, "accept_offer"),)),
        runtime,
    )
    acceptance_week = EmploymentBoundaryEngine().apply_boundary(maya, context(week=3), runtime)
    started_state, started_runtime, records = EmploymentBoundaryEngine().apply_boundary(
        maya,
        context(week=4),
        runtime,
    )

    assert acceptance_week[0].employment.status != "employed"
    assert started_state.employment.status == "employed"
    assert started_state.employment.job_search_intensity == 0.0
    assert records[0].weekly_wage == Decimal("240.00")
    assert started_state.financial.income_streams[-1].source_type == "employment"
    assert started_state.financial.income_streams[-1].source_id == started_state.employment.contract_id
    assert started_runtime.scheduled_starts == ()


def test_employment_income_reuses_m7_cashflow_and_preserves_unrelated_income() -> None:
    maya = with_income(IncomeStream("other", Decimal("5.00"), "weekly", 1.0))
    start = scheduled_start("contract_pay", job_id="pay", hourly_rate=Decimal("10.00"), weekly_hours=12.0)
    started, _, _ = EmploymentBoundaryEngine().apply_boundary(
        maya,
        context(week=2),
        EmploymentRuntimeState(scheduled_starts=(start,)),
    )
    paid, _, cashflow = PassiveCashflowEngine().apply(started, context(week=2), NoneSafePassiveRuntime())

    assert [stream.name for stream in started.financial.income_streams] == [
        "other",
        "Wages: Test Role",
    ]
    wage_entry = next(entry for entry in cashflow.entries if entry.name == "Wages: Test Role")
    assert wage_entry.amount_paid == Decimal("120.00")
    assert paid.financial.bank_balance == maya.financial.bank_balance + Decimal("100.00")
    assert started.financial.income_streams[0].source_type == "generic"


def test_work_effects_increment_tenure_once_and_preserve_forbidden_domains() -> None:
    employed = employed_state(load_agent_state(MAYA_SCENARIO))
    before = employed.to_dict()
    next_state, runtime, records = EmploymentWorkEngine().apply_work(
        employed,
        context(week=3),
        EmploymentRuntimeState(),
    )

    assert next_state.employment.tenure_weeks == employed.employment.tenure_weeks + 1
    assert records[0].effects
    for key in ("identity", "personality", "skills", "education", "social", "habits", "memory", "goals"):
        assert next_state.to_dict()[key] == before[key]
    with pytest.raises(ValueError, match="already processed"):
        EmploymentWorkEngine().apply_work(next_state, context(week=3), runtime)


def test_fixed_term_contract_ends_at_end_week_exclusive_and_removes_only_own_stream() -> None:
    base = load_agent_state(MAYA_SCENARIO)
    active = employed_state(
        with_income(
            IncomeStream("other", Decimal("5.00"), "weekly", 1.0),
            IncomeStream("Wages: Test Role", Decimal("100.00"), "weekly", 1.0, source_type="employment", source_id="contract_old"),
        ),
        contract_id="contract_old",
        end_week_exclusive=5,
    )

    still_active, _, _ = EmploymentBoundaryEngine().apply_boundary(active, context(week=4), EmploymentRuntimeState())
    ended, _, records = EmploymentBoundaryEngine().apply_boundary(active, context(week=5), EmploymentRuntimeState())

    assert still_active.employment.status == "employed"
    assert ended.employment.status == "seeking_entry_level_work"
    assert [stream.name for stream in ended.financial.income_streams] == ["other"]
    assert records[0].action == "contract_ended"
    assert base.identity == ended.identity


def test_job_replacement_is_atomic_and_avoids_double_salary() -> None:
    active = employed_state(
        with_income(
            IncomeStream("Wages: Old", Decimal("100.00"), "weekly", 1.0, source_type="employment", source_id="old"),
        ),
        contract_id="old",
    )
    start = scheduled_start("new", job_id="new_job", hourly_rate=Decimal("12.00"), weekly_hours=10.0)

    next_state, _, records = EmploymentBoundaryEngine().apply_boundary(
        active,
        context(week=3),
        EmploymentRuntimeState(scheduled_starts=(start,)),
    )

    employment_streams = [
        stream for stream in next_state.financial.income_streams if stream.source_type == "employment"
    ]
    assert len(employment_streams) == 1
    assert employment_streams[0].source_id == "new"
    assert records[0].action == "contract_started"


def test_candidate_fit_uses_skills_not_protected_identity_fields() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    altered = replace(
        maya,
        identity=replace(
            maya.identity,
            pronouns="she/they",
            origin_city="Elsewhere",
            background="Different protected/arbitrary presentation text.",
        ),
    )
    better = replace(
        maya,
        skills=replace(
            maya.skills,
            items=tuple(
                replace(skill, level=80.0) if skill.name == "customer service" else skill
                for skill in maya.skills.items
            ),
        ),
    )
    target = job("fit", skill_requirements=(SkillRequirement("customer service", 60.0, 1.0),))

    assert candidate_fit(maya, target).to_dict() == candidate_fit(altered, target).to_dict()
    assert candidate_fit(better, target).final_fit > candidate_fit(maya, target).final_fit
    assert candidate_fit(better, target).final_fit <= 1.0


def test_employment_rng_isolated_from_m4_m5_and_m7_income_rng() -> None:
    maya = with_income(IncomeStream("maybe", Decimal("0.00"), "weekly", 0.5))
    consequence = ConsequenceCatalog(
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
    base = LifeSimEngine(
        config(seed=19),
        transitions=(
            PassiveCashflowTransition(PassiveCashflowEngine()),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence)),
        ),
    ).run(initial_agent=maya)
    with_employment = LifeSimEngine(
        config(seed=19),
        transitions=(
            PassiveCashflowTransition(PassiveCashflowEngine()),
            EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT))),
            EventEngineTransition(EventEngine(event_catalog())),
            DecisionEngineTransition(DecisionEngine()),
            DecisionConsequenceTransition(ConsequenceEngine(consequence)),
        ),
    ).run(initial_agent=maya)

    assert base.states[1].passive_records[0].to_dict() == with_employment.states[1].passive_records[0].to_dict()
    assert base.states[1].event_traces[0].to_dict() == with_employment.states[1].event_traces[0].to_dict()
    base_decision = next(decision for decision in base.states[1].decisions if decision.source_event_id == "choice_event")
    employment_decision = next(
        decision for decision in with_employment.states[1].decisions if decision.source_event_id == "choice_event"
    )
    assert base_decision.evaluations[0].controlled_noise == employment_decision.evaluations[0].controlled_noise
    assert base.states[1].consequences[0].to_dict() == with_employment.states[1].consequences[0].to_dict()


def test_runtime_duplicate_processing_and_repeated_engine_runs_are_deterministic() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    market = EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT)))
    first = market.apply(maya, context())
    with pytest.raises(ValueError, match="already processed"):
        market.apply(maya, replace(context(), employment_runtime=first.employment_runtime))

    boundary = EmploymentBoundaryTransition(EmploymentBoundaryEngine())
    boundary_result = boundary.apply(maya, context())
    with pytest.raises(ValueError, match="already processed"):
        boundary.apply(maya, replace(context(), employment_runtime=boundary_result.employment_runtime))

    engine = LifeSimEngine(
        config(duration_weeks=3, seed=23),
        transitions=(
            EmploymentBoundaryTransition(EmploymentBoundaryEngine()),
            EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT))),
            DecisionEngineTransition(DecisionEngine()),
            EmploymentDecisionTransition(EmploymentProcessEngine(load_employment_catalog(EMPLOYMENT))),
            EmploymentWorkTransition(EmploymentWorkEngine()),
        ),
    )
    assert engine.run(initial_agent=maya).to_dict() == engine.run(initial_agent=maya).to_dict()


def test_no_m3_special_event_week_can_still_progress_employment_search() -> None:
    maya = load_agent_state(MAYA_SCENARIO)
    result = LifeSimEngine(
        config(seed=31),
        transitions=(
            EmploymentMarketTransition(EmploymentMarketEngine(load_employment_catalog(EMPLOYMENT))),
            EventEngineTransition(EventEngine(EventCatalog((), event_probability=0.0))),
            DecisionEngineTransition(DecisionEngine()),
        ),
    ).run(initial_agent=maya)

    assert result.states[1].event_traces[0].no_event is True
    assert result.states[1].employment_records
    assert any(event.category == "employment" for event in result.states[1].events)


def config(*, duration_weeks: int = 1, seed: int = 1) -> LifeSimConfig:
    return LifeSimConfig(
        simulation=SimulationConfig(name="employment-test", seed=seed, duration_weeks=duration_weeks),
        city=CityConfig(name="Veyra"),
    )


def context(*, week: int = 1, seed: int = 1, events=(), decisions=()) -> WeeklyContext:
    return WeeklyContext(
        week=week,
        config=config(seed=seed),
        rng=Random(seed),
        events=tuple(events),
        decisions=tuple(decisions),
    )


def job(
    job_id: str,
    *,
    version: str = "1",
    weekly_hours: float = 20.0,
    hourly_rate: Decimal = Decimal("12.00"),
    base_interview_probability: float = 0.35,
    base_offer_probability: float = 0.35,
    physical_demand: float = 0.2,
    skill_requirements: tuple[SkillRequirement, ...] = (
        SkillRequirement("customer service", 35.0, 1.0),
    ),
) -> JobDefinition:
    return JobDefinition(
        job_id=job_id,
        version=version,
        role_title="Test Role",
        employer="Test Employer",
        sector="test",
        tags=("test",),
        contract_type="fixed_term",
        weekly_hours=weekly_hours,
        hourly_rate=hourly_rate,
        stability=0.5,
        fixed_term_weeks=4,
        physical_demand=physical_demand,
        mental_demand=0.3,
        social_demand=0.4,
        base_discovery_weight=1.0,
        base_interview_probability=base_interview_probability,
        base_offer_probability=base_offer_probability,
        skill_requirements=skill_requirements,
    )


def job_raw(job_id: str, **overrides):
    raw = {
        "job_id": job_id,
        "version": "1",
        "role_title": "Test Role",
        "employer": "Test Employer",
        "sector": "test",
        "tags": ["test"],
        "contract_type": "fixed_term",
        "weekly_hours": 20.0,
        "hourly_rate": "12.00",
        "stability": 0.5,
        "fixed_term_weeks": 4,
        "physical_demand": 0.2,
        "mental_demand": 0.3,
        "social_demand": 0.4,
        "base_discovery_weight": 1.0,
        "base_interview_probability": 0.35,
        "base_offer_probability": 0.35,
        "skill_requirements": [{"skill_name": "customer service", "desired_level": 35.0, "weight": 1.0}],
    }
    raw.update(overrides)
    return raw


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
        options=(
            EventOption(
                option_id="chosen",
                label="Chosen",
                summary="Chosen option.",
                short_term_value=0.2,
            ),
        ),
    )


def decision(event, chosen_option_id: str) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"decision_{event.event_id}_{chosen_option_id}".replace(":", "_"),
        agent_id="maya",
        week=event.week,
        source_event_id=event.event_id,
        source_event_version=event.version,
        time_pressure=event.time_pressure,
        available_option_ids=(chosen_option_id,),
        unavailable_option_ids=tuple(
            option.option_id for option in event.options if option.option_id != chosen_option_id
        ),
        chosen_option_id=chosen_option_id,
        evaluations=(
            OptionEvaluation(
                option_id=chosen_option_id,
                available=True,
                unavailable_reason="",
                deterministic_score=1.0,
                controlled_noise=0.0,
                final_score=1.0,
                components=(
                    DecisionScoreComponent("fixture", 1.0, 1.0, 1.0),
                ),
            ),
        )
        + tuple(
            OptionEvaluation(
                option_id=option.option_id,
                available=False,
                unavailable_reason="fixture",
                deterministic_score=None,
                controlled_noise=None,
                final_score=None,
                components=(),
            )
            for option in event.options
            if option.option_id != chosen_option_id
        ),
        strongest_positive_factors=("fixture",),
        strongest_negative_factors=(),
    )


def scheduled_start(
    contract_id: str,
    *,
    job_id: str = "test_job",
    hourly_rate: Decimal = Decimal("10.00"),
    weekly_hours: float = 10.0,
    start_week: int = 2,
) -> ScheduledEmploymentStart:
    return ScheduledEmploymentStart(
        contract_id=contract_id,
        application_id=f"app_{contract_id}",
        job_id=job_id,
        job_version="1",
        role_title="Test Role",
        employer="Test Employer",
        contract_type="fixed_term",
        weekly_hours=weekly_hours,
        hourly_rate=hourly_rate,
        stability=0.5,
        physical_demand=0.2,
        mental_demand=0.3,
        social_demand=0.4,
        start_week=start_week,
        end_week_exclusive=start_week + 4,
    )


def with_income(*streams: IncomeStream):
    maya = load_agent_state(MAYA_SCENARIO)
    return replace(
        maya,
        financial=replace(maya.financial, income_streams=streams),
    )


def employed_state(state, *, contract_id: str = "contract_test", end_week_exclusive: int = 0):
    return replace(
        state,
        employment=EmploymentState(
            status="employed",
            role_title="Test Role",
            employer="Test Employer",
            weekly_hours=20.0,
            job_search_intensity=0.0,
            source_job_id="test_job",
            source_job_version="1",
            contract_id=contract_id,
            contract_type="fixed_term",
            hourly_rate=Decimal("10.00"),
            stability=0.5,
            physical_demand=0.2,
            mental_demand=0.3,
            social_demand=0.4,
            start_week=1,
            tenure_weeks=1,
            end_week_exclusive=end_week_exclusive,
        ),
    )


def NoneSafePassiveRuntime():
    from lifesim.passive import PassiveLifeRuntimeState

    return PassiveLifeRuntimeState()
