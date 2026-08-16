from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lifesim.agents.state import AgentState, EducationState, SkillRating, SkillsState
from lifesim.decisions.model import DecisionRecord
from lifesim.development.model import (
    DevelopmentCatalog,
    DevelopmentEffectApplication,
    DevelopmentEfficiencyAudit,
    DevelopmentPlanRecord,
    DevelopmentProfile,
    DevelopmentRuntimeState,
    DevelopmentWeekRecord,
    EducationProgressRecord,
    SkillDevelopmentRecord,
    SkillExperienceSource,
)
from lifesim.employment.model import EmploymentCatalog, EmploymentWorkWeekRecord, SkillRequirement
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

DEVELOPMENT_EVENT_ID = "weekly_development"
DEVELOPMENT_EVENT_VERSION = "1"


class DevelopmentEngine:
    def __init__(
        self,
        catalog: DevelopmentCatalog,
        *,
        employment_catalog: EmploymentCatalog | None = None,
    ) -> None:
        self._catalog = catalog
        self._employment_catalog = employment_catalog

    @property
    def catalog(self) -> DevelopmentCatalog:
        return self._catalog

    def plan(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: DevelopmentRuntimeState,
    ) -> tuple[DevelopmentRuntimeState, EventOccurrence]:
        if context.week in runtime.processed_planning_weeks:
            raise ValueError(f"Development planning already processed for week {context.week}.")
        profiles = _applicable_profiles(state, self._catalog)
        if not profiles:
            raise ValueError("Expected at least one applicable development profile.")
        occurrence = _development_occurrence(context, profiles)
        record = DevelopmentPlanRecord(
            week=context.week,
            event_id=occurrence.event_id,
            event_version=occurrence.version,
            available_profile_ids=tuple(profile.profile_id for profile in profiles),
        )
        runtime = replace(
            runtime,
            history=runtime.history.record_plan(record),
            planned_event_id=occurrence.event_id,
            planned_event_version=occurrence.version,
            planned_profile_ids=record.available_profile_ids,
            planned_week=context.week,
            processed_planning_weeks=runtime.processed_planning_weeks + (context.week,),
        )
        return runtime, occurrence

    def execute(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: DevelopmentRuntimeState,
    ) -> tuple[AgentState, DevelopmentRuntimeState, DevelopmentWeekRecord]:
        if context.week in runtime.processed_execution_weeks:
            raise ValueError(f"Development execution already processed for week {context.week}.")
        if runtime.planned_week != context.week:
            raise ValueError("Expected planned development week to match execution week.")
        if not runtime.planned_event_id or not runtime.planned_profile_ids:
            raise ValueError("Expected planned development opportunity before execution.")
        decision, event = _find_development_decision(state, context, runtime)
        if decision.chosen_option_id is None:
            raise ValueError("Expected weekly development decision to choose a profile.")
        profile = self._catalog.profile(decision.chosen_option_id)
        if decision.chosen_option_id not in runtime.planned_profile_ids:
            raise ValueError("Expected development profile to have been offered during planning.")
        if decision.decision_id in runtime.processed_decision_ids:
            raise ValueError(f"Development decision '{decision.decision_id}' already processed.")

        efficiency = _efficiency(state, profile)
        sources, consumed_work_keys = self._experience_sources(state, context, runtime, profile, efficiency)
        next_state, skill_records = _apply_skill_development(state, self._catalog, sources)
        next_state, education_record = _apply_education_progress(next_state, self._catalog, profile, efficiency, context)
        next_state, effects = _apply_development_effects(next_state, profile, efficiency)
        record = DevelopmentWeekRecord(
            week=context.week,
            profile_id=profile.profile_id,
            decision_id=decision.decision_id,
            education_hours=profile.education_hours,
            practice=profile.practice,
            efficiency=efficiency,
            education_progress=education_record,
            skill_developments=skill_records,
            effects=effects,
            consumed_work_record_keys=consumed_work_keys,
        )
        runtime = replace(
            runtime,
            history=runtime.history.record_week(record),
            planned_event_id="",
            planned_event_version="",
            planned_profile_ids=(),
            planned_week=None,
            processed_execution_weeks=runtime.processed_execution_weeks + (context.week,),
            processed_decision_ids=runtime.processed_decision_ids + (record.decision_id,),
            processed_work_record_keys=runtime.processed_work_record_keys + consumed_work_keys,
        )
        _ = event
        return next_state, runtime, record

    def _experience_sources(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: DevelopmentRuntimeState,
        profile: DevelopmentProfile,
        efficiency: DevelopmentEfficiencyAudit,
    ) -> tuple[tuple[SkillExperienceSource, ...], tuple[str, ...]]:
        sources: list[SkillExperienceSource] = []
        sources.extend(_practice_sources(self._catalog, profile, efficiency))
        sources.extend(_education_sources(state, self._catalog, profile, efficiency))
        work_sources, consumed = _work_sources(state, context, runtime, self._catalog, self._employment_catalog)
        sources.extend(work_sources)
        return tuple(sources), consumed


@dataclass(frozen=True, slots=True)
class DevelopmentPlanningTransition:
    engine: DevelopmentEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        runtime, occurrence = self.engine.plan(
            state,
            context,
            runtime,
        )
        return WeeklyTransitionResult(
            agent_state=state,
            events=(occurrence,),
            development_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class DevelopmentExecutionTransition:
    engine: DevelopmentEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.execute(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            development_records=(record,),
            development_runtime=runtime,
        )


def _runtime(context: WeeklyContext) -> DevelopmentRuntimeState:
    runtime = context.development_runtime
    if runtime is None:
        return DevelopmentRuntimeState()
    if not isinstance(runtime, DevelopmentRuntimeState):
        raise TypeError("Expected WeeklyContext.development_runtime to contain DevelopmentRuntimeState.")
    return runtime


def _applicable_profiles(
    state: AgentState,
    catalog: DevelopmentCatalog,
) -> tuple[DevelopmentProfile, ...]:
    enrolled_program_resolvable = False
    if state.education.status == "enrolled":
        try:
            catalog.program(state.education.program)
        except KeyError:
            enrolled_program_resolvable = False
        else:
            enrolled_program_resolvable = True
    return tuple(
        profile
        for profile in catalog.profiles
        if profile.education_hours == 0.0 or enrolled_program_resolvable
    )


def _development_occurrence(
    context: WeeklyContext,
    profiles: tuple[DevelopmentProfile, ...],
) -> EventOccurrence:
    return EventOccurrence(
        event_id=DEVELOPMENT_EVENT_ID,
        version=DEVELOPMENT_EVENT_VERSION,
        week=context.week,
        category="development",
        effective_weight=1.0,
        title="Weekly development",
        summary="Choose a study and skill-practice focus for the week.",
        tags=("development", "education", "skills"),
        options=tuple(_profile_option(profile) for profile in profiles),
    )


def _profile_option(profile: DevelopmentProfile) -> EventOption:
    return EventOption(
        option_id=profile.profile_id,
        label=profile.label,
        summary=profile.summary,
        time_cost_hours=profile.total_hours,
        energy_cost=profile.energy_cost,
        short_term_value=profile.short_term_value,
        future_value=profile.future_value,
        perceived_risk=profile.perceived_risk,
        uncertainty=profile.uncertainty,
        social_value=profile.social_value,
        social_pressure=profile.social_pressure,
        autonomy_value=profile.autonomy_value,
        learning_value=profile.learning_value,
        health_value=profile.health_value,
        comfort_value=profile.comfort_value,
        goal_tags=profile.goal_tags,
        behavior_tags=profile.behavior_tags,
        requires_full_estimated_cost=False,
    )


def _find_development_decision(
    state: AgentState,
    context: WeeklyContext,
    runtime: DevelopmentRuntimeState,
) -> tuple[DecisionRecord, EventOccurrence]:
    matches = tuple(
        item
        for item in context.decisions
        if isinstance(item, DecisionRecord)
        and item.source_event_id == runtime.planned_event_id
        and item.source_event_version == runtime.planned_event_version
    )
    if len(matches) != 1:
        raise ValueError("Expected exactly one same-week M4 decision for weekly development.")
    decision = matches[0]
    if decision.agent_id != state.identity.agent_id:
        raise ValueError("Development decision does not belong to the current agent.")
    if decision.week != context.week:
        raise ValueError("Expected development decision week to match WeeklyContext.week.")
    if decision.source_event_id != DEVELOPMENT_EVENT_ID or decision.source_event_version != DEVELOPMENT_EVENT_VERSION:
        raise ValueError("Expected development decision to reference weekly_development.")
    event = next(
        (
            occurrence
            for occurrence in context.events
            if isinstance(occurrence, EventOccurrence)
            and occurrence.event_id == decision.source_event_id
            and occurrence.version == decision.source_event_version
        ),
        None,
    )
    if event is None:
        raise ValueError("Expected development decision to reference a same-week event.")
    if event.week != context.week:
        raise ValueError("Expected development event week to match WeeklyContext.week.")
    if event.category != "development":
        raise ValueError("Expected development decision event category to be development.")
    option_ids = {option.option_id for option in event.options}
    if decision.chosen_option_id not in option_ids:
        raise ValueError("Expected development decision chosen option to exist on its event.")
    if option_ids != set(runtime.planned_profile_ids):
        raise ValueError("Expected development event options to match planned profile ids.")
    return decision, event


def _efficiency(state: AgentState, profile: DevelopmentProfile) -> DevelopmentEfficiencyAudit:
    employment_hours = state.employment.weekly_hours if state.employment.status == "employed" else 0.0
    development_hours = profile.total_hours
    combined = employment_hours + development_hours
    energy_factor = 0.35 + state.health.energy / 100.0 * 0.65
    stress_factor = 1.0 - state.mental.stress / 100.0 * 0.45
    mental_load_factor = 1.0 - state.mental.mental_load / 100.0 * 0.35
    recovery_factor = 1.0 - state.mental.recovery_need / 100.0 * 0.35
    overload = max(0.0, combined - 42.0)
    workload_factor = 1.0 / (1.0 + overload / 45.0)
    final = max(
        0.08,
        min(1.0, energy_factor * stress_factor * mental_load_factor * recovery_factor * workload_factor),
    )
    return DevelopmentEfficiencyAudit(
        energy_factor=round(energy_factor, 12),
        stress_factor=round(stress_factor, 12),
        mental_load_factor=round(mental_load_factor, 12),
        recovery_factor=round(recovery_factor, 12),
        workload_factor=round(workload_factor, 12),
        combined_workload_hours=round(combined, 12),
        study_ratio=_study_ratio(state, profile),
        effective_study_hours=round(profile.education_hours * final, 12),
        effective_practice_hours=round(sum(item.hours for item in profile.practice) * final, 12),
        final_efficiency=round(final, 12),
    )


def _study_ratio(state: AgentState, profile: DevelopmentProfile) -> float:
    nominal = state.education.weekly_study_hours
    if nominal <= 0:
        return 0.0 if profile.education_hours <= 0 else 1.0
    return round(min(1.5, profile.education_hours / nominal), 12)


def _practice_sources(
    catalog: DevelopmentCatalog,
    profile: DevelopmentProfile,
    efficiency: DevelopmentEfficiencyAudit,
) -> tuple[SkillExperienceSource, ...]:
    sources = []
    for allocation in profile.practice:
        if allocation.hours <= 0:
            continue
        definition = catalog.skill(allocation.skill_name)
        raw = allocation.hours * definition.practice_xp_per_hour
        sources.append(
            SkillExperienceSource(
                skill_name=allocation.skill_name,
                source_type="practice",
                source_id=profile.profile_id,
                hours=round(allocation.hours, 12),
                raw_experience=round(raw, 12),
                effective_experience=round(raw * efficiency.final_efficiency, 12),
            )
        )
    return tuple(sources)


def _education_sources(
    state: AgentState,
    catalog: DevelopmentCatalog,
    profile: DevelopmentProfile,
    efficiency: DevelopmentEfficiencyAudit,
) -> tuple[SkillExperienceSource, ...]:
    if state.education.status != "enrolled" or profile.education_hours <= 0:
        return ()
    program = catalog.program(state.education.program)
    total_weight = sum(mapping.weight for mapping in program.curriculum)
    sources = []
    for mapping in program.curriculum:
        definition = catalog.skill(mapping.skill_name)
        share = mapping.weight / total_weight
        hours = profile.education_hours * share
        raw = hours * definition.practice_xp_per_hour * 0.75
        sources.append(
            SkillExperienceSource(
                skill_name=mapping.skill_name,
                source_type="education",
                source_id=program.program,
                hours=round(hours, 12),
                raw_experience=round(raw, 12),
                effective_experience=round(raw * efficiency.final_efficiency, 12),
            )
        )
    return tuple(sources)


def _work_sources(
    state: AgentState,
    context: WeeklyContext,
    runtime: DevelopmentRuntimeState,
    catalog: DevelopmentCatalog,
    employment_catalog: EmploymentCatalog | None,
) -> tuple[tuple[SkillExperienceSource, ...], tuple[str, ...]]:
    records = tuple(
        record
        for record in context.employment_records
        if isinstance(record, EmploymentWorkWeekRecord) and record.week == context.week
    )
    if not records:
        return (), ()
    if employment_catalog is None:
        raise ValueError("Expected employment catalog to resolve work-derived skill experience.")
    sources: list[SkillExperienceSource] = []
    consumed: list[str] = []
    for record in records:
        key = f"{record.contract_id}:{record.week}"
        if key in runtime.processed_work_record_keys:
            raise ValueError(f"Employment work record '{key}' already consumed for development.")
        if record.contract_id != state.employment.contract_id:
            raise ValueError("Expected employment work record to match active employment contract.")
        job = employment_catalog.find(state.employment.source_job_id, state.employment.source_job_version)
        if job is None:
            raise ValueError("Expected active employment job to exist in employment catalog.")
        consumed.append(key)
        sources.extend(_sources_from_requirements(record, job.skill_requirements, catalog))
    return tuple(sources), tuple(consumed)


def _sources_from_requirements(
    record: EmploymentWorkWeekRecord,
    requirements: tuple[SkillRequirement, ...],
    catalog: DevelopmentCatalog,
) -> tuple[SkillExperienceSource, ...]:
    if not requirements:
        return ()
    total_weight = sum(requirement.weight for requirement in requirements)
    sources = []
    for requirement in requirements:
        definition = catalog.skill(requirement.skill_name)
        share = requirement.weight / total_weight if total_weight else 0.0
        hours = record.weekly_hours * share
        raw = hours * definition.work_xp_per_hour
        sources.append(
            SkillExperienceSource(
                skill_name=requirement.skill_name,
                source_type="work",
                source_id=record.contract_id,
                hours=round(hours, 12),
                raw_experience=round(raw, 12),
                effective_experience=round(raw, 12),
            )
        )
    return tuple(sources)


def _apply_skill_development(
    state: AgentState,
    catalog: DevelopmentCatalog,
    sources: tuple[SkillExperienceSource, ...],
) -> tuple[AgentState, tuple[SkillDevelopmentRecord, ...]]:
    by_skill: dict[str, list[SkillExperienceSource]] = {}
    for source in sources:
        if source.raw_experience <= 0:
            continue
        by_skill.setdefault(source.skill_name, []).append(source)
    if not by_skill:
        return state, ()
    existing = {skill.name: skill for skill in state.skills.items}
    records = []
    updated: dict[str, SkillRating] = dict(existing)
    for skill_name in sorted(by_skill):
        definition = catalog.skill(skill_name)
        current = existing.get(skill_name, SkillRating(skill_name, definition.category, 0.0, 0.0))
        category = current.category if skill_name in existing else definition.category
        skill_sources = tuple(by_skill[skill_name])
        raw_gain = sum(source.raw_experience for source in skill_sources)
        effective_gain = _soft_cap(sum(source.effective_experience for source in skill_sources), 55.0)
        diminishing = ((100.0 - current.level) / 100.0) ** 1.15
        level_delta = min(6.0, effective_gain * definition.learning_rate * diminishing * 0.08)
        level_after = min(100.0, current.level + level_delta)
        experience_after = current.experience + raw_gain
        updated[skill_name] = SkillRating(
            name=skill_name,
            category=category,
            level=round(level_after, 12),
            experience=round(experience_after, 12),
        )
        records.append(
            SkillDevelopmentRecord(
                skill_name=skill_name,
                category=category,
                level_before=round(current.level, 12),
                level_after=round(level_after, 12),
                level_delta=round(level_after - current.level, 12),
                experience_before=round(current.experience, 12),
                experience_after=round(experience_after, 12),
                raw_experience_gain=round(raw_gain, 12),
                effective_experience_gain=round(effective_gain, 12),
                sources=skill_sources,
            )
        )
    ordered_names = [skill.name for skill in state.skills.items] + [
        name for name in sorted(updated) if name not in existing
    ]
    next_skills = tuple(updated[name] for name in ordered_names)
    return replace(state, skills=SkillsState(items=next_skills)), tuple(records)


def _apply_education_progress(
    state: AgentState,
    catalog: DevelopmentCatalog,
    profile: DevelopmentProfile,
    efficiency: DevelopmentEfficiencyAudit,
    context: WeeklyContext,
) -> tuple[AgentState, EducationProgressRecord | None]:
    education = state.education
    if education.status != "enrolled":
        return state, None
    program = catalog.program(education.program)
    before_progress = education.progress
    nominal = max(0.000001, education.weekly_study_hours)
    useful_ratio = min(1.5, profile.education_hours / nominal)
    delta = program.progress_per_full_study_week * useful_ratio * efficiency.final_efficiency
    after_progress = min(100.0, before_progress + delta)
    year_after = _academic_year(after_progress, education.total_years)
    status_after = "completed" if after_progress >= 100.0 else education.status
    if status_after == "completed":
        year_after = education.total_years
    next_education = EducationState(
        status=status_after,
        program=education.program,
        current_year=year_after,
        total_years=education.total_years,
        progress=round(after_progress, 12),
        weekly_study_hours=education.weekly_study_hours,
    )
    record = EducationProgressRecord(
        week=context.week,
        program=education.program,
        status_before=education.status,
        status_after=status_after,
        year_before=education.current_year,
        year_after=year_after,
        progress_before=round(before_progress, 12),
        progress_after=round(after_progress, 12),
        progress_delta=round(after_progress - before_progress, 12),
        planned_study_hours=round(profile.education_hours, 12),
        effective_study_hours=efficiency.effective_study_hours,
        completed=status_after == "completed" and education.status != "completed",
    )
    return replace(state, education=next_education), record


def _academic_year(progress: float, total_years: int) -> int:
    if total_years <= 0:
        return 0
    if progress >= 100.0:
        return total_years
    year = int(progress * total_years / 100.0) + 1
    return min(total_years, max(1, year))


def _apply_development_effects(
    state: AgentState,
    profile: DevelopmentProfile,
    efficiency: DevelopmentEfficiencyAudit,
) -> tuple[AgentState, tuple[DevelopmentEffectApplication, ...]]:
    hours = profile.total_hours
    changes = (
        ("health.energy", -(profile.energy_cost * 0.26 + hours * 0.05), "development_energy"),
        ("mental.stress", hours * 0.035 + max(0.0, hours - 20.0) * 0.025, "development_stress"),
        ("mental.mental_load", hours * 0.11 + max(0.0, hours - 18.0) * 0.06, "development_load"),
        ("mental.recovery_need", profile.energy_cost * 0.055 + hours * 0.04, "development_recovery"),
        ("needs.purpose", efficiency.final_efficiency * min(2.0, hours / 12.0), "development_purpose"),
    )
    next_state = state
    effects = []
    for path, delta, reason in changes:
        next_state, effect = _bounded_replace(next_state, path, delta, reason)
        effects.append(effect)
    return next_state, tuple(effects)


def _bounded_replace(
    state: AgentState,
    path: str,
    delta: float,
    reason: str,
) -> tuple[AgentState, DevelopmentEffectApplication]:
    section_name, field_name = path.split(".", 1)
    section = getattr(state, section_name)
    before = getattr(section, field_name)
    raw_after = before + delta
    after = min(100.0, max(0.0, raw_after))
    clamped = after != raw_after
    next_state = replace(state, **{section_name: replace(section, **{field_name: after})})
    return next_state, DevelopmentEffectApplication(
        path=path,
        before=round(before, 12),
        after=round(after, 12),
        delta=round(after - before, 12),
        clamped=clamped,
        reason=reason,
    )


def _soft_cap(value: float, cap: float) -> float:
    if value <= 0:
        return 0.0
    return value / (1.0 + value / cap)


def _identity_tuple(state: AgentState) -> tuple[Any, ...]:
    return (
        state.identity,
        state.financial,
        state.employment,
        state.personality,
        state.goals,
        state.social,
        state.habits,
        state.knowledge,
        state.memory,
        state.routine,
    )


def assert_development_scope(before: AgentState, after: AgentState) -> None:
    """Test helper enforcing M9's mutation boundary."""
    if _identity_tuple(before) != _identity_tuple(after):
        raise AssertionError("M9 development mutated state outside skills, education, health, mental, or needs.")
