from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from random import Random

from lifesim.agents.state import AgentState, EmploymentState, FinancialState, IncomeStream
from lifesim.decisions.model import DecisionRecord
from lifesim.employment.model import (
    ACTIVE_APPLICATION_STATUSES,
    ApplicationStageRecord,
    CandidateFitTrace,
    EmploymentBoundaryRecord,
    EmploymentCatalog,
    EmploymentDiscoveryDraw,
    EmploymentEffectApplication,
    EmploymentHistory,
    EmploymentMarketRecord,
    EmploymentRuntimeState,
    EmploymentWorkWeekRecord,
    JobApplication,
    JobDefinition,
    MarketCandidateTrace,
    ProbabilityAudit,
    ScheduledEmploymentStart,
)
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.rng import derive_stable_seed
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

CENT = Decimal("0.01")
SEARCHING_STATUS = "seeking_entry_level_work"
APPLICATION_PREFIX = "employment_opening:"
INTERVIEW_PREFIX = "employment_interview:"
OFFER_PREFIX = "employment_offer:"


class EmploymentMarketEngine:
    def __init__(self, catalog: EmploymentCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> EmploymentCatalog:
        return self._catalog

    def advance_market(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: EmploymentRuntimeState,
    ) -> tuple[EmploymentRuntimeState, tuple[EventOccurrence, ...], EmploymentMarketRecord]:
        if context.week in runtime.processed_market_weeks:
            raise ValueError(f"Employment market already processed for week {context.week}.")
        applications = list(runtime.applications)
        stage_records = []
        events: list[EventOccurrence] = []

        for index, application in enumerate(tuple(applications)):
            job = self._require_job(application)
            if application.status == "SUBMITTED" and application.response_due_week <= context.week:
                audit = _probability_audit(state, context, application, job, "interview")
                if audit.result == "success":
                    updated = replace(
                        application,
                        status="INTERVIEW_INVITED",
                        updated_week=context.week,
                    )
                    events.append(_interview_event(job, updated, context.week))
                    detail = "interview_invited"
                else:
                    updated = replace(application, status="REJECTED", updated_week=context.week)
                    detail = "application_rejected"
                applications[index] = updated
                stage_records.append(
                    _stage_record(updated, context.week, "APPLICATION_RESOLVED", detail, audit)
                )
            elif application.status == "INTERVIEW_INVITED":
                events.append(_interview_event(job, application, context.week))
            elif application.status == "INTERVIEW_ATTENDED" and application.offer_due_week <= context.week:
                audit = _probability_audit(state, context, application, job, "offer")
                if audit.result == "success":
                    updated = replace(
                        application,
                        status="OFFER_AVAILABLE",
                        updated_week=context.week,
                    )
                    detail = "offer_available"
                else:
                    updated = replace(application, status="REJECTED", updated_week=context.week)
                    detail = "post_interview_rejected"
                applications[index] = updated
                stage_records.append(
                    _stage_record(updated, context.week, "INTERVIEW_RESOLVED", detail, audit)
                )

        offer_events = _offer_events_for_available_applications(applications, self._catalog, context.week)
        events.extend(offer_events)
        discovered, candidates, discovery_draws = _discover_jobs(
            state,
            context,
            EmploymentRuntimeState(
                history=runtime.history,
                applications=tuple(applications),
                scheduled_starts=runtime.scheduled_starts,
                processed_market_weeks=runtime.processed_market_weeks,
                processed_decision_ids=runtime.processed_decision_ids,
                processed_boundary_weeks=runtime.processed_boundary_weeks,
                processed_work_weeks=runtime.processed_work_weeks,
            ),
            self._catalog,
        )
        opening_events = tuple(_opening_event(job, context.week) for job in discovered)
        events.extend(opening_events)
        record = EmploymentMarketRecord(
            week=context.week,
            candidates=candidates,
            discovered_job_keys=tuple(_job_key(job) for job in discovered),
            event_ids_created=tuple(event.event_id for event in events),
            discovery_draws=discovery_draws,
        )
        history = EmploymentHistory(
            market_records=runtime.history.market_records + (record,),
            application_records=runtime.history.application_records + tuple(stage_records),
            boundary_records=runtime.history.boundary_records,
            work_records=runtime.history.work_records,
        )
        next_runtime = EmploymentRuntimeState(
            history=history,
            applications=tuple(applications),
            scheduled_starts=runtime.scheduled_starts,
            processed_market_weeks=runtime.processed_market_weeks + (context.week,),
            processed_decision_ids=runtime.processed_decision_ids,
            processed_boundary_weeks=runtime.processed_boundary_weeks,
            processed_work_weeks=runtime.processed_work_weeks,
        )
        return next_runtime, tuple(events), record

    def _require_job(self, application: JobApplication) -> JobDefinition:
        job = self._catalog.find(application.job_id, application.job_version)
        if job is None:
            raise ValueError(f"Unknown employment job '{application.job_id}:{application.job_version}'.")
        return job


class EmploymentProcessEngine:
    def __init__(self, catalog: EmploymentCatalog) -> None:
        self._catalog = catalog

    def process_decisions(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: EmploymentRuntimeState,
    ) -> tuple[EmploymentRuntimeState, tuple[ApplicationStageRecord, ...]]:
        applications = list(runtime.applications)
        scheduled_starts = list(runtime.scheduled_starts)
        stage_records = []
        processed = list(runtime.processed_decision_ids)
        accepted_offer = False
        for decision in context.decisions:
            if not isinstance(decision, DecisionRecord) or not _is_employment_event(decision.source_event_id):
                continue
            event = _validate_employment_decision(state, context, decision, applications)
            if decision.decision_id in processed:
                raise ValueError(f"Employment decision '{decision.decision_id}' already processed.")
            processed.append(decision.decision_id)
            if decision.source_event_id.startswith(APPLICATION_PREFIX):
                record, application = self._process_opening_decision(state, context, decision, applications)
                if application is not None:
                    applications.append(application)
                stage_records.append(record)
            elif decision.source_event_id.startswith(INTERVIEW_PREFIX):
                record = self._process_interview_decision(context, decision, applications)
                stage_records.append(record)
            elif decision.source_event_id.startswith(OFFER_PREFIX):
                record, starts, applications = self._process_offer_decision(
                    context,
                    decision,
                    applications,
                    scheduled_starts,
                )
                scheduled_starts = starts
                stage_records.extend(record)
                accepted_offer = decision.chosen_option_id == "accept_offer"
            if event.category != "employment":
                raise ValueError("Expected employment decision event category to be employment.")
        if accepted_offer:
            applications, closures = _close_active_applications_after_acceptance(
                applications,
                context.week,
            )
            stage_records.extend(closures)
        history = EmploymentHistory(
            market_records=runtime.history.market_records,
            application_records=runtime.history.application_records + tuple(stage_records),
            boundary_records=runtime.history.boundary_records,
            work_records=runtime.history.work_records,
        )
        next_runtime = EmploymentRuntimeState(
            history=history,
            applications=tuple(applications),
            scheduled_starts=tuple(scheduled_starts),
            processed_market_weeks=runtime.processed_market_weeks,
            processed_decision_ids=tuple(processed),
            processed_boundary_weeks=runtime.processed_boundary_weeks,
            processed_work_weeks=runtime.processed_work_weeks,
        )
        return next_runtime, tuple(stage_records)

    def _process_opening_decision(
        self,
        state: AgentState,
        context: WeeklyContext,
        decision: DecisionRecord,
        applications: list[JobApplication],
    ) -> tuple[ApplicationStageRecord, JobApplication | None]:
        job_id, version = _parse_event_key(decision.source_event_id, APPLICATION_PREFIX)
        job = self._require_job(job_id, version)
        if decision.chosen_option_id == "apply":
            if any(
                application.job_id == job_id
                and application.job_version == version
                and application.status in ACTIVE_APPLICATION_STATUSES
                for application in applications
            ):
                raise ValueError("Expected only one active application for a job opening.")
            application = JobApplication(
                application_id=_stable_id(
                    "application",
                    str(context.config.simulation.seed),
                    state.identity.agent_id,
                    str(context.week),
                    job_id,
                    version,
                ),
                job_id=job.job_id,
                job_version=job.version,
                status="SUBMITTED",
                created_week=context.week,
                updated_week=context.week,
                response_due_week=context.week + 1,
                source_decision_id=decision.decision_id,
            )
            return (
                _stage_record(
                    application,
                    context.week,
                    "APPLICATION_DECISION",
                    "submitted",
                    None,
                    decision.decision_id,
                ),
                application,
            )
        return (
            ApplicationStageRecord(
                application_id=_stable_id("skipped", decision.decision_id),
                week=context.week,
                stage="APPLICATION_DECISION",
                status_after="SKIPPED",
                job_id=job.job_id,
                job_version=job.version,
                decision_id=decision.decision_id,
                detail="skipped",
            ),
            None,
        )

    def _process_interview_decision(
        self,
        context: WeeklyContext,
        decision: DecisionRecord,
        applications: list[JobApplication],
    ) -> ApplicationStageRecord:
        application_id = _parse_single_key(decision.source_event_id, INTERVIEW_PREFIX)
        index, application = _find_application(applications, application_id)
        if decision.chosen_option_id == "attend_interview":
            updated = replace(
                application,
                status="INTERVIEW_ATTENDED",
                updated_week=context.week,
                offer_due_week=context.week + 1,
            )
            detail = "interview_attended"
        else:
            updated = replace(application, status="WITHDRAWN", updated_week=context.week)
            detail = "interview_declined"
        applications[index] = updated
        return _stage_record(
            updated,
            context.week,
            "INTERVIEW_DECISION",
            detail,
            None,
            decision.decision_id,
        )

    def _process_offer_decision(
        self,
        context: WeeklyContext,
        decision: DecisionRecord,
        applications: list[JobApplication],
        scheduled_starts: list[ScheduledEmploymentStart],
    ) -> tuple[tuple[ApplicationStageRecord, ...], list[ScheduledEmploymentStart], list[JobApplication]]:
        application_id = _parse_single_key(decision.source_event_id, OFFER_PREFIX)
        index, application = _find_application(applications, application_id)
        job = self._require_job(application.job_id, application.job_version)
        records = []
        if decision.chosen_option_id == "accept_offer":
            if scheduled_starts:
                raise ValueError("Expected at most one scheduled employment start.")
            contract_id = _stable_id("contract", application.application_id, decision.decision_id)
            start_week = context.week + 1
            scheduled_starts.append(
                ScheduledEmploymentStart(
                    contract_id=contract_id,
                    application_id=application.application_id,
                    job_id=job.job_id,
                    job_version=job.version,
                    role_title=job.role_title,
                    employer=job.employer,
                    contract_type=job.contract_type,
                    weekly_hours=job.weekly_hours,
                    hourly_rate=job.hourly_rate,
                    stability=job.stability,
                    physical_demand=job.physical_demand,
                    mental_demand=job.mental_demand,
                    social_demand=job.social_demand,
                    start_week=start_week,
                    end_week_exclusive=start_week + job.fixed_term_weeks
                    if job.fixed_term_weeks
                    else 0,
                )
            )
            updated = replace(application, status="ACCEPTED", updated_week=context.week)
            applications[index] = updated
            records.append(
                _stage_record(
                    updated,
                    context.week,
                    "OFFER_DECISION",
                    "accepted_start_scheduled",
                    None,
                    decision.decision_id,
                )
            )
            for other_index, other in enumerate(tuple(applications)):
                if other.application_id == application.application_id:
                    continue
                if other.status in ACTIVE_APPLICATION_STATUSES:
                    closed = replace(other, status="WITHDRAWN", updated_week=context.week)
                    applications[other_index] = closed
                    records.append(
                        _stage_record(
                            closed,
                            context.week,
                            "APPLICATION_CLOSED",
                            "closed_after_offer_acceptance",
                            None,
                        )
                    )
        else:
            updated = replace(application, status="DECLINED", updated_week=context.week)
            applications[index] = updated
            records.append(
                _stage_record(
                    updated,
                    context.week,
                    "OFFER_DECISION",
                    "declined",
                    None,
                    decision.decision_id,
                )
            )
        return tuple(records), scheduled_starts, applications

    def _require_job(self, job_id: str, version: str) -> JobDefinition:
        job = self._catalog.find(job_id, version)
        if job is None:
            raise ValueError(f"Unknown employment job '{job_id}:{version}'.")
        return job


class EmploymentBoundaryEngine:
    def apply_boundary(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: EmploymentRuntimeState,
    ) -> tuple[AgentState, EmploymentRuntimeState, tuple[EmploymentBoundaryRecord, ...]]:
        if context.week in runtime.processed_boundary_weeks:
            raise ValueError(f"Employment boundary already processed for week {context.week}.")
        next_state = state
        records: list[EmploymentBoundaryRecord] = []
        scheduled_starts = list(runtime.scheduled_starts)

        if (
            state.employment.status == "employed"
            and state.employment.end_week_exclusive
            and context.week >= state.employment.end_week_exclusive
        ):
            next_state = _clear_employment(next_state, search_intensity=55.0)
            next_state = replace(
                next_state,
                financial=_remove_employment_stream(
                    next_state.financial,
                    state.employment.contract_id,
                ),
            )
            records.append(
                EmploymentBoundaryRecord(
                    week=context.week,
                    action="contract_ended",
                    contract_id=state.employment.contract_id,
                    role_title=state.employment.role_title,
                    employer=state.employment.employer,
                    income_stream_id=state.employment.contract_id,
                )
            )

        due_starts = tuple(start for start in scheduled_starts if start.start_week <= context.week)
        if len(due_starts) > 1:
            raise ValueError("Expected at most one due employment start.")
        if due_starts:
            start = due_starts[0]
            replacing_contract_id = next_state.employment.contract_id
            if next_state.employment.status == "employed" and replacing_contract_id:
                records.append(
                    EmploymentBoundaryRecord(
                        week=context.week,
                        action="contract_replaced",
                        contract_id=replacing_contract_id,
                        role_title=next_state.employment.role_title,
                        employer=next_state.employment.employer,
                        income_stream_id=replacing_contract_id,
                    )
                )
            next_state = replace(
                next_state,
                financial=_remove_employment_stream(
                    next_state.financial,
                    replacing_contract_id,
                ),
            )
            next_state = replace(
                next_state,
                financial=_remove_employment_stream(next_state.financial, start.contract_id),
            )
            weekly_wage = _weekly_wage(start.hourly_rate, start.weekly_hours)
            stream = IncomeStream(
                name=f"Wages: {start.role_title}",
                amount=weekly_wage,
                cadence="weekly",
                reliability=1.0,
                source_type="employment",
                source_id=start.contract_id,
            )
            next_state = replace(
                next_state,
                financial=replace(
                    next_state.financial,
                    income_streams=next_state.financial.income_streams + (stream,),
                ),
                employment=EmploymentState(
                    status="employed",
                    role_title=start.role_title,
                    employer=start.employer,
                    weekly_hours=start.weekly_hours,
                    job_search_intensity=0.0,
                    source_job_id=start.job_id,
                    source_job_version=start.job_version,
                    contract_id=start.contract_id,
                    contract_type=start.contract_type,
                    hourly_rate=start.hourly_rate,
                    stability=start.stability,
                    physical_demand=start.physical_demand,
                    mental_demand=start.mental_demand,
                    social_demand=start.social_demand,
                    start_week=context.week,
                    tenure_weeks=0,
                    end_week_exclusive=start.end_week_exclusive,
                ),
            )
            scheduled_starts = [item for item in scheduled_starts if item.contract_id != start.contract_id]
            records.append(
                EmploymentBoundaryRecord(
                    week=context.week,
                    action="contract_started",
                    contract_id=start.contract_id,
                    role_title=start.role_title,
                    employer=start.employer,
                    income_stream_id=start.contract_id,
                    weekly_wage=weekly_wage,
                )
            )

        history = EmploymentHistory(
            market_records=runtime.history.market_records,
            application_records=runtime.history.application_records,
            boundary_records=runtime.history.boundary_records + tuple(records),
            work_records=runtime.history.work_records,
        )
        next_runtime = EmploymentRuntimeState(
            history=history,
            applications=runtime.applications,
            scheduled_starts=tuple(scheduled_starts),
            processed_market_weeks=runtime.processed_market_weeks,
            processed_decision_ids=runtime.processed_decision_ids,
            processed_boundary_weeks=runtime.processed_boundary_weeks + (context.week,),
            processed_work_weeks=runtime.processed_work_weeks,
        )
        return next_state, next_runtime, tuple(records)


class EmploymentWorkEngine:
    def apply_work(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: EmploymentRuntimeState,
    ) -> tuple[AgentState, EmploymentRuntimeState, tuple[EmploymentWorkWeekRecord, ...]]:
        if context.week in runtime.processed_work_weeks:
            raise ValueError(f"Employment work already processed for week {context.week}.")
        records = ()
        next_state = state
        if state.employment.status == "employed":
            next_state, effects = _apply_work_effects(state)
            employment = replace(
                next_state.employment,
                tenure_weeks=next_state.employment.tenure_weeks + 1,
            )
            next_state = replace(next_state, employment=employment)
            records = (
                EmploymentWorkWeekRecord(
                    week=context.week,
                    contract_id=employment.contract_id,
                    weekly_hours=employment.weekly_hours,
                    weekly_wage=_weekly_wage(employment.hourly_rate, employment.weekly_hours),
                    tenure_weeks_after=employment.tenure_weeks,
                    effects=effects,
                ),
            )
        history = EmploymentHistory(
            market_records=runtime.history.market_records,
            application_records=runtime.history.application_records,
            boundary_records=runtime.history.boundary_records,
            work_records=runtime.history.work_records + records,
        )
        next_runtime = EmploymentRuntimeState(
            history=history,
            applications=runtime.applications,
            scheduled_starts=runtime.scheduled_starts,
            processed_market_weeks=runtime.processed_market_weeks,
            processed_decision_ids=runtime.processed_decision_ids,
            processed_boundary_weeks=runtime.processed_boundary_weeks,
            processed_work_weeks=runtime.processed_work_weeks + (context.week,),
        )
        return next_state, next_runtime, records


@dataclass(frozen=True, slots=True)
class EmploymentBoundaryTransition:
    engine: EmploymentBoundaryEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, records = self.engine.apply_boundary(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            employment_records=records,
            employment_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class EmploymentMarketTransition:
    engine: EmploymentMarketEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        runtime, events, record = self.engine.advance_market(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=state,
            events=events,
            employment_records=(record,),
            employment_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class EmploymentDecisionTransition:
    engine: EmploymentProcessEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        runtime, records = self.engine.process_decisions(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=state,
            employment_records=records,
            employment_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class EmploymentWorkTransition:
    engine: EmploymentWorkEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, records = self.engine.apply_work(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            employment_records=records,
            employment_runtime=runtime,
        )


def _runtime(context: WeeklyContext) -> EmploymentRuntimeState:
    runtime = context.employment_runtime
    if runtime is None:
        return EmploymentRuntimeState()
    if not isinstance(runtime, EmploymentRuntimeState):
        raise TypeError("Expected WeeklyContext.employment_runtime to contain EmploymentRuntimeState.")
    return runtime


def _discover_jobs(
    state: AgentState,
    context: WeeklyContext,
    runtime: EmploymentRuntimeState,
    catalog: EmploymentCatalog,
) -> tuple[tuple[JobDefinition, ...], tuple[MarketCandidateTrace, ...], tuple[EmploymentDiscoveryDraw, ...]]:
    intensity = state.employment.job_search_intensity / 100.0
    if intensity <= 0:
        return (
            (),
            tuple(
                MarketCandidateTrace(job.job_id, job.version, False, 0.0, "no_search_intensity")
                for job in catalog.jobs
            ),
            tuple(
                EmploymentDiscoveryDraw(
                    slot=slot,
                    trigger_probability=0.0,
                    trigger_roll=_market_trigger_roll(state, context, slot),
                    triggered=False,
                )
                for slot in range(catalog.max_discoveries_per_week)
            ),
        )
    active_keys = {
        (application.job_id, application.job_version)
        for application in runtime.applications
        if application.status in ACTIVE_APPLICATION_STATUSES
    }
    if state.employment.status == "employed":
        active_keys.add((state.employment.source_job_id, state.employment.source_job_version))
    recently_seen = _recently_seen(runtime.history, context.week, catalog.relisting_cooldown_weeks)
    candidates = []
    weighted: list[tuple[JobDefinition, float]] = []
    for job in catalog.jobs:
        key = (job.job_id, job.version)
        if key in active_keys:
            candidates.append(MarketCandidateTrace(job.job_id, job.version, False, 0.0, "active_application"))
            continue
        if key in recently_seen:
            candidates.append(MarketCandidateTrace(job.job_id, job.version, False, 0.0, "relisting_cooldown"))
            continue
        weight = round(job.base_discovery_weight, 12)
        eligible = weight > 0
        candidates.append(
            MarketCandidateTrace(
                job.job_id,
                job.version,
                eligible,
                weight,
                "eligible" if eligible else "zero_weight",
            )
        )
        if eligible:
            weighted.append((job, weight))
    selected = []
    discovery_draws: list[EmploymentDiscoveryDraw] = []
    available = list(weighted)
    for slot in range(min(catalog.max_discoveries_per_week, len(available))):
        trigger_probability = _market_trigger_probability(state, catalog)
        trigger_roll = _market_trigger_roll(state, context, slot)
        if trigger_roll > trigger_probability:
            discovery_draws.append(
                EmploymentDiscoveryDraw(
                    slot=slot,
                    trigger_probability=trigger_probability,
                    trigger_roll=trigger_roll,
                    triggered=False,
                )
            )
            continue
        total = sum(weight for _, weight in available)
        if total <= 0:
            discovery_draws.append(
                EmploymentDiscoveryDraw(
                    slot=slot,
                    trigger_probability=trigger_probability,
                    trigger_roll=trigger_roll,
                    triggered=False,
                )
            )
            break
        rng = Random(
            derive_stable_seed(
                "employment-market-selection",
                str(context.config.simulation.seed),
                state.identity.agent_id,
                str(context.week),
                str(slot),
            )
        )
        roll = rng.random() * total
        cursor = 0.0
        chosen_index = 0
        for index, (job, weight) in enumerate(available):
            cursor += weight
            if roll <= cursor:
                chosen_index = index
                break
        selected.append(available[chosen_index][0])
        discovery_draws.append(
            EmploymentDiscoveryDraw(
                slot=slot,
                trigger_probability=trigger_probability,
                trigger_roll=trigger_roll,
                triggered=True,
                total_weight=round(total, 12),
                selection_roll=round(roll, 12),
                selected_job_key=_job_key(available[chosen_index][0]),
            )
        )
        del available[chosen_index]
    for slot in range(len(discovery_draws), catalog.max_discoveries_per_week):
        discovery_draws.append(
            EmploymentDiscoveryDraw(
                slot=slot,
                trigger_probability=_market_trigger_probability(state, catalog),
                trigger_roll=_market_trigger_roll(state, context, slot),
                triggered=False,
            )
        )
    return tuple(selected), tuple(candidates), tuple(discovery_draws)


def _market_trigger_probability(state: AgentState, catalog: EmploymentCatalog) -> float:
    intensity = state.employment.job_search_intensity / 100.0
    city_modifier = 0.85 + state.social.city_familiarity / 500.0
    return round(min(0.9, max(0.0, catalog.base_market_discovery_probability * intensity * city_modifier)), 12)


def _market_trigger_roll(state: AgentState, context: WeeklyContext, slot: int) -> float:
    rng = Random(
        derive_stable_seed(
            "employment-market-trigger",
            str(context.config.simulation.seed),
            state.identity.agent_id,
            str(context.week),
            str(slot),
        )
    )
    return round(rng.random(), 12)


def _recently_seen(history: EmploymentHistory, week: int, cooldown: int) -> set[tuple[str, str]]:
    output = set()
    for record in history.market_records:
        if week - record.week <= cooldown:
            for key in record.discovered_job_keys:
                job_id, version = key.split(":", 1)
                output.add((job_id, version))
    return output


def _opening_event(job: JobDefinition, week: int) -> EventOccurrence:
    weekly_wage = _weekly_wage(job.hourly_rate, job.weekly_hours)
    return EventOccurrence(
        event_id=f"{APPLICATION_PREFIX}{job.job_id}:{job.version}",
        version="1",
        week=week,
        category="employment",
        effective_weight=1.0,
        title=f"Job opening: {job.role_title}",
        summary=f"{job.employer} is hiring for {job.weekly_hours:g} hours per week.",
        tags=("employment", "finance", *job.tags),
        options=(
            EventOption(
                option_id="apply",
                label="Apply",
                summary="Spend time and energy applying for this job.",
                estimated_cost=Decimal("0.00"),
                requires_full_estimated_cost=False,
                expected_weekly_financial_gain=weekly_wage,
                ongoing_weekly_time_hours=job.weekly_hours,
                time_cost_hours=2.0,
                energy_cost=8.0,
                short_term_value=-0.05,
                future_value=0.55 + job.stability * 0.2,
                perceived_risk=0.25,
                uncertainty=0.45,
                social_value=job.social_demand * 0.15,
                autonomy_value=-0.05,
                learning_value=0.12,
                health_value=-job.physical_demand * 0.08,
                comfort_value=-0.15,
                goal_tags=("finance", "stability", "career"),
            ),
            EventOption(
                option_id="skip",
                label="Skip",
                summary="Do not apply for this opening.",
                estimated_cost=Decimal("0.00"),
                time_cost_hours=0.0,
                energy_cost=0.0,
                short_term_value=0.12,
                future_value=-0.15,
                perceived_risk=0.05,
                uncertainty=0.05,
                autonomy_value=0.2,
                comfort_value=0.15,
            ),
        ),
    )


def _interview_event(job: JobDefinition, application: JobApplication, week: int) -> EventOccurrence:
    return EventOccurrence(
        event_id=f"{INTERVIEW_PREFIX}{application.application_id}",
        version="1",
        week=week,
        category="employment",
        effective_weight=1.0,
        title=f"Interview invitation: {job.role_title}",
        summary=f"{job.employer} invited an interview for {job.role_title}.",
        tags=("employment", "interview", *job.tags),
        time_pressure=0.2,
        options=(
            EventOption(
                option_id="attend_interview",
                label="Attend interview",
                summary="Prepare for and attend the interview.",
                time_cost_hours=3.0,
                energy_cost=16.0,
                short_term_value=-0.08,
                future_value=0.65,
                perceived_risk=0.25,
                uncertainty=0.35,
                social_value=0.1 + job.social_demand * 0.1,
                autonomy_value=-0.02,
                learning_value=0.16,
                comfort_value=-0.2,
                goal_tags=("finance", "career", "stability"),
            ),
            EventOption(
                option_id="decline_interview",
                label="Decline interview",
                summary="Withdraw from the hiring process.",
                short_term_value=0.12,
                future_value=-0.25,
                perceived_risk=0.05,
                uncertainty=0.05,
                autonomy_value=0.2,
                comfort_value=0.18,
            ),
        ),
    )


def _offer_event(job: JobDefinition, application: JobApplication, week: int) -> EventOccurrence:
    weekly_wage = _weekly_wage(job.hourly_rate, job.weekly_hours)
    return EventOccurrence(
        event_id=f"{OFFER_PREFIX}{application.application_id}",
        version="1",
        week=week,
        category="employment",
        effective_weight=1.0,
        title=f"Job offer: {job.role_title}",
        summary=f"{job.employer} offered {job.role_title} at {weekly_wage} EUR/week.",
        tags=("employment", "offer", "finance", *job.tags),
        time_pressure=0.35,
        options=(
            EventOption(
                option_id="accept_offer",
                label="Accept offer",
                summary="Accept the job and start next week.",
                estimated_cost=Decimal("0.00"),
                requires_full_estimated_cost=False,
                expected_weekly_financial_gain=weekly_wage,
                ongoing_weekly_time_hours=job.weekly_hours,
                time_cost_hours=1.0,
                energy_cost=4.0,
                short_term_value=0.2,
                future_value=0.7 + job.stability * 0.2,
                perceived_risk=max(0.02, 0.35 - job.stability * 0.25),
                uncertainty=0.15,
                social_value=job.social_demand * 0.15,
                autonomy_value=-0.08,
                learning_value=0.18,
                health_value=-(job.physical_demand + job.mental_demand) * 0.05,
                comfort_value=-0.1,
                goal_tags=("finance", "career", "stability"),
            ),
            EventOption(
                option_id="decline_offer",
                label="Decline offer",
                summary="Do not take this job.",
                short_term_value=0.08,
                future_value=-0.25,
                perceived_risk=0.08,
                uncertainty=0.08,
                autonomy_value=0.28,
                comfort_value=0.1,
            ),
        ),
    )


def _offer_events_for_available_applications(
    applications: list[JobApplication],
    catalog: EmploymentCatalog,
    week: int,
) -> tuple[EventOccurrence, ...]:
    available = sorted(
        (application for application in applications if application.status == "OFFER_AVAILABLE"),
        key=lambda application: application.application_id,
    )
    if not available:
        return ()
    application = available[0]
    job = catalog.find(application.job_id, application.job_version)
    if job is None:
        raise ValueError("Expected offer application to reference a known job.")
    return (_offer_event(job, application, week),)


def _validate_employment_decision(
    state: AgentState,
    context: WeeklyContext,
    decision: DecisionRecord,
    applications: list[JobApplication],
) -> EventOccurrence:
    if decision.agent_id != state.identity.agent_id:
        raise ValueError("Employment decision does not belong to the current agent.")
    if decision.week != context.week:
        raise ValueError("Expected employment decision week to match WeeklyContext.week.")
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
        raise ValueError("Expected employment decision to reference a same-week event.")
    if event.category != "employment":
        raise ValueError("Expected employment decision event category to be employment.")
    option_ids = {option.option_id for option in event.options}
    if decision.chosen_option_id not in option_ids:
        raise ValueError("Expected employment decision chosen option to exist on its event.")
    if decision.source_event_id.startswith(APPLICATION_PREFIX):
        if decision.chosen_option_id not in {"apply", "skip"}:
            raise ValueError("Expected opening decision option to be apply or skip.")
    elif decision.source_event_id.startswith(INTERVIEW_PREFIX):
        if decision.chosen_option_id not in {"attend_interview", "decline_interview"}:
            raise ValueError("Expected interview decision option to be attend_interview or decline_interview.")
        _, application = _find_application(applications, _parse_single_key(decision.source_event_id, INTERVIEW_PREFIX))
        if application.status != "INTERVIEW_INVITED":
            raise ValueError("Expected interview decision to reference an invited application.")
    elif decision.source_event_id.startswith(OFFER_PREFIX):
        if decision.chosen_option_id not in {"accept_offer", "decline_offer"}:
            raise ValueError("Expected offer decision option to be accept_offer or decline_offer.")
        _, application = _find_application(applications, _parse_single_key(decision.source_event_id, OFFER_PREFIX))
        if application.status != "OFFER_AVAILABLE":
            raise ValueError("Expected offer decision to reference an available offer.")
    else:
        raise ValueError("Unsupported employment decision event.")
    return event


def _close_active_applications_after_acceptance(
    applications: list[JobApplication],
    week: int,
) -> tuple[list[JobApplication], tuple[ApplicationStageRecord, ...]]:
    output = list(applications)
    records = []
    for index, application in enumerate(tuple(output)):
        if application.status in ACTIVE_APPLICATION_STATUSES:
            closed = replace(application, status="WITHDRAWN", updated_week=week)
            output[index] = closed
            records.append(
                _stage_record(
                    closed,
                    week,
                    "APPLICATION_CLOSED",
                    "closed_after_same_week_offer_acceptance",
                    None,
                )
            )
    return output, tuple(records)


def _probability_audit(
    state: AgentState,
    context: WeeklyContext,
    application: JobApplication,
    job: JobDefinition,
    stage: str,
) -> ProbabilityAudit:
    fit = candidate_fit(state, job, stage=stage)
    base = job.base_interview_probability if stage == "interview" else job.base_offer_probability
    fit_adjustment = (fit.final_fit - 0.5) * (0.30 if stage == "interview" else 0.24)
    contextual = 0.0
    if stage == "offer":
        contextual += (state.personality.confidence - 0.5) * 0.06
        contextual += (state.personality.conscientiousness - 0.5) * 0.04
        contextual -= state.mental.stress / 100.0 * 0.05
    final = _clamp_probability(base + fit_adjustment + contextual)
    rng = Random(
        derive_stable_seed(
            f"employment-{stage}",
            str(context.config.simulation.seed),
            state.identity.agent_id,
            application.application_id,
            job.job_id,
            job.version,
            str(context.week),
        )
    )
    roll = rng.random()
    return ProbabilityAudit(
        stage=stage,
        application_id=application.application_id,
        job_id=job.job_id,
        job_version=job.version,
        base_probability=round(base, 12),
        fit_adjustment=round(fit_adjustment, 12),
        contextual_modifier=round(contextual, 12),
        final_probability=final,
        roll=round(roll, 12),
        result="success" if roll <= final else "failure",
        candidate_fit=fit,
    )


def candidate_fit(state: AgentState, job: JobDefinition, *, stage: str = "interview") -> CandidateFitTrace:
    skills = {skill.name: skill.level for skill in state.skills.items}
    if job.skill_requirements:
        total_weight = sum(requirement.weight for requirement in job.skill_requirements)
        weighted = 0.0
        for requirement in job.skill_requirements:
            level = skills.get(requirement.skill_name, 0.0)
            desired = max(1.0, requirement.desired_level)
            weighted += min(1.15, level / desired) * requirement.weight
        skill_score = min(1.0, weighted / total_weight / 1.15) if total_weight else 0.5
    else:
        skill_score = 0.5
    city_modifier = 0.0
    education_modifier = 0.03 if state.education.status == "enrolled" else 0.0
    personality_modifier = 0.0
    stress_modifier = 0.0
    if stage == "offer":
        personality_modifier = (
            (state.personality.confidence - 0.5) * 0.05
            + (state.personality.conscientiousness - 0.5) * 0.04
        )
        stress_modifier = -(state.mental.stress / 100.0) * 0.04
    final_fit = min(
        1.0,
        max(0.0, skill_score + city_modifier + education_modifier + personality_modifier + stress_modifier),
    )
    return CandidateFitTrace(
        job_id=job.job_id,
        job_version=job.version,
        skill_score=round(skill_score, 12),
        city_familiarity_modifier=round(city_modifier, 12),
        education_modifier=round(education_modifier, 12),
        personality_modifier=round(personality_modifier, 12),
        stress_modifier=round(stress_modifier, 12),
        final_fit=round(final_fit, 12),
    )


def _stage_record(
    application: JobApplication,
    week: int,
    stage: str,
    detail: str,
    audit: ProbabilityAudit | None,
    decision_id: str = "",
) -> ApplicationStageRecord:
    return ApplicationStageRecord(
        application_id=application.application_id,
        week=week,
        stage=stage,
        status_after=application.status,
        job_id=application.job_id,
        job_version=application.job_version,
        decision_id=decision_id,
        detail=detail,
        probability_audit=audit,
    )


def _apply_work_effects(state: AgentState) -> tuple[AgentState, tuple[EmploymentEffectApplication, ...]]:
    employment = state.employment
    hours_factor = employment.weekly_hours / 40.0
    changes = (
        ("health.energy", -(employment.physical_demand * 5.0 + hours_factor * 2.5), "work_energy"),
        ("mental.stress", employment.mental_demand * 4.0 + hours_factor * 1.5, "work_stress"),
        ("mental.mental_load", employment.mental_demand * 5.0 + hours_factor * 2.0, "work_load"),
        ("mental.recovery_need", (employment.physical_demand + employment.mental_demand) * 3.0, "work_recovery"),
        ("needs.purpose", 1.0 + employment.stability * 1.5, "work_purpose"),
    )
    next_state = state
    effects = []
    for path, delta, reason in changes:
        next_state, effect = _bounded_replace(next_state, path, float(delta), reason)
        effects.append(effect)
    return next_state, tuple(effects)


def _bounded_replace(
    state: AgentState,
    path: str,
    delta: float,
    reason: str,
) -> tuple[AgentState, EmploymentEffectApplication]:
    section_name, field_name = path.split(".", 1)
    section = getattr(state, section_name)
    before = getattr(section, field_name)
    raw_after = before + delta
    after = min(100.0, max(0.0, raw_after))
    clamped = after != raw_after
    next_state = replace(state, **{section_name: replace(section, **{field_name: after})})
    return next_state, EmploymentEffectApplication(
        path=path,
        before=before,
        after=after,
        delta=after - before,
        clamped=clamped,
        reason=reason,
    )


def _clear_employment(state: AgentState, *, search_intensity: float) -> AgentState:
    return replace(
        state,
        employment=EmploymentState(
            status=SEARCHING_STATUS,
            role_title="",
            employer="",
            weekly_hours=0.0,
            job_search_intensity=search_intensity,
        ),
    )


def _remove_employment_stream(financial: FinancialState, contract_id: str) -> FinancialState:
    if not contract_id:
        return financial
    return replace(
        financial,
        income_streams=tuple(
            stream
            for stream in financial.income_streams
            if not (stream.source_type == "employment" and stream.source_id == contract_id)
        ),
    )


def _weekly_wage(hourly_rate: Decimal, weekly_hours: float) -> Decimal:
    return (hourly_rate * Decimal(str(weekly_hours))).quantize(CENT, rounding=ROUND_HALF_UP)


def _find_application(
    applications: list[JobApplication],
    application_id: str,
) -> tuple[int, JobApplication]:
    for index, application in enumerate(applications):
        if application.application_id == application_id:
            return index, application
    raise ValueError(f"Unknown employment application '{application_id}'.")


def _is_employment_event(event_id: str) -> bool:
    return event_id.startswith((APPLICATION_PREFIX, INTERVIEW_PREFIX, OFFER_PREFIX))


def _parse_event_key(event_id: str, prefix: str) -> tuple[str, str]:
    value = event_id.removeprefix(prefix)
    job_id, version = value.rsplit(":", 1)
    return job_id, version


def _parse_single_key(event_id: str, prefix: str) -> str:
    return event_id.removeprefix(prefix)


def _job_key(job: JobDefinition) -> str:
    return f"{job.job_id}:{job.version}"


def _clamp_probability(value: float) -> float:
    return round(min(0.95, max(0.05, value)), 12)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
