from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from random import Random
from typing import Any

from lifesim.agents.state import (
    AgentState,
    Debt,
    FinancialState,
    RoutineState,
)
from lifesim.decisions.engine import DecisionEngine
from lifesim.decisions.model import DecisionHistory
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.finance import (
    MANDATORY_FUNDING_ORDER,
    OPTIONAL_FUNDING_ORDER,
    find_arrear,
    money,
    settle_liquid_amount,
    upsert_arrear,
)
from lifesim.passive.model import (
    ArrearSettlementRecord,
    CashflowEntry,
    CashflowRecord,
    FundingTransfer,
    PassiveLifeRuntimeState,
    RoutineCatalog,
    RoutineEffectApplication,
    RoutineProfile,
    RoutineWeekRecord,
)
from lifesim.rng import derive_stable_seed
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult

FUNDING_ORDER = MANDATORY_FUNDING_ORDER
ROUTINE_OPTIONAL_FUNDING_ORDER = OPTIONAL_FUNDING_ORDER


class PassiveCashflowEngine:
    def apply(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: PassiveLifeRuntimeState,
    ) -> tuple[AgentState, PassiveLifeRuntimeState, CashflowRecord]:
        if context.week in runtime.processed_cashflow_weeks:
            raise ValueError(f"Passive cashflow already processed for week {context.week}.")
        next_state = state
        entries: list[CashflowEntry] = []
        effects: list[RoutineEffectApplication] = []

        for stream in state.financial.income_streams:
            due_date = _due_date(stream.cadence, stream.due_day, context.week_start, context.week_end)
            if due_date is None:
                continue
            arrives, roll = _income_arrives(state, context, stream.name, stream.reliability, due_date)
            amount_paid = stream.amount if arrives else Decimal("0.00")
            if arrives:
                financial = next_state.financial
                next_state = replace(
                    next_state,
                    financial=replace(
                        financial,
                bank_balance=money(financial.bank_balance + stream.amount),
                    ),
                )
            entries.append(
                CashflowEntry(
                    entry_id=f"income:{stream.name}:{due_date.isoformat()}",
                    kind="income",
                    name=stream.name,
                    amount_due=stream.amount,
                    amount_paid=amount_paid,
                    cadence=stream.cadence,
                    due_date=due_date.isoformat(),
                    paid=arrives,
                    reliability=stream.reliability,
                    roll=roll,
                )
            )

        for commitment in next_state.financial.recurring_commitments:
            due_date = _due_date(commitment.cadence, commitment.due_day, context.week_start, context.week_end)
            if due_date is None:
                continue
            next_state, entry, obligation_effects = _pay_obligation(
                next_state,
                obligation_id=f"commitment:{commitment.name}",
                name=commitment.name,
                category=commitment.category,
                amount=commitment.amount,
                cadence=commitment.cadence,
                due_date=due_date,
                context=context,
                kind="commitment",
            )
            entries.append(entry)
            effects.extend(obligation_effects)

        next_state, debt_entries, debt_effects = _process_debts(next_state, context)
        entries.extend(debt_entries)
        effects.extend(debt_effects)

        record = CashflowRecord(
            week=context.week,
            week_start=context.week_start.isoformat(),
            week_end=context.week_end.isoformat(),
            entries=tuple(entries),
            effects=tuple(effects),
        )
        runtime = replace(
            runtime,
            history=runtime.history.record_cashflow(record),
            processed_cashflow_weeks=runtime.processed_cashflow_weeks + (context.week,),
        )
        return next_state, runtime, record


class RoutineEngine:
    def __init__(self, catalog: RoutineCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> RoutineCatalog:
        return self._catalog

    def plan(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: PassiveLifeRuntimeState,
        decision_engine: DecisionEngine,
        history: DecisionHistory,
    ) -> tuple[PassiveLifeRuntimeState, EventOccurrence, DecisionHistory, Any]:
        if context.week in runtime.processed_routine_planning_weeks:
            raise ValueError(f"Routine planning already processed for week {context.week}.")
        occurrence = _routine_occurrence(context, self._catalog, state)
        decision = decision_engine.decide_event(state, context, occurrence)
        next_history = history.record((decision,))
        if decision.chosen_option_id is None:
            raise ValueError("Expected routine planning to choose an available routine profile.")
        profile_id = decision.chosen_option_id
        runtime = replace(
            runtime,
            planned_routine_profile_id=profile_id,
            planned_routine_decision=decision,
            planned_routine_week=context.week,
            processed_routine_planning_weeks=(
                runtime.processed_routine_planning_weeks + (context.week,)
            ),
        )
        return runtime, occurrence, next_history, decision

    def execute(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: PassiveLifeRuntimeState,
    ) -> tuple[AgentState, PassiveLifeRuntimeState, RoutineWeekRecord]:
        if context.week in runtime.processed_routine_execution_weeks:
            raise ValueError(f"Routine execution already processed for week {context.week}.")
        if runtime.planned_routine_week != context.week:
            raise ValueError("Expected planned routine week to match execution week.")
        profile_id = runtime.planned_routine_profile_id
        if not profile_id:
            raise ValueError("Expected planned routine profile before routine execution.")
        profile = self._catalog.get(profile_id)
        decision = runtime.planned_routine_decision
        if decision is None:
            raise ValueError("Expected planned routine decision before routine execution.")

        next_state, spending, spending_effects = _apply_routine_spending(state, context, profile)
        previous_profile_id = next_state.routine.current_profile_id
        weeks_in_profile = (
            next_state.routine.weeks_in_current_profile + 1
            if profile.profile_id == next_state.routine.current_profile_id
            else 1
        )
        low_social_streak = (
            next_state.routine.low_social_streak + 1
            if profile.social_contact < 0.35
            else 0
        )
        routine = RoutineState(
            current_profile_id=profile.profile_id,
            previous_profile_id=previous_profile_id if previous_profile_id != profile.profile_id else next_state.routine.previous_profile_id,
            weeks_in_current_profile=weeks_in_profile,
            low_social_streak=low_social_streak,
        )
        next_state = replace(next_state, routine=routine)
        next_state, routine_effects = _apply_routine_effects(next_state, profile, low_social_streak)
        effects = spending_effects + routine_effects
        record = RoutineWeekRecord(
            week=context.week,
            profile_id=profile.profile_id,
            previous_profile_id=routine.previous_profile_id,
            weeks_in_current_profile=routine.weeks_in_current_profile,
            low_social_streak=routine.low_social_streak,
            decision_id=decision.decision_id,
            spending=spending,
            effects=effects,
        )
        runtime = replace(
            runtime,
            history=runtime.history.record_routine(record),
            planned_routine_profile_id="",
            planned_routine_decision=None,
            planned_routine_week=None,
            processed_routine_execution_weeks=(
                runtime.processed_routine_execution_weeks + (context.week,)
            ),
        )
        return next_state, runtime, record


class ArrearSettlementEngine:
    def apply(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: PassiveLifeRuntimeState,
    ) -> tuple[AgentState, PassiveLifeRuntimeState, ArrearSettlementRecord]:
        if context.week in runtime.processed_arrear_settlement_weeks:
            raise ValueError(f"Arrear settlement already processed for week {context.week}.")
        financial = state.financial
        entries: list[CashflowEntry] = []
        arrears = sorted(financial.arrears, key=lambda item: (item.first_missed_week, item.obligation_id))
        for arrear in arrears:
            if arrear.balance <= Decimal("0.00"):
                continue
            settlement = settle_liquid_amount(financial, arrear.balance, FUNDING_ORDER)
            financial = settlement.financial
            remaining = settlement.amount_unpaid
            updated_arrears = []
            for current in financial.arrears:
                if current.obligation_id != arrear.obligation_id:
                    updated_arrears.append(current)
                elif remaining > Decimal("0.00"):
                    updated_arrears.append(
                        replace(
                            current,
                            balance=remaining,
                            last_updated_week=context.week,
                        )
                    )
            financial = replace(financial, arrears=tuple(updated_arrears))
            entries.append(
                CashflowEntry(
                    entry_id=f"arrear-settlement:{arrear.obligation_id}:{context.week}",
                    kind="arrear_settlement",
                    name=arrear.obligation_id,
                    amount_due=arrear.balance,
                    amount_paid=settlement.amount_paid,
                    cadence="weekly",
                    due_date=context.week_start.isoformat(),
                    paid=settlement.fully_paid,
                    funding=_funding_transfers(settlement.transfers),
                    arrear_balance_after=max(Decimal("0.00"), remaining),
                )
            )
            if settlement.amount_unpaid > Decimal("0.00"):
                break
        next_state = replace(state, financial=financial)
        record = ArrearSettlementRecord(week=context.week, entries=tuple(entries))
        runtime = replace(
            runtime,
            history=runtime.history.record_arrear_settlement(record),
            processed_arrear_settlement_weeks=runtime.processed_arrear_settlement_weeks + (context.week,),
        )
        return next_state, runtime, record


@dataclass(frozen=True, slots=True)
class PassiveCashflowTransition:
    engine: PassiveCashflowEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.apply(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            passive_records=(record,),
            passive_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class ArrearSettlementTransition:
    engine: ArrearSettlementEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.apply(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            passive_records=(record,),
            passive_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class RoutinePlanningTransition:
    engine: RoutineEngine
    decision_engine: DecisionEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        history = context.decision_history
        if history is None:
            history = DecisionHistory()
        if not isinstance(history, DecisionHistory):
            raise TypeError("Expected WeeklyContext.decision_history to contain DecisionHistory.")
        runtime, occurrence, history, decision = self.engine.plan(
            state,
            context,
            runtime,
            self.decision_engine,
            history,
        )
        return WeeklyTransitionResult(
            agent_state=state,
            events=(occurrence,),
            decisions=(decision,),
            decision_history=history,
            passive_runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class RoutineExecutionTransition:
    engine: RoutineEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, runtime, record = self.engine.execute(state, context, runtime)
        return WeeklyTransitionResult(
            agent_state=next_state,
            passive_records=(record,),
            passive_runtime=runtime,
        )


def _runtime(context: WeeklyContext) -> PassiveLifeRuntimeState:
    runtime = context.passive_runtime
    if runtime is None:
        return PassiveLifeRuntimeState()
    if not isinstance(runtime, PassiveLifeRuntimeState):
        raise TypeError("Expected WeeklyContext.passive_runtime to contain PassiveLifeRuntimeState.")
    return runtime


def _routine_occurrence(
    context: WeeklyContext,
    catalog: RoutineCatalog,
    state: AgentState | None = None,
) -> EventOccurrence:
    return EventOccurrence(
        event_id="weekly_routine",
        version="1",
        week=context.week,
        category="routine",
        effective_weight=1.0,
        title="Weekly routine",
        summary="Choose an ordinary weekly routine.",
        tags=("routine",),
        options=tuple(_profile_option(profile, state) for profile in catalog.profiles),
    )


def _profile_option(
    profile: RoutineProfile,
    state: AgentState | None = None,
) -> EventOption:
    repeat_penalty = _repeat_penalty(profile, state)
    return EventOption(
        option_id=profile.profile_id,
        label=profile.label,
        summary=profile.summary,
        estimated_cost=profile.estimated_cost,
        time_cost_hours=profile.time_cost_hours,
        energy_cost=profile.energy_cost,
        short_term_value=max(-1.0, profile.short_term_value - repeat_penalty),
        future_value=max(-1.0, profile.future_value - repeat_penalty * 0.5),
        perceived_risk=profile.perceived_risk,
        uncertainty=min(1.0, profile.uncertainty + repeat_penalty * 0.35),
        social_value=profile.social_value,
        social_pressure=profile.social_pressure,
        autonomy_value=max(-1.0, profile.autonomy_value - repeat_penalty * 0.4),
        learning_value=profile.learning_value,
        health_value=profile.health_value,
        comfort_value=max(-1.0, profile.comfort_value - repeat_penalty),
        goal_tags=profile.goal_tags,
        behavior_tags=profile.behavior_tags,
        requires_full_estimated_cost=False,
    )


def _repeat_penalty(profile: RoutineProfile, state: AgentState | None) -> float:
    if state is None or state.routine.current_profile_id != profile.profile_id:
        return 0.0
    repeated_weeks = max(0, state.routine.weeks_in_current_profile - 1)
    return min(1.20, repeated_weeks * 0.12)


def _process_debts(
    state: AgentState,
    context: WeeklyContext,
) -> tuple[AgentState, tuple[CashflowEntry, ...], tuple[RoutineEffectApplication, ...]]:
    debts: list[Debt] = []
    entries: list[CashflowEntry] = []
    effects: list[RoutineEffectApplication] = []
    next_state = state
    for debt in state.financial.debts:
        updated = debt
        if debt.balance > Decimal(0) and debt.interest_rate > Decimal(0):
            weekly_interest = money(debt.balance * debt.interest_rate / Decimal(52))
            updated = replace(updated, balance=money(updated.balance + weekly_interest))
            entries.append(
                CashflowEntry(
                    entry_id=f"debt-interest:{debt.name}:{context.week}",
                    kind="debt_interest",
                    name=debt.name,
                    amount_due=weekly_interest,
                    amount_paid=weekly_interest,
                    cadence="weekly",
                    due_date=context.week_start.isoformat(),
                    paid=True,
                )
            )
        due_date = _due_date(updated.payment_cadence, updated.due_day, context.week_start, context.week_end)
        if due_date is not None and updated.balance > Decimal(0) and updated.minimum_payment > Decimal(0):
            amount = min(updated.minimum_payment, updated.balance)
            next_state = replace(next_state, financial=replace(next_state.financial, debts=tuple(debts + [updated] + list(state.financial.debts[len(debts) + 1:]))))
            next_state, entry, obligation_effects = _pay_obligation(
                next_state,
                obligation_id=f"debt:{updated.name}",
                name=updated.name,
                category="debt",
                amount=amount,
                cadence=updated.payment_cadence,
                due_date=due_date,
                context=context,
                kind="debt_payment",
            )
            effects.extend(obligation_effects)
            paid_amount = entry.amount_paid
            missed = paid_amount < amount
            updated = replace(
                updated,
                balance=money(updated.balance - paid_amount),
                consecutive_missed_payments=(
                    updated.consecutive_missed_payments + 1 if missed else 0
                ),
            )
            entries.append(entry)
        debts.append(updated)
    next_state = replace(next_state, financial=replace(next_state.financial, debts=tuple(debts)))
    return next_state, tuple(entries), tuple(effects)


def _pay_obligation(
    state: AgentState,
    *,
    obligation_id: str,
    name: str,
    category: str,
    amount: Decimal,
    cadence: str,
    due_date: date,
    context: WeeklyContext,
    kind: str,
) -> tuple[AgentState, CashflowEntry, tuple[RoutineEffectApplication, ...]]:
    financial, paid, transfers = _fund_payment(state.financial, amount)
    unpaid = money(amount - paid)
    arrear_balance = None
    effects: tuple[RoutineEffectApplication, ...] = ()
    if unpaid > Decimal(0):
        financial = upsert_arrear(
            financial,
            obligation_id=obligation_id,
            category=category,
            unpaid=unpaid,
            week=context.week,
        )
        arrear = find_arrear(financial, obligation_id)
        arrear_balance = arrear.balance if arrear is not None else unpaid
    next_state = replace(state, financial=financial)
    if unpaid > Decimal(0):
        next_state, effects = _apply_unpaid_pressure(next_state, category, unpaid, amount, obligation_id)
    entry = CashflowEntry(
        entry_id=f"{kind}:{name}:{due_date.isoformat()}",
        kind=kind,
        name=name,
        amount_due=amount,
        amount_paid=paid,
        cadence=cadence,
        due_date=due_date.isoformat(),
        paid=paid == amount,
        funding=transfers,
        arrear_balance_after=arrear_balance,
    )
    return next_state, entry, effects


def _fund_payment(
    financial: FinancialState,
    amount: Decimal,
    funding_order: tuple[str, ...] = FUNDING_ORDER,
) -> tuple[FinancialState, Decimal, tuple[FundingTransfer, ...]]:
    settlement = settle_liquid_amount(financial, amount, funding_order)
    return settlement.financial, settlement.amount_paid, _funding_transfers(settlement.transfers)


def _apply_routine_spending(
    state: AgentState,
    context: WeeklyContext,
    profile: RoutineProfile,
) -> tuple[AgentState, tuple[CashflowEntry, ...], tuple[RoutineEffectApplication, ...]]:
    next_state = state
    entries: list[CashflowEntry] = []
    effects: list[RoutineEffectApplication] = []
    minimum_food = min(profile.minimum_food_budget, profile.food_budget)
    optional_food = money(profile.food_budget - minimum_food)
    financial, minimum_paid, minimum_transfers = _fund_payment(next_state.financial, minimum_food)
    financial, optional_paid, optional_transfers = _fund_payment(
        financial,
        optional_food,
        ROUTINE_OPTIONAL_FUNDING_ORDER,
    )
    paid = money(minimum_paid + optional_paid)
    transfers = minimum_transfers + optional_transfers
    next_state = replace(next_state, financial=financial)
    minimum_shortfall = max(Decimal("0.00"), money(minimum_food - minimum_paid))
    if minimum_shortfall > Decimal(0):
        next_state, effect = _bounded_replace(
            next_state,
            "needs.food_security",
            -(float(minimum_shortfall / Decimal(10))),
            "food_shortfall",
            source=f"routine_food:{profile.profile_id}:{context.week}",
        )
        effects.append(effect)
    entries.append(
        CashflowEntry(
            entry_id=f"routine_food:{profile.profile_id}:{context.week}",
            kind="routine_food",
            name="basic food",
            amount_due=profile.food_budget,
            amount_paid=paid,
            cadence="weekly",
            due_date=context.week_start.isoformat(),
            paid=paid == profile.food_budget,
            funding=transfers,
        )
    )

    spending = (
        (
            "routine_transport",
            "routine transport",
            profile.transport_budget,
        ),
        (
            "routine_discretionary",
            "routine discretionary",
            profile.discretionary_budget,
        ),
    )
    for kind, name, amount in spending:
        financial, paid, transfers = _fund_payment(
            next_state.financial,
            amount,
            ROUTINE_OPTIONAL_FUNDING_ORDER,
        )
        next_state = replace(next_state, financial=financial)
        entries.append(
            CashflowEntry(
                entry_id=f"{kind}:{profile.profile_id}:{context.week}",
                kind=kind,
                name=name,
                amount_due=amount,
                amount_paid=paid,
                cadence="weekly",
                due_date=context.week_start.isoformat(),
                paid=paid == amount,
                funding=transfers,
            )
        )
    return next_state, tuple(entries), tuple(effects)


def _apply_routine_effects(
    state: AgentState,
    profile: RoutineProfile,
    low_social_streak: int,
) -> tuple[AgentState, tuple[RoutineEffectApplication, ...]]:
    next_state = state
    effects: list[RoutineEffectApplication] = []
    social_need = state.personality.social_need
    isolation_pressure = max(0.0, 0.35 - profile.social_contact) * social_need * min(
        1.0,
        low_social_streak / 6,
    )
    loneliness_pressure = state.mental.loneliness / 100.0 * social_need * min(
        1.0,
        low_social_streak / 8,
    )
    changes = (
        ("health.energy", profile.recovery_intensity * 8.0 - profile.energy_cost * 0.22 - profile.physical_activity * 3.0, "routine_energy"),
        ("health.sleep_debt", -(profile.recovery_intensity * 1.2) + max(0.0, profile.energy_cost - 25.0) / 80.0, "routine_sleep"),
        ("health.physical_health", profile.physical_activity * 1.1, "routine_activity"),
        ("health.mobility", profile.physical_activity * 0.8, "routine_activity"),
        ("mental.loneliness", (0.35 - profile.social_contact) * 7.0 * social_need - profile.social_contact * 4.5, "routine_social_contact"),
        ("needs.belonging", profile.social_contact * 4.0 - isolation_pressure * 8.0, "routine_belonging"),
        (
            "mental.mood",
            profile.comfort_value * 1.8
            + profile.recovery_intensity * 0.9
            + profile.social_contact * 1.6
            - isolation_pressure * 3.0
            - loneliness_pressure * 2.4,
            "routine_mood",
        ),
        (
            "mental.stress",
            -profile.recovery_intensity * 3.0
            + float(max(Decimal(0), profile.estimated_cost - Decimal(60)) / Decimal(30)),
            "routine_stress",
        ),
        ("mental.recovery_need", -profile.recovery_intensity * 5.0 + profile.energy_cost / 35.0, "routine_recovery"),
        ("mental.mental_load", profile.time_cost_hours / 15.0 - profile.recovery_intensity * 1.5, "routine_time"),
        ("social.city_familiarity", profile.physical_activity * 0.8 + profile.social_contact * 0.6, "routine_city"),
    )
    for path, delta, reason in changes:
        next_state, effect = _bounded_replace(
            next_state,
            path,
            float(delta),
            reason,
            source=f"routine:{profile.profile_id}",
        )
        effects.append(effect)
    return next_state, tuple(effects)


def _bounded_replace(
    state: AgentState,
    path: str,
    delta: float,
    reason: str,
    *,
    source: str = "",
) -> tuple[AgentState, RoutineEffectApplication]:
    section_name, field_name = path.split(".", 1)
    section = getattr(state, section_name)
    before = getattr(section, field_name)
    if path == "health.sleep_debt" or path == "needs.food_security":
        raw_after = before + delta
        after = max(0.0, raw_after)
    else:
        effective_delta = _boundary_sensitive_delta(before, delta)
        raw_after = before + effective_delta
        after = min(100.0, max(0.0, raw_after))
    clamped = after != raw_after
    next_state = replace(state, **{section_name: replace(section, **{field_name: after})})
    return next_state, RoutineEffectApplication(
        path=path,
        before=before,
        after=after,
        delta=after - before,
        clamped=clamped,
        reason=reason,
        source=source,
    )


def _boundary_sensitive_delta(before: float, delta: float) -> float:
    """Dampen ordinary recurring effects as bounded state approaches 0 or 100."""

    if delta > 0.0:
        distance = max(0.0, 100.0 - before)
    elif delta < 0.0:
        distance = max(0.0, before)
    else:
        return 0.0
    return delta * distance / (distance + 120.0)


def _apply_unpaid_pressure(
    state: AgentState,
    category: str,
    unpaid: Decimal,
    amount: Decimal,
    source: str,
) -> tuple[AgentState, tuple[RoutineEffectApplication, ...]]:
    ratio = float(unpaid / amount) if amount > Decimal(0) else 0.0
    next_state = state
    effects: list[RoutineEffectApplication] = []
    next_state, effect = _bounded_replace(
        next_state,
        "mental.stress",
        ratio * 2.0,
        "unpaid_obligation",
        source=source,
    )
    effects.append(effect)
    if category == "housing":
        next_state, effect = _bounded_replace(
            next_state,
            "needs.housing_security",
            -ratio * 2.5,
            "housing_arrear",
            source=source,
        )
        effects.append(effect)
    if category == "food":
        next_state, effect = _bounded_replace(
            next_state,
            "needs.food_security",
            -ratio * 2.5,
            "food_arrear",
            source=source,
        )
        effects.append(effect)
    return next_state, tuple(effects)


def _income_arrives(
    state: AgentState,
    context: WeeklyContext,
    name: str,
    reliability: float,
    due_date: date,
) -> tuple[bool, float | None]:
    if reliability >= 1.0:
        return True, None
    if reliability <= 0.0:
        return False, None
    rng = Random(
        derive_stable_seed(
            "passive-income",
            str(context.config.simulation.seed),
            state.identity.agent_id,
            name,
            due_date.isoformat(),
            str(context.week),
        )
    )
    roll = rng.random()
    return roll <= reliability, roll


def _due_date(cadence: str, due_day: int, week_start: date, week_end: date) -> date | None:
    if cadence == "weekly":
        return week_start
    cursor = date(week_start.year, week_start.month, 1)
    while cursor <= week_end:
        day = min(due_day, monthrange(cursor.year, cursor.month)[1])
        candidate = date(cursor.year, cursor.month, day)
        if week_start <= candidate <= week_end:
            return candidate
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return None


def _funding_transfers(transfers) -> tuple[FundingTransfer, ...]:
    return tuple(FundingTransfer(source=transfer.source, amount=transfer.amount) for transfer in transfers)
