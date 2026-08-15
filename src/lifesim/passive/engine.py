from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from random import Random
from typing import Any

from lifesim.agents.state import (
    AgentState,
    Arrear,
    Debt,
    FinancialState,
    RoutineState,
)
from lifesim.decisions.engine import DecisionEngine
from lifesim.decisions.model import DecisionHistory
from lifesim.events.model import EventOccurrence, EventOption
from lifesim.passive.model import (
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

CENT = Decimal("0.01")
FUNDING_ORDER = ("bank_balance", "cash", "savings", "emergency_fund")


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
                        bank_balance=_money(financial.bank_balance + stream.amount),
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
        occurrence = _routine_occurrence(context, self._catalog)
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


def _routine_occurrence(context: WeeklyContext, catalog: RoutineCatalog) -> EventOccurrence:
    return EventOccurrence(
        event_id="weekly_routine",
        version="1",
        week=context.week,
        category="routine",
        effective_weight=1.0,
        title="Weekly routine",
        summary="Choose an ordinary weekly routine.",
        tags=("routine",),
        options=tuple(_profile_option(profile) for profile in catalog.profiles),
    )


def _profile_option(profile: RoutineProfile) -> EventOption:
    return EventOption(
        option_id=profile.profile_id,
        label=profile.label,
        summary=profile.summary,
        estimated_cost=profile.estimated_cost,
        time_cost_hours=profile.time_cost_hours,
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
        requires_full_estimated_cost=False,
    )


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
            weekly_interest = _money(debt.balance * debt.interest_rate / Decimal(52))
            updated = replace(updated, balance=_money(updated.balance + weekly_interest))
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
                balance=_money(updated.balance - paid_amount),
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
    unpaid = _money(amount - paid)
    arrear_balance = None
    effects: tuple[RoutineEffectApplication, ...] = ()
    if unpaid > Decimal(0):
        financial = _upsert_arrear(financial, obligation_id, category, unpaid, context.week)
        arrear = _find_arrear(financial, obligation_id)
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
) -> tuple[FinancialState, Decimal, tuple[FundingTransfer, ...]]:
    remaining = amount
    paid = Decimal("0.00")
    transfers: list[FundingTransfer] = []
    values = {source: getattr(financial, source) for source in FUNDING_ORDER}
    for source in FUNDING_ORDER:
        if remaining <= Decimal(0):
            break
        available = values[source]
        used = min(available, remaining)
        if used <= Decimal(0):
            continue
        values[source] = _money(available - used)
        remaining = _money(remaining - used)
        paid = _money(paid + used)
        transfers.append(FundingTransfer(source=source, amount=used))
    return (
        replace(financial, **values),
        paid,
        tuple(transfers),
    )


def _apply_routine_spending(
    state: AgentState,
    context: WeeklyContext,
    profile: RoutineProfile,
) -> tuple[AgentState, tuple[CashflowEntry, ...], tuple[RoutineEffectApplication, ...]]:
    next_state = state
    entries: list[CashflowEntry] = []
    effects: list[RoutineEffectApplication] = []
    spending = (
        ("routine_food", "basic food", "food", profile.food_budget),
        ("routine_transport", "routine transport", "transport", profile.transport_budget),
        ("routine_discretionary", "routine discretionary", "discretionary", profile.discretionary_budget),
    )
    for kind, name, category, amount in spending:
        financial, paid, transfers = _fund_payment(next_state.financial, amount)
        next_state = replace(next_state, financial=financial)
        if category == "food" and paid < amount:
            minimum_shortfall = max(Decimal("0.00"), _money(profile.minimum_food_budget - paid))
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
    raw_after = before + delta
    if path == "health.sleep_debt":
        after = max(0.0, raw_after)
    else:
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


def _upsert_arrear(
    financial: FinancialState,
    obligation_id: str,
    category: str,
    unpaid: Decimal,
    week: int,
) -> FinancialState:
    output: list[Arrear] = []
    found = False
    for arrear in financial.arrears:
        if arrear.obligation_id == obligation_id:
            output.append(
                replace(
                    arrear,
                    balance=_money(arrear.balance + unpaid),
                    last_updated_week=week,
                    missed_occurrences=arrear.missed_occurrences + 1,
                )
            )
            found = True
        else:
            output.append(arrear)
    if not found:
        output.append(
            Arrear(
                obligation_id=obligation_id,
                category=category,
                balance=unpaid,
                first_missed_week=week,
                last_updated_week=week,
                missed_occurrences=1,
            )
        )
    return replace(financial, arrears=tuple(output))


def _find_arrear(financial: FinancialState, obligation_id: str) -> Arrear | None:
    for arrear in financial.arrears:
        if arrear.obligation_id == obligation_id:
            return arrear
    return None


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


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)
