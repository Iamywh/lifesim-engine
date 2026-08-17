from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from random import Random
from typing import Any

from lifesim.agents.state import AgentState
from lifesim.consequences.model import (
    BOUNDED_FLOAT_PATHS,
    MONEY_PATHS,
    NONNEGATIVE_FLOAT_PATHS,
    ConsequenceCatalog,
    ConsequenceRecord,
    ConsequenceRuntimeState,
    EffectApplication,
    FinancialChargeApplication,
    FinancialChargeDefinition,
    OptionConsequenceDefinition,
    OutcomeDefinition,
    ScheduledEffect,
    ScheduledFinancialCharge,
    StateEffectDefinition,
)
from lifesim.decisions.model import DecisionRecord
from lifesim.events.model import EventOccurrence
from lifesim.finance import find_arrear, settle_liquid_amount, upsert_arrear
from lifesim.rng import derive_stable_seed
from lifesim.weekly import WeeklyContext, WeeklyTransitionResult


class ConsequenceApplicationError(RuntimeError):
    """Raised when a consequence cannot be applied atomically."""


class ConsequenceEngine:
    """Applies chosen-option consequences without owning simulation state.

    Same-week decisions are resolved in the order supplied by the weekly context.
    Scheduled effects are resolved by earliest due week, preserving existing
    runtime tuple order for effects with the same due week.
    """

    def __init__(self, catalog: ConsequenceCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> ConsequenceCatalog:
        return self._catalog

    def apply_due_scheduled_effects(
        self,
        state: AgentState,
        context: WeeklyContext,
        runtime: ConsequenceRuntimeState,
    ) -> tuple[AgentState, ConsequenceRuntimeState, tuple[ConsequenceRecord, ...]]:
        due = tuple(
            effect
            for effect in runtime.pending_scheduled_effects
            if effect.due_week <= context.week
        )
        due_charges = tuple(
            charge
            for charge in runtime.pending_scheduled_financial_charges
            if charge.due_week <= context.week
        )
        remaining = tuple(
            effect
            for effect in runtime.pending_scheduled_effects
            if effect.due_week > context.week
        )
        remaining_charges = tuple(
            charge
            for charge in runtime.pending_scheduled_financial_charges
            if charge.due_week > context.week
        )
        next_state = state
        records: list[ConsequenceRecord] = []
        for scheduled in sorted(due, key=lambda item: item.due_week):
            next_state, application = _apply_effect(next_state, context, scheduled.effect, scheduled)
            record = ConsequenceRecord(
                consequence_id=_stable_id(
                    "scheduled",
                    scheduled.scheduled_effect_id,
                    str(context.week),
                ),
                source_decision_id=scheduled.source_decision_id,
                source_event_id=scheduled.source_event_id,
                source_event_version=scheduled.source_event_version,
                chosen_option_id=scheduled.chosen_option_id,
                week_resolved=context.week,
                selected_outcome_id=scheduled.source_outcome_id,
                source_scheduled_effect_id=scheduled.scheduled_effect_id,
                effect_applications=(application,),
            )
            records.append(record)
        for scheduled_charge in sorted(due_charges, key=lambda item: item.due_week):
            next_state, application = _apply_financial_charge(
                next_state,
                context,
                scheduled_charge.charge,
                scheduled_charge,
                0,
            )
            record = ConsequenceRecord(
                consequence_id=_stable_id(
                    "scheduled-charge",
                    scheduled_charge.scheduled_charge_id,
                    str(context.week),
                ),
                source_decision_id=scheduled_charge.source_decision_id,
                source_event_id=scheduled_charge.source_event_id,
                source_event_version=scheduled_charge.source_event_version,
                chosen_option_id=scheduled_charge.chosen_option_id,
                week_resolved=context.week,
                selected_outcome_id=scheduled_charge.source_outcome_id,
                source_scheduled_charge_id=scheduled_charge.scheduled_charge_id,
                financial_charge_applications=(application,),
            )
            records.append(record)
        history = runtime.history.record(tuple(records))
        next_runtime = ConsequenceRuntimeState(
            history=history,
            pending_scheduled_effects=remaining,
            pending_scheduled_financial_charges=remaining_charges,
            processed_decision_ids=runtime.processed_decision_ids,
        )
        return next_state, next_runtime, tuple(records)

    def resolve_decisions(
        self,
        state: AgentState,
        context: WeeklyContext,
        events: tuple[EventOccurrence, ...],
        decisions: tuple[DecisionRecord, ...],
        runtime: ConsequenceRuntimeState,
    ) -> tuple[AgentState, ConsequenceRuntimeState, tuple[ConsequenceRecord, ...]]:
        next_state = state
        records: list[ConsequenceRecord] = []
        pending = list(runtime.pending_scheduled_effects)
        pending_charges = list(runtime.pending_scheduled_financial_charges)
        processed = list(runtime.processed_decision_ids)

        events_by_key = _events_by_key(events, context.week)
        for decision in decisions:
            if decision.agent_id != state.identity.agent_id:
                raise ValueError(
                    f"Decision '{decision.decision_id}' does not belong to agent "
                    f"'{state.identity.agent_id}'."
                )
            if decision.week != context.week:
                raise ValueError("Expected decision week to match WeeklyContext.week.")
            if decision.decision_id in processed:
                raise ValueError(f"Decision '{decision.decision_id}' has already been processed.")
            processed.append(decision.decision_id)
            if decision.chosen_option_id is None:
                continue

            event = events_by_key.get((decision.source_event_id, decision.source_event_version))
            if event is None:
                raise ValueError(
                    f"Decision '{decision.decision_id}' references an event not present this week."
                )
            option_ids = {option.option_id for option in event.options}
            if decision.chosen_option_id not in option_ids:
                raise ValueError(
                    f"Decision '{decision.decision_id}' chose an option not present on its event."
                )
            definition = self._catalog.find(
                event_id=decision.source_event_id,
                event_version=decision.source_event_version,
                option_id=decision.chosen_option_id,
            )
            if definition is None:
                continue

            next_state, record = self._resolve_definition(
                next_state,
                context,
                decision,
                definition,
            )
            records.append(record)
            pending.extend(record.scheduled_effects_created)
            pending_charges.extend(record.scheduled_financial_charges_created)

        history = runtime.history.record(tuple(records))
        next_runtime = ConsequenceRuntimeState(
            history=history,
            pending_scheduled_effects=tuple(pending),
            pending_scheduled_financial_charges=tuple(pending_charges),
            processed_decision_ids=tuple(processed),
        )
        return next_state, next_runtime, tuple(records)

    def _resolve_definition(
        self,
        state: AgentState,
        context: WeeklyContext,
        decision: DecisionRecord,
        definition: OptionConsequenceDefinition,
    ) -> tuple[AgentState, ConsequenceRecord]:
        outcome, roll, total_weight = self._select_outcome(state, context, decision, definition)
        outcome_effects = outcome.effects if outcome is not None else ()
        outcome_charges = outcome.financial_charges if outcome is not None else ()
        effects = definition.effects + outcome_effects
        financial_charges = definition.financial_charges + outcome_charges
        consequence_id = _stable_id(
            "consequence",
            decision.decision_id,
            definition.event_id,
            definition.event_version,
            definition.option_id,
        )
        immediate = tuple(effect for effect in effects if effect.delay_weeks == 0)
        delayed = tuple(effect for effect in effects if effect.delay_weeks > 0)
        immediate_charges = tuple(charge for charge in financial_charges if charge.delay_weeks == 0)
        delayed_charges = tuple(charge for charge in financial_charges if charge.delay_weeks > 0)
        next_state, charge_applications = _apply_financial_charges_atomically(
            state,
            context,
            immediate_charges,
            consequence_id,
        )
        next_state, applications = _apply_effects_atomically(next_state, context, immediate)
        scheduled = tuple(
            ScheduledEffect(
                scheduled_effect_id=_stable_id(
                    "scheduled",
                    consequence_id,
                    str(index),
                    str(context.week + effect.delay_weeks),
                    effect.path,
                ),
                source_decision_id=decision.decision_id,
                source_consequence_id=consequence_id,
                source_event_id=definition.event_id,
                source_event_version=definition.event_version,
                chosen_option_id=definition.option_id,
                source_outcome_id=outcome.outcome_id if outcome is not None else None,
                created_week=context.week,
                due_week=context.week + effect.delay_weeks,
                effect=effect,
            )
            for index, effect in enumerate(delayed)
        )
        scheduled_charges = tuple(
            ScheduledFinancialCharge(
                scheduled_charge_id=_stable_id(
                    "scheduled-charge",
                    consequence_id,
                    str(index),
                    str(context.week + charge.delay_weeks),
                    charge.category,
                ),
                source_decision_id=decision.decision_id,
                source_consequence_id=consequence_id,
                source_event_id=definition.event_id,
                source_event_version=definition.event_version,
                chosen_option_id=definition.option_id,
                source_outcome_id=outcome.outcome_id if outcome is not None else None,
                created_week=context.week,
                due_week=context.week + charge.delay_weeks,
                charge=charge,
            )
            for index, charge in enumerate(delayed_charges)
        )
        return next_state, ConsequenceRecord(
            consequence_id=consequence_id,
            source_decision_id=decision.decision_id,
            source_event_id=definition.event_id,
            source_event_version=definition.event_version,
            chosen_option_id=definition.option_id,
            week_resolved=context.week,
            selected_outcome_id=outcome.outcome_id if outcome is not None else None,
            outcome_roll=roll,
            outcome_total_weight=total_weight,
            effect_applications=applications,
            financial_charge_applications=charge_applications,
            scheduled_effects_created=scheduled,
            scheduled_financial_charges_created=scheduled_charges,
        )

    def _select_outcome(
        self,
        state: AgentState,
        context: WeeklyContext,
        decision: DecisionRecord,
        definition: OptionConsequenceDefinition,
    ) -> tuple[OutcomeDefinition | None, float | None, float | None]:
        if not definition.outcomes:
            return None, None, None
        total_weight = sum(outcome.weight for outcome in definition.outcomes)
        rng = Random(
            derive_stable_seed(
                "consequence-outcome",
                str(context.config.simulation.seed),
                state.identity.agent_id,
                str(context.week),
                decision.decision_id,
                definition.event_id,
                definition.event_version,
                definition.option_id,
            )
        )
        roll = rng.random() * total_weight
        cumulative = 0.0
        for outcome in definition.outcomes:
            cumulative += outcome.weight
            if roll <= cumulative:
                return outcome, roll, total_weight
        return definition.outcomes[-1], roll, total_weight


def _events_by_key(
    events: tuple[EventOccurrence, ...],
    context_week: int,
) -> dict[tuple[str, str], EventOccurrence]:
    events_by_key: dict[tuple[str, str], EventOccurrence] = {}
    for event in events:
        if event.week != context_week:
            raise ValueError("Expected event occurrence week to match WeeklyContext.week.")
        key = (event.event_id, event.version)
        if key in events_by_key:
            event_id, version = key
            raise ValueError(
                "Ambiguous duplicate same-week event occurrence key "
                f"'{event_id}/{version}'."
            )
        events_by_key[key] = event
    return events_by_key


@dataclass(frozen=True, slots=True)
class ScheduledConsequenceTransition:
    engine: ConsequenceEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, next_runtime, records = self.engine.apply_due_scheduled_effects(
            state,
            context,
            runtime,
        )
        return WeeklyTransitionResult(
            agent_state=next_state,
            consequences=records,
            consequence_runtime=next_runtime,
        )


@dataclass(frozen=True, slots=True)
class DecisionConsequenceTransition:
    engine: ConsequenceEngine

    def apply(self, state: AgentState, context: WeeklyContext) -> WeeklyTransitionResult:
        runtime = _runtime(context)
        next_state, next_runtime, records = self.engine.resolve_decisions(
            state,
            context,
            context.events,
            context.decisions,
            runtime,
        )
        return WeeklyTransitionResult(
            agent_state=next_state,
            consequences=records,
            consequence_runtime=next_runtime,
        )


def _runtime(context: WeeklyContext) -> ConsequenceRuntimeState:
    runtime = context.consequence_runtime
    if runtime is None:
        return ConsequenceRuntimeState()
    if not isinstance(runtime, ConsequenceRuntimeState):
        raise TypeError("Expected WeeklyContext.consequence_runtime to contain ConsequenceRuntimeState.")
    return runtime


def _apply_effects_atomically(
    state: AgentState,
    context: WeeklyContext,
    effects: tuple[StateEffectDefinition, ...],
) -> tuple[AgentState, tuple[EffectApplication, ...]]:
    candidate = state
    applications: list[EffectApplication] = []
    for effect in effects:
        candidate, application = _apply_effect(candidate, context, effect, None)
        applications.append(application)
    return candidate, tuple(applications)


def _apply_financial_charges_atomically(
    state: AgentState,
    context: WeeklyContext,
    charges: tuple[FinancialChargeDefinition, ...],
    source_consequence_id: str,
) -> tuple[AgentState, tuple[FinancialChargeApplication, ...]]:
    candidate = state
    applications: list[FinancialChargeApplication] = []
    for index, charge in enumerate(charges):
        candidate, application = _apply_financial_charge(
            candidate,
            context,
            charge,
            None,
            index,
            source_consequence_id,
        )
        applications.append(application)
    return candidate, tuple(applications)


def _apply_financial_charge(
    state: AgentState,
    context: WeeklyContext,
    charge: FinancialChargeDefinition,
    scheduled: ScheduledFinancialCharge | None,
    index: int,
    source_consequence_id: str | None = None,
) -> tuple[AgentState, FinancialChargeApplication]:
    for condition in charge.conditions:
        if not condition.evaluate(state, context):
            return state, FinancialChargeApplication(
                amount_due=charge.amount,
                amount_paid=Decimal("0.00"),
                amount_unpaid=Decimal("0.00"),
                category=charge.category,
                shortfall_policy=charge.shortfall_policy,
                funding_order=charge.funding_order,
                skipped=True,
                skip_reason="condition_not_met",
                scheduled_charge_id=(
                    scheduled.scheduled_charge_id if scheduled is not None else None
                ),
            )

    settlement = settle_liquid_amount(state.financial, charge.amount, charge.funding_order)
    if charge.shortfall_policy == "require_full" and not settlement.fully_paid:
        raise ConsequenceApplicationError(
            "Financial charge requires full payment but available funds were insufficient."
        )
    financial = settlement.financial
    arrear_created = False
    arrear_obligation_id = ""
    arrear_balance_after = None
    if settlement.amount_unpaid > Decimal("0.00"):
        arrear_created = True
        source_id = (
            scheduled.scheduled_charge_id
            if scheduled is not None
            else _stable_id("charge", source_consequence_id or "unknown", str(index))
        )
        arrear_obligation_id = f"charge:{source_id}"
        financial = upsert_arrear(
            financial,
            obligation_id=arrear_obligation_id,
            category=charge.category,
            unpaid=settlement.amount_unpaid,
            week=context.week,
        )
        arrear = find_arrear(financial, arrear_obligation_id)
        arrear_balance_after = arrear.balance if arrear is not None else settlement.amount_unpaid
    next_state = replace(state, financial=financial)
    return next_state, FinancialChargeApplication(
        amount_due=charge.amount,
        amount_paid=settlement.amount_paid,
        amount_unpaid=settlement.amount_unpaid,
        category=charge.category,
        shortfall_policy=charge.shortfall_policy,
        funding_order=charge.funding_order,
        funding=settlement.transfers,
        fully_paid=settlement.fully_paid,
        arrear_created=arrear_created,
        arrear_obligation_id=arrear_obligation_id,
        arrear_balance_after=arrear_balance_after,
        scheduled_charge_id=scheduled.scheduled_charge_id if scheduled is not None else None,
    )


def _apply_effect(
    state: AgentState,
    context: WeeklyContext,
    effect: StateEffectDefinition,
    scheduled: ScheduledEffect | None,
) -> tuple[AgentState, EffectApplication]:
    for condition in effect.conditions:
        if not condition.evaluate(state, context):
            return state, EffectApplication(
                path=effect.path,
                requested_delta=effect.delta,
                before=None,
                after=None,
                clamped=False,
                skipped=True,
                skip_reason="condition_not_met",
                scheduled_effect_id=scheduled.scheduled_effect_id if scheduled is not None else None,
            )

    before = _get_path(state, effect.path)
    after, clamped = _add_delta(effect.path, before, effect.delta)
    return _replace_path(state, effect.path, after), EffectApplication(
        path=effect.path,
        requested_delta=effect.delta,
        before=before,
        after=after,
        clamped=clamped,
        scheduled_effect_id=scheduled.scheduled_effect_id if scheduled is not None else None,
    )


def _add_delta(path: str, before: Any, delta: Decimal | float) -> tuple[Decimal | float, bool]:
    if path in MONEY_PATHS:
        if not isinstance(before, Decimal) or not isinstance(delta, Decimal):
            raise ConsequenceApplicationError(f"Expected Decimal values for monetary path '{path}'.")
        after = before + delta
        if after < Decimal(0):
            raise ConsequenceApplicationError(
                f"Consequence effect would make monetary path '{path}' negative."
            )
        return after, False

    if not isinstance(before, int | float) or isinstance(before, bool):
        raise ConsequenceApplicationError(f"Expected numeric value at path '{path}'.")
    after_float = float(before) + float(delta)
    if path in BOUNDED_FLOAT_PATHS:
        clamped = after_float < 0.0 or after_float > 100.0
        return min(100.0, max(0.0, after_float)), clamped
    if path in NONNEGATIVE_FLOAT_PATHS:
        if after_float < 0.0:
            raise ConsequenceApplicationError(
                f"Consequence effect would make non-negative path '{path}' negative."
            )
        return after_float, False
    raise ConsequenceApplicationError(f"Unsupported consequence write path '{path}'.")


def _get_path(state: AgentState, path: str) -> Any:
    section_name, field_name = path.split(".", 1)
    return getattr(getattr(state, section_name), field_name)


def _replace_path(state: AgentState, path: str, value: Decimal | float) -> AgentState:
    section_name, field_name = path.split(".", 1)
    section = getattr(state, section_name)
    return replace(state, **{section_name: replace(section, **{field_name: value})})


def _stable_id(*parts: str) -> str:
    return f"consequence_{derive_stable_seed(*parts):016x}"
