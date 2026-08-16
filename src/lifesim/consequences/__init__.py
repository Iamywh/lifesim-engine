"""Deterministic consequence engine primitives."""

from lifesim.consequences.catalog import load_consequence_catalog, parse_consequence_catalog
from lifesim.consequences.engine import (
    ConsequenceApplicationError,
    ConsequenceEngine,
    DecisionConsequenceTransition,
    ScheduledConsequenceTransition,
)
from lifesim.consequences.model import (
    ConsequenceCatalog,
    ConsequenceHistory,
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
    validate_consequence_catalog,
)

__all__ = [
    "ConsequenceApplicationError",
    "ConsequenceCatalog",
    "ConsequenceEngine",
    "ConsequenceHistory",
    "ConsequenceRecord",
    "ConsequenceRuntimeState",
    "DecisionConsequenceTransition",
    "EffectApplication",
    "FinancialChargeApplication",
    "FinancialChargeDefinition",
    "OptionConsequenceDefinition",
    "OutcomeDefinition",
    "ScheduledConsequenceTransition",
    "ScheduledEffect",
    "ScheduledFinancialCharge",
    "StateEffectDefinition",
    "load_consequence_catalog",
    "parse_consequence_catalog",
    "validate_consequence_catalog",
]
