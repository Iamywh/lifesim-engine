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
    OptionConsequenceDefinition,
    OutcomeDefinition,
    ScheduledEffect,
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
    "OptionConsequenceDefinition",
    "OutcomeDefinition",
    "ScheduledConsequenceTransition",
    "ScheduledEffect",
    "StateEffectDefinition",
    "load_consequence_catalog",
    "parse_consequence_catalog",
    "validate_consequence_catalog",
]
