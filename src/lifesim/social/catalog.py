from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from lifesim.social.model import SocialCatalog, SocialContactDefinition


def load_social_catalog(path: str | Path) -> SocialCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_social_catalog(raw)


def parse_social_catalog(raw: dict[str, Any]) -> SocialCatalog:
    settings = raw.get("social_settings", {})
    if not isinstance(settings, dict):
        raise TypeError("Expected social_settings to be a table.")
    contacts = raw.get("contacts")
    if not isinstance(contacts, list):
        raise TypeError("Expected social catalog to contain a 'contacts' list.")
    return SocialCatalog(
        contacts=tuple(_parse_contact(contact) for contact in contacts),
        base_new_encounter_probability=settings.get("base_new_encounter_probability", 0.18),
        max_known_options=settings.get("max_known_options", 2),
        max_network_size=settings.get("max_network_size", 12),
        relisting_cooldown_weeks=settings.get("relisting_cooldown_weeks", 3),
    )


def _parse_contact(raw: dict[str, Any]) -> SocialContactDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each social contact to be a table.")
    return SocialContactDefinition(
        contact_id=_required(raw, "contact_id"),
        name=_required(raw, "name"),
        relationship=_required(raw, "relationship"),
        context=_required(raw, "context"),
        base_availability=raw.get("base_availability", 0.35),
        proximity=raw.get("proximity", 0.5),
        responsiveness=raw.get("responsiveness", 0.5),
        volatility=raw.get("volatility", 0.2),
        supportiveness=raw.get("supportiveness", 0.4),
        neglect_resistance=raw.get("neglect_resistance", 0.4),
        remote_contact=raw.get("remote_contact", False),
        initial_closeness=raw.get("initial_closeness", 18.0),
        initial_trust=raw.get("initial_trust", 18.0),
        encounter_weight=raw.get("encounter_weight", 1.0),
        tags=tuple(raw.get("tags", ())),
    )


def _required(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected social contact '{key}' to be a non-empty string.")
    return value
