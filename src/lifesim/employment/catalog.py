from __future__ import annotations

import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lifesim.employment.model import EmploymentCatalog, JobDefinition, SkillRequirement


def load_employment_catalog(path: str | Path) -> EmploymentCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_employment_catalog(raw)


def parse_employment_catalog(raw: dict[str, Any]) -> EmploymentCatalog:
    settings = raw.get("employment_settings", {})
    if not isinstance(settings, dict):
        raise TypeError("Expected employment_settings to be a table.")
    jobs = raw.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("Expected employment catalog to contain a jobs list.")
    return EmploymentCatalog(
        jobs=tuple(_parse_job(item) for item in jobs),
        max_discoveries_per_week=settings.get("max_discoveries_per_week", 2),
        relisting_cooldown_weeks=settings.get("relisting_cooldown_weeks", 3),
    )


def _parse_job(raw: dict[str, Any]) -> JobDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each job to be a table.")
    return JobDefinition(
        job_id=raw["job_id"],
        version=raw["version"],
        role_title=raw["role_title"],
        employer=raw["employer"],
        sector=raw["sector"],
        tags=raw.get("tags", []),
        contract_type=raw["contract_type"],
        weekly_hours=raw["weekly_hours"],
        hourly_rate=_money(raw["hourly_rate"], "hourly_rate"),
        stability=raw["stability"],
        fixed_term_weeks=raw.get("fixed_term_weeks", 0),
        physical_demand=raw["physical_demand"],
        mental_demand=raw["mental_demand"],
        social_demand=raw["social_demand"],
        base_discovery_weight=raw["base_discovery_weight"],
        base_interview_probability=raw["base_interview_probability"],
        base_offer_probability=raw["base_offer_probability"],
        skill_requirements=tuple(
            SkillRequirement(
                skill_name=item["skill_name"],
                desired_level=item["desired_level"],
                weight=item["weight"],
            )
            for item in _raw_items(raw, "skill_requirements")
        ),
    )


def _raw_items(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = raw.get(key, [])
    if not isinstance(values, list):
        raise TypeError(f"Expected '{key}' to be a list.")
    if not all(isinstance(item, dict) for item in values):
        raise TypeError(f"Expected '{key}' to contain tables.")
    return values


def _money(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"Expected monetary value '{name}' to be a string.")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Expected monetary value '{name}' to be a decimal string.") from error
