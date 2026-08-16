from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from lifesim.development.model import (
    CurriculumSkill,
    DevelopmentCatalog,
    DevelopmentProfile,
    EducationProgramDefinition,
    PracticeAllocation,
    SkillDefinition,
)


def load_development_catalog(path: str | Path) -> DevelopmentCatalog:
    catalog_path = Path(path)
    with catalog_path.open("rb") as file:
        raw = tomllib.load(file)
    return parse_development_catalog(raw)


def parse_development_catalog(raw: dict[str, Any]) -> DevelopmentCatalog:
    skills = raw.get("skills")
    programs = raw.get("education_programs")
    profiles = raw.get("profiles")
    if not isinstance(skills, list):
        raise TypeError("Expected development catalog to contain a 'skills' list.")
    if not isinstance(programs, list):
        raise TypeError("Expected development catalog to contain an 'education_programs' list.")
    if not isinstance(profiles, list):
        raise TypeError("Expected development catalog to contain a 'profiles' list.")
    return DevelopmentCatalog(
        skills=tuple(_parse_skill(item) for item in skills),
        programs=tuple(_parse_program(item) for item in programs),
        profiles=tuple(_parse_profile(item) for item in profiles),
    )


def _parse_skill(raw: dict[str, Any]) -> SkillDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each skill definition to be a table.")
    return SkillDefinition(
        skill_name=_required(raw, "skill_name"),
        category=_required(raw, "category"),
        learning_rate=raw.get("learning_rate", 0.0),
        practice_xp_per_hour=raw.get("practice_xp_per_hour", 0.0),
        work_xp_per_hour=raw.get("work_xp_per_hour", 0.0),
    )


def _parse_program(raw: dict[str, Any]) -> EducationProgramDefinition:
    if not isinstance(raw, dict):
        raise TypeError("Expected each education program to be a table.")
    curriculum = raw.get("curriculum")
    if not isinstance(curriculum, list):
        raise TypeError("Expected education program curriculum to be a list.")
    return EducationProgramDefinition(
        program=_required(raw, "program"),
        progress_per_full_study_week=raw.get("progress_per_full_study_week", 0.0),
        curriculum=tuple(_parse_curriculum(item) for item in curriculum),
    )


def _parse_curriculum(raw: dict[str, Any]) -> CurriculumSkill:
    if not isinstance(raw, dict):
        raise TypeError("Expected curriculum mapping to be a table.")
    return CurriculumSkill(
        skill_name=_required(raw, "skill_name"),
        weight=raw.get("weight", 0.0),
    )


def _parse_profile(raw: dict[str, Any]) -> DevelopmentProfile:
    if not isinstance(raw, dict):
        raise TypeError("Expected each development profile to be a table.")
    practice = raw.get("practice", ())
    if not isinstance(practice, list | tuple):
        raise TypeError("Expected development profile practice to be a list.")
    return DevelopmentProfile(
        profile_id=_required(raw, "profile_id"),
        label=_required(raw, "label"),
        summary=_required(raw, "summary"),
        education_hours=raw.get("education_hours", 0.0),
        practice=tuple(_parse_practice(item) for item in practice),
        energy_cost=raw.get("energy_cost", 0.0),
        short_term_value=raw.get("short_term_value", 0.0),
        future_value=raw.get("future_value", 0.0),
        perceived_risk=raw.get("perceived_risk", 0.0),
        uncertainty=raw.get("uncertainty", 0.0),
        social_value=raw.get("social_value", 0.0),
        social_pressure=raw.get("social_pressure", 0.0),
        autonomy_value=raw.get("autonomy_value", 0.0),
        learning_value=raw.get("learning_value", 0.0),
        health_value=raw.get("health_value", 0.0),
        comfort_value=raw.get("comfort_value", 0.0),
        goal_tags=tuple(raw.get("goal_tags", ())),
    )


def _parse_practice(raw: dict[str, Any]) -> PracticeAllocation:
    if not isinstance(raw, dict):
        raise TypeError("Expected practice allocation to be a table.")
    return PracticeAllocation(
        skill_name=_required(raw, "skill_name"),
        hours=raw.get("hours", 0.0),
    )


def _required(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected development catalog '{key}' to be a non-empty string.")
    return value
