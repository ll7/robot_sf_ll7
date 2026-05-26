"""Regression tests for the repo-local skill checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_check_skills_module():
    """Load scripts/dev/check_skills.py without requiring scripts to be a package."""
    module_path = Path(__file__).parents[2] / "scripts/dev/check_skills.py"
    spec = importlib.util.spec_from_file_location("check_skills", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_yaml_fails_closed_for_empty_yaml(tmp_path: Path) -> None:
    """Empty YAML files should fail with a path-qualified checker error."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="YAML file is empty"):
        check_skills._read_yaml(yaml_path)


def test_read_yaml_fails_closed_for_non_mapping_yaml(tmp_path: Path) -> None:
    """Registry and schema YAML files should be top-level mappings."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    yaml_path = tmp_path / "scalar.yaml"
    yaml_path.write_text("not-a-map\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="YAML top level must be a mapping"):
        check_skills._read_yaml(yaml_path)


def test_registry_shape_reports_non_mapping_skill_metadata() -> None:
    """Malformed skill entries should produce checker errors instead of AttributeError."""
    check_skills = _load_check_skills_module()
    schema = {
        "allowed_categories": ["general"],
        "allowed_kinds": ["atomic"],
        "allowed_phases": ["context"],
        "allowed_write_scopes": ["filesystem"],
    }

    errors = check_skills._validate_registry_shape(
        {"version": 1, "skills": {"bad-skill": "oops"}},
        schema,
    )

    assert errors == ["bad-skill: metadata must be a dictionary"]


def test_frontmatter_fails_closed_for_non_mapping_yaml(tmp_path: Path) -> None:
    """Skill frontmatter must be a YAML mapping before downstream metadata checks run."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("---\n- nope\n---\n\nBody\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="frontmatter must be a YAML mapping"):
        check_skills._frontmatter(skill_path)
