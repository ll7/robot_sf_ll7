"""Tests for the canonical Dependabot risk-lane policy."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from scripts.dev.check_dependabot_update_policy import (
    PolicyError,
    changed_files,
    changed_lock_package_names,
    classify_package_names,
    load_policy,
    validate_ci_workflow,
    validate_dependabot_config,
    validate_direct_dependency_coverage,
    validate_direct_update_lanes,
    validate_repository_structure,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "scripts/validation/dependabot_update_policy.v1.json"
SCHEMA_PATH = REPO_ROOT / "scripts/validation/dependabot_update_policy.v1.schema.json"


def test_policy_matches_its_schema() -> None:
    """The executable policy must satisfy its tracked schema."""
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(policy)


def test_live_policy_covers_direct_dependencies_and_ci_surfaces() -> None:
    """The current repository must be covered before new updates are admitted."""
    policy = validate_repository_structure(repo_root=REPO_ROOT)
    validate_ci_workflow(policy)
    validate_dependabot_config()


def test_issue_7480_package_set_is_split_into_distinct_risk_classes() -> None:
    """The historical mixed update cannot pass as one direct-risk lane."""
    policy = load_policy()
    names = {
        "numba",
        "orjson",
        "wandb",
        "pyarrow",
        "ruff",
        "pre-commit",
        "pylint",
        "mypy",
    }
    classified = classify_package_names(names, names, policy)
    assert {item["class"] for item in classified} == {
        "developer-tooling",
        "experiment-integrations",
        "high-impact-runtime",
        "serialization-data",
    }
    with pytest.raises(PolicyError, match="mixes direct risk classes"):
        validate_direct_update_lanes(classified)


def test_developer_tooling_lane_can_remain_grouped() -> None:
    """A bounded tooling-only group remains one review and rollback surface."""
    policy = load_policy()
    names = {"ruff", "pre-commit", "pylint", "mypy", "pytest"}
    classified = classify_package_names(names, names, policy)
    assert validate_direct_update_lanes(classified) == ["developer-tooling"]


def test_unknown_direct_package_fails_closed() -> None:
    """A new direct package must receive an explicit policy classification."""
    policy = load_policy()
    with pytest.raises(PolicyError, match="not classified"):
        classify_package_names({"new-runtime-package"}, {"new-runtime-package"}, policy)


def test_unlisted_transitive_package_uses_conservative_fallback() -> None:
    """An unlisted lock-only row remains visible and keeps compatibility evidence."""
    policy = load_policy()
    classified = classify_package_names({"transitive-helper"}, set(), policy)
    assert classified == [
        {
            "name": "transitive-helper",
            "direct": False,
            "class": "transitive-lock-package",
            "risk": "unknown",
            "update_lane": "individual",
            "required_jobs": ["fast-feedback", "compat-matrix"],
        }
    ]


def test_lock_row_changes_preserve_marker_variants() -> None:
    """A changed version or marker-specific row is treated as dependency evidence."""
    base = """
version = 1

[[package]]
name = "numba"
version = "0.66.0"
marker = "python_version >= '3.11'"
"""
    head = base.replace('version = "0.66.0"', 'version = "0.67.0"')
    assert changed_lock_package_names(base, head) == {"numba"}


def test_authoritative_changed_file_list_avoids_unneeded_base_lookup(tmp_path: Path) -> None:
    """Non-dependency PRs can use the collector output when base fetch is unavailable."""
    changed_file_list = tmp_path / "changed-files.txt"
    changed_file_list.write_text("README.md\n\n docs/dev/example.md \n", encoding="utf-8")
    assert changed_files(tmp_path, "missing-base", changed_file_list) == [
        "README.md",
        "docs/dev/example.md",
    ]


def test_workflow_step_uses_the_policy_checker() -> None:
    """The existing PR contract workflow must execute the policy checker."""
    workflow = (REPO_ROOT / ".github/workflows/pr-contract-check.yml").read_text(encoding="utf-8")
    assert "Validate dependency update policy" in workflow
    assert "scripts/dev/check_dependabot_update_policy.py" in workflow


def test_direct_dependency_coverage_rejects_unreviewed_name() -> None:
    """Coverage cannot be silently weakened by adding an unreviewed direct name."""
    policy = load_policy()
    with pytest.raises(PolicyError, match="missing from the canonical policy"):
        validate_direct_dependency_coverage({"unreviewed-package"}, policy)
