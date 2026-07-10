"""Tests for the baseline planner readiness matrix."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
MATRIX_PATH = REPO_ROOT / "configs/benchmarks/planner_readiness_matrix_v1.yaml"
REQUIRED_FAMILIES = {
    "ppo",
    "social_force",
    "reciprocal_avoidance",
    "orca_dwa_style",
    "random_goal",
    "hybrid_planner_candidate",
}
REQUIRED_FIELDS = {
    "planner_id",
    "family",
    "tier",
    "requires_explicit_opt_in",
    "execution_mode",
    "readiness_status",
    "availability_status",
    "row_status",
    "counts_as_success_evidence",
    "artifact_dependency",
    "source_paths",
    "reason",
    "validation_command",
}
SUCCESS_STATUSES = {"successful_evidence"}


def _assert_tracked_source_file(path_str: str, *, label: str) -> None:
    """Require matrix source references to resolve to checked-in files."""
    path = Path(path_str)
    assert not path.is_absolute(), f"{label}: {path_str}"
    assert ".." not in path.parts, f"{label}: {path_str}"
    assert not path.parts or path.parts[0] != "output", f"{label}: {path_str}"
    assert (REPO_ROOT / path).is_file(), f"{label}: {path_str}"


def _assert_validation_command(command: str, *, label: str) -> None:
    """Require matrix validation commands to point at checked-in pytest targets."""
    parts = shlex.split(command)
    assert parts[:3] == ["uv", "run", "pytest"], f"{label}: {command}"
    assert len(parts) >= 4, f"{label}: {command}"
    for target in parts[3:]:
        if target.startswith("-"):
            continue
        target_path = Path(target.split("::", maxsplit=1)[0])
        assert not target_path.is_absolute(), f"{label}: {command}"
        assert ".." not in target_path.parts, f"{label}: {command}"
        assert (REPO_ROOT / target_path).exists(), f"{label}: {command}"


def _load_matrix() -> dict[str, Any]:
    """Load the checked-in readiness matrix."""
    payload = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_planner_readiness_matrix_covers_issue_3060_families() -> None:
    """The matrix should cover the planner families named in issue #3060."""
    matrix = _load_matrix()
    rows = matrix["rows"]
    families = {row["family"] for row in rows}
    planner_ids = {row["planner_id"] for row in rows}

    assert matrix["schema_version"] == "planner-readiness-matrix.v1"
    assert matrix["issue"] == 3060
    assert REQUIRED_FAMILIES <= families
    assert {
        "goal",
        "random",
        "social_force",
        "orca",
        "hrvo",
        "ppo",
        "risk_dwa",
        "dwa",
    } <= planner_ids
    assert {"hybrid_rule_local_planner", "hybrid_portfolio"} <= planner_ids


def test_planner_readiness_matrix_rows_use_fail_closed_status_axes() -> None:
    """Rows should use explicit status axes and never count caveats as success evidence."""
    matrix = _load_matrix()
    vocab = matrix["status_vocabulary"]

    for row in matrix["rows"]:
        assert REQUIRED_FIELDS <= set(row)
        assert row["execution_mode"] in vocab["execution_mode"]
        assert row["readiness_status"] in vocab["readiness_status"]
        assert row["availability_status"] in vocab["availability_status"]
        assert row["row_status"] in vocab["row_status"]
        if row["counts_as_success_evidence"]:
            assert row["row_status"] in SUCCESS_STATUSES
            assert row["availability_status"] == "available"
            assert row["readiness_status"] in {"native", "adapter"}
        else:
            assert row["row_status"] not in SUCCESS_STATUSES


def test_planner_readiness_matrix_source_paths_exist_and_avoid_output_dependencies() -> None:
    """Matrix evidence should point at tracked sources, not disposable output artifacts."""
    matrix = _load_matrix()

    for path in matrix["policy_sources"]:
        _assert_tracked_source_file(path, label="policy_sources")

    for row in matrix["rows"]:
        for path in row["source_paths"]:
            _assert_tracked_source_file(path, label=row["planner_id"])


def test_planner_readiness_matrix_source_path_guard_rejects_traversal() -> None:
    """Source references should not escape the repository-relative path contract."""
    with pytest.raises(AssertionError, match=r"\.\./pyproject.toml"):
        _assert_tracked_source_file("../pyproject.toml", label="escape")

    with pytest.raises(AssertionError, match=r"configs/\.\./pyproject.toml"):
        _assert_tracked_source_file("configs/../pyproject.toml", label="escape")


def test_planner_readiness_matrix_validation_commands_reference_existing_pytest_targets() -> None:
    """Validation commands should stay attached to checked-in pytest targets."""
    matrix = _load_matrix()

    for row in matrix["rows"]:
        _assert_validation_command(row["validation_command"], label=row["planner_id"])
