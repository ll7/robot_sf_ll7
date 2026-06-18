"""Tests for the baseline planner readiness matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    assert {"goal", "random", "social_force", "orca", "ppo", "risk_dwa"} <= planner_ids
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
        assert not path.startswith("output/")
        assert (REPO_ROOT / path).exists(), path

    for row in matrix["rows"]:
        for path in row["source_paths"]:
            assert not path.startswith("output/")
            assert (REPO_ROOT / path).exists(), f"{row['planner_id']}: {path}"
