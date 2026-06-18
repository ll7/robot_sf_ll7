"""Tests for the ``odd_hazard_coverage.v1`` schema, loader, and report generator."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark import odd_hazard_coverage_matrix as matrix_module
from robot_sf.benchmark.odd_hazard_coverage_matrix import (
    ODD_HAZARD_COVERAGE_SCHEMA_VERSION,
    OddHazardCoverageValidationError,
    generate_json_report,
    generate_markdown_report,
    load_odd_hazard_coverage_matrix,
    load_odd_hazard_coverage_schema,
    validate_matrix_references,
)
from scripts.tools.generate_odd_hazard_coverage_matrix import _build_command

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "configs/benchmarks/odd_hazard_coverage.v1.yaml"


def test_load_fixture_as_typed_matrix() -> None:
    """The checked-in issue #2911 matrix should validate and expose typed rows."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)

    assert matrix.schema_version == ODD_HAZARD_COVERAGE_SCHEMA_VERSION
    assert matrix.id == "issue_2911_low_speed_public_space_v1"
    assert matrix.odd_contract_ref.contract_id == "low_speed_public_space_v1"
    assert matrix.hazard_traceability_ref is not None
    assert matrix.hazard_traceability_ref.contract_id == "low_speed_public_space_hazards_v1"
    assert len(matrix.coverage_rows) == 7
    assert len(matrix.known_gaps) == 8
    assert any(row.status == "weakly_covered" for row in matrix.coverage_rows)
    assert any(gap.status == "absent" for gap in matrix.known_gaps)
    assert any(gap.status == "blocked" for gap in matrix.known_gaps)

    jsonschema.validate(matrix.to_dict(), load_odd_hazard_coverage_schema())


def test_matrix_references_resolve_against_repo_root() -> None:
    """Referenced contracts and source configs should exist in the repository."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    errors = validate_matrix_references(matrix, repo_root=REPO_ROOT)

    assert errors == []


def test_hazard_classes_must_be_declared_in_traceability(tmp_path: Path) -> None:
    """Coverage rows that name an unknown hazard class should fail reference validation."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    payload = matrix.to_dict()
    payload["coverage_rows"][0]["hazard_class"] = "unknown_hazard"

    from robot_sf.benchmark.odd_hazard_coverage_matrix import (
        odd_hazard_coverage_matrix_from_dict,
    )

    modified = odd_hazard_coverage_matrix_from_dict(payload)
    errors = validate_matrix_references(modified, repo_root=REPO_ROOT)

    assert any("unknown_hazard" in error for error in errors)


def test_missing_source_config_is_reported(tmp_path: Path) -> None:
    """Reference validation should fail closed on a missing source config path."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    payload = matrix.to_dict()
    payload["coverage_rows"][0]["source_configs"].append("configs/missing_file.yaml")

    from robot_sf.benchmark.odd_hazard_coverage_matrix import (
        odd_hazard_coverage_matrix_from_dict,
    )

    modified = odd_hazard_coverage_matrix_from_dict(payload)
    errors = validate_matrix_references(modified, repo_root=REPO_ROOT)

    assert any("configs/missing_file.yaml" in error for error in errors)


def test_unknown_odd_condition_is_reported() -> None:
    """Reference validation should fail closed on an unknown ODD condition."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    payload = matrix.to_dict()
    payload["coverage_rows"][0]["odd_condition"] = "unknown_odd"

    from robot_sf.benchmark.odd_hazard_coverage_matrix import (
        odd_hazard_coverage_matrix_from_dict,
    )

    modified = odd_hazard_coverage_matrix_from_dict(payload)
    errors = validate_matrix_references(modified, repo_root=REPO_ROOT)

    assert any("unknown_odd" in error for error in errors)


def test_invalid_schema_version_is_rejected() -> None:
    """A mismatched schema version should fail JSON Schema validation."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    payload = matrix.to_dict()
    payload["schema_version"] = "odd_hazard_coverage.v2"

    with pytest.raises(OddHazardCoverageValidationError) as exc_info:
        from robot_sf.benchmark.odd_hazard_coverage_matrix import (
            odd_hazard_coverage_matrix_from_dict,
        )

        odd_hazard_coverage_matrix_from_dict(payload)

    assert "/schema_version" in str(exc_info.value)


def test_non_covered_row_requires_gap_reason() -> None:
    """A weakly_covered/blocked/absent row without a gap reason should fail semantic validation."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    payload = matrix.to_dict()
    payload["coverage_rows"][0]["gap_reason"] = ""

    with pytest.raises(OddHazardCoverageValidationError) as exc_info:
        from robot_sf.benchmark.odd_hazard_coverage_matrix import (
            odd_hazard_coverage_matrix_from_dict,
        )

        odd_hazard_coverage_matrix_from_dict(payload)

    assert "/coverage_rows/0/gap_reason" in str(exc_info.value)


def test_generated_json_report_is_machine_readable() -> None:
    """The generated JSON report should expose status counts and gap lists."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    report = generate_json_report(matrix, repo_root=REPO_ROOT, command="test-cmd")

    assert report["schema_version"] == ODD_HAZARD_COVERAGE_SCHEMA_VERSION
    assert report["summary"]["reference_valid"] is True
    assert report["summary"]["known_gap_count"] == 8
    assert "weakly_covered" in report["summary"]["status_counts"]
    assert report["generation"]["command"] == "test-cmd"
    assert report["generation"]["base_commit"] != "unknown"
    assert report["generation"]["tree_state"] in {"clean", "dirty", "unknown"}
    assert "commit" not in report["generation"]


def test_generated_json_report_marks_dirty_worktree_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON generation metadata should capture explicit source commit and tree state."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    monkeypatch.setattr(
        "robot_sf.benchmark.odd_hazard_coverage_matrix._git_commit",
        lambda *, repo_root: "basecommit",
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.odd_hazard_coverage_matrix._git_tree_state",
        lambda *, repo_root: "dirty",
    )

    report = generate_json_report(matrix, repo_root=REPO_ROOT, command="test-cmd")

    assert report["generation"]["base_commit"] == "basecommit"
    assert report["generation"]["tree_state"] == "dirty"
    assert "commit" not in report["generation"]


def test_git_tree_state_ignores_untracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tree-state provenance should not depend on unrelated untracked local files."""

    captured: dict[str, list[str]] = {}

    def fake_check_output(cmd: list[str], **_: object) -> str:
        captured["cmd"] = cmd
        return ""

    monkeypatch.setattr(matrix_module.subprocess, "check_output", fake_check_output)

    assert matrix_module._git_tree_state(repo_root=REPO_ROOT) == "clean"
    assert "--untracked-files=no" in captured["cmd"]


def test_build_command_is_shell_safe() -> None:
    """Generated command string should quote arguments with shell metacharacters safely."""

    args = argparse.Namespace(
        config=Path("/tmp/config dir/odd hazard coverage.yaml"),
        out_json=Path("/tmp/output dir/coverage matrix.json"),
        out_md=Path("/tmp/output dir/coverage matrix.md"),
        repo_root=Path("/tmp/repo root"),
    )

    command = _build_command(args)
    parts = shlex.split(command)

    assert parts == [
        "uv",
        "run",
        "python",
        "scripts/tools/generate_odd_hazard_coverage_matrix.py",
        "--config",
        "/tmp/config dir/odd hazard coverage.yaml",
        "--out-json",
        "/tmp/output dir/coverage matrix.json",
        "--out-md",
        "/tmp/output dir/coverage matrix.md",
        "--repo-root",
        "/tmp/repo root",
    ]


def test_generated_markdown_report_has_claim_boundary_first() -> None:
    """The Markdown report should put claim boundary and evidence status before interpretation."""

    matrix = load_odd_hazard_coverage_matrix(FIXTURE_PATH)
    report = generate_markdown_report(matrix, repo_root=REPO_ROOT, command="test-cmd")

    claim_pos = report.index("## Claim Boundary")
    summary_pos = report.index("## Evidence Status Summary")
    gaps_pos = report.index("## Known Gaps")
    guard_pos = report.index("## Benchmark Wording Guard")
    assert claim_pos < summary_pos < gaps_pos < guard_pos
    assert "weakly_covered" in report
    assert "blocked" in report
    assert "absent" in report
    assert "must not be described as covered" in report
    assert "Source/base commit" in report
    assert "Worktree state" in report
