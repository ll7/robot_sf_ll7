"""Tests for predictive same-seed row-summary validation."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import yaml

from robot_sf.benchmark.predictive_same_seed_row_summary import (
    load_predictive_same_seed_row_summary_input,
    validate_predictive_same_seed_row_summary_file,
    validate_predictive_same_seed_row_summary_rows,
)
from scripts.validation.validate_predictive_same_seed_row_summary import main as validate_cli_main

FIXTURE_ROOT = Path("tests/fixtures/predictive_same_seed_row_summary/v1")


def test_valid_rows_accept_ok_and_unavailable_records() -> None:
    """Executed and unavailable same-seed rows should both validate deterministically."""
    report = validate_predictive_same_seed_row_summary_file(FIXTURE_ROOT / "valid_rows.yaml")

    assert report["status"] == "valid"
    assert report["provenance_validation"] == "format_only"
    assert report["row_count"] == 2
    assert report["valid_row_count"] == 2
    assert report["rows"][0]["row_status"] == "ok"
    assert report["rows"][0]["row_key"] == (
        "schema_fixture_baseline:schema_fixture_scenario:102:schema_fixture_grid_key"
    )
    assert report["rows"][1]["row_status"] == "unavailable"
    assert report["rows"][1]["errors"] == []


def test_missing_required_field_reports_actionable_error() -> None:
    """Required-field failures should point at the specific missing field."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(FIXTURE_ROOT / "valid_rows.yaml")
    mutated_rows = copy.deepcopy(rows[:1])
    del mutated_rows[0]["artifact_pointer"]

    report = validate_predictive_same_seed_row_summary_rows(mutated_rows)

    assert report["status"] == "invalid"
    assert {"field": "artifact_pointer", "message": "is required"} in report["rows"][0]["errors"]


def test_invalid_status_boolean_and_numeric_values_are_rejected() -> None:
    """Enum, boolean, and numeric contract violations should all fail closed."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(FIXTURE_ROOT / "valid_rows.yaml")
    invalid_status = copy.deepcopy(rows[:1])
    invalid_status[0]["status"] = "complete"

    invalid_boolean = copy.deepcopy(rows[:1])
    invalid_boolean[0]["success"] = "yes"

    invalid_numeric = copy.deepcopy(rows[:1])
    invalid_numeric[0]["min_distance"] = -0.2

    cases = [
        (invalid_status, "status"),
        (invalid_boolean, "success"),
        (invalid_numeric, "min_distance"),
    ]
    for mutated_rows, expected_field in cases:
        report = validate_predictive_same_seed_row_summary_rows(mutated_rows)

        assert report["status"] == "invalid"
        error_fields = {error["field"] for error in report["rows"][0]["errors"]}
        assert expected_field in error_fields


def test_unavailable_and_unknown_rows_require_null_outcomes() -> None:
    """Unavailable and unknown rows must not invent outcome values."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(FIXTURE_ROOT / "valid_rows.yaml")
    unavailable_with_success = copy.deepcopy(rows[1:2])
    unavailable_with_success[0]["success"] = False

    unknown_row = copy.deepcopy(rows[1:2])
    unknown_row[0]["status"] = "unknown"

    unavailable_report = validate_predictive_same_seed_row_summary_rows(unavailable_with_success)
    unknown_report = validate_predictive_same_seed_row_summary_rows(unknown_row)

    assert unavailable_report["status"] == "invalid"
    assert {
        error["field"]
        for error in unavailable_report["rows"][0]["errors"]
    } == {"success"}
    assert unknown_report["status"] == "valid"


def test_duplicate_row_keys_are_rejected() -> None:
    """Contradictory rows for one variant/scenario/seed/grid key should fail closed."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(FIXTURE_ROOT / "valid_rows.yaml")
    duplicate_rows = copy.deepcopy(rows[:1])
    duplicate_rows.append(copy.deepcopy(rows[0]))
    duplicate_rows[1]["status"] = "failed"
    duplicate_rows[1]["success"] = False
    duplicate_rows[1]["min_distance"] = None

    report = validate_predictive_same_seed_row_summary_rows(duplicate_rows)

    assert report["status"] == "invalid"
    assert report["invalid_row_count"] == 1
    assert {
        error["field"]
        for error in report["rows"][1]["errors"]
    } == {"row_key"}


def test_cli_emits_json_for_invalid_input(tmp_path: Path, capsys) -> None:
    """The CLI should return a structured JSON report and the invalid-data exit code."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(FIXTURE_ROOT / "valid_rows.yaml")
    mutated_rows = copy.deepcopy(rows[:1])
    mutated_rows[0]["timeout"] = "later"
    input_path = tmp_path / "invalid_rows.yaml"
    input_path.write_text(yaml.safe_dump({"rows": mutated_rows}, sort_keys=False), encoding="utf-8")

    exit_code = validate_cli_main(["--input", str(input_path)])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid"
    assert payload["rows"][0]["errors"][0]["field"] == "timeout"


def test_git_history_validation_rejects_unknown_git_sha(tmp_path: Path) -> None:
    """Strict provenance mode must fail closed on git SHAs absent from repository history."""
    repo_root, _known_sha = _create_temp_git_repo(tmp_path)
    _input_format, rows = load_predictive_same_seed_row_summary_input(FIXTURE_ROOT / "valid_rows.yaml")
    mutated_rows = copy.deepcopy(rows[:1])
    mutated_rows[0]["artifact_pointer"] = "docs/context/proof.json"
    mutated_rows[0]["commit_artifact"] = "deadbeef, docs/context/proof.json"
    mutated_rows[0]["source_note"] = "docs/context/proof.json"
    mutated_rows[0]["scenario_matrix"] = "docs/context/proof.json"
    mutated_rows[0]["seed_manifest"] = "docs/context/proof.json"

    report = validate_predictive_same_seed_row_summary_rows(
        mutated_rows, repo_root=repo_root, check_git_history=True
    )

    assert report["status"] == "invalid"
    assert report["provenance_validation"] == "git_history"
    messages = {
        error["message"]
        for error in report["rows"][0]["errors"]
        if error["field"] == "commit_artifact"
    }
    assert "references unknown git commit SHA in repository history: 'deadbeef'" in messages


def _create_temp_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init")
    _git(repo_root, "config", "user.name", "Robot SF Tests")
    _git(repo_root, "config", "user.email", "robot-sf-tests@example.com")
    (repo_root / "docs" / "context").mkdir(parents=True)
    (repo_root / "docs" / "context" / "proof.json").write_text(
        '{"status": "ok"}\n', encoding="utf-8"
    )
    _git(repo_root, "add", "docs/context/proof.json")
    _git(repo_root, "commit", "-m", "test fixture")
    known_sha = _git(repo_root, "rev-parse", "--short=7", "HEAD")
    return repo_root, known_sha


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
