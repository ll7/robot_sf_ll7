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
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
    PREDICTIVE_LEGACY_FEATURE_SCHEMA,
    predictive_feature_schema_metadata,
)
from scripts.validation.validate_predictive_same_seed_row_summary import main as validate_cli_main

FIXTURE_ROOT = Path("tests/fixtures/predictive_same_seed_row_summary/v1")


def _predictive_feature_schema(
    *,
    model_family: str,
    producer_key: str | None = None,
) -> dict[str, object]:
    """Return valid predictive feature schema metadata for row-summary validation tests."""
    kwargs: dict[str, object] = {"model_family": model_family}
    if model_family == PREDICTIVE_EGO_FEATURE_SCHEMA:
        kwargs["ego_motion_channel_producer"] = producer_key
    return predictive_feature_schema_metadata(**kwargs)


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
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    mutated_rows = copy.deepcopy(rows[:1])
    del mutated_rows[0]["artifact_pointer"]

    report = validate_predictive_same_seed_row_summary_rows(mutated_rows)

    assert report["status"] == "invalid"
    assert {"field": "artifact_pointer", "message": "is required"} in report["rows"][0]["errors"]


def test_invalid_status_boolean_and_numeric_values_are_rejected() -> None:
    """Enum, boolean, and numeric contract violations should all fail closed."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
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
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    unavailable_with_success = copy.deepcopy(rows[1:2])
    unavailable_with_success[0]["success"] = False

    unknown_row = copy.deepcopy(rows[1:2])
    unknown_row[0]["status"] = "unknown"

    unavailable_report = validate_predictive_same_seed_row_summary_rows(unavailable_with_success)
    unknown_report = validate_predictive_same_seed_row_summary_rows(unknown_row)

    assert unavailable_report["status"] == "invalid"
    assert {error["field"] for error in unavailable_report["rows"][0]["errors"]} == {"success"}
    assert unknown_report["status"] == "valid"


def test_duplicate_row_keys_are_rejected() -> None:
    """Contradictory rows for one variant/scenario/seed/grid key should fail closed."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    duplicate_rows = copy.deepcopy(rows[:1])
    duplicate_rows.append(copy.deepcopy(rows[0]))
    duplicate_rows[1]["status"] = "failed"
    duplicate_rows[1]["success"] = False
    duplicate_rows[1]["min_distance"] = None

    report = validate_predictive_same_seed_row_summary_rows(duplicate_rows)

    assert report["status"] == "invalid"
    assert report["invalid_row_count"] == 1
    assert {error["field"] for error in report["rows"][1]["errors"]} == {"row_key"}


def test_loader_accepts_supported_shapes_and_rejects_bad_payloads(tmp_path: Path) -> None:
    """The file loader should make accepted YAML shapes explicit."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    single_row_path = tmp_path / "single_row.yaml"
    single_row_path.write_text(yaml.safe_dump(rows[0], sort_keys=False), encoding="utf-8")
    row_list_path = tmp_path / "row_list.yaml"
    row_list_path.write_text(yaml.safe_dump(rows[:1], sort_keys=False), encoding="utf-8")
    bad_rows_path = tmp_path / "bad_rows.yaml"
    bad_rows_path.write_text(yaml.safe_dump({"rows": "not-a-list"}), encoding="utf-8")
    scalar_path = tmp_path / "scalar.yaml"
    scalar_path.write_text("true\n", encoding="utf-8")

    assert load_predictive_same_seed_row_summary_input(single_row_path)[0] == "row"
    assert load_predictive_same_seed_row_summary_input(row_list_path)[0] == "rows"
    for path in (bad_rows_path, scalar_path, tmp_path / "missing.yaml"):
        try:
            load_predictive_same_seed_row_summary_input(path)
        except ValueError:
            continue
        raise AssertionError(f"expected loader failure for {path}")


def test_semantic_contradictions_and_reference_guards_are_reported() -> None:
    """Safety semantics and durable-reference boundaries should fail closed."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    row = copy.deepcopy(rows[0])
    row["success"] = True
    row["collision_event"] = True
    row["low_progress"] = True
    row["timeout"] = True
    row["artifact_pointer"] = "docs/context/a.yaml, docs/context/b.yaml"
    row["commit_artifact"] = "docs/context/issue_1550_predictive_same_seed_row_summary_schema.md"
    row["source_issue"] = "1550"
    row["unexpected"] = "value"

    report = validate_predictive_same_seed_row_summary_rows([row])

    error_fields = {error["field"] for error in report["rows"][0]["errors"]}
    assert report["status"] == "invalid"
    assert {
        "artifact_pointer",
        "collision_event",
        "commit_artifact",
        "low_progress",
        "source_issue",
        "timeout",
        "unexpected",
    } <= error_fields


def test_status_specific_nullability_and_degraded_warning() -> None:
    """Status contracts should distinguish missing, failed, unknown, and degraded rows."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    ok_missing_min_distance = copy.deepcopy(rows[0])
    ok_missing_min_distance["min_distance"] = None
    failed_success = copy.deepcopy(rows[0])
    failed_success["variant"] = "schema_fixture_failed"
    failed_success["status"] = "failed"
    failed_success["success"] = True
    unknown_with_min_distance = copy.deepcopy(rows[1])
    unknown_with_min_distance["status"] = "unknown"
    unknown_with_min_distance["min_distance"] = 1.0
    degraded_empty = copy.deepcopy(rows[1])
    degraded_empty["variant"] = "schema_fixture_degraded"
    degraded_empty["status"] = "degraded"

    report = validate_predictive_same_seed_row_summary_rows(
        [ok_missing_min_distance, failed_success, unknown_with_min_distance, degraded_empty]
    )

    assert report["status"] == "invalid"
    assert any(
        error["field"] == "min_distance" and "status is 'ok'" in error["message"]
        for error in report["rows"][0]["errors"]
    )
    assert {"success", "collision_event"} <= {
        error["field"] for error in report["rows"][1]["errors"]
    }
    assert {error["field"] for error in report["rows"][2]["errors"]} == {"min_distance"}
    assert report["rows"][3]["status"] == "valid"
    assert report["rows"][3]["warnings"] == [
        {
            "field": "status",
            "message": (
                "degraded rows should preserve any known outcome flags instead of leaving every "
                "field null"
            ),
        }
    ]


def test_same_producer_rows_remain_valid_when_feature_schema_matches() -> None:
    """Matching producer-stamped rows must remain comparable within one same-seed group."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    stamped_rows = copy.deepcopy(rows)
    runtime_schema = _predictive_feature_schema(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        producer_key=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    )
    stamped_rows[0]["feature_schema"] = runtime_schema
    stamped_rows[1]["feature_schema"] = runtime_schema

    report = validate_predictive_same_seed_row_summary_rows(stamped_rows)

    assert report["status"] == "valid"
    assert all(row["warnings"] == [] for row in report["rows"])


def test_mixed_producer_rows_fail_closed_with_named_producer_keys() -> None:
    """Mixed producer keys must invalidate the comparison group instead of looking comparable."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    stamped_rows = copy.deepcopy(rows)
    stamped_rows[0]["feature_schema"] = _predictive_feature_schema(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        producer_key=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    )
    stamped_rows[1]["feature_schema"] = _predictive_feature_schema(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        producer_key=PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
    )

    report = validate_predictive_same_seed_row_summary_rows(stamped_rows)

    assert report["status"] == "invalid"
    for row in report["rows"]:
        producer_errors = [
            error
            for error in row["errors"]
            if error["field"] == "feature_schema.ego_motion_channel_producer.producer_key"
        ]
        assert producer_errors
        assert "ego_motion_channel_producer.producer_key" in producer_errors[0]["message"]
        assert PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME in producer_errors[0]["message"]
        assert PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE in producer_errors[0]["message"]


def test_legacy_or_missing_producer_metadata_rows_emit_conservative_caveat() -> None:
    """Legacy or unstamped rows stay valid but must not be treated as proven comparable."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    mixed_rows = copy.deepcopy(rows)
    mixed_rows[0]["feature_schema"] = _predictive_feature_schema(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        producer_key=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    )
    mixed_rows[1]["feature_schema"] = _predictive_feature_schema(
        model_family=PREDICTIVE_LEGACY_FEATURE_SCHEMA,
    )

    report = validate_predictive_same_seed_row_summary_rows(mixed_rows)

    assert report["status"] == "valid"
    for row in report["rows"]:
        producer_warnings = [
            warning
            for warning in row["warnings"]
            if warning["field"] == "feature_schema.ego_motion_channel_producer.producer_key"
        ]
        assert producer_warnings
        assert "ego_motion_channel_producer.producer_key" in producer_warnings[0]["message"]
        assert PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME in producer_warnings[0]["message"]
        assert "legacy/no-metadata rows" in producer_warnings[0]["message"]


def test_all_legacy_feature_schema_rows_emit_unknown_comparability_caveat() -> None:
    """All-legacy feature schemas remain valid while carrying an explicit comparability caveat."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    legacy_rows = copy.deepcopy(rows)
    legacy_schema = _predictive_feature_schema(model_family=PREDICTIVE_LEGACY_FEATURE_SCHEMA)
    legacy_rows[0]["feature_schema"] = legacy_schema
    legacy_rows[1]["feature_schema"] = legacy_schema

    report = validate_predictive_same_seed_row_summary_rows(legacy_rows)

    assert report["status"] == "valid"
    for row in report["rows"]:
        producer_warnings = [
            warning
            for warning in row["warnings"]
            if warning["field"] == "feature_schema.ego_motion_channel_producer.producer_key"
        ]
        assert producer_warnings
        assert "ego_motion_channel_producer.producer_key" in producer_warnings[0]["message"]
        assert "direct comparability is not proven" in producer_warnings[0]["message"]


def test_repository_reference_guards_reject_unsafe_paths(tmp_path: Path) -> None:
    """Repository-relative provenance must not point outside the durable evidence boundary."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "docs" / "context").mkdir(parents=True)
    (repo_root / "docs" / "context" / "proof.md").write_text("ok\n", encoding="utf-8")
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
    cases = {
        "absolute": str((repo_root / "docs" / "context" / "proof.md").resolve()),
        "escape": "../proof.md",
        "output": "output/proof.md",
        "missing": "docs/context/missing.md",
    }
    for expected_message, value in cases.items():
        row = copy.deepcopy(rows[0])
        row["artifact_pointer"] = value

        report = validate_predictive_same_seed_row_summary_rows([row], repo_root=repo_root)

        messages = [error["message"] for error in report["rows"][0]["errors"]]
        assert any(expected_message in message for message in messages)


def test_cli_emits_json_for_invalid_input(tmp_path: Path, capsys) -> None:
    """The CLI should return a structured JSON report and the invalid-data exit code."""
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
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
    _input_format, rows = load_predictive_same_seed_row_summary_input(
        FIXTURE_ROOT / "valid_rows.yaml"
    )
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
