"""Tests for benchmark release suite metadata reference dereference validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.release_suite_contract import (
    RELEASE_SUITE_CONTRACT_SCHEMA_VERSION,
    load_release_suite_contract,
)
from robot_sf.benchmark.release_suite_reference_validation import (
    REFERENCE_VALIDATION_CLAIM_BOUNDARY,
    RELEASE_SUITE_REFERENCE_REPORT_SCHEMA_VERSION,
    ReleaseSuiteReferenceError,
    evaluate_release_suite_references,
)
from scripts.validation import check_release_suite_contract

if TYPE_CHECKING:
    from pathlib import Path

# Reuse the canonical six metadata-owner field names.
REQUIRED_FIELDS = (
    "odd_contract",
    "scenario_contract",
    "scenario_certification",
    "planner_row_status",
    "seed_schedule",
    "artifact_manifest",
)


def _suite(suite_id: str = "nominal") -> dict[str, str]:
    """Build one structurally complete suite declaration."""

    return {
        "suite_id": suite_id,
        "odd_contract": f"contracts/{suite_id}/odd.yaml",
        "scenario_contract": f"contracts/{suite_id}/scenarios.yaml",
        "scenario_certification": f"certification/{suite_id}.json",
        "planner_row_status": f"rows/{suite_id}.json",
        "seed_schedule": f"seeds/{suite_id}.yaml",
        "artifact_manifest": f"artifacts/{suite_id}.json",
    }


def _manifest_payload(suites: list[object]) -> dict[str, object]:
    """Build a minimal suite-contract manifest payload."""

    return {
        "schema_version": RELEASE_SUITE_CONTRACT_SCHEMA_VERSION,
        "release_id": "benchmark-v0.1-candidate",
        "suites": suites,
    }


def _write_manifest(path: Path, payload: object) -> None:
    """Write a YAML or JSON manifest fixture."""

    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _materialize_suite_records(base_dir: Path, suite_id: str = "nominal") -> dict[str, str]:
    """Materialize parseable record files for every reference of one suite."""

    records = _suite(suite_id)
    for field in REQUIRED_FIELDS:
        relative = records[field]
        target = base_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: object = {"suite_id": suite_id, "field": field}
        if target.suffix == ".json":
            target.write_text(json.dumps(payload), encoding="utf-8")
        else:
            target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return records


def _load_manifest(tmp_path: Path, suites: list[object]) -> object:
    """Write and structurally load a suite manifest fixture."""

    manifest_path = tmp_path / "release_suites.yaml"
    _write_manifest(manifest_path, _manifest_payload(suites))
    return load_release_suite_contract(manifest_path)


def test_all_resolved_references_pass(tmp_path: Path) -> None:
    """Every reference resolving to a parseable file yields a pass."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "pass"
    assert report["schema_version"] == RELEASE_SUITE_REFERENCE_REPORT_SCHEMA_VERSION
    assert report["claim_boundary"] == REFERENCE_VALIDATION_CLAIM_BOUNDARY
    assert report["release_id"] == "benchmark-v0.1-candidate"
    assert report["base_dir"] == str(base_dir)
    assert report["suite_count"] == 1
    assert report["resolved_suite_count"] == 1
    assert report["blocked_suite_count"] == 0
    assert report["reference_count"] == 6
    assert report["resolved_reference_count"] == 6
    assert report["blocked_reference_count"] == 0
    assert report["blockers"] == []
    assert report["suites"][0]["status"] == "resolved"
    assert len(report["suites"][0]["references"]) == 6
    assert all(ref["status"] == "resolved" for ref in report["suites"][0]["references"])


def test_missing_reference_file_is_not_available_and_blocks(tmp_path: Path) -> None:
    """A reference pointing at a non-existent file fails closed as not_available."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    assert report["blocked_reference_count"] == 6
    assert report["resolved_reference_count"] == 0
    nominal_refs = report["suites"][0]["references"]
    assert all(ref["status"] == "not_available" for ref in nominal_refs)
    assert all("does not exist" in ref["detail"] for ref in nominal_refs)
    assert len(report["blockers"]) == 6


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_single_unresolvable_reference_blocks_whole_suite(
    tmp_path: Path,
    field: str,
) -> None:
    """A single bad reference among five good ones blocks only its suite."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    suite = _suite("nominal")
    suite[field] = "contracts/nominal/missing.yaml"
    manifest = _load_manifest(tmp_path, [suite])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    assert report["resolved_reference_count"] == 5
    assert report["blocked_reference_count"] == 1
    bad = next(ref for ref in report["suites"][0]["references"] if ref["field"] == field)
    assert bad["status"] == "not_available"
    assert len(report["blockers"]) == 1
    assert report["blockers"][0].startswith(f"nominal.{field} ")


def test_path_traversal_outside_base_dir_fails_closed(tmp_path: Path) -> None:
    """References that escape the base directory must not be followed."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"escaped": True}), encoding="utf-8")
    suite = _suite("nominal")
    # All references resolve except one that escapes the base directory.
    suite["artifact_manifest"] = str(outside)
    manifest = _load_manifest(tmp_path, [suite])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    escaped = next(
        ref for ref in report["suites"][0]["references"] if ref["field"] == "artifact_manifest"
    )
    assert escaped["status"] == "blocked"
    assert escaped["detail"] == "reference escapes the base directory"
    assert escaped["reference"] == str(outside)


@pytest.mark.parametrize(
    "content",
    ["", "   \n  \t  "],
)
def test_empty_referenced_file_blocks(tmp_path: Path, content: str) -> None:
    """A referenced file that is empty or whitespace-only fails closed."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    # Overwrite one record with empty content.
    (base_dir / _suite("nominal")["seed_schedule"]).write_text(content, encoding="utf-8")
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    bad = next(ref for ref in report["suites"][0]["references"] if ref["field"] == "seed_schedule")
    assert bad["status"] == "blocked"
    assert "empty" in bad["detail"]


def test_malformed_referenced_file_blocks(tmp_path: Path) -> None:
    """A referenced file that does not parse fails closed."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    # Write malformed JSON into a .json record.
    (base_dir / _suite("nominal")["planner_row_status"]).write_text(
        "{not valid json", encoding="utf-8"
    )
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    bad = next(
        ref for ref in report["suites"][0]["references"] if ref["field"] == "planner_row_status"
    )
    assert bad["status"] == "blocked"
    assert "not parseable" in bad["detail"]


def test_null_parsed_payload_blocks(tmp_path: Path) -> None:
    """A referenced YAML/JSON file that parses to null fails closed."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    (base_dir / _suite("nominal")["odd_contract"]).write_text("null", encoding="utf-8")
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    bad = next(ref for ref in report["suites"][0]["references"] if ref["field"] == "odd_contract")
    assert bad["status"] == "blocked"
    assert "null" in bad["detail"]


def test_directory_reference_blocks(tmp_path: Path) -> None:
    """A reference that resolves to a directory rather than a file fails closed."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    # Make the certification reference point at a directory.
    suite = _suite("nominal")
    cert_dir = base_dir / "certification" / "nominal_dir"
    cert_dir.mkdir(parents=True)
    suite["scenario_certification"] = "certification/nominal_dir"
    manifest = _load_manifest(tmp_path, [suite])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    bad = next(
        ref for ref in report["suites"][0]["references"] if ref["field"] == "scenario_certification"
    )
    assert bad["status"] == "blocked"
    assert bad["detail"] == "referenced path is not a regular file"


def test_blockers_accumulate_across_multiple_suites(tmp_path: Path) -> None:
    """Reference failures accumulate across every suite instead of stopping early."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    # Second suite references nothing materialized.
    manifest = _load_manifest(tmp_path, [_suite("nominal"), _suite("stress")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    assert report["suite_count"] == 2
    assert report["resolved_suite_count"] == 1
    assert report["blocked_suite_count"] == 1
    assert report["resolved_reference_count"] == 6
    assert report["blocked_reference_count"] == 6
    assert len(report["blockers"]) == 6
    assert all(blocker.startswith("stress.") for blocker in report["blockers"])


def test_structurally_missing_reference_field_blocks(tmp_path: Path) -> None:
    """A reference field that is not a non-empty string fails closed at dereference time."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    suite = _suite("nominal")
    suite["seed_schedule"] = None
    manifest = _load_manifest(tmp_path, [suite])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "blocked"
    bad = next(ref for ref in report["suites"][0]["references"] if ref["field"] == "seed_schedule")
    assert bad["status"] == "blocked"
    assert "non-empty string" in bad["detail"]


def test_nonexistent_base_dir_raises(tmp_path: Path) -> None:
    """Evaluation must fail closed when the base directory does not exist."""

    base_dir = tmp_path / "missing"
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    with pytest.raises(ReleaseSuiteReferenceError, match="base_dir does not exist"):
        evaluate_release_suite_references(manifest, base_dir)


def test_file_base_dir_raises(tmp_path: Path) -> None:
    """Evaluation must fail closed when the base dir path is a file."""

    base_dir = tmp_path / "not_a_dir.json"
    base_dir.write_text("{}", encoding="utf-8")
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    with pytest.raises(ReleaseSuiteReferenceError, match="not a directory"):
        evaluate_release_suite_references(manifest, base_dir)


def test_report_is_deterministic_for_repeated_evaluation(tmp_path: Path) -> None:
    """Repeated evaluation of the same manifest yields identical reports."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    first = evaluate_release_suite_references(manifest, base_dir)
    second = evaluate_release_suite_references(manifest, base_dir)

    assert first == second


def test_cli_base_dir_pass_when_all_references_resolve(tmp_path: Path) -> None:
    """The CLI passes only when --base-dir dereferences resolve every reference."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    manifest_path = tmp_path / "release_suites.yaml"
    _write_manifest(manifest_path, _manifest_payload([_suite("nominal")]))

    exit_code = check_release_suite_contract.main(
        ["--manifest", str(manifest_path), "--base-dir", str(base_dir), "--json"]
    )

    assert exit_code == 0


def test_cli_base_dir_blocks_on_missing_reference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI exits blocked when --base-dir dereferences find a missing file."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    manifest_path = tmp_path / "release_suites.yaml"
    _write_manifest(manifest_path, _manifest_payload([_suite("nominal")]))

    exit_code = check_release_suite_contract.main(
        ["--manifest", str(manifest_path), "--base-dir", str(base_dir)]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "reference validation: 0 resolved, 6 blocked" in out
    assert "blocker: nominal." in out


def test_cli_without_base_dir_unchanged_and_ignores_references(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --base-dir the structural-only behavior is backward-compatible."""

    manifest_path = tmp_path / "release_suites.yaml"
    _write_manifest(manifest_path, _manifest_payload([_suite("nominal")]))

    exit_code = check_release_suite_contract.main(["--manifest", str(manifest_path), "--json"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "pass"
    assert "reference_validation" not in out


def test_cli_base_dir_with_malformed_base_dir_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed base directory surfaces as the malformed exit code (2)."""

    base_dir = tmp_path / "missing"
    manifest_path = tmp_path / "release_suites.yaml"
    _write_manifest(manifest_path, _manifest_payload([_suite("nominal")]))

    exit_code = check_release_suite_contract.main(
        ["--manifest", str(manifest_path), "--base-dir", str(base_dir)]
    )

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_reference_resolution_handles_symlink_inside_base_dir(tmp_path: Path) -> None:
    """A symlink that stays within the base directory resolves normally."""

    base_dir = tmp_path / "evidence"
    base_dir.mkdir()
    _materialize_suite_records(base_dir, "nominal")
    # Replace the artifact manifest with a symlink to an in-base-dir file.
    real = base_dir / "artifacts" / "nominal_real.json"
    real.write_text(json.dumps({"via": "symlink"}), encoding="utf-8")
    link = base_dir / "artifacts" / "nominal.json"
    link.unlink()
    link.symlink_to(real)
    manifest = _load_manifest(tmp_path, [_suite("nominal")])

    report = evaluate_release_suite_references(manifest, base_dir)

    assert report["status"] == "pass"
    bad = [
        ref
        for ref in report["suites"][0]["references"]
        if ref["field"] == "artifact_manifest" and ref["status"] != "resolved"
    ]
    assert bad == []
