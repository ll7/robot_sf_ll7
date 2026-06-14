"""Tests for manifest lineage backfill check-only and write modes."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.manifest_lineage_backfill import (
    FieldStatus,
    ManifestBackfillPlan,
    analyze_manifest,
    main,
    run_backfill_check,
)

FIXTURE_DIR = Path(__file__).resolve().parents[0] / "fixtures" / "manifest_lineage_backfill"


# Fixture helpers


def _load_fixture(name: str) -> dict:
    """Load a JSON fixture by name."""
    path = FIXTURE_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fixture(tmp_path: Path, name: str, payload: dict) -> Path:
    """Write a manifest payload to a temporary file."""
    dest = tmp_path / name
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


# analyze_manifest tests


def test_complete_lineage_classified_as_present() -> None:
    """All fields present means every field is classified as PRESENT."""
    manifest = _load_fixture("complete_lineage.json")
    plan = analyze_manifest(manifest, path="complete_lineage.json")

    assert isinstance(plan, ManifestBackfillPlan)
    assert plan.validation_errors == []
    for entry in plan.fields:
        assert entry.status == FieldStatus.PRESENT, f"{entry.field_name} should be present"
    assert not plan.has_inferred
    assert not plan.has_ambiguous
    assert not plan.has_blocked


def test_missing_all_fields_classified_missing_or_blocked() -> None:
    """Manifest missing most fields has MISSING field statuses."""
    manifest = _load_fixture("missing_all.json")
    plan = analyze_manifest(manifest, path="missing_all.json")

    # schema_version is at top level, so it is PRESENT.
    sv = next(f for f in plan.fields if f.field_name == "schema_version")
    assert sv.status == FieldStatus.PRESENT

    # source has no nearby field, so it is MISSING.
    src = next(f for f in plan.fields if f.field_name == "source")
    assert src.status == FieldStatus.MISSING

    # Other fields with no nearby source are MISSING.
    for f in plan.fields:
        if f.field_name == "schema_version":
            continue
        assert f.status == FieldStatus.MISSING, f"{f.field_name} should be missing"


def test_inferred_candidate_fields() -> None:
    """Manifest with metadata proxies has fields classified as INFERRED."""
    manifest = _load_fixture("inferred_candidate.json")
    plan = analyze_manifest(manifest, path="inferred_candidate.json")

    # schema_version is PRESENT at the top level.
    sv = next(f for f in plan.fields if f.field_name == "schema_version")
    assert sv.status == FieldStatus.PRESENT

    # generator_id is INFERRED from metadata.generator_id.
    gid = next(f for f in plan.fields if f.field_name == "generator_id")
    assert gid.status == FieldStatus.INFERRED
    assert gid.inferred_value == "inferred-gen"
    assert gid.inferred_from is not None

    # validator_version is INFERRED from metadata.validator_version.
    vv = next(f for f in plan.fields if f.field_name == "validator_version")
    assert vv.status == FieldStatus.INFERRED
    assert vv.inferred_value == "2.1.0"

    # evidence_tier is INFERRED from metadata.evidence_tier.
    et = next(f for f in plan.fields if f.field_name == "evidence_tier")
    assert et.status == FieldStatus.INFERRED
    assert et.inferred_value == "diagnostic"

    # denominator_policy is INFERRED.
    dp = next(f for f in plan.fields if f.field_name == "denominator_policy")
    assert dp.status == FieldStatus.INFERRED

    # execution_gate is INFERRED.
    eg = next(f for f in plan.fields if f.field_name == "execution_gate")
    assert eg.status == FieldStatus.INFERRED

    assert plan.has_inferred
    assert not plan.has_ambiguous


def test_ambiguous_fields_detected() -> None:
    """Manifest with conflicting proxy sources has AMBIGUOUS classification."""
    manifest = _load_fixture("ambiguous_fields.json")
    plan = analyze_manifest(manifest, path="ambiguous_fields.json")

    # validator_version is AMBIGUOUS because metadata and config conflict.
    vv = next(f for f in plan.fields if f.field_name == "validator_version")
    assert vv.status == FieldStatus.AMBIGUOUS
    assert "multiple" in vv.reason.lower() or "ambiguous" in vv.reason.lower()

    assert plan.has_ambiguous


def test_matching_duplicate_proxy_values_inferred() -> None:
    """Matching values in multiple proxy paths are safe to infer."""
    manifest = {
        "schema_version": "benchmark_claim.v1",
        "metadata": {
            "generator_id": "gen-a",
        },
        "config": {
            "generator_id": "gen-a",
        },
    }
    plan = analyze_manifest(manifest, path="matching_proxy_values.json")

    gid = next(f for f in plan.fields if f.field_name == "generator_id")
    assert gid.status == FieldStatus.INFERRED
    assert gid.inferred_value == "gen-a"
    assert "matching candidates" in gid.reason


def test_valid_and_invalid_proxy_values_ambiguous() -> None:
    """Mixed valid and invalid proxy values stay review-visible."""
    manifest = {
        "schema_version": "benchmark_claim.v1",
        "metadata": {
            "generator_id": "gen-a",
        },
        "config": {
            "generator_id": "",
        },
    }
    plan = analyze_manifest(manifest, path="mixed_proxy_values.json")

    gid = next(f for f in plan.fields if f.field_name == "generator_id")
    assert gid.status == FieldStatus.AMBIGUOUS
    assert "valid and invalid" in gid.reason


def test_blocked_fields_when_no_inference_source() -> None:
    """Manifest with invalid nearby fields for required lineage is BLOCKED."""
    manifest = _load_fixture("blocked_fields.json")
    plan = analyze_manifest(manifest, path="blocked_fields.json")

    # Only schema_version is present
    sv = next(f for f in plan.fields if f.field_name == "schema_version")
    assert sv.status == FieldStatus.PRESENT

    # source, generator_id, and claim_boundary have invalid nearby proxies.
    for name in ("source", "generator_id", "claim_boundary"):
        entry = next(f for f in plan.fields if f.field_name == name)
        assert entry.status == FieldStatus.BLOCKED, f"{name} should be blocked"

    assert plan.has_blocked


# Check-only mode tests


def test_check_only_does_not_modify_files(tmp_path: Path) -> None:
    """Check-only mode must not modify any files."""
    src = FIXTURE_DIR / "inferred_candidate.json"
    dest = tmp_path / "inferred_candidate.json"
    shutil.copy2(src, dest)
    original_content = dest.read_text(encoding="utf-8")

    plan = run_backfill_check(
        [str(dest)],
        write_backfill=False,
        json_output=False,
    )

    assert not plan.write_mode
    assert plan.written_paths == []
    assert dest.read_text(encoding="utf-8") == original_content


def test_check_only_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check-only mode with --json produces machine-readable output."""
    src = FIXTURE_DIR / "complete_lineage.json"
    dest = tmp_path / "complete_lineage.json"
    shutil.copy2(src, dest)

    run_backfill_check(
        [str(dest)],
        write_backfill=False,
        json_output=True,
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert "manifests" in output
    assert output["write_mode"] is False


# Write-backfill mode tests


def test_write_backfill_applies_inferred_fields(tmp_path: Path) -> None:
    """Write-backfill mode writes inferred fields to the manifest."""
    src = FIXTURE_DIR / "inferred_candidate.json"
    dest = tmp_path / "inferred_candidate.json"
    shutil.copy2(src, dest)

    plan = run_backfill_check(
        [str(dest)],
        write_backfill=True,
        json_output=False,
    )

    assert plan.write_mode
    assert len(plan.written_paths) == 1

    # Verify the file was updated
    updated = json.loads(dest.read_text(encoding="utf-8"))
    assert updated["generator_id"] == "inferred-gen"
    assert updated["validator_version"] == "2.1.0"
    assert updated["evidence_tier"] == "diagnostic"
    assert updated["denominator_policy"] == "successful-only"
    assert updated["execution_gate"] == "pass"


def test_write_backfill_skips_files_without_inferred_fields(tmp_path: Path) -> None:
    """Write-backfill avoids rewriting files when no fields are inferred."""
    src = FIXTURE_DIR / "complete_lineage.json"
    dest = tmp_path / "complete_lineage.json"
    shutil.copy2(src, dest)
    original_content = dest.read_text(encoding="utf-8")

    plan = run_backfill_check(
        [str(dest)],
        write_backfill=True,
        json_output=False,
    )

    assert plan.write_mode
    assert plan.written_paths == []
    assert dest.read_text(encoding="utf-8") == original_content


def test_write_backfill_refuses_directory(tmp_path: Path) -> None:
    """Write-backfill mode refuses directory paths."""
    with pytest.raises(SystemExit):
        run_backfill_check(
            [str(tmp_path)],
            write_backfill=True,
            json_output=False,
        )


def test_write_backfill_refuses_broad_path(tmp_path: Path) -> None:
    """Write-backfill mode refuses short/broad paths like '/' or '/tmp'."""
    with pytest.raises(SystemExit):
        run_backfill_check(
            ["/"],
            write_backfill=True,
            json_output=False,
        )


def test_write_backfill_refuses_repo_root_relative(tmp_path: Path) -> None:
    """Write-backfill mode refuses paths that resolve to repo root."""
    with pytest.raises(SystemExit):
        run_backfill_check(
            ["."],
            write_backfill=True,
            json_output=False,
        )


def test_write_backfill_allows_shallow_explicit_file() -> None:
    """Write-backfill allows an explicit shallow absolute file path."""
    payload = {
        "schema_version": "benchmark_claim.v1",
        "metadata": {
            "generator_id": "gen-shallow",
        },
    }
    shallow_path = Path(tempfile.gettempdir()) / "issue2723_manifest_lineage_backfill.json"
    try:
        shallow_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        plan = run_backfill_check(
            [str(shallow_path)],
            write_backfill=True,
            json_output=False,
        )

        updated = json.loads(shallow_path.read_text(encoding="utf-8"))
        assert plan.write_mode
        assert updated["generator_id"] == "gen-shallow"
    finally:
        shallow_path.unlink(missing_ok=True)


# Multiple manifest batch tests


def test_batch_check_only_multiple_manifests() -> None:
    """Check-only mode handles multiple manifests in one run."""
    paths = [
        str(FIXTURE_DIR / "complete_lineage.json"),
        str(FIXTURE_DIR / "inferred_candidate.json"),
        str(FIXTURE_DIR / "missing_all.json"),
    ]

    plan = run_backfill_check(paths, write_backfill=False, json_output=False)

    assert len(plan.manifests) == 3
    assert plan.total_inferred > 0
    assert plan.total_missing > 0 or plan.total_blocked > 0


def test_batch_check_only_json_structure() -> None:
    """Batch check-only JSON output has expected top-level keys."""
    paths = [
        str(FIXTURE_DIR / "complete_lineage.json"),
        str(FIXTURE_DIR / "ambiguous_fields.json"),
    ]

    plan = run_backfill_check(paths, write_backfill=False, json_output=False)

    d = plan.to_dict()
    assert "manifests" in d
    assert "total_missing" in d
    assert "total_inferred" in d
    assert "total_ambiguous" in d
    assert "total_blocked" in d
    assert "write_mode" in d
    assert "written_paths" in d


# CLI tests


def test_cli_check_only(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI in check-only mode runs without error."""
    path = str(FIXTURE_DIR / "complete_lineage.json")
    main([path])
    captured = capsys.readouterr()
    assert "CHECK-ONLY" in captured.out


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --json produces valid JSON."""
    path = str(FIXTURE_DIR / "complete_lineage.json")
    main([path, "--json"])
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert "manifests" in output


# FieldStatus enum tests


def test_field_status_values() -> None:
    """FieldStatus enum has all expected values."""
    expected = {"present", "missing", "inferred", "ambiguous", "blocked"}
    assert {s.value for s in FieldStatus} == expected


# Edge cases


def test_analyze_manifest_non_dict_raises_value_error() -> None:
    """analyze_manifest fails closed for non-dictionary payloads."""
    with pytest.raises(ValueError) as exc_info:
        analyze_manifest(None, path="edge.json")  # type: ignore[arg-type]
    assert "Manifest contract payload must be a dictionary mapping" in str(exc_info.value), (
        f"Expected ValueError with contract mapping message, got: {exc_info.value}"
    )


def test_empty_manifest_all_fields_blocked() -> None:
    """Empty manifest has all mandatory fields missing."""
    plan = analyze_manifest({}, path="empty.json")

    for entry in plan.fields:
        assert entry.status == FieldStatus.MISSING, f"{entry.field_name} should be missing"


def test_inferred_fields_write_backfill_only_writes_inferred(tmp_path: Path) -> None:
    """Write-backfill writes only INFERRED fields, not AMBIGUOUS/BLOCKED."""
    payload = {
        "schema_version": "v1",
        "metadata": {
            "validator_version": "2.0",
            "generator_id": "gen-x",
        },
        "config": {
            "validator_version": "3.0",
        },
    }
    dest = _write_fixture(tmp_path, "mixed.json", payload)

    run_backfill_check(
        [str(dest)],
        write_backfill=True,
        json_output=False,
    )

    updated = json.loads(dest.read_text(encoding="utf-8"))
    # schema_version is already present, so it is unchanged.
    assert updated["schema_version"] == "v1"
    # generator_id is inferred from metadata, so it is written.
    assert updated["generator_id"] == "gen-x"
    # validator_version is ambiguous (metadata vs config), so it is not written.
    assert "validator_version" not in updated


def test_human_readable_output_includes_status_markers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human-readable output includes INFERRED/AMBIGUOUS/BLOCKED markers."""
    run_backfill_check(
        [str(FIXTURE_DIR / "inferred_candidate.json")],
        write_backfill=False,
        json_output=False,
    )
    captured = capsys.readouterr()
    assert "INFERRED" in captured.out
