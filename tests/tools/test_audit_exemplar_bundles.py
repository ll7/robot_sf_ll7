"""Tests for the exemplar-bundle audit script (issue #4920)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_exemplar_bundles.py"
SPEC = importlib.util.spec_from_file_location("audit_exemplar_bundles", SCRIPT_PATH)
assert SPEC is not None
audit_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit_mod
SPEC.loader.exec_module(audit_mod)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_minimal_bundle(
    bundle_dir: Path,
    *,
    claim_boundary: str = "illustrative test only; no benchmark claim",
    review_marker: str = "AI-GENERATED NEEDS-REVIEW",
    extra_files: dict[str, str] | None = None,
) -> None:
    """Create a minimal valid bundle directory."""
    csv_content = "step,time_s,min_robot_ped_distance_m\n0,0.0,1.5\n"
    json_content = json.dumps({"frames": []})
    metadata = {
        "campaign_id": "test_campaign",
        "campaign_job": "9999",
        "claim_boundary": claim_boundary,
        "episode_id": "test_episode",
        "episode_status": "success",
        "generated_at_utc": "2026-07-09T00:00:00+00:00",
        "git_commit": "abc123",
        "issue": "https://github.com/ll7/robot_sf_ll7/issues/9999",
        "planner": "goal",
        "scenario_id": "test_scenario",
        "schema_version": "test-exemplar.v1",
        "seed": 1,
        "selection_metric": "path_efficiency",
        "selection_metric_value": 1.0,
        "selection_mode": "best",
        "summary": {
            "episode_status": "success",
            "step_count": 10,
            "termination_reason": "success",
        },
        "review_marker": review_marker,
    }
    readme = "# Test Bundle\n"
    sha_entries = []

    _write_file(bundle_dir / "metadata.json", json.dumps(metadata, indent=2))
    _write_file(bundle_dir / "min_distance_series.csv", csv_content)
    _write_file(bundle_dir / "trace_series.json", json_content)
    _write_file(bundle_dir / "trace_timeseries.csv", csv_content)
    _write_file(bundle_dir / "README.md", readme)

    if extra_files:
        for name, content in extra_files.items():
            _write_file(bundle_dir / name, content)

    for fname in [
        "metadata.json",
        "min_distance_series.csv",
        "trace_series.json",
        "trace_timeseries.csv",
        "README.md",
    ]:
        data = (bundle_dir / fname).read_bytes()
        rel = str(bundle_dir / fname)
        sha_entries.append(f"{_sha256_bytes(data)}  {rel}")

    _write_file(bundle_dir / "SHA256SUMS", "\n".join(sha_entries) + "\n")


def test_verify_checksums_valid(tmp_path: Path) -> None:
    """Verify that valid checksums pass verification."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle)
    ok, errors = audit_mod.verify_checksums(bundle, tmp_path)
    assert ok is True
    assert errors == []


def test_verify_checksums_mismatch(tmp_path: Path) -> None:
    """Verify that corrupted files are detected by checksum verification."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle)
    (bundle / "metadata.json").write_text("{}")
    ok, errors = audit_mod.verify_checksums(bundle, tmp_path)
    assert ok is False
    assert any("MISMATCH" in e for e in errors)


def test_check_metadata_valid(tmp_path: Path) -> None:
    """Verify that valid metadata passes checks."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle)
    ok, errors, claim = audit_mod.check_metadata(bundle)
    assert ok is True
    assert errors == []
    assert "illustrative" in claim


def test_check_metadata_missing_claim(tmp_path: Path) -> None:
    """Verify that empty claim_boundary is detected."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle, claim_boundary="")
    ok, errors, _ = audit_mod.check_metadata(bundle)
    assert ok is False
    assert any("empty" in e for e in errors)


def test_check_metadata_missing_review_marker(tmp_path: Path) -> None:
    """Verify that missing NEEDS-REVIEW marker is detected."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle, review_marker="none")
    ok, errors, _ = audit_mod.check_metadata(bundle)
    assert ok is False
    assert any("NEEDS-REVIEW" in e for e in errors)


def test_audit_bundle_no_raw_dumps(tmp_path: Path) -> None:
    """Verify that bundles without raw dumps pass all checks."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle)
    result = audit_mod.audit_bundle(bundle, tmp_path)
    assert result.no_raw_dumps is True
    assert result.sha256_ok is True
    assert result.metadata_ok is True
    assert result.file_count_ok is True
    assert result.total_size_bytes > 0


def test_audit_bundle_with_raw_dump(tmp_path: Path) -> None:
    """Verify that JSONL raw dumps are detected."""
    bundle = tmp_path / "planner" / "test_bundle"
    _make_minimal_bundle(bundle, extra_files={"raw_episodes.jsonl": "{}\n"})
    result = audit_mod.audit_bundle(bundle, tmp_path)
    assert result.no_raw_dumps is False
    assert "raw_episodes.jsonl" in result.raw_dump_files


def test_audit_class(tmp_path: Path) -> None:
    """Verify that class-level audit finds all bundles."""
    cls_dir = tmp_path / "issue_9999_test"
    for i, mode in enumerate(["best", "median", "worst"]):
        bundle = cls_dir / "goal" / f"test_seed{i}_{mode}"
        _make_minimal_bundle(bundle)
    result = audit_mod.audit_class(cls_dir, tmp_path)
    assert len(result.bundles) == 3
    assert result.total_size_bytes > 0


def test_format_size() -> None:
    """Verify human-readable size formatting."""
    assert audit_mod.format_size(500) == "500 B"
    assert "KB" in audit_mod.format_size(5000)
    assert "MB" in audit_mod.format_size(5_000_000)


def test_assess_exemplar_budget_preserves_migration_headroom() -> None:
    """Verify the 45 MiB trigger reserves headroom below the hard 50 MiB cap."""
    assert audit_mod.assess_exemplar_budget(44 * audit_mod.MEBIBYTE) == ("WITHIN_BUDGET", False)
    assert audit_mod.assess_exemplar_budget(45 * audit_mod.MEBIBYTE) == (
        "OFF_TREE_MIGRATION_REQUIRED",
        True,
    )
    assert audit_mod.assess_exemplar_budget(50 * audit_mod.MEBIBYTE) == (
        "OFF_TREE_MIGRATION_REQUIRED",
        True,
    )
    assert audit_mod.assess_exemplar_budget(50 * audit_mod.MEBIBYTE + 1) == (
        "EXCEEDS_BUDGET",
        True,
    )
    with pytest.raises(ValueError, match="must not be negative"):
        audit_mod.assess_exemplar_budget(-1)


def test_parse_sha256sums(tmp_path: Path) -> None:
    """Verify SHA256SUMS file parsing."""
    sums_file = tmp_path / "SHA256SUMS"
    sums_file.write_text("# comment line\nabc123  path/to/file1.txt\ndef456  path/to/file2.csv\n")
    result = audit_mod.parse_sha256sums(sums_file)
    assert result == {
        "path/to/file1.txt": "abc123",
        "path/to/file2.csv": "def456",
    }


def test_render_summary_claim_independent_of_sha() -> None:
    """The 'Claim boundary present' row must track claim status, not SHA status."""
    lines = audit_mod._render_summary(
        all_sha_ok=False,
        all_meta_ok=True,
        all_derived=True,
        all_claim_ok=True,
        total_bundles=18,
        grand_total=13_844_906,
    )
    joined = "\n".join(lines)
    assert "| SHA256SUMS verification | **FAIL** |" in joined
    assert "| Claim boundary present | **PASS** |" in joined
