"""Tests for the issue #5164 durable h600 source-report checker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT / "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
    "h600_source_reports_manifest.json"
)
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_5164_h600_source_reports.py"

_SPEC = importlib.util.spec_from_file_location("_issue_5164_source_check", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_checked_in_contract_fails_closed_on_unavailable_sources() -> None:
    """A fresh checkout cannot treat absent source reports as export-ready."""
    report = _MODULE.validate_source_reports(_manifest(), REPO_ROOT)
    assert report["status"] == "blocked"
    assert report["required_file_count"] == 6
    assert report["missing_file_count"] == 6
    assert report["checksum_mismatch_count"] == 0
    assert report["downstream_export_allowed"] is False


def test_all_six_exact_sources_make_contract_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checker opens the downstream gate only when every digest matches."""
    manifest = deepcopy(_manifest())
    expected_runs = deepcopy(_MODULE.EXPECTED_RUNS)
    for run in manifest["required_runs"]:
        job_id = run["job_id"]
        report_dir = tmp_path / run["reports_dir"]
        report_dir.mkdir(parents=True)
        for filename, metadata in run["files"].items():
            content = f"{job_id},{filename}\n".encode()
            (report_dir / filename).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            metadata["sha256"] = digest
            expected_runs[job_id]["sha256"][filename] = digest

    monkeypatch.setattr(_MODULE, "EXPECTED_RUNS", expected_runs)
    report = _MODULE.validate_source_reports(manifest, tmp_path)
    assert report["status"] == "ready"
    assert report["verified_file_count"] == 6
    assert report["missing_file_count"] == 0
    assert report["checksum_mismatch_count"] == 0
    assert report["downstream_export_allowed"] is True


def test_checksum_mismatch_remains_blocked(tmp_path: Path) -> None:
    """A present but altered source is distinct from a missing source and fails closed."""
    manifest = _manifest()
    first_run = manifest["required_runs"][0]
    report_dir = tmp_path / first_run["reports_dir"]
    report_dir.mkdir(parents=True)
    (report_dir / "scenario_breakdown.csv").write_text("altered\n", encoding="utf-8")

    report = _MODULE.validate_source_reports(manifest, tmp_path)
    assert report["status"] == "blocked"
    assert report["missing_file_count"] == 5
    assert report["checksum_mismatch_count"] == 1
    assert report["downstream_export_allowed"] is False


def test_output_path_is_rejected_before_file_checks() -> None:
    """Local output paths cannot satisfy the durable-source contract."""
    manifest = _manifest()
    manifest["required_runs"][0]["reports_dir"] = "output/13268/reports"
    with pytest.raises(_MODULE.ContractError, match="reports_dir mismatch"):
        _MODULE.validate_source_reports(manifest, REPO_ROOT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_label", "different_run"),
        ("campaign_id", "different_campaign"),
        ("source_git_hash", "0" * 40),
    ],
)
def test_run_identity_drift_is_rejected(field: str, replacement: str) -> None:
    """The source provenance cannot drift while its report digests remain unchanged."""
    manifest = _manifest()
    manifest["required_runs"][0][field] = replacement

    with pytest.raises(_MODULE.ContractError, match=f"{field} mismatch"):
        _MODULE.validate_source_reports(manifest, REPO_ROOT)


def test_cli_emits_machine_readable_blocked_packet() -> None:
    """Automation receives an explicit blocked packet and nonzero gate status."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "blocked"
    assert payload["missing_file_count"] == 6
    assert payload["downstream_export_allowed"] is False


def test_cli_default_manifest_is_independent_of_calling_directory(tmp_path: Path) -> None:
    """The default contract remains diagnosable when automation runs outside the checkout."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "blocked"
    assert payload["missing_file_count"] == 6
