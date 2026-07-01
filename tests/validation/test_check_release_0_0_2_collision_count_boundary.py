"""Tests release 0.0.2 collision-count claim-boundary checker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "validation" / "check_release_0_0_2_collision_count_boundary.py"

_SPEC = importlib.util.spec_from_file_location(
    "check_release_0_0_2_collision_count_boundary", SCRIPT_PATH
)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DEFAULT_MANIFEST = _MODULE.DEFAULT_MANIFEST
EXPECTED_COLLISION_COUNT_STATUS = _MODULE.EXPECTED_COLLISION_COUNT_STATUS
build_report = _MODULE.build_report
validate_manifest = _MODULE.validate_manifest


def _manifest() -> dict:
    """Load the tracked boundary manifest."""
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def test_tracked_manifest_keeps_collision_count_metric_withdrawn() -> None:
    """The checked-in boundary withdraws release 0.0.2 collision-count metrics."""
    report = build_report(DEFAULT_MANIFEST)

    assert report["status"] == "pass"
    assert report["release_tag"] == "0.0.2"
    assert report["collision_count_metric_status"] == EXPECTED_COLLISION_COUNT_STATUS


def test_manifest_fails_if_collision_count_metric_marked_claim_ready() -> None:
    """The checker rejects premature collision-count claim promotion."""
    payload = _manifest()
    payload["claim_boundaries"]["collision_count_metric_status"] = "paper_ready"

    violations = validate_manifest(payload)

    assert any(
        violation.field == "claim_boundaries.collision_count_metric_status"
        for violation in violations
    )


def test_manifest_fails_if_withdrawn_status_keeps_promotion_gates() -> None:
    """Withdrawn collision-count claims must not retain open promotion gates."""
    payload = _manifest()
    payload["claim_boundaries"]["open_gates"] = [
        "committed_0_0_2_reconciliation_bundle",
    ]

    violations = validate_manifest(payload)

    assert any(violation.field == "claim_boundaries.open_gates" for violation in violations)


def test_manifest_fails_if_exact_derived_discrepancy_is_erased() -> None:
    """The boundary depends on preserving the documented release discrepancy."""
    payload = _manifest()
    payload["diagnostic_summary"]["derived_collision_count_positive_episodes"] = 241

    violations = validate_manifest(payload)

    assert any(
        violation.field == "diagnostic_summary.derived_collision_count_positive_episodes"
        for violation in violations
    )


def test_cli_reports_json_boundary() -> None:
    """CLI emits a machine-readable pass report for the tracked manifest."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "pass"
    assert report["collision_count_metric_status"] == EXPECTED_COLLISION_COUNT_STATUS
