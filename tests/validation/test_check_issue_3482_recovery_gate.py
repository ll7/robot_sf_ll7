"""Tests for the issue #3482 release 0.0.2 recovery gate checker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "validation" / "check_issue_3482_recovery_gate.py"

_SPEC = importlib.util.spec_from_file_location("check_issue_3482_recovery_gate", SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DEFAULT_BOUNDARY_MANIFEST = _MODULE.DEFAULT_BOUNDARY_MANIFEST
DEFAULT_RECOVERY_MANIFEST = _MODULE.DEFAULT_RECOVERY_MANIFEST
build_report = _MODULE.build_report
validate_gate = _MODULE.validate_gate


def _boundary() -> dict:
    return json.loads(DEFAULT_BOUNDARY_MANIFEST.read_text(encoding="utf-8"))


def _recovery() -> dict:
    return json.loads(DEFAULT_RECOVERY_MANIFEST.read_text(encoding="utf-8"))


def test_tracked_manifests_are_valid_and_close_ready_after_withdrawal() -> None:
    """Tracked manifests validate the explicit claim-withdrawal close path."""
    report = build_report(DEFAULT_BOUNDARY_MANIFEST, DEFAULT_RECOVERY_MANIFEST)

    assert report["status"] == "close_ready"
    assert report["close_ready"] is True
    assert report["decision"] == "close_ready_claims_withdrawn_exact_event_provenance_unavailable"
    assert report["violations"] == []
    assert report["diagnostic_summary"]["exact_collision_events"] == 241


def test_gate_rejects_public_release_bundle_as_closure_evidence() -> None:
    """Public release bundle alone must not become closure evidence."""
    boundary = _boundary()
    recovery = _recovery()
    recovery["public_release_bundle_policy"]["may_be_used_to_close_3482"] = True

    violations = validate_gate(boundary, recovery)

    assert any(
        violation.field == "recovery.public_release_bundle_policy.may_be_used_to_close_3482"
        for violation in violations
    )


def test_gate_rejects_missing_resolution_paths() -> None:
    """The recovery record must keep all valid resolution paths explicit."""
    boundary = _boundary()
    recovery = _recovery()
    recovery["remaining_resolution_paths"] = ["recover original three artifacts"]

    violations = validate_gate(boundary, recovery)

    assert any(violation.field == "recovery.remaining_resolution_paths" for violation in violations)


def test_gate_rejects_erased_exact_derived_discrepancy() -> None:
    """The exact-vs-derived discrepancy cannot be erased in either manifest."""
    boundary = _boundary()
    recovery = _recovery()
    boundary["diagnostic_summary"]["exact_collision_events"] = 0
    boundary["diagnostic_summary"]["reconciliation_violations"] = 0

    violations = validate_gate(boundary, recovery)

    assert any(
        violation.field == "boundary.diagnostic_summary.exact_collision_events"
        for violation in violations
    )
    assert any(
        violation.field == "cross_manifest.exact_collision_events" for violation in violations
    )


def test_cli_reports_close_ready_gate_json() -> None:
    """The CLI emits a machine-readable close-ready gate report."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "close_ready"
    assert report["close_ready"] is True
    assert report["violations"] == []


def test_cli_require_close_ready_passes_after_claim_withdrawal() -> None:
    """Close-readiness mode passes after affected claims are withdrawn."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--require-close-ready"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
