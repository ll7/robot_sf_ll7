"""Tests for the issue #2445 ORCA-residual progress-probe decision classifier."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import classify_orca_residual_progress_probe_issue_2445 as clf  # noqa: E402

FIXTURE_12913 = (
    REPO_ROOT
    / "docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json"
)


def _decision(report: dict) -> dict:
    return report["orca_residual_progress_probe_decision"]


def test_12913_fixture_yields_stop():
    """The real failed-closed v1 smoke (job 12913) must classify as stop."""
    summary = clf.load_summary(FIXTURE_12913)
    assert summary is not None, "12913 fixture must exist and be valid"
    report = clf.classify_decision(summary, FIXTURE_12913)
    block = _decision(report)

    assert block["decision"] == "stop"
    assert block["prior_decision_issue"] == 2408
    assert block["v1_smoke_success_rate"] == 0.0
    assert block["v1_smoke_failure_mode"] == "timeout_low_progress"
    # All four required fields are null in the fixture => named as missing.
    assert set(block["missing_required_fields"]) == set(clf.REQUIRED_SMOKE_FIELDS)
    assert "redesign" in block["reopen_condition"].lower()


def _passing_summary() -> dict:
    """A synthetic clean passing smoke: success>0, no timeout, all fields present."""
    return {
        "issue": 1475,
        "status": "completed",
        "metrics": {
            "episodes": 1,
            "success_rate": 1.0,
            "failure_mode_counts": {},
        },
        "required_smoke_evidence": {
            "residual_clipping_rate": 0.12,
            "guard_veto_rate": 0.0,
            "fallback_degraded_status": "nominal",
            "artifact_pointer_status": "present",
            "missing_required_fields": [],
        },
        "nominal_escalation_allowed": True,
    }


def test_passing_smoke_yields_continue(tmp_path):
    """A clean passing smoke with all required fields => continue."""
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_passing_summary()), encoding="utf-8")
    summary = clf.load_summary(path)
    report = clf.classify_decision(summary, path)
    block = _decision(report)

    assert block["decision"] == "continue"
    assert block["v1_smoke_success_rate"] == 1.0
    assert block["missing_required_fields"] == []
    assert block["artifact_pointer_status"] == "present"


def test_missing_artifact_fails_closed(tmp_path):
    """A missing/invalid artifact must fail closed to stop, naming the input."""
    missing = tmp_path / "does_not_exist.json"
    summary = clf.load_summary(missing)
    assert summary is None
    report = clf.classify_decision(summary, missing)
    block = _decision(report)

    assert block["decision"] == "stop"
    assert block["artifact_pointer_status"] == "missing_or_invalid"
    assert "missing or invalid" in block["rationale"].lower()
    assert str(missing) in block["rationale"]


def test_invalid_json_fails_closed(tmp_path):
    """Malformed JSON is treated as a missing/invalid artifact (fail closed)."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    summary = clf.load_summary(bad)
    assert summary is None
    report = clf.classify_decision(summary, bad)
    assert _decision(report)["decision"] == "stop"


def test_missing_required_fields_fails_closed(tmp_path):
    """A smoke with absent required fields must fail closed even if success>0."""
    data = _passing_summary()
    data["required_smoke_evidence"]["guard_veto_rate"] = None
    data["required_smoke_evidence"]["artifact_pointer_status"] = None
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    summary = clf.load_summary(path)
    report = clf.classify_decision(summary, path)
    block = _decision(report)

    assert block["decision"] == "stop"
    assert "guard_veto_rate" in block["missing_required_fields"]
    assert "artifact_pointer_status" in block["missing_required_fields"]
    assert "missing required" in block["rationale"].lower()


def test_missing_required_evidence_block_fails_closed(tmp_path):
    """An absent required_smoke_evidence block names all required fields as missing."""
    data = _passing_summary()
    del data["required_smoke_evidence"]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    summary = clf.load_summary(path)
    report = clf.classify_decision(summary, path)
    block = _decision(report)

    assert block["decision"] == "stop"
    assert set(block["missing_required_fields"]) == set(clf.REQUIRED_SMOKE_FIELDS)


@pytest.mark.parametrize("rate", [0.0])
def test_zero_success_with_timeout_is_stop_even_with_fields(tmp_path, rate):
    """success_rate=0 + timeout_low_progress => stop even if all fields present."""
    data = _passing_summary()
    data["metrics"]["success_rate"] = rate
    data["metrics"]["failure_mode_counts"] = {"timeout_low_progress": 1}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    summary = clf.load_summary(path)
    report = clf.classify_decision(summary, path)
    block = _decision(report)

    assert block["decision"] == "stop"
    assert block["v1_smoke_failure_mode"] == "timeout_low_progress"
    assert "reproduces the v0 failure pattern" in block["rationale"]
