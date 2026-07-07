"""Tests issue #1358 executable parent acceptance audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validation.build_issue_1358_acceptance_audit import DEFAULT_OUTPUT, build_audit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_issue_1358_parent_audit_stays_fail_closed_on_child_blocker() -> None:
    """Tracked parent evidence must not promote closure before #1475 evidence exists."""

    report = build_audit(repo_root=REPO_ROOT)

    assert report["schema_version"] == "issue-1358-acceptance-audit.v1"
    assert report["issue"] == 1358
    assert report["status"] == "blocked"
    assert report["closure_call"] == "keep_open"
    assert report["readiness"]["overall_status"] == "blocked_on_followup"
    assert report["readiness"]["integration_status"] == "local_handoff_ready_parent_blocked"
    assert report["issue_1475_audit"]["status"] == "blocked"
    assert report["state_surface"]["status"] == "valid"
    child_state = report["issue_1475_audit"]["state_surface"]
    assert child_state["status"] == "valid"
    assert child_state["path"] == "docs/context/issue_1475_state.yaml"
    assert "latest_recorded_at_utc" not in child_state
    assert "entry_status" not in child_state

    statuses = {item["criterion"]: item["status"] for item in report["acceptance_evidence"]}
    assert (
        statuses["Candidate design records exact observation additions and residual action bounds."]
        == "met"
    )
    assert (
        statuses[
            "Parent stays open until Issue #1475 classifies lane continue/revise/stop durable evidence."
        ]
        == "met"
    )
    assert statuses["Trained checkpoint durable lineage explicit artifact pointer."] == "not_met"
    assert len(report["remaining_criteria"]) == 4


def test_issue_1358_parent_audit_cli_writes_json_artifact(tmp_path: Path) -> None:
    """CLI emits machine-readable parent audit artifact for reviewers."""

    output = tmp_path / "issue_1358_acceptance_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validation/build_issue_1358_acceptance_audit.py",
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(output),
            "--write",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["issue"] == 1358
    assert payload["status"] == "blocked"
    assert payload["closure_call"] == "keep_open"
    assert payload["state_surface"]["path"] == "docs/context/issue_1358_state.yaml"
    assert payload["state_surface"]["errors"] == []
    assert "Slurm-capable lane" in payload["next_empirical_action"]


def test_issue_1358_tracked_audit_artifact_matches_builder() -> None:
    """Tracked parent evidence must match current merged child-state inputs."""
    tracked_payload = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))

    assert tracked_payload == build_audit(repo_root=REPO_ROOT)
