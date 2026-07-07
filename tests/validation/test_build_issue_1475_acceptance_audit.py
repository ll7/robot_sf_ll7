"""Tests for the issue #1475 executable acceptance audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validation.build_issue_1475_acceptance_audit import build_audit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_issue_1475_acceptance_audit_stays_fail_closed_on_tracked_smoke() -> None:
    """Tracked evidence must not be promoted to complete issue closure."""

    report = build_audit(repo_root=REPO_ROOT)

    assert report["schema_version"] == "issue-1475-acceptance-audit.v1"
    assert report["status"] == "blocked"
    assert report["closure_call"] == "keep_open"
    assert report["smoke_gate"]["status"] == "invalid"
    assert "smoke summary missing required entries" in report["smoke_gate"]["error"]
    assert report["state_surface"]["status"] == "valid"
    assert report["state_surface"]["entry_status"] == (
        "post_4726_closure_audit_delivered__external_slurm_rerun_pending"
    )

    statuses = {item["criterion"]: item["status"] for item in report["acceptance_evidence"]}
    assert (
        statuses["Fallback/degraded rows are not counted learned-residual success evidence."]
        == "met"
    )
    assert statuses["Smoke result recorded before nominal escalation."] == "met"
    assert (
        statuses["Nominal result classified ready #1358 continuation, revise, stop."] == "not_met"
    )
    assert len(report["remaining_criteria"]) == 4
    integration_report = report["integration_report"]
    assert integration_report["status"] == "blocked"
    assert integration_report["blockers_new"] == []
    assert len(integration_report["blockers_remaining"]) == 4
    assert "Integration slice" in integration_report["fragmentation_guard_response"]
    assert any(
        item["blocker"] == "No Slurm/GPU submission in this PR."
        for item in integration_report["blockers_intentional"]
    )
    pr_evidence_by_pr = {item["pr"]: item for item in report["merged_pr_evidence"]}
    assert {"#4561", "#4661", "#4667", "#4678", "#4721"} <= set(pr_evidence_by_pr)
    assert pr_evidence_by_pr["#4561"]["closure_effect"] == "partial"
    assert pr_evidence_by_pr["#4721"]["closure_effect"] == "partial_keep_open"
    assert "integration report" in pr_evidence_by_pr["#4721"]["evidence"]


def test_issue_1475_acceptance_audit_cli_writes_json_artifact(tmp_path: Path) -> None:
    """The CLI should emit the same machine-readable audit artifact reviewers inspect."""

    output = tmp_path / "issue_1475_acceptance_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validation/build_issue_1475_acceptance_audit.py",
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
    assert payload["issue"] == 1475
    assert payload["status"] == "blocked"
    assert payload["acceptance_evidence"]
    assert payload["state_surface"]["path"] == "docs/context/issue_1475_state.yaml"
    assert payload["state_surface"]["errors"] == []
    assert payload["integration_report"]["canonical_state_surface"] == (
        "docs/context/issue_1475_state.yaml"
    )
    assert [item["pr"] for item in payload["merged_pr_evidence"]] == [
        "#4561",
        "#4661",
        "#4667",
        "#4678",
        "#4721",
        "#4726",
    ]
    assert payload["merged_pr_evidence"][-1]["closure_effect"] == "post_merge_audit_keep_open"
    assert "post-PR #4726 state update" in payload["source_thread_summary"]
    assert "Slurm-capable host" in payload["next_empirical_action"]
