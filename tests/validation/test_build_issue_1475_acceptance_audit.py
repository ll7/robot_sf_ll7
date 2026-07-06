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
    assert "Slurm-capable host" in payload["next_empirical_action"]
