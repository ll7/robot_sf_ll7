"""Tests issue #4013 acceptance audit generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.analysis.audit_issue_4013_acceptance import (
    build_acceptance_audit,
    write_acceptance_audit,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence_dir(tmp_path: Path) -> Path:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    _write_json(
        evidence_dir / "comparison_report.v1.json",
        {
            "status": "diagnostic_ready",
            "evidence_tier": "diagnostic-only",
            "claim_boundary": "diagnostic matched-scenario comparison",
            "paired_seed_count": 1,
            "blockers": [],
            "fallback_degraded_rows": {
                "excluded": 0,
                "included_as_non_evidence": 0,
            },
            "roles": {
                "learned_prediction_mpc": {},
                "cv_prediction_mpc": {},
                "model_free_baseline": {},
            },
        },
    )
    _write_json(
        evidence_dir / "real_trajectory_readiness.v1.json",
        {
            "status": "blocked_manifest_contract",
            "availability": "validated",
            "blockers": [
                {
                    "code": "staging.env_unresolved_for_validated",
                    "message": "set ROBOT_SF_EXTERNAL_DATA_ROOT",
                }
            ],
            "next_action": "Stage data and rerun.",
        },
    )
    _write_json(
        evidence_dir / "training_manifest.v1.json",
        {"evidence_tier": "diagnostic-only"},
    )
    _write_json(
        evidence_dir / "training_metrics.v1.json",
        {"initial_train_loss": 0.2, "final_train_loss": 0.01},
    )
    return evidence_dir


def test_acceptance_audit_keeps_issue_partial_until_real_data_lane_runs(tmp_path: Path) -> None:
    """Diagnostic criteria are met, but real-data closure criteria stay blocked."""

    audit = build_acceptance_audit(evidence_dir=_evidence_dir(tmp_path))

    assert audit["closure_status"] == "partial"
    statuses = {item["criterion"]: item["status"] for item in audit["criteria"]}
    assert statuses["Model-based action selection runs on a smoke scenario."] == "met"
    assert (
        statuses["Real pedestrian trajectory dataset is reachable and checksum-pinned."]
        == "blocked"
    )
    assert statuses["Real-trajectory predictor training has run on validated data."] == "blocked"
    assert "checksum-pinned ETH/BIWI" in audit["next_empirical_action"]


def test_acceptance_audit_fails_closed_on_missing_comparator_role(tmp_path: Path) -> None:
    """The 3-arm comparator criterion blocks when a role is missing."""

    evidence_dir = _evidence_dir(tmp_path)
    comparison_path = evidence_dir / "comparison_report.v1.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["roles"].pop("model_free_baseline")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    audit = build_acceptance_audit(evidence_dir=evidence_dir)

    comparator_statuses = [
        item["status"]
        for item in audit["criteria"]
        if item["criterion"].startswith("Comparator smoke includes")
    ]
    assert comparator_statuses == ["blocked"]


def test_write_acceptance_audit_outputs_json_and_markdown(tmp_path: Path) -> None:
    """The writer emits both machine-readable and human-readable artifacts."""

    evidence_dir = _evidence_dir(tmp_path)
    output_json = tmp_path / "acceptance_audit.v1.json"
    output_markdown = tmp_path / "acceptance_audit.v1.md"

    audit = write_acceptance_audit(
        evidence_dir=evidence_dir,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))["schema_version"] == (
        "issue_4013.acceptance_audit.v1"
    )
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "# Issue #4013 Acceptance Audit" in markdown
    assert "Real pedestrian trajectory dataset" in markdown
    assert audit["closure_status"] == "partial"
