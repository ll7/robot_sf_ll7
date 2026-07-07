"""Tests issue #4016 acceptance audit generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis.audit_issue_4016_acceptance import (
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
        evidence_dir / "summary.json",
        {
            "schema_version": "issue_4016.smoke_manifest_materialization.v1",
            "issue": 4016,
            "evidence_tier": "diagnostic-only",
            "claim_boundary": "risk-selection diagnostic only; not benchmark evidence",
            "fallback_or_degraded": False,
        },
    )
    base_manifest = {
        "policy_id": "qr_dqn_issue_4016_mean",
        "algorithm": "qr_dqn",
        "evidence_tier": "diagnostic-only",
        "claim_boundary": "risk-selection diagnostic only; not benchmark evidence",
        "risk_alpha": 0.2,
        "seed": 4016,
        "total_timesteps": 128,
        "checkpoint_path": "output/models/distributional_rl/issue_4016/qr_dqn_issue_4016_smoke.pt",
        "fallback_or_degraded": False,
        "metrics": {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "near_miss_rate": 0.0,
            "mean_min_clearance": 1.0,
            "mean_path_efficiency": 0.5,
        },
    }
    mean = dict(base_manifest, risk_objective="mean")
    cvar = dict(base_manifest, policy_id="qr_dqn_issue_4016_cvar", risk_objective="cvar_lower")
    _write_json(evidence_dir / "qr_dqn_mean_manifest.json", mean)
    _write_json(evidence_dir / "qr_dqn_cvar_manifest.json", cvar)
    _write_json(
        evidence_dir / "distributional_rl_risk_comparison.json",
        {
            "evidence_tier": "diagnostic-only",
            "effect": {
                "benchmark_safety_claim": False,
                "comparison_status": "valid_diagnostic",
            },
            "fallback_degraded_rows": {
                "excluded": 0,
                "included_as_non_evidence": 0,
            },
        },
    )
    return evidence_dir


def test_acceptance_audit_keeps_measured_metrics_partial(tmp_path: Path) -> None:
    """Smoke metrics keep the closure audit partial until measured evidence exists."""
    audit = build_acceptance_audit(evidence_dir=_evidence_dir(tmp_path))

    assert audit["closure_status"] == "partial"
    statuses = {item["criterion"]: item["status"] for item in audit["criteria"]}
    measured_metric_statuses = [
        status for criterion, status in statuses.items() if criterion.startswith("Reports include")
    ]
    assert measured_metric_statuses == ["partial"]
    assert any("benchmark-runner measured comparison" in blocker for blocker in audit["blockers"])


def test_acceptance_audit_accepts_benchmark_runner_measured_metrics(tmp_path: Path) -> None:
    """Measured benchmark-runner manifests satisfy the remaining metric criterion."""
    evidence_dir = _evidence_dir(tmp_path)
    for filename in ("qr_dqn_mean_manifest.json", "qr_dqn_cvar_manifest.json"):
        path = evidence_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["benchmark_runner_measured"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

    audit = build_acceptance_audit(evidence_dir=evidence_dir)

    assert audit["closure_status"] == "complete"
    statuses = {item["criterion"]: item["status"] for item in audit["criteria"]}
    measured_metric_statuses = [
        status for criterion, status in statuses.items() if criterion.startswith("Reports include")
    ]
    assert measured_metric_statuses == ["met"]
    assert audit["blockers"] == []


def test_acceptance_audit_fails_closed_on_degraded_comparison(tmp_path: Path) -> None:
    """Fallback/degraded comparison rows block closure evidence."""
    evidence_dir = _evidence_dir(tmp_path)
    comparison_path = evidence_dir / "distributional_rl_risk_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["fallback_degraded_rows"]["included_as_non_evidence"] = 1
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    audit = build_acceptance_audit(evidence_dir=evidence_dir)

    degraded_statuses = [
        item["status"]
        for item in audit["criteria"]
        if item["criterion"].startswith("Fallback/degraded rows")
    ]
    assert degraded_statuses == ["blocked"]


def test_write_acceptance_audit_outputs_json_and_markdown(tmp_path: Path) -> None:
    """The writer emits both machine and human-readable audit artifacts."""
    evidence_dir = _evidence_dir(tmp_path)
    output_json = tmp_path / "acceptance_audit.json"
    output_markdown = tmp_path / "acceptance_audit.md"

    audit = write_acceptance_audit(
        evidence_dir=evidence_dir,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))["schema_version"] == (
        "issue_4016.acceptance_audit.v1"
    )
    assert "# Issue #4016 Acceptance Audit" in output_markdown.read_text(encoding="utf-8")
    assert audit["closure_status"] == "partial"


def test_acceptance_audit_requires_json_objects(tmp_path: Path) -> None:
    """Malformed evidence files fail closed instead of producing an audit."""
    evidence_dir = _evidence_dir(tmp_path)
    (evidence_dir / "summary.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        build_acceptance_audit(evidence_dir=evidence_dir)
