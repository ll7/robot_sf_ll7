"""Tests for issue #4016 distributional-RL diagnostic comparison reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.analysis.compare_distributional_rl_issue_4016 import build_report_from_config

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(*, risk_objective: str, fallback_or_degraded: bool = False) -> dict[str, object]:
    return {
        "policy_id": f"qr_dqn_{risk_objective}",
        "algorithm": "qr_dqn",
        "evidence_tier": "diagnostic-only",
        "claim_boundary": "risk-selection diagnostic only; not benchmark evidence",
        "risk_objective": risk_objective,
        "risk_alpha": 0.2,
        "seed": 7,
        "total_timesteps": 128,
        "checkpoint_path": "output/models/distributional_rl/issue_4016/qr_dqn.pt",
        "fallback_or_degraded": fallback_or_degraded,
        "metrics": {
            "success_rate": 0.5,
            "collision_rate": 0.25 if risk_objective == "mean" else 0.0,
            "near_miss_rate": 0.5,
            "mean_min_clearance": 0.8 if risk_objective == "mean" else 0.9,
            "mean_path_efficiency": 0.7,
        },
        "risk_selection_diagnostics": {
            "record_count": 4,
            "mean_selected_count": 3 if risk_objective == "mean" else 1,
            "risk_selected_count": 1 if risk_objective == "mean" else 3,
            "mean_risk_disagreement_count": 2,
        },
    }


def _write_config(tmp_path: Path, *, risk_fallback: bool = False) -> Path:
    mean_path = tmp_path / "mean.json"
    risk_path = tmp_path / "risk.json"
    mean_path.write_text(json.dumps(_manifest(risk_objective="mean")), encoding="utf-8")
    risk_path.write_text(
        json.dumps(_manifest(risk_objective="cvar_lower", fallback_or_degraded=risk_fallback)),
        encoding="utf-8",
    )
    config = {
        "schema_version": "issue_4016.distributional_rl_risk_comparison.v1",
        "issue": 4016,
        "evidence_tier": "diagnostic-only",
        "claim_boundary": "risk-selection diagnostic only; not benchmark evidence",
        "mean_manifest": str(mean_path),
        "risk_manifest": str(risk_path),
        "output_json": str(tmp_path / "report.json"),
        "output_markdown": str(tmp_path / "report.md"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path


def test_distributional_rl_comparison_builds_diagnostic_report(tmp_path: Path) -> None:
    """Matched mean/CVaR manifests produce a diagnostic-only comparison report."""

    report = build_report_from_config(_write_config(tmp_path))

    assert report["schema_version"] == "issue_4016.distributional_rl_risk_comparison.v1"
    assert report["evidence_tier"] == "diagnostic-only"
    assert report["effect"]["comparison_status"] == "valid_diagnostic"
    assert report["effect"]["benchmark_safety_claim"] is False
    assert report["effect"]["metric_deltas"]["collision_rate_delta"] == pytest.approx(-0.25)
    assert report["effect"]["metric_deltas"]["mean_min_clearance_delta"] == pytest.approx(0.1)
    assert report["fallback_degraded_rows"] == {
        "excluded": 0,
        "included_as_non_evidence": 0,
    }
    assert (tmp_path / "report.json").exists()
    assert "diagnostic-only" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_distributional_rl_comparison_excludes_degraded_rows(tmp_path: Path) -> None:
    """Fallback/degraded manifests block diagnostic validity rather than count as evidence."""

    report = build_report_from_config(_write_config(tmp_path, risk_fallback=True))

    assert report["effect"]["comparison_status"] == "blocked"
    assert "risk_fallback_or_degraded_excluded" in report["blockers"]
    assert report["fallback_degraded_rows"] == {
        "excluded": 1,
        "included_as_non_evidence": 0,
    }
