"""Tests for the fail-closed issue #4830 campaign contract audit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.benchmark.audit_issue4830_safety_wrapper_contract import audit_campaign, write_audit

if TYPE_CHECKING:
    from pathlib import Path


def _episode(*, planner: str, wrapper_arm: str, scenario: str, seed: int) -> dict[str, object]:
    """Return a minimal camera-ready episode without normalized #3501 metrics."""
    wrapper = (
        {
            "enabled": True,
            "arm_key": "wrapper_on",
            "thresholds_source": "predeclared_fixed_no_per_planner_tuning",
            "intervention_rate": 0.0,
        }
        if wrapper_arm == "wrapper_on"
        else None
    )
    metrics: dict[str, object] = {
        "success": False,
        "collisions": 0,
        "near_misses": 0,
        "clearing_distance_min": 1.0,
        "time_to_goal_norm": 1.0,
    }
    if wrapper_arm == "wrapper_on":
        metrics["wrapper_intervention_rate"] = 0.0
    return {
        "scenario_id": scenario,
        "seed": seed,
        "algo": planner,
        "git_hash": "abc123",
        "metrics": metrics,
        "algorithm_metadata": {"safety_wrapper": wrapper},
        "event_ledger": {"schema_version": "EpisodeEventLedger.v2"},
    }


# evidence-writer-exempt: tests construct temporary campaign fixtures for the audit;
# they do not write repository evidence artifacts.
def _write_fixture(root: Path) -> None:
    """Write a complete standard-artifact fixture with two arms."""
    for relative in (
        "campaign_manifest.json",
        "manifest.json",
        "preflight.json",
        "run_meta.json",
        "reports/campaign_summary.json",
        "reports/campaign_integrity.json",
        "reports/campaign_credibility_scorecard.json",
        "reports/matrix_summary.json",
        "reports/comparability_matrix.json",
        "reports/post_campaign_stage_status.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "reports/campaign_report.md").write_text("# report\n", encoding="utf-8")
    summary = {
        "campaign": {
            "campaign_id": "issue4830_safety_wrapper_factorial_v1",
            "git_hash": "abc123",
            "campaign_execution_status": "completed",
            "evidence_status": "valid",
            "benchmark_success": True,
            "total_episodes": 2,
            "total_runs": 2,
            "successful_runs": 2,
            "unexpected_failed_runs": 0,
            "row_status_summary": {"fallback_or_degraded_rows": 0},
        }
    }
    (root / "reports/campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "reports/campaign_integrity.json").write_text(
        json.dumps({"status": "valid"}), encoding="utf-8"
    )
    (root / "reports/post_campaign_stage_status.json").write_text(
        json.dumps({"post_campaign_stage": {"status": "completed"}}), encoding="utf-8"
    )
    for planner in ("orca",):
        for arm in ("wrapper_off", "wrapper_on"):
            path = root / f"runs/{planner}__{arm}__differential_drive/episodes.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(_episode(planner=planner, wrapper_arm=arm, scenario="s1", seed=111))
                + "\n",
                encoding="utf-8",
            )


def test_audit_preserves_standard_success_but_blocks_unmapped_factorial_metrics(
    tmp_path: Path,
) -> None:
    """A completed camera-ready run cannot silently become a #3501 report."""
    root = tmp_path / "campaign"
    _write_fixture(root)

    audit = audit_campaign(root)

    assert audit["standard_campaign"]["status"] == "valid"
    assert audit["factorial_contract"]["status"] == "blocked"
    assert audit["factorial_contract"]["normalized_metric_values_record_count"] == 0
    assert "min_predicted_separation_m" in audit["factorial_contract"]["blocked_metrics"]
    assert audit["roster"]["observed_arm_keys"] == ["orca__wrapper_off", "orca__wrapper_on"]


def test_audit_writes_compact_outputs(tmp_path: Path) -> None:
    """The audit output is deterministic JSON plus a human-readable boundary note."""
    root = tmp_path / "campaign"
    _write_fixture(root)
    audit = audit_campaign(
        root,
        config_path="configs/benchmarks/example.yaml",
        config_sha256="a" * 64,
        artifact_prefix="docs/context/evidence/issue_4830_example",
        source_location="private_ops:job-13775",
    )
    paths = write_audit(audit, tmp_path / "audit")

    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot_sf.issue_4830_safety_wrapper_contract_audit.v1"
    assert payload["config_path"] == "configs/benchmarks/example.yaml"
    assert payload["config_sha256"] == "a" * 64
    assert payload["campaign_root"] == "private_ops:job-13775"
    source_manifest = payload["source_artifacts"]["campaign_manifest.json"]
    assert source_manifest["artifact_path"].startswith("private_ops:job-13775/")
    assert source_manifest["location"].startswith("private_ops:job-13775/")
    assert "does not infer missing metric semantics" in paths["readme"].read_text(encoding="utf-8")
