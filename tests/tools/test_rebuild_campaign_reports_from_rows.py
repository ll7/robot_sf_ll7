"""Tests for the frozen-row publication reconstruction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.metrics import snqi as curvature_aware_snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)
from scripts.tools import rebuild_campaign_reports_from_rows as rebuild

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
BASELINE = ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"


def _write_episode(
    campaign_root: Path,
    *,
    arm: str,
    episode_id: str,
    metrics: dict[str, float],
    runtime_commit: str = "execution-commit",
    boundary_note: str | None = None,
) -> None:
    arm_dir = campaign_root / "runs" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "episode_id": episode_id,
        "metrics": metrics,
        "event_ledger": {
            "software_commit": runtime_commit,
            "exact_events": {
                "goal_reached": boundary_note is not None,
                "timeout": boundary_note is not None,
            },
        },
    }
    if boundary_note is not None:
        row["goal_timeout_boundary_note"] = boundary_note
    with (arm_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _metrics(*, curvature_mean: float) -> dict[str, float]:
    weights = load_weight_mapping(WEIGHTS)
    baseline = load_baseline_mapping(BASELINE)
    values = {
        "success": 1.0,
        "time_to_goal_norm": 1.0,
        "collisions": 0.0,
        "near_misses": 0.0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0.0,
        "jerk_mean": 0.0,
        "curvature_mean": curvature_mean,
    }
    values["snqi"] = curvature_aware_snqi(values, weights, baseline_stats=baseline)
    return values


def test_record_row_reconstruction_provenance_names_both_commit_roles(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    (campaign_root / "run_meta.json").write_text(
        json.dumps({"repo": {"commit": "publication-commit"}}),
        encoding="utf-8",
    )
    _write_episode(
        campaign_root,
        arm="goal__differential_drive",
        episode_id="ep-1",
        metrics=_metrics(curvature_mean=0.1),
        boundary_note="Frozen row retained but excluded from timing-boundary interpretation.",
    )

    rebuild._record_row_reconstruction_provenance(campaign_root)

    run_meta = json.loads((campaign_root / "run_meta.json").read_text(encoding="utf-8"))
    reconciliation = run_meta["commit_reconciliation"]
    assert reconciliation["status"] == "explained"
    assert reconciliation["execution_commit"] == "execution-commit"
    assert reconciliation["runtime_commits"] == ["execution-commit"]
    assert reconciliation["publication_commit"] == "publication-commit"
    assert set(reconciliation["roles"]) == {"execution_commit", "publication_commit"}
    assert run_meta["goal_timeout_boundary"] == {
        "annotated_rows": 1,
        "unresolved_rows": 0,
        "policy": (
            "Frozen rows lacking a reached-goal step must carry an explicit note and are excluded "
            "from timing-boundary interpretation; no timing evidence is fabricated."
        ),
    }


def test_reconcile_snqi_diagnostics_uses_curvature_aware_stored_field(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    reports_dir = campaign_root / "reports"
    reports_dir.mkdir(parents=True)
    baseline = load_baseline_mapping(BASELINE)
    _write_episode(
        campaign_root,
        arm="smooth__differential_drive",
        episode_id="smooth",
        metrics=_metrics(curvature_mean=baseline["curvature_mean"]["med"]),
    )
    _write_episode(
        campaign_root,
        arm="curved__differential_drive",
        episode_id="curved",
        metrics=_metrics(curvature_mean=baseline["curvature_mean"]["p95"]),
    )
    (reports_dir / "snqi_diagnostics.json").write_text(
        json.dumps({"planner_ordering": []}),
        encoding="utf-8",
    )

    rebuild._reconcile_snqi_diagnostics(
        campaign_root,
        SimpleNamespace(snqi_weights_path=WEIGHTS, snqi_baseline_path=BASELINE),
    )

    diagnostics = json.loads((reports_dir / "snqi_diagnostics.json").read_text(encoding="utf-8"))
    assert [row["planner_key"] for row in diagnostics["planner_ordering"]] == [
        "smooth",
        "curved",
    ]
    reconciliation = diagnostics["score_basis_reconciliation"]
    assert reconciliation["verified_episode_rows"] == 2
    assert reconciliation["effective_weights"]["w_curvature"] == 1.0
    assert "omits w_curvature" in reconciliation["declared_weights_labeling_disposition"]


def test_reconcile_snqi_diagnostics_rejects_changed_stored_field(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    reports_dir = campaign_root / "reports"
    reports_dir.mkdir(parents=True)
    metrics = _metrics(curvature_mean=0.1)
    metrics["snqi"] += 0.5
    _write_episode(
        campaign_root,
        arm="goal__differential_drive",
        episode_id="ep-1",
        metrics=metrics,
    )
    (reports_dir / "snqi_diagnostics.json").write_text(
        json.dumps({"planner_ordering": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stored SNQI does not match"):
        rebuild._reconcile_snqi_diagnostics(
            campaign_root,
            SimpleNamespace(snqi_weights_path=WEIGHTS, snqi_baseline_path=BASELINE),
        )
