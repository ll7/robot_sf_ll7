"""Persisted dev-seed counterexamples for publication ordering; no rollouts."""

import argparse
import json

import pytest

from robot_sf.benchmark.artifact_publication import _check_snqi_field_consistency
from robot_sf.benchmark.camera_ready._artifacts import _write_snqi_diagnostics_artifacts
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, SnqiContractConfig
from robot_sf.benchmark.camera_ready.campaign import _build_and_write_snqi_section
from robot_sf.benchmark.metrics import snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)
from robot_sf.common.artifact_paths import get_repository_root
from scripts.tools import analyze_issue_1462_h500_failure_modes as h500
from scripts.tools import rebuild_campaign_reports_from_rows as rebuild
from scripts.tools.revalidate_benchmark_release import _stored_snqi_ordering
from scripts.validation.check_release_snqi_field_consistency import _AuditState, _check_episode


def _campaign(tmp_path):
    root = get_repository_root()
    weights_path = root / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
    baseline_path = root / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
    weights = load_weight_mapping(weights_path)
    baseline = load_baseline_mapping(baseline_path)
    # a's eligible score is -1; its excluded successes make its raw mean > b's -0.5.
    rows_by_arm = {}
    for arm, values in {"a": [0, 1, 1, 1], "b": [0.5], "excluded": [1]}.items():
        rows = []
        for index, success in enumerate(values):
            metrics = {"success": success, "collisions": 0, "curvature_mean": 0.0}
            metrics["snqi"] = snqi(metrics, weights, baseline_stats=baseline)
            row = {"scenario_id": "s", "seed": 1001 + index, "metrics": metrics}
            if (arm == "a" and index > 0) or arm == "excluded":
                row["spawn_validity"] = {"invalid_run": True, "invalid_reason": "spawn_overlap"}
            rows.append(row)
        path = tmp_path / "runs" / f"{arm}__differential_drive" / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        rows_by_arm[arm] = rows
    reports = tmp_path / "reports"
    reports.mkdir()
    cfg = CampaignConfig(
        name="mixed",
        scenario_matrix_path=tmp_path / "unused.yaml",
        planners=(),
        snqi_weights_path=weights_path,
        snqi_baseline_path=baseline_path,
        snqi_contract=SnqiContractConfig(calibration_trials=1),
    )
    entries = [
        {
            "planner": {"key": arm},
            "kinematics": "differential_drive",
            "episodes_path": str(
                tmp_path / "runs" / f"{arm}__differential_drive" / "episodes.jsonl"
            ),
        }
        for arm in rows_by_arm
    ]
    _build_and_write_snqi_section(
        cfg,
        campaign_id="mixed",
        campaign_finished_at_utc="2026-09-30T00:00:00Z",
        reports_dir=reports,
        planner_rows=[],
        run_entries=entries,
        snqi_weights=weights,
        snqi_baseline=baseline,
        warnings=[],
    )
    return rows_by_arm, weights, baseline, cfg


def test_publication_mixed_cohort_passes_and_wrong_order_fails(tmp_path):
    """The real campaign SNQI section and publication precheck use the same cohort."""
    _campaign(tmp_path)
    diagnostic_path = tmp_path / "reports/snqi_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    assert [row["planner_key"] for row in diagnostic["planner_ordering"]] == ["b", "a"]
    result = _check_snqi_field_consistency(tmp_path)
    assert result["violation_count"] == 0, result["violations"]
    for row in diagnostic["planner_ordering"]:
        row["rank"] = 3 - row["rank"]
    diagnostic_path.write_text(json.dumps(diagnostic))
    result = _check_snqi_field_consistency(tmp_path)
    assert result["violation_count"] == 1
    assert "arm ordering disagrees" in result["violations"][0]


@pytest.mark.parametrize("reader", ["archive", "revalidate", "rebuild"])
def test_recovery_orderings_exclude_invalid_rows_and_absent_arms(tmp_path, reader):
    """Independent release consumers retain b,a ordering and omit the excluded-only arm."""
    rows_by_arm, weights, baseline, cfg = _campaign(tmp_path)
    if reader == "archive":
        state = _AuditState()
        for arm, rows in rows_by_arm.items():
            for row in rows:
                _check_episode(
                    row, arm=arm, source="fixture", weights=weights, baseline=baseline, state=state
                )
        means = {
            arm: value / state.per_arm_field_count[arm]
            for arm, value in state.per_arm_field_sum.items()
        }
        assert sorted(means, key=lambda arm: -means[arm]) == ["b", "a"]
        assert state.snqi_recomputed_rows == 6
        assert not state.violations
    elif reader == "revalidate":
        assert [row["planner_key"] for row in _stored_snqi_ordering(tmp_path)] == ["b", "a"]
    else:
        rebuild._reconcile_snqi_diagnostics(tmp_path, cfg)
        diagnostic = json.loads((tmp_path / "reports/snqi_diagnostics.json").read_text())
        assert [row["planner_key"] for row in diagnostic["planner_ordering"]] == ["b", "a"]
        assert diagnostic["score_basis_reconciliation"]["verified_episode_rows"] == 6


def test_excluded_rows_still_fail_snqi_formula_audit(tmp_path):
    """Excluded metrics may not evade the all-row formula check."""
    _campaign(tmp_path)
    path = tmp_path / "runs/excluded__differential_drive/episodes.jsonl"
    row = json.loads(path.read_text())
    row["metrics"]["snqi"] += 5
    path.write_text(json.dumps(row) + "\n")
    result = _check_snqi_field_consistency(tmp_path)
    assert result["violation_count"] == 1
    assert "curvature-aware recompute" in result["violations"][0]


def test_exclusion_markdown_has_readable_reason_list(tmp_path):
    """Reason counts remain legible instead of exposing a Python mapping repr."""
    _, md, _ = _write_snqi_diagnostics_artifacts(
        tmp_path,
        {
            "evidence_cohort": {
                "exclusion_reasons": {"invalid_or_unmeasured_spawn": 4, "foresight_ineligible": 2}
            }
        },
    )
    body = md.read_text()
    assert "  - invalid or unmeasured spawn: 4" in body
    assert "  - foresight ineligible: 2" in body


def test_h500_skips_zero_episode_nan_rows(tmp_path):
    """An excluded-only arm must not lower the measured success mean."""
    row = dict.fromkeys(
        (
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "comfort_exposure_mean",
            "snqi_mean",
        ),
        "0",
    )
    row.update(scenario_id="s", planner_key="orca", episodes="1", success_mean="1")
    excluded = {**row, "planner_key": "excluded", "episodes": "0", "success_mean": "nan"}
    args = argparse.Namespace(
        unsolved_success_max=0.05,
        candidate_delta_min=0.1,
        broad_success_min=0.8,
        seed_range_min=0.2,
    )
    difficulty, _ = h500._scenario_rows(
        [row, excluded], {"orca": "core", "excluded": "core"}, {}, {}, args
    )
    assert difficulty[0]["success_mean_all"] == 1.0
    assert difficulty[0]["planner_count"] == 1
    assert h500._scenario_rows([excluded], {}, {}, {}, args) == ([], [])
