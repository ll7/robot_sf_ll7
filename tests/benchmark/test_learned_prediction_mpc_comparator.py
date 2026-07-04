"""Tests for issue #4013 learned-prediction MPC comparator contract."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.learned_prediction_mpc_comparator import (
    DIAGNOSTIC_BOUNDARY,
    build_comparator_preflight,
)

ROOT = Path(__file__).resolve().parents[2]


def test_issue_4013_comparator_configs_are_ready_diagnostic_smoke() -> None:
    """Paired configs stay pinned, diagnostic, and world-model excluded."""

    report = build_comparator_preflight(
        model_free_config=Path("configs/benchmarks/issue_4013_model_free_smoke.yaml"),
        model_based_config=Path("configs/benchmarks/issue_4013_model_based_smoke.yaml"),
        repo_root=ROOT,
    ).to_dict()

    assert report["status"] == "ready_diagnostic_smoke"
    assert report["blockers"] == []
    assert report["scenario_matrix"] == "configs/scenarios/single/francis2023_blind_corner.yaml"
    assert report["seeds"] == [4013]
    assert report["model_free_planner"] == "prediction_mpc_cv"
    assert report["model_based_planner"] == "learned_prediction_mpc_diagnostic"
    assert report["predictor_source"] == "diagnostic_untrained_smoke"
    assert report["fallback_status"] == "diagnostic_only_not_benchmark_evidence"
    assert report["claim_boundary"] == DIAGNOSTIC_BOUNDARY
    assert "dreamerv3" in report["world_model_exclusions"]


def test_issue_4013_comparator_fails_closed_when_seed_lists_diverge(tmp_path: Path) -> None:
    """Preflight reports blockers rather than accepting incomparable smoke configs."""

    model_free = tmp_path / "model_free.yaml"
    model_based = tmp_path / "model_based.yaml"
    model_free.write_text(
        """
name: model_free
scenario_matrix: configs/scenarios/single/francis2023_blind_corner.yaml
seed_policy:
  mode: fixed-list
  seeds: [1]
planners:
  - key: prediction_mpc_cv
    algo: prediction_mpc
    algo_config: configs/algos/prediction_mpc_cv.yaml
    issue_4013_role: model_free_comparator
    claim_boundary: diagnostic_smoke_only
""",
        encoding="utf-8",
    )
    model_based.write_text(
        """
name: model_based
scenario_matrix: configs/scenarios/single/francis2023_blind_corner.yaml
seed_policy:
  mode: fixed-list
  seeds: [2]
planners:
  - key: learned_prediction_mpc_diagnostic
    algo: learned_prediction_mpc
    algo_config: configs/algos/learned_prediction_mpc_issue_4013_smoke.yaml
    issue_4013_role: model_based_candidate
    claim_boundary: diagnostic_smoke_only
""",
        encoding="utf-8",
    )

    report = build_comparator_preflight(
        model_free_config=model_free,
        model_based_config=model_based,
        repo_root=ROOT,
    ).to_dict()

    assert report["status"] == "blocked"
    assert "seed_list_mismatch" in report["blockers"]
