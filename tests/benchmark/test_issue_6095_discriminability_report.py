"""Focused tests for the issue #6095 discriminability report."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.benchmark.build_issue_6095_discriminability_report import (
    EXPECTED_KINEMATICS,
    EpisodeRow,
    RegimeData,
    _provenance_limitation_lines,
    bootstrap_mean_ci,
    classify_stress_floor,
)


def _episode(
    planner_key: str,
    scenario_id: str,
    seed: int,
    *,
    success: float = 0.0,
    collision: float = 0.0,
    near_misses: float = 0.0,
) -> EpisodeRow:
    """Build a minimal validated episode row for pure classification tests."""
    return EpisodeRow(
        planner_key=planner_key,
        scenario_id=scenario_id,
        seed=seed,
        success=success,
        collision=collision,
        near_misses=near_misses,
        near_miss_any=float(near_misses > 0.0),
        execution_mode="native",
        observation_level="tracked_agents_no_noise",
        model_id=None,
        horizon=100,
        dt=0.1,
    )


def _regime() -> RegimeData:
    """Build a four-scenario stress fixture covering every floor class."""
    scenarios = ("both_some", "one_some", "collision_only", "near_miss_only")
    seeds = (111, 112, 113)
    rows: dict[tuple[str, str, int], EpisodeRow] = {}
    for planner in ("orca", "ppo"):
        for scenario in scenarios:
            for seed in seeds:
                kwargs: dict[str, float] = {}
                if scenario == "both_some" and seed == 111:
                    kwargs["success"] = 1.0
                if scenario == "one_some" and planner == "ppo" and seed == 111:
                    kwargs["success"] = 1.0
                if scenario == "collision_only" and planner == "orca" and seed == 111:
                    kwargs["collision"] = 1.0
                if scenario == "near_miss_only" and planner == "ppo" and seed == 111:
                    kwargs["near_misses"] = 2.0
                rows[(planner, scenario, seed)] = _episode(planner, scenario, seed, **kwargs)
    return RegimeData(
        name="stress",
        root=Path("."),
        campaign_id="fixture",
        scenario_matrix="fixture",
        scenario_matrix_hash="fixture",
        git_commit="fixture",
        scenario_ids=scenarios,
        seeds=seeds,
        rows=rows,
        blockers=[],
        warnings=[],
        checkpoint={},
        metadata={"kinematics": EXPECTED_KINEMATICS},
    )


def test_bootstrap_mean_ci_is_deterministic_and_seed_scenario_aware() -> None:
    """The declared bootstrap returns repeatable finite bounds for a 2-D matrix."""
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    first = bootstrap_mean_ci(matrix, bootstrap_seed=6095, bootstrap_samples=200)
    second = bootstrap_mean_ci(matrix, bootstrap_seed=6095, bootstrap_samples=200)

    assert first == second
    assert first[0] == 0.5
    assert 0.0 <= first[1] <= first[2] <= 1.0


def test_classify_stress_floor_counts_both_zero_metric_discriminability() -> None:
    """Both-zero scenarios are separated from one-planner and shared successes."""
    result = classify_stress_floor(_regime(), seeds=(111, 112, 113))

    assert result["class_counts"] == {
        "both_planners_some_success": 1,
        "both_planners_zero_success": 2,
        "exactly_one_planner_some_success": 1,
    }
    assert result["both_zero_count"] == 2
    assert result["both_zero_distinguished_count"] == 2
    assert result["both_zero_distinguished_by_collision_count"] == 1
    assert result["both_zero_distinguished_by_near_miss_count"] == 1


def test_provenance_markdown_tracks_staged_receipts() -> None:
    """Human-readable provenance caveats must reflect staged receipt status."""
    receipt = {
        "status": "staged_receipt",
        "identity_matches_expected": True,
        "hash_source": "computed_file",
        "submit_safe": True,
        "load_status": "not_run",
    }

    lines = _provenance_limitation_lines({"nominal": receipt, "stress": receipt})

    rendered = "\n".join(lines)
    assert "staged" in rendered
    assert "metadata-only" not in rendered
    assert "nominal=not_run, stress=not_run" in rendered
