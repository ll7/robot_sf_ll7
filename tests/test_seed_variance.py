"""TODO docstring. Document this module."""

from __future__ import annotations

from robot_sf.benchmark.seed_variance import (
    build_seed_variability_csv_rows,
    build_seed_variability_rows,
    compute_seed_variance,
)


def _make_record(group: str, seed: int, metrics: dict[str, float]) -> dict:
    """TODO docstring. Document this function.

    Args:
        group: TODO docstring.
        seed: TODO docstring.
        metrics: TODO docstring.

    Returns:
        TODO docstring.
    """
    return {
        "episode_id": f"{group}-{seed}",
        "scenario_id": group,
        "seed": seed,
        "metrics": metrics,
    }


def test_compute_seed_variance_basic():
    """TODO docstring. Document this function."""
    records = [
        _make_record("g1", 0, {"a": 1.0, "b": 2.0}),
        _make_record("g1", 1, {"a": 3.0, "b": 2.0}),
        _make_record("g2", 0, {"a": 10.0}),
    ]
    res = compute_seed_variance(records, group_by="scenario_id")
    assert "g1" in res and "g2" in res
    a_stats = res["g1"]["a"]
    # mean for a in g1: (1+3)/2 = 2
    assert abs(a_stats["mean"] - 2.0) < 1e-9
    # std with population (ddof=0): sqrt(((1-2)^2 + (3-2)^2)/2) = 1.0
    assert abs(a_stats["std"] - 1.0) < 1e-9
    # cv = std/mean = 0.5
    assert abs(a_stats["cv"] - 0.5) < 1e-9
    # metric missing for some seeds should be ignored; count equals number of finite values
    assert res["g2"]["a"]["count"] == 1.0


def test_compute_seed_variance_metric_filter():
    """TODO docstring. Document this function."""
    records = [
        _make_record("g", 0, {"a": 1.0, "b": 2.0}),
        _make_record("g", 1, {"a": 3.0, "b": 4.0}),
    ]
    res = compute_seed_variance(records, metrics=["a"], group_by="scenario_id")
    assert set(res["g"].keys()) == {"a"}


def test_build_seed_variability_rows_groups_by_scenario_and_planner():
    """Paper-facing seed export should preserve per-seed metrics and provenance."""
    records = [
        {
            "episode_id": "s1-a-111",
            "scenario_id": "s1",
            "seed": 111,
            "algo": "ppo",
            "planner_key": "ppo_candidate",
            "planner_group": "experimental",
            "benchmark_profile": "experimental",
            "kinematics": "differential_drive",
            "metrics": {"success": 1.0, "collisions": 0.0, "snqi": 0.4},
        },
        {
            "episode_id": "s1-a-222",
            "scenario_id": "s1",
            "seed": 222,
            "algo": "ppo",
            "planner_key": "ppo_candidate",
            "planner_group": "experimental",
            "benchmark_profile": "experimental",
            "kinematics": "differential_drive",
            "metrics": {"success": 0.0, "collisions": 1.0, "snqi": -0.2},
        },
    ]
    rows = build_seed_variability_rows(
        records,
        metrics=("success", "collisions", "snqi"),
        campaign_id="cid",
        config_hash="cfg123",
        git_hash="deadbeef",
        seed_policy={"mode": "fixed-list", "seed_set": "paper"},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["scenario_id"] == "s1"
    assert row["planner_key"] == "ppo_candidate"
    assert row["seed_list"] == [111, 222]
    assert row["summary"]["success"]["mean"] == 0.5
    assert row["provenance"]["campaign_id"] == "cid"


def test_build_seed_variability_csv_rows_flattens_per_seed_metrics():
    """CSV export should flatten per-seed and across-seed values for downstream paper use."""
    rows = [
        {
            "scenario_id": "s1",
            "planner_key": "ppo_candidate",
            "algo": "ppo",
            "planner_group": "experimental",
            "kinematics": "differential_drive",
            "benchmark_profile": "experimental",
            "seed_list": [111, 222],
            "per_seed": [
                {"seed": 111, "episode_count": 1, "metrics": {"success": 1.0}},
                {"seed": 222, "episode_count": 1, "metrics": {"success": 0.0}},
            ],
            "summary": {"success": {"mean": 0.5, "std": 0.5, "cv": 1.0, "count": 2.0}},
            "provenance": {
                "campaign_id": "cid",
                "config_hash": "cfg123",
                "git_hash": "deadbeef",
                "seed_policy": {"mode": "fixed-list", "seed_set": "paper"},
            },
        }
    ]
    csv_rows = build_seed_variability_csv_rows(rows, metrics=("success",))
    assert len(csv_rows) == 2
    assert csv_rows[0]["campaign_id"] == "cid"
    assert csv_rows[0]["success_across_seed_mean"] == 0.5
    assert csv_rows[0]["success_per_seed_mean"] in {0.0, 1.0}
