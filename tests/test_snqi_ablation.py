"""Module test_snqi_ablation auto-generated docstring."""

from __future__ import annotations

from robot_sf.benchmark.ablation import compute_snqi_ablation


def _episodes_two_groups():
    """Episodes two groups.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Two groups (A, B) with two episodes each; metrics chosen so that
    # base ranking favors B, but removing collisions weight favors A.
    base = {
        "scenario_id": "sc-1",
        "seed": 0,
    }
    return [
        {
            **base,
            "episode_id": "A-1",
            "scenario_params": {"algo": "A"},
            "algo": "A",
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.5, "collisions": 2.0},
        },
        {
            **base,
            "episode_id": "A-2",
            "scenario_params": {"algo": "A"},
            "algo": "A",
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.5, "collisions": 2.0},
        },
        {
            **base,
            "episode_id": "B-1",
            "scenario_params": {"algo": "B"},
            "algo": "B",
            "metrics": {"success": 0.5, "time_to_goal_norm": 0.2, "collisions": 0.0},
        },
        {
            **base,
            "episode_id": "B-2",
            "scenario_params": {"algo": "B"},
            "algo": "B",
            "metrics": {"success": 0.5, "time_to_goal_norm": 0.2, "collisions": 0.0},
        },
    ]


def test_compute_snqi_ablation_rank_shifts():
    """Test compute snqi ablation rank shifts.

    Returns:
        Any: Auto-generated placeholder description.
    """
    records = _episodes_two_groups()
    weights = {"w_success": 2.0, "w_time": 1.0, "w_collisions": 2.0}
    baseline = {"collisions": {"med": 0.0, "p95": 2.0}}
    rows = compute_snqi_ablation(
        records,
        weights=weights,
        baseline=baseline,
        group_by="scenario_params.algo",
    )
    # Expect two groups
    assert len(rows) == 2
    # Base SNQI: B > A (see analysis), hence base ranks: B=1, A=2
    ranks = {r.group: r.base_rank for r in rows}
    assert ranks == {"B": 1, "A": 2}
    # Ablating collisions should flip: A improves by -1, B degrades by +1
    deltas = {r.group: r.deltas for r in rows}
    assert deltas["A"].get("w_collisions") == -1.0
    assert deltas["B"].get("w_collisions") == +1.0
