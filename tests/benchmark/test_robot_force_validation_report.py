"""Diagnostic force report handles monotonicity and duplicate custody."""

import pytest

from scripts.analysis.issue_9666_robot_force_validation import analyze


def test_signed_distance_correlation_and_disagreements():
    rows = [
        {
            "scenario_id": "test",
            "seed": seed,
            "algo": "goal",
            "metrics": {
                "robot_force_impulse_total": value,
                "min_distance": 4 - value,
                "near_misses": value,
            },
        }
        for seed, value in enumerate((1, 2, 3))
    ]
    report = analyze(rows)
    first = report["correlations"][0]
    assert first["spearman_rho"] == 1
    assert first["n"] == 3
    assert len(report["largest_rank_disagreements"]) == 3
    assert report["correlations"][2]["spearman_rho"] is None
    with pytest.raises(ValueError, match="duplicate"):
        analyze(rows + rows[:1])
