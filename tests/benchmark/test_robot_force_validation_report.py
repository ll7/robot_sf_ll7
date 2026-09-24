"""Diagnostic force report handles monotonicity and duplicate custody."""

import pytest

from scripts.analysis.issue_9666_robot_force_validation import analyze, trace_evidence


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


def test_null_metrics_and_zero_exposure_are_not_invented():
    rows = [
        {
            "scenario_id": "empty",
            "seed": seed,
            "algo": "goal",
            "metrics": {
                "robot_force_impulse_total": impulse,
                "min_distance": distance,
                "robot_force_exposed_ped_count": 0,
            },
        }
        for seed, (impulse, distance) in enumerate(((None, 2), (0, None), (0, 2)))
    ]
    report = analyze(rows)
    assert report["correlations"][0]["n"] == 1
    assert report["correlations"][0]["spearman_rho"] is None
    assert len(report["largest_rank_disagreements"]) == 1
    assert report["largest_rank_disagreements"][0]["observed_pattern"] == "no_pedestrians_exposed"


def test_trace_duration_distinguishes_pair_exposure_from_elapsed_time():
    row = {
        "scenario_params": {"run_dt": 0.1},
        "termination_reason": "collision",
        "metrics": {
            "robot_force_samples": [
                {"forces": [[3, 4], [0, 2]]},
                {"forces": [[0, 0], [0, 2]]},
                {"forces": [[0, 0], [0, 0]]},
            ]
        },
    }
    summary = trace_evidence(row)
    assert summary["duration_s"] == pytest.approx(0.3)
    assert summary["force_active_duration_s"] == pytest.approx(0.2)
    assert summary["force_active_pedestrian_seconds"] == pytest.approx(0.3)
    assert summary["max_concurrently_exposed_pedestrians"] == 2
    assert summary["termination_reason"] == "collision"


def test_serialized_nested_human_discomfort_is_used():
    rows = [
        {
            "scenario_id": "proxy",
            "seed": i,
            "algo": "goal",
            "metrics": {
                "robot_force_impulse_total": i,
                "min_distance": 4 - i,
                "human_interaction_proxy": {
                    "canonical_reductions": {"human_discomfort_exposure_m_s": 2 * i}
                },
            },
        }
        for i in range(3)
    ]
    report = analyze(rows)
    correlation = report["correlations"][2]
    assert correlation["n"] == 3
    assert correlation["spearman_rho"] == 1
    assert report["largest_rank_disagreements"][0]["metrics"]["human_discomfort_exposure_m_s"] == 0
