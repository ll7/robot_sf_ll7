"""AUD5 public-API integrity, producer-bound and null-calibration regressions."""

import random
from itertools import combinations

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect
from robot_sf.analysis_workbench.audit_scan import scan_campaign


def cohort(draw=0, spread=0.1, planners=3, shift=0.0):
    """Independent same-distribution draws, paired by development seed."""
    rng = random.Random(draw)
    return [
        {
            "episode_id": f"{scenario}-p{p}-{i}",
            "planner_id": f"p{p}",
            "scenario_id": scenario,
            "seed": 1001 + i,
            "outcome": {"label": "success"},
            "metrics": {
                "avg_speed": 0.85 * (1 + rng.uniform(-spread, spread) + (shift if p == 0 else 0)),
                "time_to_goal_norm": 0.45 * (1 + rng.uniform(-spread, spread)),
            },
        }
        for scenario in ("s1", "s2")
        for p in range(planners)
        for i in range(30)
    ]


def cell_signals(rows):
    """One public detect call per cell; every peer is validated by detect."""
    return [detect("planner_cohort_shift", row, cohort=rows) for row in rows if row["seed"] == 1001]


@pytest.mark.parametrize("planners", [3, 5, 8])
@pytest.mark.parametrize("spread", [0.02, 0.1, 0.4])
def test_same_distribution_cell_false_positive_rate_is_at_most_five_percent(planners, spread):
    signals = [s for draw in range(10) for s in cell_signals(cohort(draw, spread, planners))]
    assert all(s.status in {"clear", "flagged"} for s in signals)
    assert sum(s.status == "flagged" for s in signals) / len(signals) <= 0.05


@pytest.mark.parametrize("planners", [3, 5, 8])
@pytest.mark.parametrize("spread", [0.02, 0.1, 0.4])
def test_one_shifted_planner_flags_without_contaminating_other_planners(planners, spread):
    signals = cell_signals(cohort(7, spread, planners, shift=0.5))
    assert all(
        s.status == ("flagged" if s.measured["planner_id"] == "p0" else "clear") for s in signals
    )
    for s in signals:
        assert s.threshold["family_alpha"] == 0.05
        assert s.threshold["comparison_count"] == planners * (planners - 1)  # two features


@pytest.mark.parametrize("detector", ["outcome_incidence", "planner_cohort_shift"])
@pytest.mark.parametrize("field", ["planner_id", "scenario_id"])
def test_readable_unassigned_rows_are_counted_in_scan_and_direct_detection(detector, field):
    rows = cohort()
    for row in rows[:26]:
        row.pop(field)
    report = scan_campaign(rows, detector_ids=[detector])
    s = next(s for s in report.signals if s.episode_id == "s1-p0-26")
    assert (s.status, s.reason_code) == ("unavailable", "cohort_admission_incomplete")
    assert s.measured["cohort_dropped_counts"] == {"unassigned": 26}
    direct = detect(detector, rows[26], cohort=rows)
    assert direct.status == "unavailable"
    assert direct.measured["cohort_dropped_counts"] == {"unassigned": 26}


@pytest.mark.parametrize("detector", ["outcome_incidence", "planner_cohort_shift"])
@pytest.mark.parametrize("defect", ["repeated", "absent", "different", "missing_seed"])
def test_seed_coverage_is_required_with_exact_accounting(detector, defect):
    rows = cohort()[:90]
    if defect == "absent":
        rows = rows[:4] + rows[30:]
    elif defect == "repeated":
        for row in rows[:30]:
            row["seed"] = 1001 + int(row["episode_id"].rsplit("-", 1)[1]) % 4
    elif defect == "different":
        rows[29]["seed"] = 9999
    else:
        rows[29].pop("seed")
    s = detect(detector, rows[0], cohort=rows)
    assert (s.status, s.reason_code) == ("unavailable", "cohort_seed_coverage_incomplete")
    counts = s.measured["planner_seed_counts"]
    assert counts["p1"]["unique_seeds"] == 30
    assert counts["p0"]["unique_seeds"] == (
        4 if defect in {"repeated", "absent"} else 29 if defect == "missing_seed" else 30
    )
    assert counts["p0"]["repeated_seed_rows"] == (26 if defect == "repeated" else 0)


@pytest.mark.parametrize(
    "states", list(combinations(["route_complete", "timeout_event", "collision_event"], 2))
)
@pytest.mark.parametrize("reason", [None, "success", "max_steps", "collision", "other"])
def test_multiple_terminal_states_flag_independently_of_reason(states, reason):
    row = {
        "episode_id": "terminal",
        "steps": 400,
        "horizon": 400,
        "outcome": dict.fromkeys(states, True),
        "termination_reason": reason,
    }
    s = scan_campaign([row], detector_ids=["horizon_consistency"]).signals[0]
    assert s.status == "flagged"
    assert "multiple_terminal_outcomes" in s.measured["signatures"]


@pytest.mark.parametrize(
    "metrics",
    [
        {"avg_speed": -0.1},
        {"avg_speed": 2.1},
        {"path_length": -1},
        {"success_path_length": -1},
        {"socnavbench_path_length": -1},
        {"time_to_goal_norm": 1.01},
        {"time_to_goal_norm_success_only": -0.1},
        {"path_efficiency": 1.01},
        {"min_separation_corrupted_m": -0.1},
        {"stalled_time": -1},
        {"near_misses": -1},
        {"energy": -1},
        {"force_exceed_events": -1},
        {"signal_metrics_denominator": -1},
    ],
)
def test_real_producer_scalar_bounds_flag_impossible_values(metrics):
    s = detect("extreme_measurements", {"episode_id": "metric", "metrics": metrics})
    assert s.status == "flagged"
    assert s.measured["extreme"] == metrics


def test_valid_producer_boundaries_signed_scores_and_declared_speed_cap_clear():
    row = {
        "episode_id": "valid",
        "metrics": {
            "avg_speed": 2,
            "time_to_goal_norm": 1,
            "path_efficiency": 0,
            "energy": 0,
            "near_misses": 0,
            "snqi": -5,
            "clear_mota": -1,
        },
    }
    assert detect("extreme_measurements", row).status == "clear"
    row["scenario_params"] = {"robot_config": {"max_linear_speed": 3}}
    row["metrics"]["avg_speed"] = 2.5
    assert detect("extreme_measurements", row).status == "clear"
    row["scenario_params"]["robot_config"]["max_linear_speed"] = 2
    assert detect("extreme_measurements", row).status == "flagged"


@pytest.mark.parametrize(
    "name", ["mystery", "mystery_force", "mystery_distance", "mystery_collision"]
)
def test_unknown_numeric_metrics_are_explicitly_unavailable(name):
    s = detect("extreme_measurements", {"episode_id": "unknown", "metrics": {name: 1, "energy": 1}})
    assert (s.status, s.reason_code) == ("unavailable", "bounds_not_declared")
    assert s.measured["unbounded_metric_names"] == [name]
    assert s.missingness == (f"metrics.{name}.bounds",)


@pytest.mark.parametrize("extra", [{"p99": "5"}, {"q99": 5}, {"q50": "1"}, {"metadata": "bad"}])
def test_force_quantiles_use_only_fixed_numeric_quantiles(extra):
    row = {
        "episode_id": "force",
        "metrics": {"force_quantiles": {"q50": 1, "q90": 2, "q95": 3, **extra}},
    }
    assert detect("extreme_measurements", row).status == "error"


def test_step_alias_conflict_keeps_all_recorded_paths():
    row = {
        "episode_id": "steps",
        "steps": 400,
        "episode_steps": 400,
        "metrics": {"steps": 300, "episode_steps": 400},
        "horizon": 400,
        "outcome": {"label": "success"},
    }
    s = detect("horizon_consistency", row)
    assert s.status == "flagged"
    assert "conflicting_recorded_step_counts" in s.measured["signatures"]
    assert s.measured["recorded_step_values"] == {
        "row.steps": 400,
        "row.episode_steps": 400,
        "metrics.steps": 300,
        "metrics.episode_steps": 400,
    }


def test_actual_producer_scalar_names_have_declared_bounds():
    from robot_sf.benchmark.metrics import METRIC_NAMES

    # Only the producer's declared records are excluded from this scalar check.
    records = {"distributional_disruption", "force_sample_stats", "signal_metrics_evidence"}
    for name in METRIC_NAMES:
        if name in records:
            continue
        row = {"episode_id": "producer", "metrics": {"energy": 0, name: 0}}
        signal = detect("extreme_measurements", row)
        assert signal.status == "clear", (name, signal.reason_code, signal.measured)


def test_real_sha_checked_release_rows_keep_physical_bounds_clear():
    import hashlib
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/analysis_workbench/aud_release_0_0_7.json"
    )
    for sample in json.loads(fixture.read_text())["samples"]:
        assert hashlib.sha256(sample["raw_line"].encode()).hexdigest() == sample["line_sha256"]
        row = json.loads(sample["raw_line"])
        row["_source_member"] = sample["member"]
        row["_release_arm"] = sample["member"].split("/")[2].split("__")[0]
        report = scan_campaign(
            [row],
            detector_ids=["extreme_measurements", "horizon_consistency", "telemetry_integrity"],
        )
        assert all(signal.status == "clear" for signal in report.signals), sample["kind"]
