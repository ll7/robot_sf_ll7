"""RVAUD regressions through public scans, with explicit physical/coverage oracles."""

from copy import deepcopy

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect
from robot_sf.analysis_workbench.audit_scan import scan_campaign


def signal(rows, detector, episode):
    report = scan_campaign(rows)
    assert all(item.readable for item in report.inventory)
    matches = [s for s in report.signals if s.detector_id == detector and s.episode_id == episode]
    assert len(matches) == 1, f"missing independent {detector} channel"
    return matches[0]


@pytest.mark.parametrize("field", ["q50", "q90", "q95"])
@pytest.mark.parametrize("value", [{"value": -1}, True, False, "-1", "corrupt", [], -1])
def test_named_force_record_validates_each_physical_quantile(field, value):
    row = {
        "episode_id": "named",
        "metrics": {"clearance_m": 1, "force_quantiles": {"q50": 1, "q90": 1, "q95": 1}},
    }
    row["metrics"]["force_quantiles"][field] = value
    s = signal([row], "extreme_measurements", "named")
    assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert f"metrics.force_quantiles.{field}" in s.missingness
    # Explicit missingness and valid zero magnitude remain admitted.
    row["metrics"]["force_quantiles"][field] = None
    assert signal([row], "extreme_measurements", "named").status == "clear"
    row["metrics"]["force_quantiles"][field] = 0
    assert signal([row], "extreme_measurements", "named").status == "clear"
    for invalid in (float("nan"), float("inf"), -float("inf")):
        row["metrics"]["force_quantiles"][field] = invalid
        s = detect("extreme_measurements", row)
        assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement")


@pytest.mark.parametrize(
    "block,path",
    [
        ("deadlock_stall", ("stall_window_count",)),
        ("force_sample_stats", ("raw_samples",)),
        ("distributional_disruption", ("cohort_metrics", "slow_speed_tier", "delay_mean_s")),
        ("metric_values", ("stop_yield_latency_s",)),
        ("social_compliance", ("metrics", "comfort_exposure_person_s", "value")),
        ("social_mini_game", ("rows", 0, "value")),
    ],
)
@pytest.mark.parametrize("value", [True, False, "corrupt", [], {"value": 1}, -1])
def test_named_diagnostics_validate_documented_numeric_fields(block, path, value):
    record = {"status": "diagnostic", "interpretation": {"arbitrary": ["metadata", True]}}
    if block == "metric_values":
        record = {}
    if block == "social_mini_game":
        record["rows"] = [{"metric": "invasiveness", "status": "available", "value": 1}]
    container = record
    for key in path[:-1]:
        if isinstance(key, str):
            container = container.setdefault(key, {})
        else:
            container = container[key]
    container[path[-1]] = value
    row = {"episode_id": "schema", "metrics": {"clearance_m": 1, block: record}}
    s = signal([row], "extreme_measurements", "schema")
    assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert "metrics." + block + "." + ".".join(map(str, path)) in s.missingness
    container[path[-1]] = None
    assert signal([row], "extreme_measurements", "schema").status == "clear"
    container[path[-1]] = 0
    assert signal([row], "extreme_measurements", "schema").status == "clear"


def test_diagnostic_container_corruption_cannot_hide_documented_fields():
    row = {
        "episode_id": "container",
        "metrics": {"clearance_m": 1, "distributional_disruption": {"cohort_metrics": []}},
    }
    s = signal([row], "extreme_measurements", "container")
    assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    row["metrics"]["distributional_disruption"] = {
        "missing_data": {"slow_speed_tier": {"status": "missing", "reason": "no samples"}},
        "cohort_metrics": {},
    }
    assert signal([row], "extreme_measurements", "container").status == "clear"
    # This diagnostic contains only strings; arbitrary extension data is not a measurement.
    row["metrics"]["signal_metrics_evidence"] = {
        "state": "unavailable",
        "exclusion_reason": "no signal",
        "extension": {"value": "text"},
    }
    assert signal([row], "extreme_measurements", "container").status == "clear"


def test_structured_scalar_bounds_preserve_signed_values_and_metadata():
    row = {"episode_id": "bounds", "metrics": {"clearance_m": 1}}
    invalid_blocks = [
        ("deadlock_stall", {"window_steps": 0}),
        ("deadlock_stall", {"progress_eps_m": -1}),
        ("force_sample_stats", {"finite_samples": 1.5}),
        ("force_sample_stats", {"valid_fraction": 1.01}),
        ("metric_values", {"completion_probability": 1.01}),
        ("distributional_disruption", {"missing_data": {"slow_speed_tier": {"support_count": -1}}}),
        ("social_compliance", {"parameters": {"comfort_radius_m": True}}),
        ("social_mini_game", {"rows": [{"metric": "invasiveness", "support_count": "1"}]}),
    ]
    for block, record in invalid_blocks:
        row["metrics"] = {"clearance_m": 1, block: record}
        s = signal([row], "extreme_measurements", "bounds")
        assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement"), block
    row["metrics"] = {
        "clearance_m": 1,
        "metric_values": {"min_predicted_separation_m": -0.5},
        "social_mini_game": {
            "rows": [
                {"metric": "path_deviation_ratio", "value": -1},
                {"metric": "diagnostic_indicator", "value": True},
            ]
        },
        "force_sample_stats": {"status": "no-pedestrians", "valid_fraction": 0},
    }
    assert signal([row], "extreme_measurements", "bounds").status == "clear"


def test_named_quantiles_also_enforce_the_force_ceiling():
    row = {
        "episode_id": "force",
        "metrics": {"force_quantiles": {"q50": 0, "q90": 101, "q95": None}},
    }
    s = signal([row], "extreme_measurements", "force")
    assert (s.status, s.reason_code) == ("flagged", "extreme_measurement")
    assert s.measured["extreme"] == {"force_quantiles.q90": 101}
    assert detect("extreme_measurements", row, config={"force_max_N": 102}).status == "clear"


@pytest.mark.parametrize(
    "limit", ["run_horizon", "simulator_max_episode_steps", "effective_budget_steps"]
)
@pytest.mark.parametrize("outcome", ["success", "collision", "timeout", "failure", "unknown"])
@pytest.mark.parametrize("other_limits_present", [True, False])
def test_every_recorded_maximum_is_checked_independently(limit, outcome, other_limits_present):
    limits = {"run_horizon": 600, "simulator_max_episode_steps": 600, "effective_budget_steps": 600}
    limits[limit] = 400
    if not other_limits_present:
        limits = {limit: 400}
    params = {key: value for key, value in limits.items() if key != "simulator_max_episode_steps"}
    if "simulator_max_episode_steps" in limits:
        params["simulation_config"] = {"max_episode_steps": limits["simulator_max_episode_steps"]}
    row = {
        "episode_id": "limit",
        "steps": 500,
        "outcome": {"label": outcome},
        "scenario_params": params,
    }
    s = signal([row], "horizon_consistency", "limit")
    assert s.status == "flagged"
    signature = "episode_steps_exceed_" + limit
    assert signature in s.measured["signatures"]
    assert s.measured["limit_comparisons"][limit]["status"] == "flagged"
    for name in {
        "run_horizon",
        "simulator_max_episode_steps",
        "effective_budget_steps",
    } - limits.keys():
        assert s.measured["limit_comparisons"][name]["status"] == "unavailable"
        assert name in s.missingness
    # Equal/early termination does not exceed any maximum (timeouts retain the early-timeout check).
    for steps in (400, 300):
        row["steps"] = steps
        control = signal([row], "horizon_consistency", "limit")
        assert not any(
            item.startswith("episode_steps_exceed_") for item in control.measured["signatures"]
        )
        if outcome in {"success", "collision", "failure"}:
            assert control.status == "clear"
    row["scenario_params"] = {}
    assert signal([row], "horizon_consistency", "limit").status == "unavailable"
    for invalid in (True, "400", -1, 1.5, {}, []):
        row["scenario_params"] = (
            {"simulation_config": {"max_episode_steps": invalid}}
            if limit == "simulator_max_episode_steps"
            else {limit: invalid}
        )
        s = signal([row], "horizon_consistency", "limit")
        assert (s.status, s.reason_code) == ("error", f"malformed_horizon_contract:{limit}")


def cohort():
    return [
        {
            "episode_id": f"{planner}-{i}",
            "planner_id": planner,
            "scenario_id": "same",
            "seed": 1001 + i,
            "outcome": {"label": "success"},
            "metrics": {"time_to_goal_norm": 10},
        }
        for planner in ("bad", "good-a", "good-b")
        for i in range(30)
    ]


@pytest.mark.parametrize("valid", [0, 4, 29])
@pytest.mark.parametrize("missing_value", [None, "corrupt", True, [], {}])
def test_incomplete_target_feature_is_unavailable_with_explicit_denominators(valid, missing_value):
    rows = cohort()
    for row in rows[valid:30]:
        row["metrics"]["time_to_goal_norm"] = deepcopy(missing_value)
    s = signal(rows, "planner_cohort_shift", "bad-0")
    assert s.status == "unavailable"
    assert s.measured["feature_sample_sizes"]["bad"]["time_to_goal_norm"] == valid
    assert s.measured["feature_missing_counts"]["bad"]["time_to_goal_norm"] == 30 - valid
    assert s.measured["planner_sizes"]["bad"] == 30
    assert s.threshold["minimum_feature_valid_fraction"] == 1.0
    assert s.measured["feature_excluded_planners"]["time_to_goal_norm"] == ["bad"]
    for planner in ("good-a", "good-b"):
        assert s.measured["feature_sample_sizes"][planner]["time_to_goal_norm"] == 30
        assert s.measured["feature_missing_counts"][planner]["time_to_goal_norm"] == 0
    rows = cohort()
    complete = signal(rows, "planner_cohort_shift", "bad-0")
    assert complete.status == "clear"
    assert complete.measured["feature_sample_sizes"]["bad"]["time_to_goal_norm"] == 30


def test_incomplete_controls_are_named_and_never_used_in_feature_medians():
    rows = cohort()
    for row in rows[30:60]:
        row["metrics"]["time_to_goal_norm"] = None
    s = signal(rows, "planner_cohort_shift", "bad-0")
    assert s.status == "clear"
    assert s.measured["peer_feature_medians"]["time_to_goal_norm"] == [10]
    assert s.measured["feature_excluded_planners"]["time_to_goal_norm"] == ["good-a"]
    assert s.measured["feature_sample_sizes"]["good-a"]["time_to_goal_norm"] == 0
    for row in rows[60:]:
        row["metrics"].pop("time_to_goal_norm")
    assert signal(rows, "planner_cohort_shift", "bad-0").status == "unavailable"
    # A separate complete feature still has an independent, disclosed comparison.
    for row in rows:
        row["metrics"]["energy"] = 20 if row["planner_id"] == "bad" else 10
    s = signal(rows, "planner_cohort_shift", "bad-0")
    assert s.status == "flagged"
    assert set(s.measured["feature_z_scores"]) == {"energy"}
    assert s.measured["feature_excluded_planners"]["time_to_goal_norm"] == ["good-a", "good-b"]
    assert s.measured["feature_sample_sizes"]["bad"]["energy"] == 30
    # Too-small controls also stay visible in the per-feature disclosure.
    rows = [r for r in rows if r["planner_id"] != "good-a"] + cohort()[30:33]
    s = signal(rows, "planner_cohort_shift", "bad-0")
    assert s.measured["planner_sizes"]["good-a"] == 3
    assert "good-a" in s.measured["feature_excluded_planners"]["energy"]
