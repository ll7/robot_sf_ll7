"""RV10 sensitivity regressions; recorded rows and named counterfactuals only."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect
from robot_sf.analysis_workbench.audit_scan import scan_campaign


def row(i, planner="bad", outcome="success", value=0.2):
    return {
        "episode_id": f"{planner}-{i}",
        "planner_id": planner,
        "scenario_id": "same",
        "config_id": planner + "-config",
        "seed": 1001 + i,
        "outcome": {"label": outcome},
        "metrics": {"time_to_goal_norm": value},
    }


def channel(rows, detector):
    # Use the public default scan, so the pre-fix failure identifies the missing
    # independent channel rather than an unknown-detector exception.
    signals = [s for s in scan_campaign(rows).signals if s.detector_id == detector]
    assert len(signals) == len(rows), f"missing independent {detector} channel"
    return {s.episode_id: s for s in signals}


def test_failure_subgroup_has_outcome_incidence_without_external_control():
    rows = [row(i) for i in range(24)] + [row(24 + i, outcome="timeout", value=1) for i in range(6)]
    signals = channel(rows, "outcome_incidence")
    for r in rows[24:]:
        s = signals[r["episode_id"]]
        assert s.status == "flagged"
        assert s.measured["planner_rates"]["timeout"] == 0.2
        assert s.measured["planner_size"] == 30
        assert s.measured["cross_planner_median_rates"] is None
    assert all(signals[r["episode_id"]].status == "clear" for r in rows[:24])


def test_planner_uniform_failures_compare_against_scenario_not_planner_config():
    rows = [row(i, outcome="collision") for i in range(4)]
    rows += [row(i, planner=p) for p in ("good-a", "good-b") for i in range(4)]
    signals = channel(rows, "outcome_incidence")
    s = signals["bad-0"]
    assert s.status == "flagged"
    assert s.measured["planner_rates"]["collision"] == 1
    assert s.measured["cross_planner_median_rates"]["collision"] == 0
    assert s.measured["excess_rates"]["collision"] == 1
    assert signals["good-a-0"].status == "clear"
    # A completely failing single-planner cohort still has an absolute signal.
    single = detect("outcome_incidence", rows[0], cohort=rows[:4])
    assert single.status == "flagged"
    assert single.measured["cross_planner_median_rates"] is None
    # Controls from another scenario must not erase its failure incidence.
    isolated = [dict(r, scenario_id="other") for r in rows[4:]]
    s = detect("outcome_incidence", rows[0], cohort=[*rows[:4], *isolated])
    assert s.measured["planner_count"] == 1
    assert s.status == "flagged"


def test_uniform_metric_shift_has_cross_planner_median_signal():
    rows = [row(i, value=0.4) for i in range(4)]
    rows += [row(i, planner=p) for p in ("good-a", "good-b") for i in range(4)]
    signals = channel(rows, "planner_cohort_shift")
    assert signals["bad-0"].status == "flagged"
    assert signals["bad-0"].measured["planner_feature_medians"]["time_to_goal_norm"] == 0.4
    assert signals["bad-0"].measured["peer_feature_medians"]["time_to_goal_norm"] == [0.2, 0.2]
    # No external metric control is explicitly unavailable, never confirmed clear.
    assert detect("planner_cohort_shift", rows[0], cohort=rows[:4]).status == "unavailable"


@pytest.mark.parametrize("detector", ["seed_outlier", "cohort_multivariate_outlier"])
def test_mad_floor_keeps_percent_level_signal_and_rejects_micro_noise(detector):
    rows = [row(i, value=10) for i in range(4)]
    target = row(4, value=10.1)
    assert detect(detector, target, cohort=[*rows, target]).status == "flagged"
    target["metrics"]["time_to_goal_norm"] = 10.000001
    assert detect(detector, target, cohort=[*rows, target]).status == "clear"


def horizon_row(steps=400, reason="terminated", outcome="timeout"):
    r = row(0, outcome=outcome)
    r.update(steps=steps, termination_reason=reason, scenario_params={"run_horizon": 600})
    return r


def test_horizon_timeout_steps_cannot_disappear_inside_small_outcome_cohort():
    r = horizon_row()
    s = channel([r], "horizon_consistency")[r["episode_id"]]
    assert s.status == "flagged"
    assert s.reason_code == "horizon_termination_inconsistent"
    assert "timeout_before_run_horizon" in s.measured["signatures"]
    assert detect("horizon_consistency", horizon_row(600)).status == "clear"
    assert detect("horizon_consistency", horizon_row(400, "success", "success")).status == "clear"
    assert (
        detect("horizon_consistency", horizon_row(400, "collision", "collision")).status == "clear"
    )
    assert detect("horizon_consistency", horizon_row(601)).status == "flagged"
    assert (
        detect("horizon_consistency", horizon_row(600, "max_steps", "success")).status == "flagged"
    )
    assert detect("horizon_consistency", horizon_row(400, "success", "timeout")).status == "flagged"
    assert (
        detect("horizon_consistency", horizon_row(400, "unknown", "unknown")).status
        == "unavailable"
    )
    assert detect("horizon_consistency", row(1)).status == "unavailable"
    assert detect("horizon_consistency", horizon_row(1.5)).status == "error"


def test_published_unresolved_candidate_exposes_runner_simulator_horizon_mismatch():
    p = Path(__file__).parents[1] / "fixtures/analysis_workbench/aud2_horizon_0_0_7.json"
    fixture = json.loads(p.read_text())
    assert hashlib.sha256(fixture["raw_line"].encode()).hexdigest() == fixture["line_sha256"]
    r = json.loads(fixture["raw_line"])
    r["_release_arm"] = "scenario_adaptive_hybrid_orca_v2_bottleneck_yield"
    r["_source_member"] = fixture["source_member"]
    assert r["steps"] == 400
    assert r["scenario_params"]["run_horizon"] == 600
    assert r["metrics"]["deadlock_stall"]["stall_window_count"] == 0
    s = channel([r], "horizon_consistency")[r["_release_arm"] + "::" + r["episode_id"]]
    assert s.status == "flagged"
    assert s.measured["simulator_max_episode_steps"] == 400
    assert "runner_simulator_horizon_mismatch" in s.measured["signatures"]
    assert s.measured["simulator_limit_explains_early_timeout"] is True


@pytest.mark.parametrize(
    "metric", ["clearance_m", "time_to_collision_min", "ped_force_mean", "total_collision_count"]
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_physical_scalar_has_measurement_error(metric, value):
    s = detect("extreme_measurements", {"episode_id": "corrupt", "metrics": {metric: value}})
    assert s.status == "error"
    assert s.reason_code == "nonfinite_or_malformed_measurement"


def test_only_named_structured_diagnostics_may_bypass_scalar_validation():
    r = {"episode_id": "structured", "metrics": {"clearance_m": 1, "mystery": {"value": -1}}}
    s = detect("extreme_measurements", r)
    assert s.status == "error"
    assert s.reason_code == "nonfinite_or_malformed_measurement"
    # The block name, not an arbitrary Mapping, grants structured admission.
    r["metrics"].pop("mystery")
    r["metrics"]["deadlock_stall"] = {"status": "ok", "stall_window_count": 0}
    assert detect("extreme_measurements", r).status == "clear"
    r["metrics"]["deadlock_stall"] = {"stall_window_count": math.inf}
    assert detect("extreme_measurements", r).status == "error"


@pytest.mark.parametrize(
    "metric", ["clearance_m", "time_to_collision_min", "ped_force_mean", "total_collision_count"]
)
def test_boolean_physical_scalar_is_malformed(metric):
    s = detect("extreme_measurements", {"episode_id": "bool", "metrics": {metric: True}})
    assert s.status == "error"
    assert s.reason_code == "nonfinite_or_malformed_measurement"
