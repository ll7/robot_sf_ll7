"""Independent schema, budget, and admission-loss oracles for the AUD4 fix."""

from copy import deepcopy

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect
from robot_sf.analysis_workbench.audit_scan import scan_campaign

NAMED = (
    "deadlock_stall",
    "distributional_disruption",
    "force_quantiles",
    "force_sample_stats",
    "metric_values",
    "signal_metrics_evidence",
    "social_compliance",
    "social_mini_game",
)


def extreme(metrics):
    report = scan_campaign(
        [{"episode_id": "physical", "metrics": {"clearance_m": 1, **metrics}}],
        detector_ids=["extreme_measurements"],
    )
    assert report.inventory[0].readable
    return report.signals[0]


@pytest.mark.parametrize("name", NAMED)
@pytest.mark.parametrize("value", [0, 3.5, -1, True, "record", []])
def test_every_named_record_requires_mapping_or_null(name, value):
    s = extreme({name: value})
    assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert f"metrics.{name}" in s.missingness
    assert extreme({name: None}).status == "clear"
    assert extreme({name: {}}).status == "clear"


@pytest.mark.parametrize("name", ["metric_values", "signal_metrics_evidence"])
@pytest.mark.parametrize(
    "extra", [{"clearance_m": -5}, {"collision_count": 7}, {"force_max_N": 1e9}, {"unknown": 0}]
)
def test_undeclared_numeric_record_leaves_are_rejected(name, extra):
    s = extreme({name: extra})
    assert s.status == "error"
    assert f"metrics.{name}.{next(iter(extra))}" in s.missingness
    # Numeric-looking content in an explicit metadata subtree is not a measurement.
    assert extreme({name: {"extension": {"value": 1}}}).status == "clear"


@pytest.mark.parametrize(
    "record",
    [
        {"q50": 3, "q90": 2, "q95": 4},
        {"q50": 1, "q90": 3, "q95": 2},
        {"q50": 3, "q90": None, "q95": 2},
        {"q50": 1, "q99": -5},
        {"q50": 1, "q99": 2},
    ],
)
def test_quantiles_are_ordered_and_have_only_documented_quantile_names(record):
    assert extreme({"force_quantiles": record}).status == "error"
    assert extreme({"force_quantiles": {"q50": 0, "q90": 1, "q95": 2}}).status == "clear"


@pytest.mark.parametrize("raw,finite", [(0, 1), (3, 4), (100, 101)])
def test_finite_force_sample_count_cannot_exceed_raw(raw, finite):
    s = extreme({"force_sample_stats": {"raw_samples": raw, "finite_samples": finite}})
    assert s.status == "error"
    assert "metrics.force_sample_stats.finite_samples" in s.missingness
    assert (
        extreme({"force_sample_stats": {"raw_samples": finite, "finite_samples": finite}}).status
        == "clear"
    )


@pytest.mark.parametrize("name", ["max_force_N", "force_max_N", "force_mean_N", "robot_force_N"])
def test_negative_force_magnitudes_are_not_abs_normalized(name):
    s = extreme({name: -5})
    assert s.status == "flagged"
    assert s.measured["extreme"][name] == -5
    assert extreme({name: 0}).status == "clear"


def horizon(extra, steps=300, label="timeout"):
    row = {
        "episode_id": "budget",
        "steps": steps,
        "termination_reason": "max_steps" if label == "timeout" else label,
        "outcome": {"label": label},
        **extra,
    }
    return scan_campaign([row], detector_ids=["horizon_consistency"]).signals[0]


@pytest.mark.parametrize(
    "extra,budget",
    [
        ({"horizon": 500}, 500),
        ({"scenario_params": {"simulation_config": {"max_episode_steps": 400}}}, 400),
        ({"effective_budget_steps": 400}, 400),
        (
            {"horizon": 400, "scenario_params": {"simulation_config": {"max_episode_steps": 600}}},
            400,
        ),
    ],
)
def test_timeout_uses_available_budget_and_requires_exact_equality(extra, budget):
    s = horizon(extra)
    assert s.status == "flagged"
    assert s.measured["timeout_budget_steps"] == budget
    assert horizon(extra, steps=budget).status == "clear"
    assert horizon(extra, steps=budget + 1).status == "flagged"


@pytest.mark.parametrize("budget", [300, 700])
@pytest.mark.parametrize("label", ["success", "timeout"])
def test_budget_must_equal_minimum_of_runner_and_simulator(budget, label):
    extra = {
        "horizon": 500,
        "effective_budget_steps": budget,
        "scenario_params": {"simulation_config": {"max_episode_steps": 600}},
    }
    s = horizon(extra, steps=200, label=label)
    assert s.status == "flagged"
    assert "effective_budget_not_minimum_limit" in s.measured["signatures"]
    extra["effective_budget_steps"] = 500
    assert horizon(extra, steps=500, label=label).status == "clear"


@pytest.mark.parametrize(
    "field",
    ["run_horizon", "horizon_steps", "horizon", "effective_budget_steps", "simulation_config"],
)
def test_conflicting_top_level_and_scenario_limits_are_visible(field):
    top = {"max_episode_steps": 600} if field == "simulation_config" else 600
    nested = {"max_episode_steps": 400} if field == "simulation_config" else 400
    s = horizon({field: top, "scenario_params": {field: nested}}, steps=300, label="success")
    assert s.status == "flagged"
    assert "conflicting_recorded_horizon_limits" in s.measured["signatures"]
    assert s.measured["recorded_limit_values"]
    assert (
        horizon(
            {field: top, "scenario_params": {field: deepcopy(top)}}, steps=300, label="success"
        ).status
        == "clear"
    )


def test_correct_9999_scheduled_timeout_is_clear_below_campaign_default():
    # #9999 records the selected schedule in horizon/run_horizon; simulator can be larger.
    extra = {
        "horizon": 400,
        "effective_budget_steps": 400,
        "scenario_params": {"run_horizon": 400, "simulation_config": {"max_episode_steps": 600}},
    }
    s = horizon(extra, steps=400)
    assert (s.status, s.reason_code) == ("clear", "horizon_termination_consistent")
    assert s.measured["timeout_budget_steps"] == 400
    assert s.measured["signatures"] == []
    assert horizon(extra, steps=399).status == "flagged"


def rows():
    return [
        {
            "episode_id": f"{p}-{i}",
            "planner_id": p,
            "scenario_id": "same",
            "seed": 1001 + i,
            "outcome": {"label": "success"},
            "metrics": {"time_to_goal_norm": 10},
        }
        for p in ("target", "peer-a", "peer-b")
        for i in range(6)
    ]


@pytest.mark.parametrize("detector", ["outcome_incidence", "planner_cohort_shift"])
@pytest.mark.parametrize("planner", ["target", "peer-a"])
@pytest.mark.parametrize("corruption", [float("inf"), float("nan"), "duplicate", "unsupported"])
def test_target_or_control_admission_loss_gates_entire_cohort(detector, planner, corruption):
    data = rows()
    bad = next(r for r in data if r["episode_id"] == f"{planner}-5")
    if corruption == "duplicate":
        data.append(deepcopy(bad))
        dropped = 2
    elif corruption == "unsupported":
        bad["row_status"] = "fallback"
        dropped = 1
    else:
        bad["metrics"]["time_to_goal_norm"] = corruption
        dropped = 1
    report = scan_campaign(data, detector_ids=[detector])
    s = next(s for s in report.signals if s.episode_id == "target-0")
    assert (s.status, s.reason_code) == ("unavailable", "cohort_admission_incomplete")
    assert s.measured["cohort_dropped_counts"] == {planner: dropped}
    assert s.measured["cohort_dropped_rows"] == dropped
    assert s.measured["planner_admitted_sizes"][planner] == 5
    assert s.measured["planner_recorded_sizes"][planner] == 5 + dropped
    assert scan_campaign(rows(), detector_ids=[detector]).signals[0].status == "clear"


def test_dropped_rows_in_other_scenario_do_not_poison_complete_cell():
    data = rows()
    data.append(
        {
            "episode_id": "other",
            "planner_id": "target",
            "scenario_id": "other",
            "metrics": {"time_to_goal_norm": float("nan")},
        }
    )
    report = scan_campaign(data, detector_ids=["outcome_incidence", "planner_cohort_shift"])
    assert all(s.status == "clear" for s in report.signals if s.episode_id == "target-0")


def test_unassignable_invalid_row_cannot_make_cohort_clear():
    data = rows() + [{"episode_id": "unknown", "metrics": {"x": float("inf")}}]
    s = next(
        s
        for s in scan_campaign(data, detector_ids=["outcome_incidence"]).signals
        if s.episode_id == "target-0"
    )
    assert s.status == "unavailable"
    assert s.measured["cohort_dropped_counts"] == {"unassigned": 1}


def test_direct_cohort_detector_also_refuses_corrupt_peers():
    data = rows()
    data[-1]["metrics"]["time_to_goal_norm"] = float("nan")
    s = detect("planner_cohort_shift", data[0], cohort=data)
    assert (s.status, s.reason_code) == ("unavailable", "cohort_admission_incomplete")
    assert s.measured["cohort_dropped_counts"] == {"peer-b": 1}


@pytest.mark.parametrize("name", ["metric_values", "signal_metrics_evidence"])
def test_undeclared_containers_cannot_hide_numeric_measurements(name):
    s = extreme({name: {"clearance_m": {"value": -5}}})
    assert s.status == "error"
    assert f"metrics.{name}.clearance_m" in s.missingness


@pytest.mark.parametrize("field", ["state", "exclusion_reason"])
def test_signal_evidence_text_fields_cannot_be_measurements(field):
    assert extreme({"signal_metrics_evidence": {field: -5}}).status == "error"
    assert extreme({"signal_metrics_evidence": {field: "unavailable"}}).status == "clear"


@pytest.mark.parametrize("field", ["max_force_N", "force_max_N", "force_N"])
def test_flattened_negative_force_is_not_hidden_by_other_metrics(field):
    row = {"episode_id": "flat", field: -5, "metrics": {"clearance_m": 1}}
    s = scan_campaign([row], detector_ids=["extreme_measurements"]).signals[0]
    assert s.status == "flagged"
    assert s.measured["extreme"][field] == -5


@pytest.mark.parametrize("name", NAMED)
def test_flattened_named_record_scalar_is_malformed(name):
    s = scan_campaign(
        [{"episode_id": "flat", name: 1, "metrics": {"clearance_m": 1}}],
        detector_ids=["extreme_measurements"],
    ).signals[0]
    assert s.status == "error"
    assert name in s.missingness


@pytest.mark.parametrize(
    "detector",
    [
        "seed_outlier",
        "cohort_multivariate_outlier",
        "trajectory_shape_outlier",
        "common_mode_anomaly",
        "planner_disagreement",
    ],
)
def test_other_cohort_detectors_also_disclose_rejected_observations(detector):
    data = rows()
    for row in data:
        row.update(config_id="same-config", initial_state={"position": [0, 0]}, seed=1001)
    data[-1]["metrics"]["time_to_goal_norm"] = float("inf")
    # Use the affected planner for configuration-conditioned channels.
    report = scan_campaign(data, detector_ids=[detector])
    s = next(s for s in report.signals if s.episode_id == "peer-b-0")
    assert (s.status, s.reason_code) == ("unavailable", "cohort_admission_incomplete")
    assert s.measured["cohort_dropped_counts"] == {"peer-b": 1}


def test_direct_unassignable_corrupt_peer_is_not_silently_omitted():
    data = rows() + [{"episode_id": "unknown", "metrics": {"x": float("inf")}}]
    s = detect("planner_cohort_shift", data[0], cohort=data)
    assert (s.status, s.reason_code) == ("unavailable", "cohort_admission_incomplete")
    assert s.measured["cohort_dropped_counts"] == {"unassigned": 1}


@pytest.mark.parametrize("name", ["metric_values", "signal_metrics_evidence"])
def test_unknown_measurement_names_cannot_evade_validation_with_text(name):
    assert extreme({name: {"clearance_m": "-5"}}).status == "error"


@pytest.mark.parametrize("origin", ["row", "operational_metrics"])
@pytest.mark.parametrize("name", NAMED)
def test_corrupt_named_alias_cannot_hide_behind_a_valid_metric_record(origin, name):
    row = {"episode_id": "shadowed", "metrics": {"clearance_m": 1, name: {}}}
    if origin == "row":
        row[name] = 1
        path = name
    else:
        row[origin] = {name: 1}
        path = f"{origin}.{name}"
    s = scan_campaign([row], detector_ids=["extreme_measurements"]).signals[0]
    assert (s.status, s.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert path in s.missingness


@pytest.mark.parametrize("origin", ["row", "operational_metrics"])
@pytest.mark.parametrize(
    "value,expected", [(-5, "flagged"), (1e9, "flagged"), (True, "error"), ("-5", "error")]
)
def test_corrupt_force_alias_cannot_hide_behind_valid_compact_measurement(origin, value, expected):
    row = {"episode_id": "shadowed-force", "metrics": {"max_force_N": 5}}
    if origin == "row":
        row["max_force_N"] = value
    else:
        row[origin] = {"max_force_N": value}
    s = scan_campaign([row], detector_ids=["extreme_measurements"]).signals[0]
    assert s.status == expected
    path = f"{origin}.max_force_N"
    if expected == "error":
        assert path in s.missingness
    else:
        assert s.measured["extreme"][path] == value


@pytest.mark.parametrize(
    "field,value",
    [("seed", True), ("seed", float("nan")), ("initial_state", {"position": [float("nan"), 0]})],
)
def test_corrupt_grouping_key_makes_rejected_observation_unassignable(field, value):
    data = rows()
    for row in data:
        row.update(config_id="same-config", initial_state={"position": [0, 0]}, seed=1001)
    data[-1][field] = value
    report = scan_campaign(data, detector_ids=["planner_disagreement"])
    s = next(s for s in report.signals if s.episode_id == "target-0")
    assert (s.status, s.reason_code) == ("unavailable", "cohort_admission_incomplete")
    assert s.measured["cohort_dropped_counts"] == {"unassigned": 1}
