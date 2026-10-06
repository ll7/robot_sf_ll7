"""RRAUD4 refutations for PR #9997 at 1be4c547; each asserts the 'never clear on corrupt input' contract."""

import pytest

from robot_sf.analysis_workbench.audit_scan import scan_campaign

PLANNERS = ("bad", "good-a", "good-b")
COHORT = ("outcome_incidence", "planner_cohort_shift")


def _row(planner, i, ok=True, seed=None):
    return {
        "episode_id": f"{planner}-{i}",
        "planner_id": planner,
        "scenario_id": "same",
        "seed": 1001 + i if seed is None else seed,
        "outcome": {"label": "success" if ok else "timeout"},
        "metrics": {"time_to_goal_norm": 10 if ok else 40},
    }


def _target(rows, detector):
    report = scan_campaign(rows, detector_ids=[detector])
    return next(s for s in report.signals if s.episode_id == "bad-0")


def _drop_planner(row):
    row.pop("planner_id")


def _null_planner(row):
    row["planner_id"] = None


def _null_scenario(row):
    row["scenario_id"] = None


# F1: readable rows whose grouping key is missing/null are silently excluded from the
# cell (26 of 30 target rows); the target cohort clears on 4 rows. Empty-string, blank
# and integer planner ids are rejected and gate the cell, so this is an inconsistency.
@pytest.mark.parametrize("detector", COHORT)
@pytest.mark.parametrize("corrupt", [_drop_planner, _null_planner, _null_scenario])
def test_readable_rows_with_missing_grouping_key_do_not_shrink_cell(detector, corrupt):
    rows = [_row(p, i, ok=(p != "bad" or i < 4)) for p in PLANNERS for i in range(30)]
    for row in rows:
        if row["planner_id"] == "bad" and row["outcome"]["label"] == "timeout":
            corrupt(row)
    signal = _target(rows, detector)
    assert signal.status != "clear", (
        signal.reason_code,
        dict(signal.measured or {}).get("planner_sizes"),
    )


# F2: the target planner's 26 timeouts are replaced by repeated copies of its 4 successful
# seeds (distinct episode ids). Seed-paired controls cover 30 distinct seeds; the target
# cell covers 4. Both cohort channels report clear on the duplicated observations.
@pytest.mark.parametrize("detector", COHORT)
def test_repeated_planner_scenario_seed_observations_are_not_clear(detector):
    rows = [_row(p, i) for p in PLANNERS[1:] for i in range(30)]
    rows += [_row("bad", i, seed=1001 + (i % 4)) for i in range(30)]
    signal = _target(rows, detector)
    assert signal.status != "clear", signal.reason_code


# F3: a row whose outcome says success while termination_reason and timeout_event say
# max_steps timeout is reported horizon-consistent (no detector in a full scan flags it).
@pytest.mark.parametrize(
    "outcome",
    [
        {"label": "success", "timeout_event": True},
        {"route_complete": True, "timeout_event": True, "collision_event": False},
    ],
)
def test_success_and_timeout_contradiction_is_not_horizon_consistent(outcome):
    row = {
        "episode_id": "h",
        "steps": 400,
        "horizon": 400,
        "effective_budget_steps": 400,
        "termination_reason": "max_steps",
        "outcome": outcome,
    }
    signal = scan_campaign([row], detector_ids=["horizon_consistency"]).signals[0]
    assert signal.status != "clear", signal.measured


# F4: physically impossible values under real 0.0.7 metric names that have no declared
# bound are reported "measurements_within_declared_bounds" (pre-existing at base).
@pytest.mark.parametrize(
    "name,value",
    [
        ("avg_speed", -1.0),
        ("socnavbench_path_length", -3.0),
        ("time_to_goal_norm", -0.5),
        ("stalled_time", -2.0),
        ("near_misses", -4.0),
        ("energy", -10.0),
    ],
)
def test_impossible_value_under_unbounded_real_metric_is_not_clear(name, value):
    row = {"episode_id": "m", "metrics": {name: value}}
    signal = scan_campaign([row], detector_ids=["extreme_measurements"]).signals[0]
    assert signal.status != "clear", signal.reason_code


# P3: conflicting step-count aliases clear although conflicting limit aliases now flag.
@pytest.mark.parametrize("extra", [{"episode_steps": 300}, {"metrics": {"steps": 300}}])
def test_conflicting_step_aliases_are_not_clear(extra):
    row = {
        "episode_id": "h",
        "steps": 400,
        "horizon": 400,
        "effective_budget_steps": 400,
        "termination_reason": "max_steps",
        "outcome": {"label": "timeout"},
        **extra,
    }
    signal = scan_campaign([row], detector_ids=["horizon_consistency"]).signals[0]
    assert signal.status != "clear", signal.measured


# P3: a non-q extra key with a string value in force_quantiles is admitted.
def test_force_quantiles_extra_string_key_is_not_clear():
    row = {
        "episode_id": "f",
        "metrics": {
            "clearance_m": 1.0,
            "force_quantiles": {"q50": 1, "q90": 2, "q95": 3, "p99": "-5"},
        },
    }
    signal = scan_campaign([row], detector_ids=["extreme_measurements"]).signals[0]
    assert signal.status != "clear", signal.reason_code


# Control: PR #9999 scheduled shape stays clear when correct and flags one step early.
@pytest.mark.parametrize("steps,expected", [(400, "clear"), (399, "flagged")])
def test_pr9999_scheduled_shape_control(steps, expected):
    row = {
        "episode_id": "s",
        "steps": steps,
        "horizon": 400,
        "effective_budget_steps": 400,
        "termination_reason": "max_steps",
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": True},
        "scenario_params": {"run_horizon": 400, "simulation_config": {"max_episode_steps": 600}},
    }
    assert scan_campaign([row], detector_ids=["horizon_consistency"]).signals[0].status == expected
