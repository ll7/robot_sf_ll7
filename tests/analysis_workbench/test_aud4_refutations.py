"""RRAUD refutations for PR #9997 at f626d156; each asserts the documented contract."""

import pytest

from robot_sf.analysis_workbench.audit_scan import scan_campaign


# F1: a named structured record replaced by a bare scalar is admitted as an ordinary scalar.
@pytest.mark.parametrize(
    "name,value",
    [
        ("force_quantiles", -5),
        ("force_sample_stats", -3),
        ("deadlock_stall", -1),
        ("metric_values", -1),
        ("signal_metrics_evidence", -5),
    ],
)
def test_named_record_replaced_by_scalar_is_not_clear(name, value):
    row = {"episode_id": "named", "metrics": {"clearance_m": 1, name: value}}
    report = scan_campaign([row], detector_ids=["extreme_measurements"])
    assert report.inventory[0].readable
    assert report.signals[0].status in {"error", "flagged"}, report.signals[0].reason_code


# F2: a max_steps timeout far before every recorded maximum is reported consistent
# when the runner maximum is recorded only as the top-level `horizon` (PR #9999 row shape)
# or only the simulator/budget maxima are available.
@pytest.mark.parametrize(
    "extra",
    [
        {
            "horizon": 500,
            "effective_budget_steps": 500,
            "scenario_params": {"simulation_config": {"max_episode_steps": 500}},
        },
        {"scenario_params": {"simulation_config": {"max_episode_steps": 400}}},
        {"effective_budget_steps": 400},
    ],
)
def test_max_steps_timeout_before_every_available_maximum_is_not_clear(extra):
    row = {
        "episode_id": "early",
        "steps": 300,
        "outcome": {"label": "timeout"},
        "termination_reason": "max_steps",
        **extra,
    }
    report = scan_campaign([row], detector_ids=["horizon_consistency"])
    assert report.inventory[0].readable
    assert report.signals[0].status == "flagged", report.signals[0].measured


# F3: rows excluded from a planner cell (non-finite corrupt value) silently shrink
# the target denominator 30 -> 4; both cohort channels report clear without disclosure.
def _rows():
    return [
        {
            "episode_id": f"{p}-{i}",
            "planner_id": p,
            "scenario_id": "same",
            "seed": 1001 + i,
            "outcome": {"label": "success" if (p != "bad" or i < 4) else "timeout"},
            "metrics": {"time_to_goal_norm": 10 if (p != "bad" or i < 4) else float("nan")},
        }
        for p in ("bad", "good-a", "good-b")
        for i in range(30)
    ]


@pytest.mark.parametrize("detector", ["outcome_incidence", "planner_cohort_shift"])
def test_cohort_channels_disclose_or_refuse_rows_dropped_from_the_cell(detector):
    report = scan_campaign(_rows(), detector_ids=[detector])
    target = next(s for s in report.signals if s.episode_id == "bad-0")
    measured = dict(target.measured or {})
    disclosed = (
        any("excluded" in k or "invalid" in k or "dropped" in k for k in measured)
        and target.status != "clear"
    )
    assert target.status in {"unavailable", "flagged"} or disclosed, (
        target.status,
        measured.get("planner_sizes"),
    )
