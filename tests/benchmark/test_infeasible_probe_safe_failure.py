"""Doorway probe safe-failure metric and its comparator gate (issue #9974).

Rows are synthetic; no planner steps run for seeds 111-140.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.infeasible_probe_safe_failure import (
    CONTACT,
    CRASH,
    DEFECT_SUCCESS,
    SAFE_FAILURE,
    UNRESOLVED,
    classify_probe_row,
    safe_failure_summary,
)
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import _probe_gate


def _row(
    termination: str = "max_steps",
    *,
    route_complete: bool = False,
    collision: bool = False,
    timeout: bool | None = None,
    **metrics: float,
) -> dict:
    return {
        "termination_reason": termination,
        "outcome": {
            "route_complete": route_complete,
            "collision_event": collision,
            "timeout_event": termination in {"max_steps", "truncated"}
            if timeout is None
            else timeout,
        },
        "metrics": {"collisions": 1.0 if collision else 0.0, **metrics},
        "integrity": {"contradictions": []},
    }


def test_timeout_without_contact_is_safe_failure() -> None:
    assert classify_probe_row(_row()) == SAFE_FAILURE
    assert classify_probe_row(_row("truncated", robot_force_impulse_total=0.0)) == SAFE_FAILURE


def test_crash_is_not_safe_failure() -> None:
    assert classify_probe_row(_row("error", timeout=False)) == CRASH


@pytest.mark.parametrize(
    "row",
    [
        _row("collision", collision=True, timeout=False),
        _row(robot_force_impulse_total=0.4),
        _row(robot_force_peak=2.0),
        _row(robot_force_exposed_ped_count=1.0),
    ],
)
def test_robot_attributable_contact_is_not_safe_failure(row: dict) -> None:
    assert classify_probe_row(row) == CONTACT


def test_timeout_with_contact_is_not_counted_even_when_flagged_timeout() -> None:
    assert classify_probe_row(_row(collisions=2.0)) == CONTACT


def test_success_on_probe_is_a_defect() -> None:
    row = _row("success", route_complete=True, timeout=False)
    assert classify_probe_row(row) == DEFECT_SUCCESS
    summary = safe_failure_summary([row, *[_row()] * 3], expected_rows=4)
    assert summary["status"] == "defect"
    assert summary["safe_failure_rate"] == 0.75


@pytest.mark.parametrize(
    "row",
    [
        {**_row(), "termination_reason": None},
        {**_row(), "integrity": {"contradictions": ["x"]}},
        _row(robot_force_peak=float("nan")),
        _row(timeout=False),
        {"termination_reason": "max_steps", "outcome": {}, "metrics": {}},
    ],
)
def test_unresolved_rows_fail_admission(row: dict) -> None:
    assert classify_probe_row(row) == UNRESOLVED
    assert safe_failure_summary([row], expected_rows=1)["status"] == "fail_admission"


def test_summary_rate_and_row_count_check() -> None:
    rows = [_row()] * 3 + [_row("error", timeout=False)]
    summary = safe_failure_summary(rows, expected_rows=4)
    assert summary["status"] == "computed"
    assert summary["safe_failure_rate"] == 0.75
    assert safe_failure_summary(rows, expected_rows=5)["status"] == "fail_admission"
    assert safe_failure_summary([], expected_rows=0)["status"] == "fail_admission"


def test_gate_refuses_incomplete_or_unresolved_probe_rows() -> None:
    good = [_row()] * 420
    assert _probe_gate(good, 420)["status"] == "computed"
    assert _probe_gate([], 0) is None
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(good[:-1], 420)
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate([*good[:-1], {**_row(), "termination_reason": None}], 420)
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(good[:10], 10)


def test_gate_reports_success_as_defect_without_refusing() -> None:
    rows = [*[_row()] * 419, _row("success", route_complete=True, timeout=False)]
    assert _probe_gate(rows, 420)["status"] == "defect"
