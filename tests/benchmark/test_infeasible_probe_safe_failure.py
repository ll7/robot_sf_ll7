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
    classify_probe_slots,
    safe_failure_summary,
)
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import _probe_gate, _root_failure_slots

PROBE = "francis2023_narrow_doorway"
ARMS = [(f"planner{i}", "differential_drive") for i in range(14)]
SEEDS = list(range(111, 141))
SLOTS = {(p, k, PROBE, seed, "") for p, k in ARMS for seed in SEEDS}


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


def test_force_above_zero_without_collision_is_safe_failure() -> None:
    row = _row(
        robot_force_impulse_total=0.4, robot_force_peak=2.0, robot_force_exposed_ped_count=1.0
    )
    assert classify_probe_row(row) == SAFE_FAILURE
    summary = safe_failure_summary([row], expected_rows=1)
    assert summary["robot_force_descriptive"]["robot_force_peak"]["max"] == 2.0
    assert summary["robot_force_descriptive"]["robot_force_peak"]["rows_above_zero"] == 1


def test_obstacle_or_wall_collision_flag_is_contact() -> None:
    assert classify_probe_row(_row(collision=True)) == CONTACT


def test_error_termination_is_not_safe_failure() -> None:
    assert classify_probe_row(_row("error", timeout=False)) not in {SAFE_FAILURE, CRASH}


@pytest.mark.parametrize(
    "row",
    [
        _row("collision", collision=True, timeout=False),
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
        _row(collision=False, collisions=float("nan")),
        _row(timeout=False),
        {"termination_reason": "max_steps", "outcome": {}, "metrics": {}},
    ],
)
def test_unresolved_rows_fail_admission(row: dict) -> None:
    assert classify_probe_row(row) == UNRESOLVED
    assert safe_failure_summary([row], expected_rows=1)["status"] == "fail_admission"


def test_summary_rate_and_row_count_check() -> None:
    rows = [_row()] * 3 + [_row("collision", collision=True, timeout=False)]
    summary = safe_failure_summary(rows, expected_rows=4)
    assert summary["status"] == "computed"
    assert summary["safe_failure_rate"] == 0.75
    assert safe_failure_summary(rows, expected_rows=5)["status"] == "fail_admission"
    assert safe_failure_summary([], expected_rows=0)["status"] == "fail_admission"


def _rows(slots=SLOTS) -> dict:
    return {slot: _row() for slot in slots}


def test_slot_fixture_is_420_real_slots() -> None:
    assert len(SLOTS) == 420
    assert len({(s[0], s[3]) for s in SLOTS}) == 420


def test_gate_computes_over_exact_slots() -> None:
    summary = _probe_gate(_rows(), SLOTS)
    assert summary["status"] == "computed"
    assert summary["class_counts"] == {SAFE_FAILURE: 420}
    assert _probe_gate({}, set()) is None


def test_gate_ignores_other_scenarios_and_width_slices() -> None:
    other = ("planner0", "differential_drive", "francis2023_narrow_doorway_width_2p20", 111, "")
    new = {**_rows(), other: _row()}
    assert _probe_gate(new, {*SLOTS, other})["status"] == "computed"


def test_gate_refuses_missing_slot_without_failure_record() -> None:
    new = _rows()
    del new[sorted(SLOTS)[0]]
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(new, SLOTS)


def test_gate_refuses_duplicate_that_offsets_a_missing_slot() -> None:
    new = _rows()
    missing, dup = sorted(SLOTS)[:2]
    del new[missing]
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(new, SLOTS, {dup: 2})


def test_gate_refuses_wrong_slot_set() -> None:
    swapped = {(p, k, PROBE, seed + 1000, "") for p, k, _, seed, _ in SLOTS}
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(_rows(swapped), SLOTS)
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(_rows(sorted(SLOTS)[:419]), set(sorted(SLOTS)[:419]))
    extra = ("planner99", "differential_drive", PROBE, 111, "")
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate({**_rows(), extra: _row()}, SLOTS)


def test_gate_refuses_unresolved_row() -> None:
    new = _rows()
    new[sorted(SLOTS)[0]] = {**_row(), "termination_reason": None}
    with pytest.raises(ValueError, match="safe_failure_metric not computed"):
        _probe_gate(new, SLOTS)


def test_missing_row_with_failure_record_is_crash() -> None:
    new = _rows()
    victim = sorted(SLOTS)[0]
    del new[victim]
    summary = _probe_gate(new, SLOTS, failure_slots={victim[:4]})
    assert summary["class_counts"] == {CRASH: 1, SAFE_FAILURE: 419}
    assert summary["status"] == "computed"
    assert classify_probe_slots(SLOTS, new)["status"] == "fail_admission"


def test_failure_records_are_read_from_run_summary(tmp_path) -> None:
    run = tmp_path / "runs" / "planner0__differential_drive"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        f'{{"failures": [{{"scenario_id": "{PROBE}", "seed": 111, "error": "RuntimeError()"}}]}}'
    )
    assert _root_failure_slots(tmp_path) == {("planner0", "differential_drive", PROBE, 111)}
    (run / "summary.json").write_text('{"failures": "x"}')
    with pytest.raises(ValueError, match="not a list"):
        _root_failure_slots(tmp_path)


def test_gate_reports_success_as_defect_without_refusing() -> None:
    new = _rows()
    new[sorted(SLOTS)[0]] = _row("success", route_complete=True, timeout=False)
    assert _probe_gate(new, SLOTS)["status"] == "defect"
