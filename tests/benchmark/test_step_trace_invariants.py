"""Tests for the step-trace invariant checker (#9979): one violating and one clean trace each."""

# robot-sf-test-lane: fast-contract

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from robot_sf.benchmark.step_trace_invariants import (
    INVARIANTS,
    Tolerances,
    check_episode,
    resolve_limits,
    summarize,
    to_markdown,
)
from scripts.validation.check_step_trace_invariants import main as cli_main

DT = 0.1


def _row(  # noqa: PLR0913
    positions: list[tuple[float, float]],
    *,
    goal: tuple[float, float] = (20.0, 0.0),
    goal_next: tuple[float, float] | None = None,
    headings: list[float] | None = None,
    peds: list[list[dict[str, Any]]] | None = None,
    collisions: list[dict[str, bool]] | None = None,
    reason: str = "max_steps",
    start: tuple[float, float] | None = None,
    robot_config: dict[str, Any] | None = None,
    goals: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    steps = []
    for i, pos in enumerate(positions):
        step: dict[str, Any] = {
            "step": i,
            "robot": {"position": list(pos), "heading": headings[i] if headings else 0.0},
            "pedestrians": peds[i] if peds else [],
            "goal": {
                "current": list(goals[i] if goals else goal),
                "next": list(goal_next) if goal_next else None,
            },
        }
        if collisions:
            step["collision"] = collisions[i]
        steps.append(step)
    # Extrapolate the reset pose one step back so constant-velocity traces have no start-up jerk.
    if start is None:
        start = (
            (2 * positions[0][0] - positions[1][0], 2 * positions[0][1] - positions[1][1])
            if len(positions) > 1
            else positions[0]
        )
    first = start
    h0 = 0.0
    if headings and len(headings) > 1:
        h0 = 2 * headings[0] - headings[1]
    return {
        "episode_id": "ep-1",
        "scenario_id": "scn",
        "algo": "arm",
        "termination_reason": reason,
        "steps": len(steps),
        "horizon": len(steps),
        "event_ledger": {
            "exact_events": {
                "collision": reason == "collision",
                "goal_reached": reason == "success",
                "timeout": reason in {"max_steps", "terminated"},
            }
        },
        "scenario_params": {"robot_config": robot_config or {"type": "differential_drive"}},
        "algorithm_metadata": {
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "dt": DT,
                "steps": steps,
                "reset": {"robot": {"position": list(first), "heading": h0}},
            }
        },
    }


def _line(n: int, speed: float, origin: tuple[float, float], direction: tuple[float, float]):
    return [
        (
            origin[0] + direction[0] * speed * DT * (i + 1),
            origin[1] + direction[1] * speed * DT * (i + 1),
        )
        for i in range(n)
    ]


def _kinds(row: dict[str, Any], invariant: str, **kw: Any) -> set[str]:
    viols, _ = check_episode(row, **kw)
    return {v.kind for v in viols if v.invariant == invariant}


# (a) heading towards goal.current
def test_a_flags_motion_towards_origin_on_final_leg() -> None:
    # Robot at (10,10) drives to the origin while goal.current is at (20,10) and next is absent.
    d = (-1 / math.sqrt(2), -1 / math.sqrt(2))
    row = _row(_line(120, 1.0, (10.0, 10.0), d), goal=(20.0, 10.0), start=(10.0, 10.0))
    assert _kinds(row, "a_goal_heading") == {"towards_origin_not_current"}


def test_a_flags_motion_towards_next_waypoint() -> None:
    row = _row(_line(120, 1.0, (10.0, 0.0), (0.0, 1.0)), goal=(20.0, 0.0), goal_next=(10.0, 40.0))
    assert _kinds(row, "a_goal_heading") == {"towards_next_not_current"}


def test_a_clean_when_heading_to_current() -> None:
    row = _row(_line(120, 1.0, (0.0, 0.0), (1.0, 0.0)), goal=(30.0, 0.0), goal_next=(30.0, 30.0))
    assert _kinds(row, "a_goal_heading") == set()


def test_a_clean_for_short_detour_around_obstacle() -> None:
    # 1.5 s of retreat is an avoidance manoeuvre, below the sustained-motion requirement.
    pts = _line(10, 1.0, (10.0, 10.0), (-0.7, -0.7)) + _line(100, 1.0, (3.0, 3.0), (1.0, 0.0))
    assert _kinds(_row(pts, goal=(20.0, 10.0)), "a_goal_heading") == set()


# (b) drive limits from the env config
def test_b_flags_speed_above_limit_and_reads_limit_from_config() -> None:
    row = _row(_line(30, 3.0, (0.0, 0.0), (1.0, 0.0)))  # 3 m/s, default limit is 2 m/s
    assert "speed" in _kinds(row, "b_drive_limits")
    fast = _row(
        _line(30, 3.0, (0.0, 0.0), (1.0, 0.0)),
        robot_config={
            "type": "differential_drive",
            "max_linear_speed": 4.0,
            "max_linear_accel": 100.0,
        },
    )
    assert "speed" not in _kinds(fast, "b_drive_limits")


def test_b_flags_yaw_rate_and_acceleration() -> None:
    headings = [0.5 * i for i in range(30)]  # 5 rad/s
    row = _row(_line(30, 1.0, (0.0, 0.0), (1.0, 0.0)), headings=headings)
    assert "yaw_rate" in _kinds(row, "b_drive_limits")
    # 0 -> 1.9 m/s in one step is a 19 m/s^2 acceleration.
    pts = [(0.0, 0.0)] * 5 + [(0.19 * (i + 1), 0.0) for i in range(20)]
    assert "accel" in _kinds(_row(pts, start=(0.0, 0.0)), "b_drive_limits")


def test_b_clean_within_limits() -> None:
    headings = [0.05 * i for i in range(30)]  # 0.5 rad/s
    row = _row(_line(30, 1.0, (0.0, 0.0), (1.0, 0.0)), headings=headings)
    assert _kinds(row, "b_drive_limits") == set()


# (c) clearance and contact
def _ped(pos: tuple[float, float], clearance: float) -> dict[str, Any]:
    return {"position": list(pos), "surface_clearance_m": clearance}


def test_c_flags_clearance_mismatch_and_flagless_contact() -> None:
    pts = [(0.0, 0.0)] * 3
    # true clearance for a ped 1.0 m away is 1.0 - 1.0 - 0.4 = -0.4 (real radii 1.0 and 0.4)
    peds = [[_ped((1.0, 0.0), -0.4)], [_ped((1.0, 0.0), 0.9)], [_ped((1.0, 0.0), -0.4)]]
    coll = [{"pedestrian": False, "obstacle": False, "robot": False}] * 3
    kinds = _kinds(_row(pts, peds=peds, collisions=coll), "c_clearance_contact")
    assert kinds == {"clearance_mismatch", "contact_without_flag"}


def test_c_flags_collision_flag_without_contact() -> None:
    peds = [[_ped((6.0, 0.0), 4.6)]]
    coll = [{"pedestrian": True, "obstacle": False, "robot": False}]
    assert _kinds(_row([(0.0, 0.0)], peds=peds, collisions=coll), "c_clearance_contact") == {
        "flag_without_contact"
    }


def test_c_clean_when_contact_matches_flag() -> None:
    peds = [[_ped((3.0, 0.0), 1.6)], [_ped((1.0, 0.0), -0.4)]]
    coll = [
        {"pedestrian": False, "obstacle": False, "robot": False},
        {"pedestrian": True, "obstacle": False, "robot": False},
    ]
    assert (
        _kinds(
            _row([(0.0, 0.0)] * 2, peds=peds, collisions=coll, reason="collision"),
            "c_clearance_contact",
        )
        == set()
    )


# (d) position jumps
def test_d_flags_teleport_and_clean_run() -> None:
    pts = _line(20, 1.0, (0.0, 0.0), (1.0, 0.0))
    jumped = pts[:10] + [(x + 15.0, y) for x, y in pts[10:]]
    assert _kinds(_row(jumped), "d_position_jump") == {"jump"}
    assert _kinds(_row(pts), "d_position_jump") == set()


def test_d_flags_respawn_right_after_reset() -> None:
    assert _kinds(
        _row(_line(5, 1.0, (30.0, 0.0), (1.0, 0.0)), start=(0.0, 0.0)), "d_position_jump"
    ) == {"jump"}


# (e) termination
def test_e_success_far_from_goal_and_clean_success() -> None:
    far = _row(_line(20, 1.0, (0.0, 0.0), (1.0, 0.0)), goal=(50.0, 0.0), reason="success")
    assert "success_far_from_goal" in _kinds(far, "e_termination")
    near = _row(_line(20, 1.0, (0.0, 0.0), (1.0, 0.0)), goal=(3.5, 0.0), reason="success")
    assert _kinds(near, "e_termination") == set()


def test_e_collision_without_flag_and_timeout_with_flag() -> None:
    quiet = [{"pedestrian": False, "obstacle": False, "robot": False}] * 5
    assert "collision_without_flag_in_tail" in _kinds(
        _row(_line(5, 1.0, (0, 0), (1, 0)), collisions=quiet, reason="collision"), "e_termination"
    )
    hit = quiet[:-1] + [{"pedestrian": False, "obstacle": True, "robot": False}]
    assert "timeout_with_final_collision_flag" in _kinds(
        _row(_line(5, 1.0, (0, 0), (1, 0)), collisions=hit, reason="max_steps"), "e_termination"
    )
    assert (
        _kinds(
            _row(_line(5, 1.0, (0, 0), (1, 0)), collisions=hit, reason="collision"), "e_termination"
        )
        == set()
    )


def test_e_row_level_checks_without_trace() -> None:
    row = {
        "episode_id": "r",
        "termination_reason": "success",
        "steps": 50,
        "horizon": 600,
        "event_ledger": {
            "exact_events": {"collision": True, "goal_reached": True, "timeout": False}
        },
    }
    viols, has_trace = check_episode(row)
    assert not has_trace
    assert {v.kind for v in viols} >= {
        "success_without_goal_or_with_collision",
        "goal_and_collision_both_set",
    }
    clean = {
        "episode_id": "r",
        "termination_reason": "terminated",
        "steps": 500,
        "horizon": 600,
        "scenario_params": {"simulation_config": {"max_episode_steps": 500}},
        "event_ledger": {
            "exact_events": {"collision": False, "goal_reached": False, "timeout": True}
        },
    }
    assert check_episode(clean)[0] == []


def test_resolve_limits_reads_env_config() -> None:
    row = {
        "scenario_params": {
            "robot_config": {"type": "differential_drive", "radius": 0.7, "max_angular_speed": 3.0},
            "simulation_config": {"ped_radius": 0.3},
        }
    }
    lim = resolve_limits(row)
    assert (lim.robot_radius, lim.ped_radius, lim.max_angular_speed) == (0.7, 0.3, 3.0)
    default = resolve_limits({})
    assert (default.robot_radius, default.ped_radius) == (1.0, 0.4)


def test_summary_markdown_and_cli(tmp_path) -> None:
    d = (-1 / math.sqrt(2), -1 / math.sqrt(2))
    bad = _row(_line(120, 1.0, (10.0, 10.0), d), goal=(20.0, 10.0), start=(10.0, 10.0))
    good = _row(_line(120, 1.0, (0.0, 0.0), (1.0, 0.0)), goal=(30.0, 0.0))
    good["episode_id"] = "ep-2"
    run = tmp_path / "risk_dwa__differential_drive"
    run.mkdir()
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in (bad, good)) + "\n")
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    assert (
        cli_main(
            [
                str(tmp_path),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
                "--fail-on-violation",
            ]
        )
        == 1
    )
    summary = json.loads(out_json.read_text())
    cell = summary["table"]["risk_dwa"]["scn"]["invariants"]["a_goal_heading"]
    assert cell["episodes"] == 1 and cell["examples"][0]["episode_id"] == "ep-1"
    assert "risk_dwa" in out_md.read_text()
    assert set(summary["arms"]["risk_dwa"]["flagged"]) == set(INVARIANTS)


def test_summarize_and_markdown_handle_empty() -> None:
    assert "Flagged episodes per arm" in to_markdown(summarize([]))


@pytest.mark.parametrize("bad", [None, {}, {"steps": []}])
def test_rows_with_missing_trace_do_not_crash(bad: Any) -> None:
    row = {"episode_id": "x", "algorithm_metadata": {"simulation_step_trace": bad}}
    assert check_episode(row, tol=Tolerances())[1] is False
