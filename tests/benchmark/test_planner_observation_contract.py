"""Observation-format contract for benchmark planners (anti-"silent blind planner" guard).

Motivation: ``stream_gap`` was found to be silently blind in the real benchmark runner (#3556) — it
read the nested SOCNAV observation (``obs["robot"]``/``obs["pedestrians"]``) while ``map_runner``
feeds a flat observation (``robot_position``/``pedestrians_positions``), so it extracted
``robot=[0,0], n_peds=0`` and drove blind every episode while still emitting collision "results".

This contract pins the invariant that protects the headline benchmark comparison: a planner's
observation extractor must return a *non-degenerate* view of the flat benchmark observation — it must
see the robot pose and the actual pedestrian count, not an empty/origin default. A behavioural
"reacts to pedestrians" check would false-positive on the trivial reference planner (which ignores
pedestrians by design), so the contract checks *what the planner sees*, not how it reacts.

The 7-planner headline comparison's classical planners all share ``_socnav_fields`` (which handles
both nested and flat observations); ``stream_gap``'s standalone extractor is covered in
``tests/benchmark/test_scenario_belief_policy_hook_issue_3556.py``.
"""

from __future__ import annotations

import numpy as np

from robot_sf.planner.socnav import SocialForcePlannerAdapter, SocNavPlannerConfig


def _flat_benchmark_observation(n_peds: int) -> dict:
    """Build a flat map_runner-style observation with a known robot pose and ``n_peds`` pedestrians."""
    positions = [[8.0 + i, 5.0] for i in range(n_peds)]
    return {
        "robot_position": [5.0, 5.0],
        "robot_heading": [0.0],
        "robot_speed": [0.0],
        "robot_radius": [0.4],
        "robot_velocity_xy": [0.0, 0.0],
        "goal_current": [12.0, 5.0],
        "goal_next": [12.0, 5.0],
        "pedestrians_positions": positions,
        "pedestrians_velocities": [[0.0, 0.0] for _ in range(n_peds)],
        "pedestrians_count": [n_peds],
        "pedestrians_radius": [0.3],
        "map_size": [40.0, 40.0],
        "sim_timestep": 0.1,
    }


def _nested_socnav_observation(n_peds: int) -> dict:
    """Build the nested SOCNAV equivalent for the backward-compatibility check."""
    positions = [[8.0 + i, 5.0] for i in range(n_peds)]
    return {
        "robot": {"position": [5.0, 5.0], "heading": [0.0]},
        "goal": {"current": [12.0, 5.0], "next": [12.0, 5.0]},
        "pedestrians": {
            "positions": positions,
            "velocities": [[0.0, 0.0] for _ in range(n_peds)],
            "count": [n_peds],
        },
    }


def _shared_extractor():
    """Return the ``_socnav_fields`` extractor shared by the classical headline planners."""
    return SocialForcePlannerAdapter(config=SocNavPlannerConfig())._socnav_fields


def test_shared_extractor_sees_flat_benchmark_observation():
    """The classical-planner extractor must read the flat benchmark observation, not come back blind."""
    robot_state, goal_state, ped_state = _shared_extractor()(_flat_benchmark_observation(n_peds=2))
    robot_pos = np.asarray(robot_state["position"], dtype=float).reshape(-1)[:2]
    goal = np.asarray(goal_state["current"], dtype=float).reshape(-1)[:2]
    positions = np.asarray(ped_state.get("positions"), dtype=float)
    count = int(np.asarray(ped_state.get("count", [0]), dtype=float).reshape(-1)[0])

    assert list(robot_pos) == [5.0, 5.0]  # not the [0, 0] blind default
    assert list(goal) == [12.0, 5.0]
    assert count == 2
    assert positions.shape[0] == 2


def test_shared_extractor_still_reads_nested_observation():
    """Backward compatibility: the nested SOCNAV observation must still extract correctly."""
    robot_state, _goal_state, ped_state = _shared_extractor()(_nested_socnav_observation(n_peds=2))
    robot_pos = np.asarray(robot_state["position"], dtype=float).reshape(-1)[:2]
    count = int(np.asarray(ped_state.get("count", [0]), dtype=float).reshape(-1)[0])
    assert list(robot_pos) == [5.0, 5.0]
    assert count == 2


def test_extractor_does_not_invent_pedestrians_when_absent():
    """With no pedestrians the extractor reports zero — fail-closed, no phantom agents."""
    _robot_state, _goal_state, ped_state = _shared_extractor()(
        _flat_benchmark_observation(n_peds=0)
    )
    positions = np.asarray(ped_state.get("positions") or [], dtype=float).reshape(-1, 2)
    assert positions.shape[0] == 0
