"""Tests for map-runner safety predicate emission."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.map_runner_episode import _safety_predicates_for_episode


def test_map_runner_safety_predicates_feed_event_ledger_surrogates() -> None:
    """Episode arrays should produce ledger-ready safety predicate records."""
    robot_pos_arr = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
        ],
        dtype=float,
    )
    robot_vel_arr = np.tile(np.asarray([[1.0, 0.0]], dtype=float), (robot_pos_arr.shape[0], 1))
    robot_headings = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    ped_pos_arr = np.tile(
        np.asarray([[[0.2, 0.0]]], dtype=float),
        (robot_pos_arr.shape[0], 1, 1),
    )

    safety_predicates = _safety_predicates_for_episode(
        robot_pos_arr=robot_pos_arr,
        robot_vel_arr=robot_vel_arr,
        robot_headings=robot_headings,
        ped_pos_arr=ped_pos_arr,
        dt=0.1,
    )

    assert safety_predicates["oscillatory_control_predicate"]["oscillation"] is True
    assert safety_predicates["late_evasive_predicate"]["late_evasive"] is True
    assert safety_predicates["occlusion_near_miss_predicate"]["occlusion_near_miss"] is False

    ledger = build_event_ledger(
        {
            "scenario_id": "scenario-1",
            "seed": 7,
            "algo": "goal",
            "git_hash": "abc123",
            "metrics": {"collisions": 0.0},
            "termination_reason": "success",
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
            "safety_predicates": safety_predicates,
        }
    )

    surrogate_events = ledger["surrogate_events"]
    assert surrogate_events["oscillation"] is True
    assert surrogate_events["late_evasive"] is True
    assert surrogate_events["occlusion_near_miss"] is False
    assert (
        surrogate_events["oscillatory_control_predicate"]["schema_version"]
        == "safety_predicate.oscillatory_control.v1"
    )
