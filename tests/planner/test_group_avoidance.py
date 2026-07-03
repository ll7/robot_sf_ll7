"""Tests for the TAGA-like group-avoidance wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.nav.map_config import SocialGroupDefinition
from robot_sf.planner.group_avoidance import (
    GroupAvoidanceConfig,
    TangentSubgoalGroupAvoidanceAdapter,
    build_group_avoidance_config,
)


def _obs(*, robot=(0.0, 0.0), heading=0.0, goal=(10.0, 0.0)) -> dict[str, object]:
    return {
        "robot": {"position": list(robot), "heading": [heading]},
        "goal": {"current": list(goal)},
    }


def _env_with_group(group: SocialGroupDefinition) -> SimpleNamespace:
    return SimpleNamespace(simulator=SimpleNamespace(social_groups=[group]))


def _conversation_group() -> SocialGroupDefinition:
    return SocialGroupDefinition(
        group_id="conversation_a",
        type="conversation",
        members=("ped_a", "ped_b"),
        formation="face_to_face",
        centroid=(2.0, 0.0),
        radius=0.8,
    )


def test_group_avoidance_defers_to_goal_outside_safety_margin() -> None:
    """Adapter returns the wrapped goal command when no group is close."""

    adapter = TangentSubgoalGroupAvoidanceAdapter(
        GroupAvoidanceConfig(safety_margin_m=0.3, tangent_clearance_m=0.1, max_speed=1.0)
    )
    adapter.bind_env(_env_with_group(_conversation_group()))

    command = adapter.plan(_obs(robot=(-2.0, 0.0), goal=(10.0, 0.0)))

    assert command == pytest.approx((1.0, 0.0))
    assert adapter.diagnostics()["group_avoidance"]["trigger_count"] == 0


def test_group_avoidance_selects_deterministic_tangent_subgoal_near_group() -> None:
    """Adapter selects a tangent subgoal outside the configured group buffer."""

    adapter = TangentSubgoalGroupAvoidanceAdapter(
        GroupAvoidanceConfig(
            safety_margin_m=0.5,
            tangent_clearance_m=0.2,
            tangent_side="left",
            max_speed=1.0,
        )
    )
    adapter.bind_env(_env_with_group(_conversation_group()))

    command = adapter.plan(_obs(robot=(1.0, 0.0), goal=(5.0, 0.0)))
    diagnostics = adapter.diagnostics()["group_avoidance"]

    assert diagnostics["trigger_count"] == 1
    assert diagnostics["last_selected_group_id"] == "conversation_a"
    subgoal = np.asarray(diagnostics["last_subgoal"], dtype=float)
    assert np.linalg.norm(subgoal - np.asarray([2.0, 0.0])) == pytest.approx(1.5)
    assert abs(command[1]) > 0.01


def test_group_avoidance_auto_tangent_selection_is_stable() -> None:
    """Auto tangent selection is deterministic for the same observation."""

    adapter = TangentSubgoalGroupAvoidanceAdapter(
        GroupAvoidanceConfig(safety_margin_m=0.5, tangent_clearance_m=0.2)
    )
    adapter.bind_env(_env_with_group(_conversation_group()))

    adapter.plan(_obs(robot=(0.0, -0.2), goal=(5.0, 1.5)))
    first = adapter.diagnostics()["group_avoidance"]["last_subgoal"]
    adapter.plan(_obs(robot=(0.0, -0.2), goal=(5.0, 1.5)))
    second = adapter.diagnostics()["group_avoidance"]["last_subgoal"]

    assert second == pytest.approx(first)


def test_group_avoidance_invalid_wrapped_algo_fails_closed() -> None:
    """Unsupported wrapped planners fail before benchmark execution."""

    with pytest.raises(ValueError, match="wrapped_algo='goal'"):
        build_group_avoidance_config({"wrapped_algo": "orca"})


def test_group_avoidance_config_validation_fails_closed() -> None:
    """Invalid tangent mode, trigger mode, and numeric bounds are rejected."""

    with pytest.raises(ValueError, match="tangent_side"):
        build_group_avoidance_config({"tangent_side": "middle"})
    with pytest.raises(ValueError, match="trigger_mode"):
        build_group_avoidance_config({"trigger_mode": "distance"})
    with pytest.raises(ValueError, match="safety_margin_m"):
        build_group_avoidance_config({"safety_margin_m": -0.1})
    with pytest.raises(ValueError, match="tangent_clearance_m"):
        build_group_avoidance_config({"tangent_clearance_m": -0.1})
    with pytest.raises(ValueError, match="max_speed"):
        build_group_avoidance_config({"max_speed": -0.1})


def test_group_avoidance_accepts_dict_backed_group_specs() -> None:
    """Adapter can bind plain JSON-safe group dictionaries from compatible sources."""

    adapter = TangentSubgoalGroupAvoidanceAdapter(GroupAvoidanceConfig(safety_margin_m=0.5))
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            social_groups=[
                {
                    "group_id": "conversation_dict",
                    "centroid": [2.0, 0.0],
                    "radius": 0.8,
                }
            ]
        )
    )
    adapter.bind_env(env)

    adapter.plan(_obs(robot=(1.0, 0.0), goal=(5.0, 0.0)))
    diagnostics = adapter.diagnostics()["group_avoidance"]

    assert diagnostics["group_count"] == 1
    assert diagnostics["last_selected_group_id"] == "conversation_dict"
