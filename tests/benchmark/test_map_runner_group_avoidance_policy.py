"""Map-runner wiring tests for the TAGA-like group-avoidance policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.nav.map_config import SocialGroupDefinition


def _fake_env() -> SimpleNamespace:
    group = SocialGroupDefinition(
        group_id="conversation_a",
        type="conversation",
        members=("ped_a", "ped_b"),
        formation="face_to_face",
        centroid=(2.0, 0.0),
        radius=0.8,
    )
    return SimpleNamespace(simulator=SimpleNamespace(social_groups=[group]))


def test_build_policy_registers_taga_group_avoidance_with_provenance() -> None:
    """Map-runner builds the opt-in wrapper and exposes runtime diagnostics."""

    policy, meta = _build_policy(
        "taga_group_avoidance",
        {
            "wrapped_algo": "goal",
            "safety_margin_m": 0.5,
            "tangent_clearance_m": 0.2,
            "tangent_side": "left",
        },
        robot_kinematics="differential_drive",
    )

    assert meta["algorithm"] == "taga_group_avoidance"
    assert meta["group_avoidance"]["schema_version"] == "taga-like-group-avoidance.v1"
    assert meta["group_avoidance"]["diagnostic_only"] is True
    assert "not a collision-safety guarantee" in meta["group_avoidance"]["claim_boundary"]
    assert meta["planner_kinematics"]["adapter_name"] == "TangentSubgoalGroupAvoidanceAdapter"
    assert hasattr(policy, "_planner_bind_env")
    assert hasattr(policy, "_planner_stats")

    policy._planner_bind_env(_fake_env())
    command = policy(
        {
            "robot": {"position": [1.0, 0.0], "heading": [0.0]},
            "goal": {"current": [5.0, 0.0]},
        }
    )
    diagnostics = policy._planner_stats()["group_avoidance"]

    assert command[0] >= 0.0
    assert diagnostics["trigger_count"] == 1
    assert diagnostics["last_selected_group_id"] == "conversation_a"


def test_group_avoidance_readiness_requires_explicit_opt_in() -> None:
    """Readiness metadata classifies the wrapper as experimental opt-in."""

    readiness = get_algorithm_readiness("group_avoidance")

    assert readiness.canonical_name == "taga_group_avoidance"
    assert readiness.tier == "experimental"
    assert readiness.requires_explicit_opt_in is True


def test_build_policy_group_avoidance_rejects_non_goal_wrapped_algo() -> None:
    """Map-runner policy construction fails closed for unsupported wrapping."""

    with pytest.raises(ValueError, match="wrapped_algo='goal'"):
        _build_policy("group_avoidance", {"wrapped_algo": "orca"})
