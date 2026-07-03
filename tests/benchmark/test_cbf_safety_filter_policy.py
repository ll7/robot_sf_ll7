"""Benchmark policy wiring tests for the optional CBF safety filter."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.map_runner import _build_policy


def test_adapter_policy_cbf_filter_is_explicit_opt_in() -> None:
    """Adapter-backed planners expose CBF metadata and shield stats only when enabled."""

    policy, meta = _build_policy(
        "safety_barrier",
        {
            "max_linear_speed": 0.8,
            "cbf_safety_filter": {
                "enabled": True,
                "alpha": 1.0,
                "safety_margin": 0.1,
                "max_linear_speed": 1.0,
                "max_angular_speed": 2.0,
            },
        },
        robot_kinematics="differential_drive",
        robot_command_mode="unicycle",
    )

    command = policy(
        {
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.8, 0.0],
                "heading": [0.0],
                "radius": [0.3],
            },
            "goal": {"current": [5.0, 0.0], "next": [5.0, 0.0]},
            "agents": [
                {
                    "position": [0.8, 0.0],
                    "velocity": [-0.6, 0.0],
                    "radius": 0.3,
                }
            ],
        }
    )

    assert command[0] < 0.8
    assert meta["cbf_safety_filter"]["status"] == "enabled"
    assert meta["safety_shield_contract"]["shield_name"] == "CollisionConeCbfSafetyFilter"
    assert meta["shield_stats"]["decision_count"] == 1
    assert meta["shield_stats"]["intervention_count"] == 1
    stats = policy._planner_stats()
    assert stats["cbf_safety_filter"]["decision_count"] == 1


def test_adapter_policy_without_cbf_filter_has_no_shield_metadata() -> None:
    """Default adapter-backed planner construction remains shield-free."""

    _policy, meta = _build_policy(
        "safety_barrier",
        {"max_linear_speed": 0.8},
        robot_kinematics="differential_drive",
        robot_command_mode="unicycle",
    )

    assert "cbf_safety_filter" not in meta
    assert "safety_shield_contract" not in meta
    assert "shield_stats" not in meta


def test_adapter_policy_accepts_dynamic_parabolic_cbf_variant() -> None:
    """Adapter-backed planners can opt into the versioned DPCBF filter."""

    policy, meta = _build_policy(
        "safety_barrier",
        {
            "max_linear_speed": 0.8,
            "cbf_safety_filter": {
                "enabled": True,
                "variant": "dynamic_parabolic_cbf_v1",
                "max_linear_speed": 1.0,
            },
        },
        robot_kinematics="differential_drive",
        robot_command_mode="unicycle",
    )

    command = policy(
        {
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.8, 0.0],
                "heading": [0.0],
                "radius": [0.3],
            },
            "goal": {"current": [5.0, 0.0], "next": [5.0, 0.0]},
            "agents": [
                {
                    "position": [0.8, 0.0],
                    "velocity": [-0.6, 0.0],
                    "radius": 0.3,
                }
            ],
        }
    )

    assert command[0] < 0.8
    assert meta["cbf_safety_filter"]["variant"] == "dynamic_parabolic_cbf_v1"
    assert meta["safety_shield_contract"]["shield_name"] == "DynamicParabolicCbfSafetyFilter"


def test_adapter_policy_rejects_unknown_cbf_variant() -> None:
    """Unsupported variants fail closed at policy construction."""

    with pytest.raises(ValueError, match="variant"):
        _build_policy(
            "safety_barrier",
            {"cbf_safety_filter": {"enabled": True, "variant": "custom_cbf"}},
            robot_kinematics="differential_drive",
            robot_command_mode="unicycle",
        )
