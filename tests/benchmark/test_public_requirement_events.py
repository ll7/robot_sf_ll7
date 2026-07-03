"""Unit coverage for public-requirement diagnostic event predicates."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.public_requirement_events import evaluate_public_requirement_events


def test_pedestrian_steps_in_front_triggers_at_conflict_point() -> None:
    """Named pedestrian reaching conflict point marks authored trigger."""
    scenario = _scenario("safe_braking", "pedestrian_steps_in_front")
    robot_positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    robot_velocities = np.zeros_like(robot_positions)
    ped_positions = np.asarray(
        [
            [[4.0, 4.0]],
            [[0.1, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=float,
    )

    event = evaluate_public_requirement_events(
        scenario=scenario,
        robot_positions=robot_positions,
        robot_velocities=robot_velocities,
        ped_positions=ped_positions,
        dt=0.1,
    )

    assert event["schema_version"] == "public-requirement-events.v1"
    assert event["category"] == "safe_braking"
    assert event["event_type"] == "pedestrian_steps_in_front"
    assert event["triggered"] is True
    assert event["trigger_step"] == 1
    assert event["trigger_time_s"] == 0.2


def test_sudden_obstacle_proxy_respects_robot_trigger_radius() -> None:
    """Emergency proxy waits for both actor and robot to be near conflict point."""
    scenario = _scenario(
        "emergency_reaction",
        "sudden_obstacle_proxy",
        actor_key="actor_id",
        extra_contract={"robot_trigger_radius_m": 0.5},
    )
    ped_positions = np.asarray([[[0.0, 0.0]], [[0.0, 0.0]]], dtype=float)
    robot_velocities = np.zeros((2, 2), dtype=float)

    far = evaluate_public_requirement_events(
        scenario=scenario,
        robot_positions=np.asarray([[3.0, 0.0], [2.0, 0.0]], dtype=float),
        robot_velocities=robot_velocities,
        ped_positions=ped_positions,
        dt=0.1,
    )
    near = evaluate_public_requirement_events(
        scenario=scenario,
        robot_positions=np.asarray([[3.0, 0.0], [0.2, 0.0]], dtype=float),
        robot_velocities=robot_velocities,
        ped_positions=ped_positions,
        dt=0.1,
    )

    assert far["triggered"] is False
    assert near["triggered"] is True
    assert near["trigger_step"] == 1


def test_speed_limit_monitor_records_violation_fields() -> None:
    """Speed-limit diagnostics record violation count and max excess."""
    scenario = {
        "metadata": {
            "public_requirement": {
                "category": "speed_limit",
                "event_contract": {
                    "type": "speed_limit_monitor",
                    "speed_limit_m_s": 0.8,
                    "violation_margin_m_s": 0.05,
                },
            }
        }
    }
    robot_velocities = np.asarray([[0.5, 0.0], [0.9, 0.0], [1.1, 0.0]], dtype=float)

    event = evaluate_public_requirement_events(
        scenario=scenario,
        robot_positions=np.zeros((3, 2), dtype=float),
        robot_velocities=robot_velocities,
        ped_positions=np.zeros((3, 0, 2), dtype=float),
        dt=0.1,
    )

    assert event["triggered"] is True
    assert event["speed_limit_m_s"] == 0.8
    assert event["speed_limit_violation_count"] == 2
    assert event["max_speed_m_s"] == 1.1
    assert event["max_excess_m_s"] == pytest.approx(0.3)


def test_missing_public_requirement_metadata_is_not_applicable() -> None:
    """Ordinary scenarios remain compatible with the diagnostic helper."""
    event = evaluate_public_requirement_events(
        scenario={"name": "ordinary"},
        robot_positions=np.zeros((1, 2), dtype=float),
        robot_velocities=np.zeros((1, 2), dtype=float),
        ped_positions=np.zeros((1, 0, 2), dtype=float),
        dt=0.1,
    )

    assert event["status"] == "not_applicable"
    assert event["triggered"] is False


def _scenario(
    category: str,
    event_type: str,
    *,
    actor_key: str = "pedestrian_id",
    extra_contract: dict | None = None,
) -> dict:
    contract = {
        "type": event_type,
        actor_key: "h1",
        "conflict_point": [0.0, 0.0],
        "trigger_radius_m": 0.25,
    }
    if extra_contract:
        contract.update(extra_contract)
    return {
        "single_pedestrians": [{"id": "h1"}],
        "metadata": {
            "public_requirement": {
                "category": category,
                "event_contract": contract,
            }
        },
    }
