"""Tests for pedestrian collision benchmark helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts import benchmark_ped_policy_collisions as mod


def test_collect_collision_samples_requires_ped_specific_collision(monkeypatch) -> None:
    """Pedestrian-specific telemetry should only be collected for robot-ped collisions."""
    robot_speeds: list[float] = []
    ped_speeds: list[float] = []
    angles: list[float] = []
    zone_counts = {"front": 0, "back": 0, "side": 0}

    monkeypatch.setattr(mod, "_extract_robot_speed", lambda meta, env: 1.5)
    monkeypatch.setattr(mod, "_extract_ped_speed", lambda env: 0.5)

    generic_robot_collision = {
        "is_robot_collision": True,
        "is_robot_pedestrian_collision": False,
        "robot_ped_collision_zone": "",
    }
    mod._collect_collision_samples(
        meta=generic_robot_collision,
        env=SimpleNamespace(),
        robot_speed_at_collision=robot_speeds,
        ped_speed_at_collision=ped_speeds,
        impact_angle_deg_at_collision=angles,
        zone_counts=zone_counts,
    )

    assert robot_speeds == []
    assert ped_speeds == []
    assert angles == []
    assert zone_counts == {"front": 0, "back": 0, "side": 0}

    ped_collision = {
        "is_robot_collision": True,
        "is_robot_pedestrian_collision": True,
        "robot_ped_collision_zone": "front",
        "collision_impact_angle_deg": 45.0,
    }
    mod._collect_collision_samples(
        meta=ped_collision,
        env=SimpleNamespace(),
        robot_speed_at_collision=robot_speeds,
        ped_speed_at_collision=ped_speeds,
        impact_angle_deg_at_collision=angles,
        zone_counts=zone_counts,
    )

    assert robot_speeds == [1.5]
    assert ped_speeds == [0.5]
    assert angles == [45.0]
    assert zone_counts == {"front": 1, "back": 0, "side": 0}


def test_parse_args_requires_explicit_ped_model(monkeypatch) -> None:
    """Benchmark CLI should require an explicit pedestrian model unless the caller opts into latest."""
    monkeypatch.setattr(sys, "argv", ["benchmark_ped_policy_collisions.py"])
    with pytest.raises(SystemExit):
        mod.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_ped_policy_collisions.py", "--ped-model", "latest"],
    )
    args = mod.parse_args()
    assert args.ped_model == "latest"
