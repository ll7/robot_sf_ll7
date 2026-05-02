"""Tests for simulator pedestrian-force factory wiring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.sim import simulator as simulator_module


class _ForceStub:
    """Simple force stub storing config and bound callback for inspection."""

    def __init__(self, kind: str, config, peds, getter) -> None:
        self.kind = kind
        self.config = config
        self.peds = peds
        self.getter = getter


def test_make_ped_forces_binds_robot_specific_callbacks(monkeypatch) -> None:
    """Each generated force must reference the robot it was created for."""
    base_force = object()
    monkeypatch.setattr(simulator_module, "pysf_make_forces", lambda sim, config: [base_force])
    monkeypatch.setattr(
        simulator_module,
        "PedRobotForce",
        lambda config, peds, get_robot_pos: _ForceStub("prf", config, peds, get_robot_pos),
    )
    monkeypatch.setattr(
        simulator_module,
        "AdversarialPedForce",
        lambda config, peds, get_robot_pose: _ForceStub("apf", config, peds, get_robot_pose),
    )

    robots = [
        SimpleNamespace(
            pos=(1.0, 2.0),
            pose=((1.0, 2.0), 0.1),
            config=SimpleNamespace(radius=0.5),
        ),
        SimpleNamespace(
            pos=(3.0, 4.0),
            pose=((3.0, 4.0), 0.2),
            config=SimpleNamespace(radius=0.8),
        ),
    ]

    prf_config = PedRobotForceConfig(is_active=True, robot_radius=9.0)
    apf_config = AdversarialPedForceConfig(is_active=True, robot_radius=9.0)

    forces = simulator_module._make_ped_forces(
        sim=SimpleNamespace(peds=SimpleNamespace()),
        config=SimpleNamespace(),
        robots=robots,
        peds_have_obstacle_forces=True,
        prf_config=prf_config,
        apf_config=apf_config,
    )

    assert forces[0] is base_force

    prf_forces = forces[1:3]
    apf_forces = forces[3:5]

    assert [force.getter() for force in prf_forces] == [robot.pos for robot in robots]
    assert [force.getter() for force in apf_forces] == [robot.pose for robot in robots]
    assert [force.kind for force in prf_forces] == ["prf", "prf"]
    assert [force.kind for force in apf_forces] == ["apf", "apf"]
    assert [force.config.robot_radius for force in prf_forces] == [0.5, 0.8]
    assert [force.config.robot_radius for force in apf_forces] == [0.5, 0.8]
    assert prf_config.robot_radius == 9.0
    assert apf_config.robot_radius == 9.0


def test_ped_simulator_reset_uses_npc_velocity_for_ego_heading(monkeypatch) -> None:
    """When the ego pedestrian respawns independently, use NPC velocity instead of tau."""

    class _EgoPedStub:
        def __init__(self) -> None:
            self.pose = ((0.0, 0.0), 0.25)
            self.reset_calls: list[tuple[tuple[float, float], float]] = []

        def reset_state(self, new_pose) -> None:
            self.reset_calls.append(new_pose)
            self.pose = new_pose

    sim = object.__new__(simulator_module.PedSimulator)
    sim.robots = [SimpleNamespace()]
    sim.robot_navs = [SimpleNamespace(reached_waypoint=True, reached_destination=False)]
    sim.spawn_near_robot = False
    sim.map_def = SimpleNamespace(ped_spawn_zones=["zone"])
    sim.pysf_state = SimpleNamespace(
        num_peds=2,
        pysf_states=lambda: np.array(
            [
                [1.0, 2.0, 0.0, 2.0, 5.0, 6.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
    )
    sim.ego_ped = _EgoPedStub()
    sim._sync_ego_ped_social_force_state = lambda: None

    monkeypatch.setattr(simulator_module, "sample", lambda population, k: [population[0]])
    monkeypatch.setattr(simulator_module, "sample_zone", lambda zone, count: [(9.0, 9.0)])

    simulator_module.PedSimulator.reset_state(sim)

    assert sim.ego_ped.reset_calls[0][0] == (9.0, 9.0)
    assert sim.ego_ped.reset_calls[0][1] == pytest.approx(np.pi / 2)


def test_ped_simulator_reset_requires_spawn_zone_when_spawn_near_robot_disabled() -> None:
    """The explicit random-zone spawn mode should fail clearly without pedestrian zones."""

    class _EgoPedStub:
        def __init__(self) -> None:
            self.pose = ((0.0, 0.0), 0.25)

        def reset_state(self, new_pose) -> None:
            raise AssertionError(f"unexpected reset_state call: {new_pose}")

    sim = object.__new__(simulator_module.PedSimulator)
    sim.robots = [SimpleNamespace()]
    sim.robot_navs = [SimpleNamespace(reached_waypoint=True, reached_destination=False)]
    sim.spawn_near_robot = False
    sim.map_def = SimpleNamespace(ped_spawn_zones=[])
    sim.pysf_state = SimpleNamespace(num_peds=0, pysf_states=lambda: np.zeros((0, 7), dtype=float))
    sim.ego_ped = _EgoPedStub()
    sim._sync_ego_ped_social_force_state = lambda: None

    with pytest.raises(
        ValueError,
        match="spawn_near_robot=False requires at least one pedestrian spawn zone",
    ):
        simulator_module.PedSimulator.reset_state(sim)
