"""Tests for simulator pedestrian-force factory wiring."""

from __future__ import annotations

from types import SimpleNamespace

from robot_sf.ped_npc.adversial_ped_force import AdversialPedForceConfig
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.sim import simulator as simulator_module


class _ForceStub:
    """Simple force stub storing config and bound callback for inspection."""

    def __init__(self, config, peds, getter) -> None:
        self.config = config
        self.peds = peds
        self.getter = getter


def test_make_ped_forces_binds_robot_specific_callbacks(monkeypatch) -> None:
    """Each generated force must reference the robot it was created for."""
    base_force = object()
    monkeypatch.setattr(simulator_module, "pysf_make_forces", lambda sim, config: [base_force])
    monkeypatch.setattr(simulator_module, "PedRobotForce", _ForceStub)
    monkeypatch.setattr(simulator_module, "AdversialPedForce", _ForceStub)

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
    apf_config = AdversialPedForceConfig(is_active=True, robot_radius=9.0)

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
    assert [force.config.robot_radius for force in prf_forces] == [0.5, 0.8]
    assert [force.config.robot_radius for force in apf_forces] == [0.5, 0.8]
    assert prf_config.robot_radius == 9.0
    assert apf_config.robot_radius == 9.0
