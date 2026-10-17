"""Physical capsule witnesses through existing config and environment factories.

The public factories already exist on base; failures there assert wrong contact
behaviour rather than relying on missing capsule imports or missing config files.
"""

from math import pi
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.env_util import (
    create_spaces,
    init_collision_and_sensors,
    init_collision_and_sensors_with_image,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.robot.bicycle_drive import BicycleDriveRobot
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


def capsule_config():
    """Parse authored robot fields through the production scenario loader."""
    return build_robot_config_from_scenario(
        {
            "robot_config": {
                "type": "bicycle_drive",
                "radius": 0.64,
                "wheelbase": 0.9,
                "max_velocity": 1.34,
                "capsule": {"length": 1.183, "width": 0.478},
            }
        },
        scenario_path=Path("configs/robots/diagnostic.yaml"),
    )


def occupancy(*, heading=0.0, pedestrians=(), walls=(), other_pose=None, image=False):
    """Construct production occupancies; callbacks read live plant poses."""
    config = capsule_config()
    robots = [BicycleDriveRobot(config.robot_config)]
    robots[0].reset_state(((5.0, 5.0), heading))
    if other_pose is not None:
        robots.append(BicycleDriveRobot(config.robot_config))
        robots[1].reset_state(other_pose)
    sim = SimpleNamespace(
        robots=robots,
        map_def=SimpleNamespace(width=10.0, height=10.0, max_target_dist=15.0),
        ped_pos=np.asarray(pedestrians).reshape(-1, 2),
        get_obstacle_lines=lambda: np.asarray(walls).reshape(-1, 4),
        goal_pos=[(9.0, 9.0)] * len(robots),
    )

    # Property access must stay live as the plant rotates/moves.
    class LiveSimulator(SimpleNamespace):
        @property
        def robot_pos(self):
            return [r.pos for r in self.robots]

        @property
        def robot_poses(self):
            return [r.pose for r in self.robots]

    sim = LiveSimulator(**vars(sim))
    initializer = init_collision_and_sensors_with_image if image else init_collision_and_sensors
    occupancies, _ = initializer(sim, config, create_spaces(config, sim.map_def)[2])
    return occupancies, robots


@pytest.mark.parametrize("image", [False, True])
def test_capsule_side_clearance_and_live_heading(image):
    """A side gap clears the narrow body; rotation puts an end cap in contact."""
    occs, robots = occupancy(pedestrians=[(5.0, 5.7)], image=image)
    assert occs[0].is_pedestrian_collision is False
    robots[0].reset_state(((5.0, 5.0), pi / 2))
    assert occs[0].is_pedestrian_collision is True


@pytest.mark.parametrize("image", [False, True])
def test_capsule_side_wall_clearance_and_rotation(image):
    """Wall 0.3 m from centre clears the side but hits a rotated end cap."""
    occs, robots = occupancy(walls=[(4.0, 5.3, 6.0, 5.3)], image=image)
    assert occs[0].is_obstacle_collision is False
    robots[0].reset_state(((5.0, 5.0), pi / 2))
    assert occs[0].is_obstacle_collision is True


@pytest.mark.parametrize("image", [False, True])
def test_two_capsules_use_both_headings(image):
    """Parallel bodies clear at 0.8 m separation; a turned body makes contact."""
    occs, robots = occupancy(other_pose=((5.0, 5.8), 0.0), image=image)
    assert occs[0].is_dynamic_collision is False
    assert occs[1].is_dynamic_collision is False
    robots[1].reset_state(((5.0, 5.8), pi / 2))
    assert occs[0].is_dynamic_collision is True
    assert occs[1].is_dynamic_collision is True


def test_capsule_bounds_include_body_not_only_centre():
    """A centre inside the map cannot hide an end cap outside it."""
    occs, robots = occupancy()
    robots[0].reset_state(((0.3, 5.0), 0.0))
    assert occs[0].is_obstacle_collision is True
    robots[0].reset_state(((0.3, 5.0), pi / 2))
    assert occs[0].is_obstacle_collision is False


def test_capsule_hash_binds_geometry_without_changing_disc_serialization():
    """An opt-in footprint cannot share a config identity with its disc proxy."""
    from dataclasses import asdict

    from robot_sf.gym_env.robot_env import _stable_config_hash
    from robot_sf.robot.bicycle_drive import BicycleDriveSettings

    capsule = capsule_config()
    disc = RobotSimulationConfig(
        robot_config=BicycleDriveSettings(radius=0.64, wheelbase=0.9, max_velocity=1.34)
    )
    assert asdict(capsule.robot_config) == asdict(disc.robot_config)
    assert _stable_config_hash(capsule) != _stable_config_hash(disc)


@pytest.mark.parametrize(
    "dimensions",
    [
        {},
        {"length": 1.183},
        {"length": 1.183, "width": -0.478},
        {"length": 0.4, "width": 0.478},
        {"length": float("nan"), "width": 0.478},
        {"length": True, "width": 0.478},
        {"length": 1.183, "width": 0.478, "shape": "disc"},
    ],
)
def test_invalid_capsule_never_silently_falls_back_to_disc(dimensions):
    """Reject malformed authored bodies before any simulation is constructed."""
    with pytest.raises(ValueError, match="capsule"):
        build_robot_config_from_scenario(
            {
                "robot_config": {
                    "type": "bicycle_drive",
                    "capsule": dimensions,
                }
            },
            scenario_path=Path("configs/robots/diagnostic.yaml"),
        )


@pytest.mark.parametrize("degrees", [30, 45])
def test_authored_t60_profile_loads_exact_case_and_full_body(degrees):
    """A rounded disc proxy cannot accidentally replace an authored capsule case."""
    import math

    import yaml

    path = Path(f"configs/robots/t60_capsule_bicycle_{degrees}deg_v1.yaml")
    config = build_robot_config_from_scenario(yaml.safe_load(path.read_text()), scenario_path=path)
    body = config.robot_config.capsule
    assert body.length == 1.183 and body.width == 0.478
    assert config.robot_config.max_steer == math.radians(degrees)
    assert config.robot_config.wheelbase == 0.9
    assert config.robot_config.max_velocity == 1.34
    assert config.robot_config.radius == 0.64
    assert not config.robot_config.limited_reverse
    assert config.robot_config.creep_speed == 0.0


def test_named_legacy_reference_equals_default_robot_settings():
    """The named legacy config is the released default disc/drivetrain payload."""
    from dataclasses import asdict

    import yaml

    path = Path("configs/robots/legacy_reference_robot_v1.yaml")
    config = build_robot_config_from_scenario(yaml.safe_load(path.read_text()), scenario_path=path)
    assert asdict(config.robot_config) == asdict(RobotSimulationConfig().robot_config)


@pytest.mark.parametrize("drive", ["differential_drive", "holonomic"])
def test_capsule_on_unimplemented_drive_is_rejected(drive):
    """Unknown drive/body combinations must not drop the opt-in selector."""
    with pytest.raises(ValueError, match="capsule currently requires type bicycle_drive"):
        build_robot_config_from_scenario(
            {
                "robot_config": {
                    "type": drive,
                    "capsule": {"length": 1.183, "width": 0.478},
                }
            },
            scenario_path=Path("configs/robots/diagnostic.yaml"),
        )


def test_real_environment_uses_selected_body_and_executed_pose():
    """Exercise the full simulator/environment construction and stepping route."""
    from robot_sf.gym_env.environment_factory import make_robot_env

    config = capsule_config()
    env = make_robot_env(config=config, seed=1001)
    try:
        env.reset(seed=1001)
        env.step(np.array([0.2, 0.1], dtype=np.float32))
        assert env.state.occupancy.capsule is config.robot_config.capsule
        assert env.state.occupancy.capsule_pose == env.simulator.robots[0].pose
    finally:
        env.close()


@pytest.mark.parametrize("image", [False, True])
def test_capsule_collision_provider_does_not_hide_robots_from_lidar(image):
    """Until sensor migration, other capsules remain visible as explicit covering discs."""
    from robot_sf.sensor.range_sensor import LidarScannerSettings, lidar_ray_scan_ranges_only

    occs, robots = occupancy(other_pose=((5.0, 5.8), 0.0), image=image)
    assert not occs[0].is_dynamic_collision
    settings = LidarScannerSettings(num_rays=4, scan_noise=[0.0, 0.0])
    ranges = lidar_ray_scan_ranges_only(robots[0].pose, occs[0], settings)
    # The ray at pi/2 meets the retained 0.64 m covering disc at 0.8 - 0.64 m.
    assert ranges[3] == pytest.approx(0.16)


def test_dynamic_disc_peer_uses_its_own_contact_radius():
    """A disc peer must not inherit the capsule's conservative sensor proxy."""
    from robot_sf.robot.bicycle_drive import BicycleDriveSettings

    occs, robots = occupancy()
    neighbor = BicycleDriveRobot(BicycleDriveSettings(radius=0.2))
    neighbor.reset_state(((5.0, 5.7), 0.0))
    robots.append(neighbor)
    assert not occs[0].is_dynamic_collision  # 0.7 - 0.239 - 0.2 > 0
    neighbor.reset_state(((5.0, 5.4), 0.0))
    assert occs[0].is_dynamic_collision


def test_ego_pedestrian_side_gap_and_tangent_contact():
    """Both query directions retain an open side gap and inclusive tangency."""
    from robot_sf.nav.occupancy import EgoPedContinuousOccupancy

    config = capsule_config().robot_config
    body = getattr(config, "capsule", None)
    pedestrian = [(5.0, 5.6391)]
    body_fields = {"enemy_capsule": body, "get_enemy_pose": lambda: ((5.0, 5.0), 0.0)}
    ego = EgoPedContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: pedestrian[0],
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: [],
        get_pedestrian_coords=lambda: [],
        agent_radius=0.4,
        get_enemy_coords=lambda: (5.0, 5.0),
        enemy_radius=config.radius,
        **(body_fields if body is not None else {}),
    )
    assert not ego.is_agent_agent_collision
    pedestrian[0] = (5.0, 5.639)
    assert ego.is_agent_agent_collision
