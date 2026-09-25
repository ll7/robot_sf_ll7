"""Grid-resolution metamorphism for occupancy-aware planner inputs."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner import socnav
from robot_sf.planner.dwa import DWAPlannerAdapter, DWAPlannerConfig
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter, SocNavPlannerConfig

BASE_RESOLUTION = 0.2
GRID_RESOLUTIONS = (BASE_RESOLUTION, BASE_RESOLUTION / 2.0, BASE_RESOLUTION * 2.0)
COMMAND_ATOL = 0.05
FORCE_RTOL = 0.05
FORCE_ATOL = 0.05
SOCIAL_FORCE_V2_FIELD = "social_force_planner_version"
SOCIAL_FORCE_V2_VERSION = "resolution_independent_v2"


def _observation_with_wall(
    resolution: float,
    *,
    wall_present: bool = True,
    wall_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (0.8, 1.6),
        (-0.4, 0.4),
    ),
) -> dict:
    """Return a wall raster at one cell resolution, or its free-space control."""
    origin = (-2.0, -4.0)
    size = (8.0, 8.0)
    width = round(size[0] / resolution)
    height = round(size[1] / resolution)
    grid = np.zeros((4, height, width), dtype=np.float32)

    x_centers = origin[0] + (np.arange(width, dtype=float) + 0.5) * resolution
    y_centers = origin[1] + (np.arange(height, dtype=float) + 0.5) * resolution
    if wall_present:
        (wall_x_min, wall_x_max), (wall_y_min, wall_y_max) = wall_bounds
        wall_columns = (x_centers >= wall_x_min) & (x_centers < wall_x_max)
        wall_rows = (y_centers >= wall_y_min) & (y_centers < wall_y_max)
        wall_mask = np.ix_(wall_rows, wall_columns)
        grid[0][wall_mask] = 1.0
        grid[3][wall_mask] = 1.0

    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=np.float32),
            "heading": np.asarray([0.0], dtype=np.float32),
            "speed": np.asarray([0.0, 0.0], dtype=np.float32),
            "radius": np.asarray([0.3], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray([8.0, 0.0], dtype=np.float32),
            "next": np.asarray([8.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((1, 2), dtype=np.float32),
            "velocities": np.zeros((1, 2), dtype=np.float32),
            "radius": np.asarray([0.3], dtype=np.float32),
            "count": np.asarray([0.0], dtype=np.float32),
        },
        "sim": {"timestep": np.asarray([0.1], dtype=np.float32)},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray(origin, dtype=np.float32),
        "occupancy_grid_meta_resolution": np.asarray([resolution], dtype=np.float32),
        "occupancy_grid_meta_size": np.asarray(size, dtype=np.float32),
        "occupancy_grid_meta_use_ego_frame": np.asarray([0.0], dtype=np.float32),
        "occupancy_grid_meta_center_on_robot": np.asarray([0.0], dtype=np.float32),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=np.float32),
        "occupancy_grid_meta_robot_pose": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }


class _ForceRecordingSocialForcePlanner(socnav.SocialForcePlannerAdapter):
    """Capture the obstacle force used by ``plan`` for the metamorphic assertion."""

    def _compute_obstacle_force(
        self,
        observation: dict,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_vel: np.ndarray,
        robot_state: dict,
    ) -> np.ndarray:
        force = super()._compute_obstacle_force(
            observation,
            robot_pos,
            robot_heading,
            robot_vel,
            robot_state,
        )
        self.obstacle_force = np.array(force, dtype=float, copy=True)
        return force


def _measure_social_force(
    resolution: float, config: socnav.SocNavPlannerConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Run one normal planner call and return its command and obstacle force."""
    planner = _ForceRecordingSocialForcePlanner(config)
    command = np.asarray(planner.plan(_observation_with_wall(resolution)), dtype=float)
    return command, planner.obstacle_force


def _measure_dwa(resolution: float, *, wall_present: bool) -> np.ndarray:
    """Run DWA on the float32 occupancy observation used by the resolution probe."""
    observation = _observation_with_wall(
        resolution,
        wall_present=wall_present,
        wall_bounds=((1.2, 2.4), (-0.8, 0.8)),
    )
    planner = DWAPlannerAdapter(DWAPlannerConfig(prediction_steps=25))
    return np.asarray(planner.plan(observation), dtype=float)


def _occupied_wall_bounds(observation: dict) -> tuple[float, float, float, float]:
    """Return the rasterized occupied extent in world coordinates."""
    occupied_rows, occupied_columns = np.where(observation["occupancy_grid"][0] > 0.0)
    assert occupied_rows.size > 0, "wall raster must contain occupied cells"
    origin_x, origin_y = observation["occupancy_grid_meta_origin"]
    resolution = float(observation["occupancy_grid_meta_resolution"][0])
    return (
        float(origin_x + occupied_columns.min() * resolution),
        float(origin_x + (occupied_columns.max() + 1) * resolution),
        float(origin_y + occupied_rows.min() * resolution),
        float(origin_y + (occupied_rows.max() + 1) * resolution),
    )


# The ORCA wall sits 2.0-3.2 m ahead and slightly left, so its command is neither
# speed- nor turn-rate-saturated: an obstacle-geometry error that scales with the
# cell size (a margin counted in cells, a radius off by one cell) moves the
# command by more than the tolerance instead of being hidden by a clip.
ORCA_WALL_BOUNDS = ((2.0, 3.2), (-0.8, 0.4))
SATURATION_MARGIN = 0.05


@pytest.mark.skipif(socnav.rvo2 is None, reason="rvo2 is required for native ORCA execution")
def test_orca_command_is_invariant_to_grid_resolution_with_wall_influence() -> None:
    """Native ORCA keeps an unsaturated wall response across resolutions."""
    config = SocNavPlannerConfig()

    def command(resolution: float, *, wall_present: bool) -> np.ndarray:
        planner = ORCAPlannerAdapter(config, allow_fallback=False)
        return np.asarray(
            planner.plan(
                _observation_with_wall(
                    resolution, wall_present=wall_present, wall_bounds=ORCA_WALL_BOUNDS
                )
            ),
            dtype=float,
        )

    wall_commands = np.asarray(
        [command(resolution, wall_present=True) for resolution in GRID_RESOLUTIONS]
    )
    free_commands = np.asarray(
        [command(resolution, wall_present=False) for resolution in GRID_RESOLUTIONS]
    )

    assert np.all(np.linalg.norm(wall_commands - free_commands, axis=1) > COMMAND_ATOL), (
        "the static wall must change each resolution's ORCA command"
    )
    assert np.all(wall_commands[:, 0] < config.max_linear_speed - SATURATION_MARGIN), (
        f"wall response saturates the linear speed cap: {wall_commands[:, 0]}"
    )
    assert np.all(np.abs(wall_commands[:, 1]) < config.max_angular_speed - SATURATION_MARGIN), (
        f"wall response saturates the turn-rate cap: {wall_commands[:, 1]}"
    )
    np.testing.assert_allclose(
        wall_commands[1:],
        np.broadcast_to(wall_commands[0], wall_commands[1:].shape),
        rtol=0.0,
        atol=COMMAND_ATOL,
        err_msg="ORCA command changed after halving or doubling occupancy resolution",
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known DWA grid-resolution dependence tracked by #9740; remove after that fix merges.",
)
def test_dwa_command_is_invariant_to_grid_resolution_with_wall_influence() -> None:
    """The same float32 wall and free control should give stable DWA commands."""
    wall_bounds = ((1.2, 2.4), (-0.8, 0.8))
    observations = [
        _observation_with_wall(
            resolution,
            wall_bounds=wall_bounds,
        )
        for resolution in GRID_RESOLUTIONS
    ]
    for observation in observations:
        np.testing.assert_allclose(
            _occupied_wall_bounds(observation),
            (1.2, 2.4, -0.8, 0.8),
            rtol=0.0,
            atol=1e-6,
            err_msg="each raster must represent the same physical wall bounds",
        )

    wall_commands = np.asarray(
        [_measure_dwa(resolution, wall_present=True) for resolution in GRID_RESOLUTIONS]
    )
    free_commands = np.asarray(
        [_measure_dwa(resolution, wall_present=False) for resolution in GRID_RESOLUTIONS]
    )

    np.testing.assert_allclose(
        free_commands[1:],
        np.broadcast_to(free_commands[0], free_commands[1:].shape),
        rtol=0.0,
        atol=COMMAND_ATOL,
        err_msg="free-space DWA command changed with occupancy-grid resolution",
    )
    assert np.any(np.linalg.norm(wall_commands - free_commands, axis=1) > COMMAND_ATOL), (
        "the wall must affect at least one DWA command so the obstacle is exercised"
    )
    np.testing.assert_allclose(
        wall_commands[1:],
        np.broadcast_to(wall_commands[0], wall_commands[1:].shape),
        rtol=0.0,
        atol=COMMAND_ATOL,
        err_msg="DWA command changed after halving or doubling occupancy resolution",
    )


def _assert_social_force_resolution_invariant(
    config: socnav.SocNavPlannerConfig,
) -> None:
    """Compare commands and obstacle forces at half, base and double resolution."""
    measurements = [_measure_social_force(resolution, config) for resolution in GRID_RESOLUTIONS]
    commands = np.asarray([command for command, _force in measurements])
    forces = np.asarray([force for _command, force in measurements])

    assert commands.shape == forces.shape == (len(GRID_RESOLUTIONS), 2)
    assert np.all(np.linalg.norm(forces, axis=1) > 0.0), "wall force must be exercised"
    np.testing.assert_allclose(
        forces[1:],
        np.broadcast_to(forces[0], forces[1:].shape),
        rtol=FORCE_RTOL,
        atol=FORCE_ATOL,
        err_msg="obstacle forces changed after halving or doubling occupancy resolution",
    )
    np.testing.assert_allclose(
        commands[1:],
        np.broadcast_to(commands[0], commands[1:].shape),
        rtol=0.0,
        atol=COMMAND_ATOL,
        err_msg="planner commands changed after halving or doubling occupancy resolution",
    )


@pytest.mark.skipif(
    socnav.sf_forces is None,
    reason="pysocialforce (fast-pysf) is required for the SocialForcePlannerAdapter path",
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "#9724: the legacy grid_cell_sum_v1 obstacle term (still the SocNavPlannerConfig "
        "default) sums one force per occupied cell; the release arm now uses v2, whose "
        "twin below passes"
    ),
)
def test_social_force_command_and_force_are_invariant_to_grid_resolution() -> None:
    """The legacy (code-default) social-force obstacle term sums one force per cell."""
    _assert_social_force_resolution_invariant(socnav.SocNavPlannerConfig())


@pytest.mark.skipif(
    socnav.sf_forces is None,
    reason="pysocialforce (fast-pysf) is required for the SocialForcePlannerAdapter path",
)
def test_social_force_v2_command_and_force_are_invariant_to_grid_resolution() -> None:
    """The v2 obstacle term used by the release social_force arm is resolution invariant."""
    config = socnav.SocNavPlannerConfig(
        **{SOCIAL_FORCE_V2_FIELD: SOCIAL_FORCE_V2_VERSION},
    )
    _assert_social_force_resolution_invariant(config)
