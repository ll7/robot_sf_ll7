"""Surface-distance pedestrian repulsion v3 for the social-force planner (issue #9758)."""

import numpy as np
import pytest

from robot_sf.planner import socnav_social_force as sf
from robot_sf.planner.socnav_base import (
    SOCIAL_FORCE_PED_LEGACY_KERNEL,
    SOCIAL_FORCE_PED_SURFACE_V3,
    SocNavPlannerConfig,
    resolve_social_force_ped_version,
)


def _v3_config(**overrides):
    kwargs = {"social_force_ped_version": SOCIAL_FORCE_PED_SURFACE_V3}
    kwargs.update(overrides)
    return SocNavPlannerConfig(**kwargs)


def _ped_state(positions, velocities=None):
    positions = np.asarray(positions, dtype=float)
    if velocities is None:
        velocities = np.zeros_like(positions)
    return {
        "positions": positions,
        "velocities": np.asarray(velocities, dtype=float),
        "count": [len(positions)],
    }


def test_ped_version_resolution_defaults_to_legacy() -> None:
    """Unset or blank ped version keeps the historical kernel path."""
    assert resolve_social_force_ped_version(None) == SOCIAL_FORCE_PED_LEGACY_KERNEL
    assert resolve_social_force_ped_version("  ") == SOCIAL_FORCE_PED_LEGACY_KERNEL
    assert resolve_social_force_ped_version("surface_v3") == SOCIAL_FORCE_PED_SURFACE_V3


@pytest.mark.parametrize("value", ["unsupported", 3, object()])
def test_config_rejects_invalid_ped_version_at_construction(value) -> None:
    """Invalid selectors fail before an adapter can run with ambiguous semantics."""
    with pytest.raises((TypeError, ValueError)):
        SocNavPlannerConfig(social_force_ped_version=value)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_config_preserves_legacy_ped_version_for_unset_or_blank(value) -> None:
    """Missing and blank selectors retain the historical kernel."""
    config = SocNavPlannerConfig(social_force_ped_version=value)
    assert config.social_force_ped_version == SOCIAL_FORCE_PED_LEGACY_KERNEL


def test_standing_pedestrian_ahead_repels_beyond_goal_force() -> None:
    """At contact surface distance the v3 term exceeds the ~2 goal force (issue #9758)."""
    adapter = sf.SocialForcePlannerAdapter(_v3_config())
    ped = _ped_state([[1.4, 0.0]])  # 1.4 m centre == 0.0 m surface at release radii
    force = adapter._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), ped, 0.0, robot_radius=1.0
    )
    assert np.all(np.isfinite(force))
    assert force[0] < 0.0  # opposes forward motion
    assert float(np.linalg.norm(force)) >= 2.0


def test_v3_dominates_legacy_kernel_at_contact() -> None:
    """The legacy centre-distance kernel (~0.09 here) cannot act before contact."""
    legacy = sf.SocialForcePlannerAdapter(SocNavPlannerConfig())
    v3 = sf.SocialForcePlannerAdapter(_v3_config())
    ped = _ped_state([[1.4, 0.0]])
    legacy_force = legacy._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), ped, 0.0, robot_radius=1.0
    )
    v3_force = v3._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), ped, 0.0, robot_radius=1.0
    )
    assert float(np.linalg.norm(v3_force)) > 10.0 * float(np.linalg.norm(legacy_force))


def test_v3_fades_with_surface_distance() -> None:
    """Far pedestrians contribute negligibly (derivation table: d_s = 2 m -> ~0.09*w)."""
    adapter = sf.SocialForcePlannerAdapter(_v3_config())
    near = _ped_state([[1.4, 0.0]])
    far = _ped_state([[3.4, 0.0]])  # 2.0 m surface distance
    near_force = adapter._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), near, 0.0, robot_radius=1.0
    )
    far_force = adapter._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), far, 0.0, robot_radius=1.0
    )
    assert float(np.linalg.norm(far_force)) < 0.2 * float(np.linalg.norm(near_force))


def test_v3_mirror_symmetry() -> None:
    """Mirrored pedestrians give mirrored lateral forces."""
    adapter = sf.SocialForcePlannerAdapter(_v3_config())
    left = _ped_state([[2.0, 0.5]])
    right = _ped_state([[2.0, -0.5]])
    force_left = adapter._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), left, 0.0, robot_radius=1.0
    )
    force_right = adapter._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), right, 0.0, robot_radius=1.0
    )
    np.testing.assert_allclose(force_left[0], force_right[0], rtol=1e-9)
    np.testing.assert_allclose(force_left[1], -force_right[1], rtol=1e-9)


def test_v3_empty_pedestrians_is_zero() -> None:
    """No pedestrians means no repulsion."""
    adapter = sf.SocialForcePlannerAdapter(_v3_config())
    force = adapter._compute_social_force(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), _ped_state(np.zeros((0, 2))), 0.0
    )
    np.testing.assert_array_equal(force, np.zeros(2))


def test_v3_diagnostics_reports_ped_version() -> None:
    """Diagnostics names the active ped-term version."""
    adapter = sf.SocialForcePlannerAdapter(_v3_config())
    assert adapter.diagnostics()["ped_version"] == SOCIAL_FORCE_PED_SURFACE_V3
    legacy = sf.SocialForcePlannerAdapter(SocNavPlannerConfig())
    assert legacy.diagnostics()["ped_version"] == SOCIAL_FORCE_PED_LEGACY_KERNEL


def _head_on_obs(
    ped_xy: tuple[float, float],
    *,
    ped_velocity: tuple[float, float] = (0.0, 0.0),
    robot_xy: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    robot_speed: float = 1.0,
    robot_radius: float = 0.5,
    dt: float = 0.1,
) -> dict:
    """Build a deterministic world-frame SocNav observation for one pedestrian."""
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    world_velocity = np.asarray(ped_velocity, dtype=float)
    # SocNav pedestrian velocities are ego-frame; keep the requested crossing
    # speed constant in world coordinates as the robot turns.
    ego_velocity = np.array(
        [
            cos_h * world_velocity[0] + sin_h * world_velocity[1],
            -sin_h * world_velocity[0] + cos_h * world_velocity[1],
        ],
        dtype=float,
    )
    return {
        "robot": {
            "position": np.asarray(robot_xy, dtype=float),
            "heading": np.array([heading]),
            "speed": np.array([robot_speed, 0.0]),
            "radius": np.array([robot_radius]),
        },
        "goal": {
            "current": np.array([5.0, 0.0]),
            "next": np.array([0.0, 0.0]),
        },
        "pedestrians": {
            "positions": np.array([ped_xy]),
            "velocities": np.array([ego_velocity]),
            "radius": np.array([0.4]),
            "count": np.array([1.0]),
        },
        "map": {"size": np.array([10.0, 10.0])},
        "sim": {"timestep": np.array([dt])},
    }


def test_v3_brakes_for_close_head_on_pedestrian() -> None:
    """Plan-level: v3 commands less forward speed than legacy with a close ped ahead."""
    legacy = sf.SocialForcePlannerAdapter(SocNavPlannerConfig())
    v3 = sf.SocialForcePlannerAdapter(_v3_config())
    obs = _head_on_obs((1.4, 0.0))
    legacy_cmd = legacy.plan(obs)
    v3_cmd = v3.plan(obs)
    assert np.all(np.isfinite(v3_cmd))
    assert 0.0 <= v3_cmd[0] <= 3.0 + 1e-9
    assert v3_cmd[0] < legacy_cmd[0]


def _unicycle_step(
    position: np.ndarray, heading: float, command: tuple[float, float], duration: float
) -> tuple[np.ndarray, float]:
    """Integrate one constant unicycle command exactly over ``duration``."""
    linear, angular = (float(command[0]), float(command[1]))
    if abs(angular) < 1e-12:
        return position + duration * linear * np.array([np.cos(heading), np.sin(heading)]), heading
    next_heading = heading + duration * angular
    next_position = position + (linear / angular) * np.array(
        [np.sin(next_heading) - np.sin(heading), -np.cos(next_heading) + np.cos(heading)],
    )
    return next_position, next_heading


def _rollout_v3(
    *,
    ped_start: tuple[float, float],
    ped_velocity: tuple[float, float],
    dt: float,
    horizon_s: float,
) -> tuple[float, float]:
    """Roll out the adapter and return swept minimum clearance and final speed."""
    adapter = sf.SocialForcePlannerAdapter(_v3_config())
    robot_position = np.zeros(2, dtype=float)
    heading = 0.0
    robot_speed = 0.0
    ped_start_arr = np.asarray(ped_start, dtype=float)
    ped_velocity_arr = np.asarray(ped_velocity, dtype=float)
    minimum_clearance = float("inf")
    steps = round(horizon_s / dt)
    for step in range(steps):
        ped_position = ped_start_arr + step * dt * ped_velocity_arr
        observation = _head_on_obs(
            tuple(ped_position),
            ped_velocity=ped_velocity,
            robot_xy=tuple(robot_position),
            heading=heading,
            robot_speed=robot_speed,
            robot_radius=1.0,
            dt=dt,
        )
        command = adapter.plan(observation)
        # Check the commanded arc against the pedestrian's linear path, rather
        # than only checking sampled endpoint states for contact.
        for fraction in np.linspace(0.0, 1.0, num=21):
            sample_robot, _ = _unicycle_step(robot_position, heading, command, dt * float(fraction))
            sample_pedestrian = ped_position + dt * float(fraction) * ped_velocity_arr
            minimum_clearance = min(
                minimum_clearance,
                float(np.linalg.norm(sample_robot - sample_pedestrian) - 1.0 - 0.4),
            )
        robot_position, heading = _unicycle_step(robot_position, heading, command, dt)
        robot_speed = float(command[0])
    return minimum_clearance, robot_speed


@pytest.mark.parametrize("dt", [0.1, 0.05])
def test_v3_standing_pedestrian_rollout_stops_before_contact(dt: float) -> None:
    """A standing pedestrian is kept clear by the actual commanded trajectory."""
    minimum_clearance, final_speed = _rollout_v3(
        ped_start=(1.8, 0.0), ped_velocity=(0.0, 0.0), dt=dt, horizon_s=4.0
    )
    assert minimum_clearance > 0.05
    assert final_speed < 0.05


@pytest.mark.parametrize("dt", [0.1, 0.05])
def test_v3_crossing_pedestrian_rollout_avoids_contact(dt: float) -> None:
    """A 1.3 m/s crossing pedestrian completes the encounter without contact."""
    minimum_clearance, _ = _rollout_v3(
        ped_start=(2.0, -1.3), ped_velocity=(0.0, 1.3), dt=dt, horizon_s=3.0
    )
    assert minimum_clearance > 0.05


def test_v3_exact_overlap_fails_closed_without_changing_legacy() -> None:
    """An exact v3 overlap stops, while the legacy selector remains historical."""
    overlap = _head_on_obs((0.0, 0.0), robot_speed=0.0)
    v3_command = sf.SocialForcePlannerAdapter(_v3_config()).plan(overlap)
    legacy_command = sf.SocialForcePlannerAdapter(SocNavPlannerConfig()).plan(overlap)
    assert v3_command == (0.0, 0.0)
    assert legacy_command[0] > 0.0
