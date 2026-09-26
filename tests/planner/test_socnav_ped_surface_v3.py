"""Surface-distance pedestrian repulsion v3 for the social-force planner (issue #9758)."""

import numpy as np

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


def _head_on_obs(ped_xy: tuple[float, float]) -> dict:
    """Minimal SocNav observation: robot at origin facing a goal at (5, 0)."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "speed": np.array([1.0, 0.0]),
            "radius": np.array([0.5]),
        },
        "goal": {
            "current": np.array([5.0, 0.0]),
            "next": np.array([0.0, 0.0]),
        },
        "pedestrians": {
            "positions": np.array([ped_xy]),
            "velocities": np.zeros((1, 2)),
            "radius": np.array([0.4]),
            "count": np.array([1.0]),
        },
        "map": {"size": np.array([10.0, 10.0])},
        "sim": {"timestep": np.array([0.1])},
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
