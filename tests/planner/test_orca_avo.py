"""Physical-limit, swept-exclusion and default-ORCA compatibility contracts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString

from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.planner.orca_avo import (
    ORCAAVOConfig,
    ORCAAVOPlannerAdapter,
    acceleration_paths,
    reachable_targets,
    trajectories_clear,
)
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def observation(speed=0.0, heading=0.0):
    """Provide the real nested observation schema without simulator truth."""
    return {
        "robot": {
            "position": np.zeros(2),
            "heading": [heading],
            "speed": [speed, 0.0],
            "radius": [0.2],
        },
        "goal": {"current": [5.0, 2.0], "next": [5.0, 2.0]},
        "pedestrians": {
            "positions": np.array([[2.0, 0.5]]),
            "velocities": np.array([[0.1, 0.0]]),
            "count": [1],
            "radius": [0.3],
        },
        "sim": {"timestep": [0.1]},
    }


def bind(adapter, drive=None, segments=()):
    """Bind actual typed settings via the production adapter entrypoint."""
    drive = drive or DifferentialDriveSettings(radius=0.2)
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=drive),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: segments),
        )
    )


@pytest.mark.parametrize("current", [[0, 0], [1.5, 0], [0, -2.0]])
@pytest.mark.parametrize("delta", [0.05, 0.1, 0.2])
def test_reachable_disk_and_asymmetric_real_limits(current, delta):
    """Every candidate obeys both disks even at the physical speed boundary."""
    drive = DifferentialDriveSettings(max_linear_accel=0.4, max_linear_decel=0.2)
    targets = reachable_targets([3, 2], current, delta, drive)
    assert np.all(np.linalg.norm(targets - current, axis=1) <= delta * 0.2 + 1e-12)
    assert np.all(np.linalg.norm(targets, axis=1) <= drive.max_linear_speed + 1e-12)


def test_exact_projection_and_exponential_trajectory():
    """Unobstructed nearest target and exact lag integration have analytic answers."""
    drive = DifferentialDriveSettings()
    targets = reachable_targets([2, 0], [1, 0], 0.1, drive)
    np.testing.assert_allclose(targets[0], [1.1, 0])
    times = np.array([0.0, 0.1, 1.0])
    paths = acceleration_paths([[1.1, 0]], [1, 0], [0, 0], 0.1, times)
    np.testing.assert_allclose(paths[0, :, 0], 1.1 * times - 0.01 * (1 - np.exp(-times / 0.1)))
    np.testing.assert_allclose(paths[0, :, 1], 0)


def test_excludes_crossing_between_samples_and_moving_pedestrian():
    """Safe endpoints alone cannot admit an intervening pedestrian crossing."""
    paths = np.array([[[0, 0], [2, 0]], [[0, 3], [2, 3]]], dtype=float)
    clear = trajectories_clear(
        paths,
        np.array([0, 1]),
        np.array([[1, -1]]),
        np.array([[0, 2]]),
        (0.1, 0.1),
        GeometryCollection(),
        0,
    )
    assert clear.tolist() == [False, True]


def test_excludes_static_wall_and_tangent_contact():
    """Whole swept polylines reject a thin wall and exact body tangency."""
    paths = np.array([[[0, 0], [2, 0]], [[0, 2], [2, 2]]], dtype=float)
    clear = trajectories_clear(
        paths,
        np.array([0, 1]),
        np.empty((0, 2)),
        np.empty((0, 2)),
        (0.2, 0.1),
        LineString([(1, -1), (1, 0.2)]),
        0,
    )
    assert clear.tolist() == [False, True]
    tangent = trajectories_clear(
        paths[:1],
        np.array([0, 1]),
        np.array([[1, 0.3]]),
        np.zeros((1, 2)),
        (0.2, 0.1),
        GeometryCollection(),
        0,
    )
    assert not tangent[0]


def test_real_binding_nearest_choice_and_explicit_empty_safe_set(monkeypatch):
    """Integration reads live limits and marks the braking escape unresolved."""
    adapter = ORCAAVOPlannerAdapter()
    with pytest.raises(RuntimeError, match="Bind"):
        adapter.plan_velocity_world(observation())
    drive = DifferentialDriveSettings(radius=0.2, max_linear_accel=0.3, max_linear_decel=0.2)
    bind(adapter, drive)
    monkeypatch.setattr(ORCAPlannerAdapter, "plan_velocity_world", lambda *_: np.array([2.0, 0.0]))
    obs = observation(speed=0.5)
    obs["pedestrians"]["count"] = [0]
    np.testing.assert_allclose(adapter.plan_velocity_world(obs), [0.52, 0])
    assert adapter._avo_last["status"] == "safe_sampled_target"
    bind(adapter, drive, [((0, -2), (0, 2))])
    np.testing.assert_allclose(adapter.plan_velocity_world(obs), [0.48, 0])
    assert adapter._avo_last["status"] == "no_safe_target_braking"
    assert adapter.diagnostics()["avo"]["no_safe_target_steps"] == 1
    adapter.reset()
    assert adapter.diagnostics()["avo"]["no_safe_target_steps"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"avo_horizon": 0}, {"avo_sample_dt": float("nan")}, {"avo_rings": 0}, {"avo_angles": 1.5}],
)
def test_invalid_config_fails_closed(kwargs):
    """Malformed candidate contracts cannot become silent degraded runs."""
    with pytest.raises(ValueError):
        ORCAAVOConfig(**kwargs)


def test_reject_unsupported_plant_and_fallback():
    """There is no guessed drive or heuristic fallback on the research path."""
    with pytest.raises(ValueError, match="fallback"):
        ORCAAVOPlannerAdapter(allow_fallback=True)
    with pytest.raises(ValueError, match="forward"):
        bind(ORCAAVOPlannerAdapter(), DifferentialDriveSettings(allow_backwards=True))


def test_opt_in_registration_and_default_byte_equivalence():
    """Plain ORCA command/trace bytes match the unchanged baseline implementation."""
    pytest.importorskip("rvo2")
    config = {"max_linear_speed": 2.0, "max_angular_speed": 1.0, "orca_adapter_trace_enabled": True}
    policy, _ = _build_policy("orca", config, robot_kinematics="differential_drive")
    legacy = ORCAPlannerAdapter(SocNavPlannerConfig(**config))
    assert type(policy._planner_adapter) is ORCAPlannerAdapter
    for speed, heading in [(0.0, 0.0), (0.6, 0.4), (1.8, -2.0), (0.2, 1.0)]:
        obs = observation(speed, heading)
        assert np.asarray(policy(obs)).tobytes() == np.asarray(legacy.plan(obs)).tobytes()
    assert (
        json.dumps(policy._planner_adapter.adapter_trace(), sort_keys=True).encode()
        == json.dumps(legacy.adapter_trace(), sort_keys=True).encode()
    )
    source = Path(__file__).resolve().parents[2] / "robot_sf/planner/socnav_orca.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "88a27eece259b3d6a57881f7ecf55bbce483d5d0e78b26c161aa6429c87f54f2"
    )
    variant, meta = _build_policy("orca_avo", config, robot_kinematics="differential_drive")
    assert isinstance(variant._planner_adapter, ORCAAVOPlannerAdapter)
    assert meta["status"] == "ok"


def test_filter_rejects_local_grid_and_preserves_cv_velocity_frame(monkeypatch):
    """The real integration excludes grid contacts and rotates ego CV predictions."""
    adapter = ORCAAVOPlannerAdapter(ORCAAVOConfig(avo_horizon=1.0))
    bind(adapter)
    monkeypatch.setattr(ORCAPlannerAdapter, "plan_velocity_world", lambda *_: np.array([0.0, 1.0]))
    obs = observation(speed=0.0, heading=np.pi / 2)
    obs["pedestrians"]["count"] = [0]
    np.testing.assert_allclose(adapter.plan_velocity_world(obs), [0, 0.1], atol=1e-12)
    monkeypatch.setattr(
        adapter,
        "_extract_obstacles_from_grid",
        lambda *_: (np.array([[0.0, 0.0]]), np.array([0.1])),
    )
    adapter.plan_velocity_world(obs)
    assert adapter._avo_last["status"] == "no_safe_target_braking"
    monkeypatch.setattr(
        adapter, "_extract_obstacles_from_grid", lambda *_: (np.empty((0, 2)), np.empty(0))
    )
    # A pedestrian at world (0,1) moving ego-left travels toward the stationary
    # robot at heading pi/2: ego (+1,0) -> world (0,+1), ego (-1,0) -> (0,-1).
    obs["pedestrians"].update(
        positions=np.array([[0.0, 1.0]]), velocities=np.array([[-1.0, 0.0]]), count=[1]
    )
    adapter.plan_velocity_world(obs)
    assert adapter._avo_last["status"] == "no_safe_target_braking"
