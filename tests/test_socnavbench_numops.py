"""Regression tests for vendored SocNavBench NumPy operation paths."""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np


def _socnav_root() -> Path:
    """Return the vendored SocNavBench root."""
    return Path(__file__).resolve().parents[1] / "third_party" / "socnavbench"


def test_voxel_map_interpolation_uses_numpy_indices(monkeypatch) -> None:
    """VoxelMap should interpolate wrapped NumPy indices and reject invalid points."""
    monkeypatch.syspath_prepend(str(_socnav_root()))
    voxel_module = importlib.import_module("utils.voxel_map_utils")

    voxel_map = voxel_module.VoxelMap(
        scale=np.float32(1.0),
        origin_2=np.array([0.0, 0.0], dtype=np.float32),
        map_size_2=np.array([3, 3], dtype=np.int32),
        function_array_mn=np.array(
            [
                [0.0, 10.0, 20.0],
                [100.0, 110.0, 120.0],
                [200.0, 210.0, 220.0],
            ],
            dtype=np.float32,
        ),
    )

    positions = np.array([[[0.5, 0.5], [1.0, 1.0], [2.2, 0.5]]], dtype=np.float32)

    values = voxel_map.compute_voxel_function(positions, invalid_value=999.0)

    np.testing.assert_allclose(values, np.array([[55.0, 110.0, 999.0]], dtype=np.float32))


def test_control_pipeline_waypoint_selection_wraps_heading(monkeypatch) -> None:
    """Closest waypoint selection should compare headings with wraparound semantics."""
    monkeypatch.syspath_prepend(str(_socnav_root()))
    helper_module = importlib.import_module("control_pipelines.control_pipeline_v0_helper")
    trajectory_module = importlib.import_module("trajectory.trajectory")

    desired = trajectory_module.SystemConfig.from_pos3([0.0, 0.0, -math.pi + 0.05])
    waypoints = trajectory_module.Trajectory.from_pos3_array(
        np.array([[[0.0, 0.0, math.pi - 0.05], [0.0, 0.0, 0.0]]], dtype=np.float32)
    )

    idx = helper_module.ControlPipelineV0Helper().compute_closest_waypt_idx(
        desired,
        waypoints,
    )

    assert idx == 0


def test_dubins_egocentric_world_roundtrip_preserves_pose(monkeypatch) -> None:
    """Dubins coordinate transforms should stay NumPy-native and angle-normalized."""
    monkeypatch.syspath_prepend(str(_socnav_root()))
    dubins_module = importlib.import_module("systems.dubins_car")
    trajectory_module = importlib.import_module("trajectory.trajectory")

    ref = trajectory_module.SystemConfig.from_pos3([1.0, 2.0, math.pi / 2])
    world = trajectory_module.Trajectory.from_pos3_array(
        np.array([[[1.0, 3.0, -math.pi + 0.1], [2.0, 2.0, math.pi - 0.2]]], dtype=np.float32)
    )

    ego = dubins_module.DubinsCar.to_egocentric_coordinates(ref, world, mode="new")
    roundtrip = dubins_module.DubinsCar.to_world_coordinates(ref, ego, mode="new")

    np.testing.assert_allclose(roundtrip.position_nk2(), world.position_nk2(), atol=1e-6)
    np.testing.assert_allclose(roundtrip.heading_nk1(), world.heading_nk1(), atol=1e-6)
