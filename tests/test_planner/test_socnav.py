"""Unit tests for SocNav planner adapters and occupancy helpers."""
# ruff: noqa: D103

import configparser
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from robot_sf.planner.socnav import (
    SamplingPlannerAdapter,
    SocNavBenchSamplingAdapter,
    SocNavPlannerConfig,
)


def _base_observation() -> dict:
    obs = {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "speed": np.array([0.0]),
            "radius": np.array([0.3]),
        },
        "goal": {"current": np.array([2.0, 0.0]), "next": None},
        "pedestrians": {
            "positions": np.zeros((0, 2)),
            "count": np.array([0]),
            "radius": np.array([0.3]),
        },
    }
    meta = {
        "occupancy_grid_meta_origin": np.array([0.0, 0.0]),
        "occupancy_grid_meta_resolution": np.array([1.0]),
        "occupancy_grid_meta_size": np.array([3.0, 3.0]),
        "occupancy_grid_meta_use_ego_frame": np.array([0.0]),
        "occupancy_grid_meta_center_on_robot": np.array([0.0]),
        "occupancy_grid_meta_channel_indices": np.array([0, 1, 2, 3]),
        "occupancy_grid_meta_robot_pose": np.array([0.0, 0.0, 0.0]),
    }
    obs.update(meta)
    return obs


def _with_grid(observation: dict, grid: np.ndarray) -> dict:
    obs = dict(observation)
    obs["occupancy_grid"] = grid
    return obs


def test_extract_grid_payload_handles_flattened_meta():
    adapter = SamplingPlannerAdapter()
    obs = _base_observation()
    grid = np.zeros((4, 3, 3), dtype=float)
    obs = _with_grid(obs, grid)

    payload = adapter._extract_grid_payload(obs)

    assert payload is not None
    grid_arr, meta = payload
    assert grid_arr.shape == (4, 3, 3)
    assert meta["resolution"][0] == 1.0


def test_path_penalty_increases_when_obstacle_present():
    adapter = SamplingPlannerAdapter()
    obs = _base_observation()
    clear_grid = np.zeros((4, 3, 3), dtype=float)
    obstacle_grid = clear_grid.copy()
    # Place obstacle in combined channel directly ahead of the robot
    obstacle_grid[3, 0, 1] = 1.0

    penalty_clear = adapter._path_penalty(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        _with_grid(obs, clear_grid),
        base_distance=2.0,
        num_samples=2,
    )[0]
    penalty_blocked = adapter._path_penalty(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        _with_grid(obs, obstacle_grid),
        base_distance=2.0,
        num_samples=2,
    )[0]

    assert penalty_blocked > penalty_clear


def test_select_safe_heading_deflects_from_occupied_direction():
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(occupancy_weight=3.0))
    obs = _base_observation()
    grid = np.zeros((4, 3, 3), dtype=float)
    grid[3, 0, 1] = 1.0  # obstacle ahead
    obs = _with_grid(obs, grid)

    best_dir, penalty = adapter._select_safe_heading(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        obs,
        sweep=adapter.config.occupancy_heading_sweep,
        num_candidates=5,
        lookahead=2.0,
        weight=adapter.config.occupancy_weight,
        angle_weight=adapter.config.occupancy_angle_weight,
    )

    assert best_dir[0] < 0.99  # deflected away from straight ahead
    assert abs(best_dir[1]) > 0.0
    assert penalty >= 0.0


def test_sampling_planner_respects_goal_tolerance_and_occupancy():
    adapter = SamplingPlannerAdapter()
    # Within tolerance → zero command
    near_goal_obs = _base_observation()
    near_goal_obs["goal"]["current"] = np.array([0.05, 0.05])
    v, w = adapter.plan(_with_grid(near_goal_obs, np.zeros((4, 3, 3), dtype=float)))
    assert v == 0.0
    assert w == 0.0

    # Farther goal with obstacle reduces speed versus clear grid
    far_obs = _base_observation()
    clear_grid = np.zeros((4, 3, 3), dtype=float)
    obstacle_grid = np.ones((4, 3, 3), dtype=float)

    v_clear, _ = adapter.plan(_with_grid(far_obs, clear_grid))
    v_blocked, _ = adapter.plan(_with_grid(far_obs, obstacle_grid))

    assert v_blocked < v_clear


def test_socnavbench_root_validation():
    """Validate that the vendored SocNavBench root has required modules."""
    root = Path(__file__).resolve().parents[2] / "third_party" / "socnavbench"
    missing = SocNavBenchSamplingAdapter._validate_socnav_root(root)
    assert not missing


def test_socnavbench_adapter_loads_vendored_upstream(monkeypatch):
    """Ensure the adapter can load the vendored planner without generating pipelines."""
    root = Path(__file__).resolve().parents[2] / "third_party" / "socnavbench"
    monkeypatch.setattr(
        configparser,
        "SafeConfigParser",
        configparser.ConfigParser,
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "skfmm",
        types.SimpleNamespace(distance=lambda phi, dx: np.zeros_like(phi, dtype=float)),
    )
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)

    central = importlib.import_module("params.central_params")
    monkeypatch.setattr(central, "base_data_dir", lambda: str(root))
    monkeypatch.setattr(central, "get_sbpd_data_dir", lambda: str(root))
    monkeypatch.setattr(central, "get_traversible_dir", lambda: str(root))
    monkeypatch.setattr(central, "get_surreal_mesh_dir", lambda: str(root))
    monkeypatch.setattr(central, "get_surreal_texture_dir", lambda: str(root))

    cp = importlib.import_module("control_pipelines.control_pipeline_v0")

    class FakePipeline:
        def __init__(self, params):
            self.params = params

        def does_pipeline_exist(self):
            return True

        def load_control_pipeline(self):
            return None

        def generate_control_pipeline(self):
            raise AssertionError("Pipeline generation should be skipped in smoke tests.")

    monkeypatch.setattr(
        cp.ControlPipelineV0,
        "get_pipeline",
        classmethod(lambda cls, params: FakePipeline(params)),
    )

    adapter = SocNavBenchSamplingAdapter(socnav_root=root, allow_fallback=True)
    obs = _base_observation()
    v, w = adapter.plan(obs)
    assert np.isfinite(v)
    assert np.isfinite(w)


def test_socnavbench_upstream_end_to_end_when_data_present(monkeypatch):
    """End-to-end SocNavBench planner load if full datasets are available."""
    root = Path(__file__).resolve().parents[2] / "third_party" / "socnavbench"
    required = [
        root / "wayptnav_data",
        root / "sd3dis" / "stanford_building_parser_dataset",
        root / "surreal" / "code" / "human_meshes",
        root / "surreal" / "code" / "human_textures",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("SocNavBench datasets not available for end-to-end planner test.")

    monkeypatch.chdir(root)
    adapter = SocNavBenchSamplingAdapter(socnav_root=root, allow_fallback=False)
    obs = _base_observation()
    v, w = adapter.plan(obs)
    assert np.isfinite(v)
    assert np.isfinite(w)


def test_socnav_fields_supports_flattened_observation():
    adapter = SamplingPlannerAdapter()
    flat_obs = {
        "robot_position": np.array([1.0, 2.0]),
        "robot_heading": np.array([0.5]),
        "robot_speed": np.array([0.1]),
        "robot_radius": np.array([0.3]),
        "goal_current": np.array([3.0, 4.0]),
        "goal_next": np.array([4.0, 5.0]),
        "pedestrians_positions": np.array([[1.0, 1.0]]),
        "pedestrians_count": np.array([1]),
        "pedestrians_radius": np.array([0.2]),
    }

    robot_state, goal_state, ped_state = adapter._socnav_fields(flat_obs)

    assert robot_state["position"][0] == 1.0
    assert goal_state["current"][0] == 3.0
    assert ped_state["count"][0] == 1


def test_grid_helpers_handle_missing_channels_and_oob():
    adapter = SamplingPlannerAdapter()
    obs = _base_observation()
    grid = np.zeros((4, 3, 3), dtype=float)
    obs = _with_grid(obs, grid)
    grid_arr, meta = adapter._extract_grid_payload(obs)

    # No channel requested -> returns 0
    assert adapter._grid_value(np.array([0.0, 0.0]), grid_arr, meta, -1) == 0.0
    # Out-of-bounds returns occupied=1.0
    assert adapter._grid_value(np.array([5.0, 5.0]), grid_arr, meta, 3) == 1.0


def test_select_safe_heading_returns_base_when_direction_tiny():
    adapter = SamplingPlannerAdapter()
    obs = _with_grid(_base_observation(), np.zeros((4, 3, 3), dtype=float))

    direction, penalty = adapter._select_safe_heading(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        obs,
        sweep=1.0,
        num_candidates=1,
        lookahead=1.0,
        weight=1.0,
        angle_weight=1.0,
    )

    assert penalty == 0.0
    assert np.allclose(direction, np.array([0.0, 0.0]))


def test_social_force_and_variants_produce_actions():
    obs = _with_grid(_base_observation(), np.zeros((4, 3, 3), dtype=float))

    sf_adapter = SamplingPlannerAdapter()
    assert sf_adapter.plan(obs) == sf_adapter.plan(obs)  # deterministic

    social_force = sf_adapter.__class__()
    v, w = social_force.plan(obs)
    assert isinstance(v, float)
    assert isinstance(w, float)


def test_policy_wrappers_and_factory_helpers():
    from robot_sf.planner.socnav import (
        ORCAPlannerAdapter,
        SACADRLPlannerAdapter,
        SocNavPlannerPolicy,
        make_orca_policy,
        make_sacadrl_policy,
        make_social_force_policy,
    )

    obs = _with_grid(_base_observation(), np.zeros((4, 3, 3), dtype=float))

    policy = SocNavPlannerPolicy()
    assert len(policy.act(obs)) == 2

    assert isinstance(make_social_force_policy().adapter, SamplingPlannerAdapter)
    assert isinstance(make_orca_policy().adapter, ORCAPlannerAdapter)
    assert isinstance(make_sacadrl_policy().adapter, SACADRLPlannerAdapter)


def test_socnavbench_adapter_uses_upstream_when_available():
    class FakeTraj:
        def position_nk2(self):
            return np.array([[[1.0, 0.0]]])

    class FakeOpt:
        @classmethod
        def from_pos3(cls, arr):
            inst = cls()
            inst.arr = np.array(arr)
            return inst

    class FakePlanner:
        def __init__(self):
            self.opt_waypt = FakeOpt()

        def optimize(self, start_config, goal_config):
            # Ensure called with provided configs
            assert isinstance(start_config, FakeOpt)
            assert isinstance(goal_config, FakeOpt)
            return {"trajectory": FakeTraj()}

    adapter = __import__(
        "robot_sf.planner.socnav", fromlist=["SocNavBenchSamplingAdapter"]
    ).SocNavBenchSamplingAdapter(planner_factory=lambda: FakePlanner())
    obs = _with_grid(_base_observation(), np.zeros((4, 3, 3), dtype=float))

    v, w = adapter.plan(obs)

    assert isinstance(v, float)
    assert isinstance(w, float)
