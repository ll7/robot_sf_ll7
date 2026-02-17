"""Tests for map_runner helper utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner import (
    _build_policy,
    _build_socnav_config,
    _goal_policy,
    _normalize_xy_rows,
    _parse_algo_config,
    _ppo_action_to_unicycle,
    _ppo_paper_gate_status,
    _resolve_seed_list,
    _robot_kinematics_label,
    _robot_max_speed,
    _run_map_episode,
    _scenario_robot_kinematics_label,
    _select_seeds,
    _stack_ped_positions,
    _suite_key,
    _validate_behavior_sanity,
    _vel_and_acc,
    run_map_batch,
)
from robot_sf.common.types import Rect
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition


def test_parse_algo_config_validates_yaml(tmp_path: Path) -> None:
    """Ensure YAML config parsing handles missing, valid, and invalid files."""
    assert _parse_algo_config(None) == {}
    with pytest.raises(FileNotFoundError):
        _parse_algo_config(str(tmp_path / "missing.yaml"))

    cfg_path = tmp_path / "algo.yaml"
    cfg_path.write_text("max_speed: 2.0\n", encoding="utf-8")
    assert _parse_algo_config(str(cfg_path))["max_speed"] == 2.0

    list_path = tmp_path / "algo_list.yaml"
    list_path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(TypeError):
        _parse_algo_config(str(list_path))


def test_goal_policy_and_build_policy() -> None:
    """Validate goal policy behavior and metadata wiring."""
    obs_at_goal = {
        "robot": {"position": [1.0, 1.0], "heading": [0.0]},
        "goal": {"current": [1.0, 1.0]},
    }
    assert _goal_policy(obs_at_goal) == (0.0, 0.0)

    policy, meta = _build_policy("goal", {"max_speed": 1.5})
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [1.0, 0.0]},
    }
    linear, angular = policy(obs)
    assert meta["status"] == "ok"
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "deterministic_goal_seeking"
    assert meta["planner_kinematics"]["execution_mode"] == "native"
    assert linear > 0.0
    assert abs(angular) <= 1.0


def test_build_policy_handles_unknown_and_placeholder() -> None:
    """Ensure unknown algorithms raise and placeholders emit metadata."""
    with pytest.raises(ValueError):
        _build_policy("unknown_algo", {})

    _, meta = _build_policy("rvo", {})
    assert meta["status"] == "placeholder"
    assert meta["fallback_reason"] == "unimplemented"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"


def test_build_policy_socnav_bench_forwards_allow_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure socnav_bench policy wiring respects allow_fallback setting."""

    captured: dict[str, bool] = {}

    class _DummyAdapter:
        def __init__(self, config, allow_fallback: bool = False) -> None:
            del config
            captured["allow_fallback"] = bool(allow_fallback)

        def plan(self, _obs):
            return (0.0, 0.0)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SocNavBenchSamplingAdapter", _DummyAdapter)
    _, meta = _build_policy("socnav_bench", {"allow_fallback": True})
    assert captured["allow_fallback"] is True
    assert meta["status"] == "ok"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"


def test_suite_seed_selection_and_behavior_sanity() -> None:
    """Check suite key selection and behavior sanity validation."""
    assert _suite_key(Path("classic_interactions.yaml")) == "classic_interactions"
    assert _suite_key(Path("francis2023.yaml")) == "francis2023"
    assert _suite_key(Path("other.yaml")) == "default"

    suite_seeds = {"default": [7, 8], "classic_interactions": [1, 2]}
    assert _select_seeds({"seeds": [3, 4]}, suite_seeds=suite_seeds, suite_key="default") == [
        3,
        4,
    ]
    assert _select_seeds({}, suite_seeds=suite_seeds, suite_key="classic_interactions") == [1, 2]

    errors = _validate_behavior_sanity({"metadata": {"behavior": "wait"}, "single_pedestrians": []})
    assert errors

    ok = _validate_behavior_sanity(
        {
            "metadata": {"behavior": "wait"},
            "single_pedestrians": [{"role": "companion", "wait_at": [0.0, 0.0]}],
        }
    )
    assert not ok


def test_velocity_and_ped_stack_helpers() -> None:
    """Ensure velocity/acceleration and trajectory stacking behave for small inputs."""
    positions = np.array([[0.0, 0.0]])
    vel, acc = _vel_and_acc(positions, dt=0.1)
    assert np.allclose(vel, 0.0)
    assert np.allclose(acc, 0.0)

    traj = [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0], [2.0, 2.0]])]
    stacked = _stack_ped_positions(traj)
    assert stacked.shape == (2, 2, 2)
    assert _stack_ped_positions([]).shape == (0, 0, 2)


def test_map_runner_metadata_and_normalization_helpers() -> None:
    """Cover helper branches for kinematics metadata and PPO payload conversion."""
    assert _normalize_xy_rows([]).shape == (0, 2)
    assert _normalize_xy_rows([1.0, 2.0]).shape == (1, 2)
    assert _normalize_xy_rows([1.0]).shape == (0, 2)
    trimmed = _normalize_xy_rows(np.array([[1.0, 2.0, 3.0]]))
    assert trimmed.shape == (1, 2)

    class DifferentialDriveSettings:
        max_linear_speed = 2.5

    class BicycleDriveSettings:
        max_velocity = 3.5

    diff = type("Cfg", (), {"robot_config": DifferentialDriveSettings()})()
    bike = type("Cfg", (), {"robot_config": BicycleDriveSettings()})()
    unknown = type("Cfg", (), {"robot_config": object()})()
    none_cfg = type("Cfg", (), {"robot_config": None})()

    assert _robot_kinematics_label(diff) == "differential_drive"
    assert _robot_kinematics_label(bike) == "bicycle_drive"
    assert _robot_kinematics_label(unknown) != ""
    assert _robot_kinematics_label(none_cfg) == "differential_drive"
    assert _robot_max_speed(diff) == 2.5
    assert _robot_max_speed(bike) == 3.5
    assert _robot_max_speed(none_cfg) is None

    assert _scenario_robot_kinematics_label({}) == "differential_drive"
    assert (
        _scenario_robot_kinematics_label({"robot_config": {"type": "bicycle_drive"}})
        == "bicycle_drive"
    )
    assert (
        _scenario_robot_kinematics_label({"robot_config": {"type": "skid_steer"}}) == "skid_steer"
    )

    ok, reason = _ppo_paper_gate_status(
        {
            "profile": "paper",
            "provenance": {
                "training_config": "cfg",
                "training_commit": "abc",
                "dataset_version": "v1",
                "checkpoint_id": "ckpt",
                "normalization_id": "norm",
                "deterministic_seed_set": "eval",
            },
            "quality_gate": {"min_success_rate": 0.5, "measured_success_rate": 0.7},
        }
    )
    assert ok is True and reason is None
    ok, reason = _ppo_paper_gate_status({"profile": "paper"})
    assert ok is False and "missing 'provenance'" in str(reason)

    native = _ppo_action_to_unicycle({"v": 0.2, "omega": 0.1}, {"robot": {}}, {})
    assert native[2] == "native"
    adapted = _ppo_action_to_unicycle(
        {"vx": 0.0, "vy": 0.0},
        {"robot": {"heading": [0.0]}},
        {"v_max": 1.0, "omega_max": 1.0},
    )
    assert adapted[2] == "adapter"


def test_build_socnav_config_and_seed_loading(tmp_path: Path) -> None:
    """Verify SocNav config fallback and seed list parsing."""
    cfg = _build_socnav_config({"invalid_key": 123})
    assert hasattr(cfg, "social_force_desired_speed")

    seed_path = tmp_path / "seeds.yaml"
    seed_path.write_text("suite_a: [1, 2]\ninvalid: 3\n", encoding="utf-8")
    assert _resolve_seed_list(seed_path) == {"suite_a": [1, 2]}
    assert _resolve_seed_list(tmp_path / "missing.yaml") == {}


def _minimal_map_def() -> MapDefinition:
    width = 5.0
    height = 4.0
    spawn_zone: Rect = ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0))
    goal_zone: Rect = ((3.5, 2.5), (4.0, 2.5), (3.5, 3.0))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.75, 0.75), (3.75, 2.75)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def test_run_map_episode_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a stubbed map episode and ensure record fields are produced."""

    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.zeros((0, 2), dtype=float)
            self.goal_pos = [np.array([1.0, 1.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((0, 2), dtype=float)

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)

        def reset(self, seed: int | None = None):
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            return obs, {}

        def step(self, action):
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            return obs, 0.0, True, False, {"success": True}

        def close(self) -> None:
            return None

    map_def = _minimal_map_def()
    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(map_def),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 1.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    scenario = {"name": "s1", "simulation_config": {"max_episode_steps": 1}}
    record = _run_map_episode(
        scenario,
        seed=1,
        horizon=None,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )
    assert record["scenario_id"] == "s1"
    assert record["metrics"]["success"] == 1.0
    algo_md = record["algorithm_metadata"]
    assert algo_md["baseline_category"] == "classical"
    assert algo_md["planner_kinematics"]["robot_kinematics"] in {"unknown", "differential_drive"}


def test_run_map_batch_serial_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise serial batch execution and resume skipping."""
    scenario = {"name": "s1", "metadata": {"supported": True}}
    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.validate_scenario_list", lambda scenarios: []
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.load_schema", lambda path: {})
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._run_map_job_worker",
        lambda job: {"episode_id": "ep1"},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._write_validated",
        lambda *args, **kwargs: None,
    )

    # Serial run writes one record
    result = run_map_batch(
        [scenario],
        out_path,
        schema_path=tmp_path / "schema.json",
        workers=1,
        resume=False,
    )
    assert result["written"] == 1
    assert result["algorithm_metadata_contract"]["baseline_category"] == "classical"
    assert result["algorithm_metadata_contract"]["planner_kinematics"]["robot_kinematics"] in {
        "unknown",
        "differential_drive",
    }

    # Resume path skips existing episode
    monkeypatch.setattr("robot_sf.benchmark.map_runner.index_existing", lambda path: {"ep1"})
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._compute_map_episode_id",
        lambda sc, seed: "ep1",
    )
    result = run_map_batch(
        [scenario],
        out_path,
        schema_path=tmp_path / "schema.json",
        workers=1,
        resume=True,
    )
    assert result["written"] == 0


def test_run_map_batch_filters_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover validation failures and unsupported scenario filtering."""
    bad_scenarios = [{"name": "bad"}]
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.validate_scenario_list",
        lambda scenarios: ["error"],
    )
    with pytest.raises(ValueError):
        run_map_batch(bad_scenarios, tmp_path / "out.jsonl", schema_path=tmp_path / "schema.json")

    supported = [
        {"name": "skip", "supported": False},
        {"name": "ok", "metadata": {"supported": True}},
    ]
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.validate_scenario_list", lambda scenarios: []
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.load_schema", lambda path: {})
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._run_map_job_worker",
        lambda job: {"episode_id": "ep1"},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._write_validated",
        lambda *args, **kwargs: None,
    )
    result = run_map_batch(
        supported,
        tmp_path / "out.jsonl",
        schema_path=tmp_path / "schema.json",
        workers=1,
        resume=False,
    )
    assert result["total_jobs"] == 1
