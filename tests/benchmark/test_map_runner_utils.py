"""Tests for map_runner helper utilities."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner import (
    _build_policy,
    _build_socnav_config,
    _default_robot_command_space,
    _extract_ppo_pedestrians,
    _finalize_feasibility_metadata,
    _goal_policy,
    _normalize_xy_rows,
    _parse_algo_config,
    _planner_kinematics_compatibility,
    _policy_command_to_env_action,
    _ppo_action_to_unicycle,
    _ppo_paper_gate_status,
    _preflight_policy,
    _project_with_feasibility,
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
from robot_sf.robot.action_adapters import holonomic_to_diff_drive_action
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings


class _KinematicsStub:
    """Minimal kinematics model test double for map-runner wiring tests."""

    name: str = "stub"

    def __init__(self, projected: tuple[float, float], *, feasible: bool = True) -> None:
        self._projected = projected
        self._feasible = feasible
        self.calls: list[tuple[float, float]] = []

    def is_feasible(self, command: tuple[float, float]) -> bool:
        del command
        return self._feasible

    def project(self, command: tuple[float, float]) -> tuple[float, float]:
        self.calls.append((float(command[0]), float(command[1])))
        return self._projected

    def diagnostics(
        self,
        command: tuple[float, float],
        projected: tuple[float, float],
    ) -> dict[str, object]:
        del command, projected
        return {}


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
    assert meta["planner_kinematics"]["planner_command_space"] == "unicycle_vw"
    assert linear > 0.0
    assert abs(angular) <= 1.0


def test_goal_policy_supports_flat_map_runner_observation() -> None:
    """Goal baseline should work with the flat observation keys emitted by the env."""
    obs = {
        "robot_position": [0.0, 0.0],
        "robot_heading": [0.0],
        "goal_current": [1.0, 0.0],
    }
    linear, angular = _goal_policy(obs, max_speed=1.5)
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


def test_build_policy_teb_wires_teb_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TEB-inspired key should build the new adapter instead of placeholder sampling."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return (0.4, 0.2)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.TEBCommitmentPlannerAdapter", _DummyAdapter)
    policy, meta = _build_policy("teb", {"commit_gain": 0.8})
    linear, angular = policy(
        {"robot": {"position": [0.0, 0.0], "heading": [0.0]}, "goal": {"current": [1.0, 0.0]}}
    )
    assert (linear, angular) == (0.4, 0.2)
    assert meta["status"] == "ok"
    assert meta["policy_semantics"] == "corridor_commitment_local_planner"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"


def test_build_policy_nmpc_social_wires_nmpc_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NMPC key should build the native optimizer adapter."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return (0.3, -0.1)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.NMPCSocialPlannerAdapter", _DummyAdapter)
    policy, meta = _build_policy("nmpc_social", {"horizon_steps": 4})
    linear, angular = policy(
        {"robot": {"position": [0.0, 0.0], "heading": [0.0]}, "goal": {"current": [1.0, 0.0]}}
    )
    assert (linear, angular) == (0.3, -0.1)
    assert meta["status"] == "ok"
    assert meta["policy_semantics"] == "nonlinear_model_predictive_local_planner"
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


def test_build_policy_socnav_sampling_uses_local_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure socnav_sampling maps to local SamplingPlannerAdapter, not SocNavBench wrapper."""

    calls: dict[str, int] = {"local": 0, "upstream": 0}

    class _DummyLocalAdapter:
        def __init__(self, config) -> None:
            del config
            calls["local"] += 1

        def plan(self, _obs):
            return (0.0, 0.0)

    class _DummyUpstreamAdapter:
        def __init__(self, config, allow_fallback: bool = False) -> None:
            del config, allow_fallback
            calls["upstream"] += 1

        def plan(self, _obs):
            return (0.0, 0.0)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SamplingPlannerAdapter", _DummyLocalAdapter)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SocNavBenchSamplingAdapter",
        _DummyUpstreamAdapter,
    )

    _, meta = _build_policy("socnav_sampling", {"allow_fallback": True})
    assert calls["local"] == 1
    assert calls["upstream"] == 0
    assert meta["status"] == "ok"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"


def test_build_policy_orca_preserves_provenance_metadata() -> None:
    """Ensure ORCA metadata carries explicit upstream provenance and projection fields."""
    _, meta = _build_policy(
        "orca",
        {
            "allow_fallback": False,
            "provenance": {
                "upstream_repo": "https://github.com/mit-acl/Python-RVO2",
                "upstream_commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
            },
        },
        robot_kinematics="differential_drive",
    )
    assert meta["provenance"]["upstream_repo"] == "https://github.com/mit-acl/Python-RVO2"
    assert meta["planner_kinematics"]["projection_policy"] == (
        "heading_safe_velocity_to_unicycle_vw"
    )


def test_build_policy_hrvo_preserves_local_provenance_metadata() -> None:
    """HRVO metadata should expose its local implementation boundary explicitly."""
    _, meta = _build_policy(
        "hrvo",
        {
            "allow_testing_algorithms": True,
            "provenance": {
                "reference_repo": "https://github.com/snape/HRVO",
            },
        },
        robot_kinematics="differential_drive",
    )
    assert meta["provenance"]["reference_repo"] == "https://github.com/snape/HRVO"
    assert meta["policy_semantics"] == "hybrid_reciprocal_velocity_obstacle"
    assert meta["planner_kinematics"]["projection_policy"] == (
        "heading_safe_velocity_to_unicycle_vw"
    )


@pytest.mark.parametrize("algo", ["hrvo", "socnav_hrvo"])
def test_build_policy_hrvo_holonomic_vx_vy_uses_world_velocity_command(
    algo: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holonomic HRVO aliases should expose an explicit world-frame velocity command payload."""

    bind_calls: list[tuple[np.ndarray, float]] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = SimpleNamespace(orca_obstacle_margin=0.15)

        def plan(self, _obs):
            raise AssertionError("Holonomic HRVO should not call plan() in vx_vy mode.")

        def plan_velocity_world(self, _obs):
            return np.array([0.3, 0.4], dtype=float)

        def bind_static_obstacle_points(self, points, *, spacing):
            bind_calls.append((np.asarray(points, dtype=float), float(spacing)))

    class _DummySim:
        def iter_obstacle_segments(self):
            return [((0.0, 0.0), (1.0, 0.0))]

    class _DummyEnv:
        simulator = _DummySim()

    monkeypatch.setattr("robot_sf.benchmark.map_runner.HRVOPlannerAdapter", _DummyAdapter)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.sample_obstacle_points",
        lambda segments, spacing: np.array([[0.5, 0.0]], dtype=float),
    )
    policy, meta = _build_policy(
        algo,
        {"allow_testing_algorithms": True},
        robot_kinematics="holonomic",
        robot_command_mode="vx_vy",
    )

    assert hasattr(policy, "_planner_bind_env")
    policy._planner_bind_env(_DummyEnv())
    command = policy({})
    assert len(bind_calls) == 1
    np.testing.assert_allclose(bind_calls[0][0], np.array([[0.5, 0.0]], dtype=float))
    assert bind_calls[0][1] == pytest.approx(0.3)
    assert command == {
        "command_kind": "holonomic_vxy_world",
        "vx": 0.3,
        "vy": 0.4,
    }
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"
    assert meta["planner_kinematics"]["projection_policy"] == ("world_velocity_passthrough")


def test_build_policy_hrvo_reset_hook_tolerates_adapters_without_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter reset hooks should not require every adapter to accept a seed kwarg."""

    reset_calls: list[str] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return 0.1, 0.2

        def reset(self) -> None:
            reset_calls.append("reset")

    monkeypatch.setattr("robot_sf.benchmark.map_runner.HRVOPlannerAdapter", _DummyAdapter)
    policy, _ = _build_policy(
        "hrvo",
        {"allow_testing_algorithms": True},
        robot_kinematics="differential_drive",
    )

    policy._planner_reset(seed=7)
    assert reset_calls == ["reset"]


def test_build_policy_orca_holonomic_vx_vy_uses_world_velocity_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holonomic ORCA should expose an explicit world-frame velocity command payload."""

    class _DummyAdapter:
        def __init__(self, config, allow_fallback: bool = False) -> None:
            del config, allow_fallback

        def plan(self, _obs):
            raise AssertionError("Holonomic ORCA should not call plan() in vx_vy mode.")

        def plan_velocity_world(self, _obs):
            return np.array([0.6, -0.2], dtype=float)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.ORCAPlannerAdapter", _DummyAdapter)
    policy, meta = _build_policy(
        "orca",
        {"allow_fallback": False},
        robot_kinematics="holonomic",
        robot_command_mode="vx_vy",
    )

    command = policy({})
    assert command == {
        "command_kind": "holonomic_vxy_world",
        "vx": pytest.approx(0.6),
        "vy": pytest.approx(-0.2),
    }
    assert meta["planner_kinematics"]["planner_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["benchmark_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["projection_policy"] == "world_velocity_passthrough"


def test_build_policy_social_navigation_pyenvs_orca_preserves_provenance_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure external ORCA prototype metadata carries explicit upstream provenance."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return (0.2, 0.1)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsORCAAdapter",
        _DummyAdapter,
    )
    _, meta = _build_policy(
        "social_navigation_pyenvs_orca",
        {
            "repo_root": "output/repos/Social-Navigation-PyEnvs",
            "provenance": {
                "upstream_repo": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
                "upstream_policy": "crowd_nav.policy_no_train.orca.ORCA",
            },
        },
        robot_kinematics="differential_drive",
    )
    assert (
        meta["provenance"]["upstream_repo"]
        == "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs"
    )
    assert meta["planner_kinematics"]["projection_policy"] == (
        "heading_safe_velocity_to_unicycle_vw"
    )
    assert meta["upstream_reference"]["upstream_policy"] == "crowd_nav.policy_no_train.orca.ORCA"


def test_build_policy_crowdnav_height_preserves_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experimental CrowdNav_HEIGHT wiring should expose the upstream checkpoint boundary."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def bind_env(self, env) -> None:
            del env

        def plan(self, _obs):
            return (0.15, 0.05)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.CrowdNavHeightAdapter", _DummyAdapter)
    _, meta = _build_policy(
        "crowdnav_height",
        {
            "repo_root": "output/repos/CrowdNav_HEIGHT",
            "model_dir": "output/external_checkpoints/crowdnav_height_extracted/HEIGHT/HEIGHT",
            "checkpoint_name": "237800.pt",
        },
        robot_kinematics="differential_drive",
    )
    assert meta["baseline_category"] == "learning"
    assert meta["planner_kinematics"]["projection_policy"] == (
        "upstream_discrete_delta_vw_to_unicycle_vw_stateful"
    )
    assert (
        meta["upstream_reference"]["repo_url"] == "https://github.com/Shuijing725/CrowdNav_HEIGHT"
    )
    assert meta["upstream_reference"]["default_checkpoint"] == "HEIGHT/checkpoints/237800.pt"


def test_build_policy_sonic_crowdnav_wires_external_checkpoint_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SoNIC key should build the upstream checkpoint wrapper path."""

    planners: list[object] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config
            planners.append(self)

        def plan(self, obs):
            assert obs["goal"]["current"] == [1.0, 0.0]
            return (0.4, 0.1)

        def reset(self, *, seed: int | None = None) -> None:
            del seed

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SonicCrowdNavAdapter", _DummyAdapter)
    policy, meta = _build_policy(
        "sonic_gst",
        {
            "repo_root": "output/repos/SoNIC-Social-Nav",
            "checkpoint_name": "05207.pt",
        },
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0], "velocity_xy": [0.0, 0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )
    assert (linear, angular) == (0.4, 0.1)
    policy._planner_reset(seed=9)
    assert planners
    assert meta["policy_semantics"] == "upstream_sonic_checkpoint_wrapper"
    assert meta["planner_kinematics"]["adapter_name"] == "SonicCrowdNavAdapter"
    assert planners[0].config.model_name == "SoNIC_GST"
    assert planners[0].config.repo_root.name == "SoNIC-Social-Nav"
    assert planners[0].config.checkpoint_name == "05207.pt"


def test_build_policy_gensafenav_ours_wires_external_checkpoint_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GenSafeNav Ours_GST key should build the learned checkpoint wrapper path."""

    planners: list[object] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config
            planners.append(self)

        def plan(self, obs):
            assert obs["goal"]["current"] == [1.0, 0.0]
            return (0.35, 0.2)

        def reset(self, *, seed: int | None = None) -> None:
            del seed

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SonicCrowdNavAdapter", _DummyAdapter)
    policy, meta = _build_policy(
        "ours_gst",
        {},
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0], "velocity_xy": [0.0, 0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )
    assert (linear, angular) == (0.35, 0.2)
    assert planners
    assert planners[0].config.repo_root.name == "GenSafeNav"
    assert planners[0].config.model_name == "Ours_GST"
    assert planners[0].config.checkpoint_name == "05207.pt"
    assert meta["policy_semantics"] == "upstream_gensafenav_checkpoint_wrapper"
    assert meta["planner_kinematics"]["adapter_name"] == "SonicCrowdNavAdapter"


def test_build_policy_gensafenav_gst_predictor_rand_wires_external_checkpoint_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GenSafeNav CrowdNav++-style key should build the learned checkpoint wrapper path."""

    planners: list[object] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config
            planners.append(self)

        def plan(self, obs):
            assert obs["goal"]["current"] == [1.0, 0.0]
            return (0.25, -0.1)

        def reset(self, *, seed: int | None = None) -> None:
            del seed

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SonicCrowdNavAdapter", _DummyAdapter)
    policy, meta = _build_policy(
        "gst_predictor_rand",
        {},
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0], "velocity_xy": [0.0, 0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )
    assert (linear, angular) == (0.25, -0.1)
    assert planners
    assert planners[0].config.repo_root.name == "GenSafeNav"
    assert planners[0].config.model_name == "GST_predictor_rand"
    assert planners[0].config.checkpoint_name == "05207.pt"
    assert meta["policy_semantics"] == "upstream_gensafenav_checkpoint_wrapper"
    assert meta["planner_kinematics"]["adapter_name"] == "SonicCrowdNavAdapter"


def test_build_policy_gensafenav_ours_guarded_uses_guard_and_goal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guarded Ours_GST alias should wire guard decisions and expose mixed metadata."""

    planners: list[object] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config
            planners.append(self)

        def plan(self, _obs):
            return (0.35, 0.2)

        def reset(self, *, seed: int | None = None) -> None:
            del seed

    class _DummyGuard:
        def __init__(self, config, *, fallback_adapter) -> None:
            del config
            self.fallback_adapter = fallback_adapter

        def choose_command(self, observation, ppo_command):
            del ppo_command
            return self.fallback_adapter.plan(observation), "fallback_safe"

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SonicCrowdNavAdapter", _DummyAdapter)
    monkeypatch.setattr("robot_sf.benchmark.map_runner.GuardedPPOAdapter", _DummyGuard)

    policy, meta = _build_policy(
        "ours_gst_guarded",
        {},
        robot_kinematics="differential_drive",
        adapter_impact_eval=True,
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0], "velocity_xy": [0.0, 0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )

    assert planners
    assert planners[0].config.repo_root.name == "GenSafeNav"
    assert planners[0].config.model_name == "Ours_GST"
    assert (linear, angular) == (1.0, 0.0)
    assert meta["policy_semantics"] == "guarded_upstream_gensafenav_checkpoint_wrapper"
    assert meta["planner_kinematics"]["execution_mode"] == "mixed"
    assert meta["planner_kinematics"]["fallback_policy"] == "goal"
    assert meta["guard_stats"]["fallback_safe"] == 1
    assert meta["adapter_impact"]["native_steps"] == 0
    assert meta["adapter_impact"]["adapted_steps"] == 1


def test_build_policy_gensafenav_gst_predictor_rand_guarded_defaults_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guarded GST_predictor_rand alias should default to the correct upstream checkpoint."""

    planners: list[object] = []

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config
            planners.append(self)

        def plan(self, _obs):
            return (0.25, -0.1)

        def reset(self, *, seed: int | None = None) -> None:
            del seed

    class _DummyGuard:
        def __init__(self, config, *, fallback_adapter) -> None:
            del config, fallback_adapter

        def choose_command(self, observation, ppo_command):
            del observation
            return ppo_command, "ppo_safe"

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SonicCrowdNavAdapter", _DummyAdapter)
    monkeypatch.setattr("robot_sf.benchmark.map_runner.GuardedPPOAdapter", _DummyGuard)

    policy, meta = _build_policy(
        "gst_predictor_rand_guarded",
        {},
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0], "velocity_xy": [0.0, 0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )

    assert planners
    assert planners[0].config.repo_root.name == "GenSafeNav"
    assert planners[0].config.model_name == "GST_predictor_rand"
    assert planners[0].config.checkpoint_name == "05207.pt"
    assert (linear, angular) == (0.25, -0.1)
    assert meta["policy_semantics"] == "guarded_upstream_gensafenav_checkpoint_wrapper"
    assert meta["planner_kinematics"]["execution_mode"] == "mixed"
    assert meta["guard_stats"]["ppo_safe"] == 1


def test_build_policy_sonic_gst_holonomic_vx_vy_uses_direct_world_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holonomic SoNIC runs should forward ActionXY directly instead of round-tripping via `(v, w)`."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return (0.4, 0.1)

        def plan_velocity_world(self, obs):
            assert obs["goal"]["current"] == [1.0, 0.0]
            return (0.6, -0.2)

        def reset(self, *, seed: int | None = None) -> None:
            del seed

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SonicCrowdNavAdapter", _DummyAdapter)

    policy, meta = _build_policy(
        "sonic_gst",
        {},
        robot_kinematics="holonomic",
        robot_command_mode="vx_vy",
    )
    command = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.4], "velocity_xy": [0.0, 0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )

    assert command == {"command_kind": "holonomic_vxy_world", "vx": 0.6, "vy": -0.2}
    assert meta["planner_kinematics"]["planner_command_space"] == "holonomic_vxy_world"


def test_build_policy_guarded_gensafenav_holonomic_vx_vy_fails_closed() -> None:
    """Guarded GenSafeNav aliases should fail closed until the guard supports ActionXY passthrough."""

    with pytest.raises(
        ValueError, match="do not support holonomic vx_vy benchmark action space yet"
    ):
        _build_policy(
            "ours_gst_guarded",
            {},
            robot_kinematics="holonomic",
            robot_command_mode="vx_vy",
        )


def test_build_policy_sicnav_wires_external_mpc_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SICNav key should build the external MPC wrapper path."""

    planners: list[object] = []

    class _DummyPlanner:
        def __init__(self, config, seed=None) -> None:
            self.config = config
            self.seed = seed
            self.reset_calls: list[int | None] = []
            planners.append(self)

        def get_metadata(self):
            return {"status": "ok"}

        def step(self, obs):
            assert obs["robot"]["goal"] == [1.0, 0.0]
            return {"v": 0.3, "omega": 0.2}

        def reset(self, *, seed: int | None = None) -> None:
            self.reset_calls.append(seed)

        def close(self) -> None:
            return None

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SICNavPlanner", _DummyPlanner)
    policy, meta = _build_policy(
        "sicnav",
        {"repo_root": "third_party/external_mpc_repos/sicnav"},
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )
    assert (linear, angular) == (0.3, 0.2)
    policy._planner_reset(seed=11)
    assert planners[0].reset_calls == [11]
    assert meta["policy_semantics"] == "upstream_sicnav_checkpoint_or_policy_wrapper"
    assert meta["planner_kinematics"]["adapter_name"] == "SICNavPlanner"


def test_build_policy_dr_mpc_wires_external_mpc_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DR-MPC key should build the external residual-MPC wrapper path."""

    class _DummyPlanner:
        def __init__(self, config, seed=None) -> None:
            self.config = config
            self.seed = seed

        def get_metadata(self):
            return {"status": "ok"}

        def step(self, obs):
            assert obs["robot"]["goal"] == [1.0, 0.0]
            return {"vx": 0.25, "vy": -0.05}

        def reset(self) -> None:
            pass

    monkeypatch.setattr("robot_sf.benchmark.map_runner.DRMPCPlanner", _DummyPlanner)
    policy, meta = _build_policy(
        "dr_mpc",
        {"repo_root": "third_party/external_mpc_repos/dr_mpc"},
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {},
            "sim": {"timestep": 0.1},
        }
    )
    assert linear == pytest.approx(math.hypot(0.25, -0.05))
    assert angular == pytest.approx(math.atan2(-0.05, 0.25))
    assert meta["policy_semantics"] == "upstream_dr_mpc_residual_mpc_wrapper"
    assert meta["planner_kinematics"]["adapter_name"] == "DRMPCPlanner"


def test_build_policy_social_navigation_pyenvs_orca_holonomic_vx_vy_uses_world_velocity_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holonomic Social-Navigation-PyEnvs ORCA should forward upstream ActionXY as world velocity."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            raise AssertionError("Holonomic upstream ORCA should not call plan() in vx_vy mode.")

        def plan_velocity_world(self, _obs):
            return np.array([0.3, 0.7], dtype=float)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsORCAAdapter",
        _DummyAdapter,
    )
    policy, meta = _build_policy(
        "social_navigation_pyenvs_orca",
        {"repo_root": "output/repos/Social-Navigation-PyEnvs"},
        robot_kinematics="holonomic",
        robot_command_mode="vx_vy",
    )

    command = policy({})
    assert command == {
        "command_kind": "holonomic_vxy_world",
        "vx": pytest.approx(0.3),
        "vy": pytest.approx(0.7),
    }
    assert meta["planner_kinematics"]["planner_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["benchmark_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["projection_policy"] == "world_velocity_passthrough"
    assert meta["planner_kinematics"]["execution_detail"] == "direct_holonomic_world_velocity"


def test_build_policy_social_force_holonomic_vx_vy_uses_world_velocity_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holonomic local SocialForce should forward world-frame velocity directly."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            raise AssertionError(
                "Holonomic social-force path should not call plan() in vx_vy mode."
            )

        def plan_velocity_world(self, _obs):
            return np.array([0.15, 0.45], dtype=float)

    monkeypatch.setattr("robot_sf.benchmark.map_runner.SocialForcePlannerAdapter", _DummyAdapter)
    policy, meta = _build_policy(
        "social_force",
        {},
        robot_kinematics="holonomic",
        robot_command_mode="vx_vy",
    )

    command = policy({})
    assert command == {
        "command_kind": "holonomic_vxy_world",
        "vx": pytest.approx(0.15),
        "vy": pytest.approx(0.45),
    }
    assert meta["planner_kinematics"]["planner_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["benchmark_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["projection_policy"] == "world_velocity_passthrough"
    assert meta["planner_kinematics"]["execution_detail"] == "direct_holonomic_world_velocity"


def test_build_policy_social_navigation_pyenvs_force_models_preserve_provenance_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure external force-model prototype metadata carries explicit upstream provenance."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return (0.2, 0.1)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsForceModelAdapter",
        _DummyAdapter,
    )
    _, socialforce = _build_policy(
        "social_navigation_pyenvs_socialforce",
        {
            "repo_root": "output/repos/Social-Navigation-PyEnvs",
            "policy_name": "socialforce",
            "provenance": {
                "upstream_repo": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
                "upstream_policy": "crowd_nav.policy_no_train.socialforce.SocialForce",
            },
        },
        robot_kinematics="differential_drive",
    )
    _, sfm = _build_policy(
        "social_navigation_pyenvs_sfm_helbing",
        {
            "repo_root": "output/repos/Social-Navigation-PyEnvs",
            "policy_name": "sfm_helbing",
            "provenance": {
                "upstream_repo": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
                "upstream_policy": "crowd_nav.policy_no_train.sfm_helbing.SFMHelbing",
            },
        },
        robot_kinematics="differential_drive",
    )
    assert socialforce["upstream_reference"]["upstream_policy"] == (
        "crowd_nav.policy_no_train.socialforce.SocialForce"
    )
    assert sfm["upstream_reference"]["upstream_policy"] == (
        "crowd_nav.policy_no_train.sfm_helbing.SFMHelbing"
    )


@pytest.mark.parametrize(
    ("algo_key", "policy_name"),
    [
        ("social_navigation_pyenvs_socialforce", "socialforce"),
        ("social_navigation_pyenvs_sfm_helbing", "sfm_helbing"),
    ],
)
def test_build_policy_social_navigation_pyenvs_force_models_holonomic_vx_vy_use_world_velocity_command(
    monkeypatch: pytest.MonkeyPatch,
    algo_key: str,
    policy_name: str,
) -> None:
    """Holonomic force-model wrappers should forward upstream ActionXY as world velocity."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            raise AssertionError("Holonomic force-model path should not call plan() in vx_vy mode.")

        def plan_velocity_world(self, _obs):
            return np.array([0.25, -0.4], dtype=float)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsForceModelAdapter",
        _DummyAdapter,
    )
    policy, meta = _build_policy(
        algo_key,
        {"repo_root": "output/repos/Social-Navigation-PyEnvs", "policy_name": policy_name},
        robot_kinematics="holonomic",
        robot_command_mode="vx_vy",
    )

    command = policy({})
    assert command == {
        "command_kind": "holonomic_vxy_world",
        "vx": pytest.approx(0.25),
        "vy": pytest.approx(-0.4),
    }
    assert meta["planner_kinematics"]["planner_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["benchmark_command_space"] == "holonomic_vxy_world"
    assert meta["planner_kinematics"]["projection_policy"] == "world_velocity_passthrough"
    assert meta["planner_kinematics"]["execution_detail"] == "direct_holonomic_world_velocity"


def test_build_policy_social_navigation_pyenvs_hsfm_preserves_provenance_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure external HSFM prototype metadata carries explicit upstream provenance."""

    class _DummyAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def plan(self, _obs):
            return (0.2, 0.1)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsHSFMAdapter",
        _DummyAdapter,
    )
    _, meta = _build_policy(
        "social_navigation_pyenvs_hsfm_new_guo",
        {
            "repo_root": "output/repos/Social-Navigation-PyEnvs",
            "policy_name": "hsfm_new_guo",
            "provenance": {
                "upstream_repo": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
                "upstream_policy": "crowd_nav.policy_no_train.hsfm_new_guo.HSFMNewGuo",
            },
        },
        robot_kinematics="differential_drive",
    )
    assert meta["upstream_reference"]["upstream_policy"] == (
        "crowd_nav.policy_no_train.hsfm_new_guo.HSFMNewGuo"
    )
    assert meta["planner_kinematics"]["projection_policy"] == (
        "body_velocity_heading_safe_to_unicycle_vw"
    )


def test_preflight_policy_treats_social_navigation_pyenvs_force_models_as_socnav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permissive prereq policies should apply to the new Social-Navigation-PyEnvs aliases."""

    def _fake_build_policy(algo, cfg, *, robot_kinematics=None, adapter_impact_eval=False):
        del cfg, robot_kinematics, adapter_impact_eval
        raise RuntimeError(f"missing upstream prereq for {algo}")

    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _fake_build_policy)
    for algo, policy_name in (
        ("social_navigation_pyenvs_socialforce", "socialforce"),
        ("social_navigation_pyenvs_sfm_helbing", "sfm_helbing"),
    ):
        cfg, preflight = _preflight_policy(
            algo=algo,
            algo_config={
                "repo_root": "output/repos/Social-Navigation-PyEnvs",
                "policy_name": policy_name,
            },
            benchmark_profile="experimental",
            missing_prereq_policy="skip-with-warning",
            robot_kinematics="differential_drive",
        )
        assert cfg["policy_name"] == policy_name
        assert preflight["status"] == "skipped"
        assert preflight["policy"] == "skip-with-warning"
        assert "missing upstream prereq" in str(preflight["error"])


def test_preflight_policy_treats_social_navigation_pyenvs_hsfm_as_socnav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permissive prereq policies should apply to the HSFM alias too."""

    def _fake_build_policy(algo, cfg, *, robot_kinematics=None, adapter_impact_eval=False):
        del cfg, robot_kinematics, adapter_impact_eval
        raise RuntimeError(f"missing upstream prereq for {algo}")

    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _fake_build_policy)
    cfg, preflight = _preflight_policy(
        algo="social_navigation_pyenvs_hsfm_new_guo",
        algo_config={
            "repo_root": "output/repos/Social-Navigation-PyEnvs",
            "policy_name": "hsfm_new_guo",
        },
        benchmark_profile="experimental",
        missing_prereq_policy="skip-with-warning",
        robot_kinematics="differential_drive",
    )
    assert cfg["policy_name"] == "hsfm_new_guo"
    assert preflight["status"] == "skipped"
    assert preflight["policy"] == "skip-with-warning"
    assert "missing upstream prereq" in str(preflight["error"])


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


def test_policy_command_to_env_action_passthroughs_world_velocity_for_holonomic_vx_vy() -> None:
    """Structured holonomic world-velocity commands should pass straight into vx_vy robots."""
    config = SimpleNamespace(
        robot_config=HolonomicDriveSettings(
            radius=0.3,
            max_speed=2.0,
            max_angular_speed=1.5,
            command_mode="vx_vy",
        ),
        sim_config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    robot = SimpleNamespace(pose=((0.0, 0.0), 0.7), current_speed=(0.0, 0.0))
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[robot]))

    action = _policy_command_to_env_action(
        env=env,
        config=config,
        command={"command_kind": "holonomic_vxy_world", "vx": 0.4, "vy": -0.2},
    )

    assert np.allclose(action, np.array([0.4, -0.2], dtype=float))


def test_policy_command_to_env_action_converts_world_velocity_for_differential_drive() -> None:
    """Structured world velocities should reuse the differential-drive adapter path."""
    config = SimpleNamespace(
        robot_config=DifferentialDriveSettings(
            radius=0.3,
            max_linear_speed=2.0,
            max_angular_speed=1.5,
            allow_backwards=False,
        ),
        sim_config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    robot = SimpleNamespace(pose=((0.0, 0.0), 0.4), current_speed=(0.3, -0.1))
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[robot]))

    action = _policy_command_to_env_action(
        env=env,
        config=config,
        command={"command_kind": "holonomic_vxy_world", "vx": 0.9, "vy": 0.2},
    )

    expected_vw = holonomic_to_diff_drive_action(
        np.array([0.9, 0.2], dtype=float),
        robot.pose,
        max_linear_speed=2.0,
        max_angular_speed=1.5,
    )
    assert np.allclose(
        action,
        np.array([expected_vw[0] - 0.3, expected_vw[1] + 0.1], dtype=float),
    )


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
    assert _scenario_robot_kinematics_label({"robot_config": {"type": "holonomic"}}) == "holonomic"
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


def test_map_runner_feasibility_helpers_accumulate_and_finalize() -> None:
    """Feasibility helpers should tolerate missing meta and compute summary stats."""
    passthrough = _project_with_feasibility(
        model=_KinematicsStub((0.5, 0.1)),
        command=(0.5, 0.1),
        meta={},
    )
    assert passthrough == (0.5, 0.1)

    meta = {
        "kinematics_feasibility": {
            "commands_evaluated": 0,
            "infeasible_native_count": 0,
            "projected_count": 0,
            "_sum_abs_delta_linear": 0.0,
            "_sum_abs_delta_angular": 0.0,
            "_max_abs_delta_linear": 0.0,
            "_max_abs_delta_angular": 0.0,
        }
    }
    projected = _project_with_feasibility(
        model=_KinematicsStub((0.2, -0.1), feasible=False),
        command=(0.5, 0.3),
        meta=meta,
    )
    assert projected == (0.2, -0.1)
    _finalize_feasibility_metadata(meta)
    feasibility = meta["kinematics_feasibility"]
    assert feasibility["commands_evaluated"] == 1
    assert feasibility["infeasible_native_count"] == 1
    assert feasibility["projected_count"] == 1
    assert feasibility["projection_rate"] == pytest.approx(1.0)
    assert feasibility["infeasible_rate"] == pytest.approx(1.0)
    assert feasibility["max_abs_delta_linear"] > 0.0

    empty_meta = {"kinematics_feasibility": {}}
    _finalize_feasibility_metadata(empty_meta)
    assert empty_meta["kinematics_feasibility"]["projection_rate"] == 0.0


def test_map_runner_paper_gate_and_compatibility_branches() -> None:
    """Exercise paper-gate validation and planner/kinematics compatibility branches."""
    ok, reason = _ppo_paper_gate_status({"profile": "experimental"})
    assert ok is False and reason is None

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
            "quality_gate": {"min_success_rate": "bad", "measured_success_rate": 0.7},
        }
    )
    assert ok is False and "numeric" in str(reason)

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
            "quality_gate": {"min_success_rate": 0.9, "measured_success_rate": 0.7},
        }
    )
    assert ok is False and "quality gate failed" in str(reason)

    compatible, reason = _planner_kinematics_compatibility(
        algo="rvo",
        robot_kinematics="holonomic",
        algo_config={},
    )
    assert compatible is False and "disabled" in str(reason)

    compatible, reason = _planner_kinematics_compatibility(
        algo="ppo",
        robot_kinematics="holonomic",
        algo_config={"obs_mode": "image"},
    )
    assert compatible is False and "non-image" in str(reason)

    compatible, reason = _planner_kinematics_compatibility(
        algo="ppo",
        robot_kinematics="differential_drive",
        algo_config={"obs_mode": "image"},
    )
    assert compatible is True and reason is None


def test_extract_ppo_pedestrians_respects_count_and_radius() -> None:
    """Pedestrian extraction should slice by count and pad malformed velocity payloads."""
    ped_pos, ped_vel, ped_radius = _extract_ppo_pedestrians(
        {
            "positions": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "velocities": [[0.5, 0.0]],
            "count": [2],
            "radius": [0.4],
        }
    )
    assert ped_pos.shape == (2, 2)
    assert ped_vel.shape == (2, 2)
    assert ped_vel[1].tolist() == [0.0, 0.0]
    assert ped_radius == pytest.approx(0.4)


def test_build_policy_for_portfolio_adapters_tracks_feasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter-backed planner policies should expose feasibility metadata and callable output."""

    class _GapPredictionStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def plan(self, _observation: dict[str, object]) -> tuple[float, float]:
            return 0.2, 0.0

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.GapAwarePredictionAdapter",
        _GapPredictionStub,
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.RiskDWAPlannerAdapter", _GapPredictionStub)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.SafetyBarrierPlannerAdapter",
        _GapPredictionStub,
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.GridRoutePlannerAdapter", _GapPredictionStub)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.StreamGapPlannerAdapter",
        _GapPredictionStub,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.MPPISocialPlannerAdapter",
        _GapPredictionStub,
    )

    for algo_name in (
        "risk_dwa",
        "safety_barrier",
        "grid_route",
        "stream_gap",
        "gap_prediction",
        "mppi_social",
    ):
        policy, meta = _build_policy(
            algo_name,
            {"max_linear_speed": 0.8, "max_angular_speed": 0.5},
            robot_kinematics="differential_drive",
        )
        linear, angular = policy(
            {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
                "goal": {"current": [2.0, 0.0], "next": [2.0, 0.0]},
                "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.3]},
            }
        )
        assert np.isfinite(linear)
        assert np.isfinite(angular)
        feasibility = meta.get("kinematics_feasibility")
        assert isinstance(feasibility, dict)
        assert feasibility["commands_evaluated"] >= 1


def test_ppo_action_to_unicycle_uses_kinematics_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPO command conversion should route through the kinematics contract."""
    model = _KinematicsStub((0.7, -0.3))
    resolver_calls: list[dict[str, object]] = []

    def _resolver(**kwargs):
        resolver_calls.append(dict(kwargs))
        return model

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.resolve_benchmark_kinematics_model",
        _resolver,
    )

    native = _ppo_action_to_unicycle(
        {"v": 1.8, "omega": 2.0},
        {"robot": {"heading": [0.0]}},
        {"v_max": 1.0, "omega_max": 1.0},
        robot_kinematics="differential_drive",
    )
    assert native[0] == pytest.approx(0.7)
    assert native[1] == pytest.approx(-0.3)
    assert native[2] == "native"
    assert model.calls == [(1.8, 2.0)]
    assert resolver_calls
    assert resolver_calls[0].get("robot_kinematics") == "differential_drive"


def test_build_policy_goal_path_uses_kinematics_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Goal policy should project commands through kinematics contract wiring."""
    model = _KinematicsStub((0.2, 0.1))
    resolver_calls: list[dict[str, object]] = []

    def _resolver(**kwargs):
        resolver_calls.append(dict(kwargs))
        return model

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.resolve_benchmark_kinematics_model",
        _resolver,
    )
    policy, meta = _build_policy("goal", {"max_speed": 10.0}, robot_kinematics="bicycle_drive")
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [10.0, 0.0]},
    }
    assert policy(obs) == pytest.approx((0.2, 0.1))
    assert model.calls == [(10.0, 0.0)]
    assert resolver_calls
    assert resolver_calls[0].get("robot_kinematics") == "bicycle_drive"
    assert meta["planner_kinematics"]["execution_mode"] == "native"


def test_ppo_action_to_unicycle_adapter_converts_heading_error_to_angular_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter path should project speed with gain-scaled angular velocity."""
    model = _KinematicsStub((0.0, 0.0))
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.resolve_benchmark_kinematics_model",
        lambda **kwargs: model,
    )
    _ppo_action_to_unicycle(
        {"vx": 0.0, "vy": 1.0},
        {"robot": {"heading": [0.0]}},
        {"omega_kp": 2.0, "omega_max": 1.0},
        robot_kinematics="differential_drive",
    )
    assert model.calls
    assert model.calls[0][0] == pytest.approx(1.0)
    # heading error = pi/2, with kp=2 and omega_max=1 clips to 1.0
    assert model.calls[0][1] == pytest.approx(1.0)


def test_preflight_policy_passes_robot_kinematics_to_build_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should build policy with the requested robot kinematics."""
    captured: dict[str, object] = {}

    def _fake_build_policy(algo, cfg, *, robot_kinematics=None, adapter_impact_eval=False):
        del adapter_impact_eval
        captured["algo"] = algo
        captured["cfg"] = cfg
        captured["robot_kinematics"] = robot_kinematics

        def _policy(_obs):
            return (0.0, 0.0)

        return _policy, {}

    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _fake_build_policy)
    cfg, preflight = _preflight_policy(
        algo="goal",
        algo_config={"max_speed": 1.0},
        benchmark_profile="baseline-safe",
        missing_prereq_policy="fail-fast",
        robot_kinematics="bicycle_drive",
    )
    assert cfg["max_speed"] == 1.0
    assert preflight["status"] == "ok"
    assert captured["robot_kinematics"] == "bicycle_drive"


def test_preflight_policy_treats_social_navigation_pyenvs_orca_as_socnav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permissive prereq policies should apply to the Social-Navigation-PyEnvs ORCA alias."""

    def _fake_build_policy(algo, cfg, *, robot_kinematics=None, adapter_impact_eval=False):
        del cfg, robot_kinematics, adapter_impact_eval
        raise RuntimeError(f"missing upstream prereq for {algo}")

    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _fake_build_policy)
    cfg, preflight = _preflight_policy(
        algo="social_navigation_pyenvs_orca",
        algo_config={"repo_root": "output/repos/Social-Navigation-PyEnvs"},
        benchmark_profile="experimental",
        missing_prereq_policy="skip-with-warning",
        robot_kinematics="differential_drive",
    )
    assert cfg["repo_root"] == "output/repos/Social-Navigation-PyEnvs"
    assert preflight["status"] == "skipped"
    assert preflight["policy"] == "skip-with-warning"
    assert "missing upstream prereq" in str(preflight["error"])


def test_preflight_policy_treats_crowdnav_height_as_socnav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permissive prereq policies should apply to CrowdNav_HEIGHT as a structured external wrapper."""

    def _fake_build_policy(algo, cfg, *, robot_kinematics=None, adapter_impact_eval=False):
        del cfg, robot_kinematics, adapter_impact_eval
        raise RuntimeError(f"missing upstream prereq for {algo}")

    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _fake_build_policy)
    cfg, preflight = _preflight_policy(
        algo="crowdnav_height",
        algo_config={"repo_root": "output/repos/CrowdNav_HEIGHT"},
        benchmark_profile="experimental",
        missing_prereq_policy="skip-with-warning",
        robot_kinematics="differential_drive",
    )
    assert cfg["repo_root"] == "output/repos/CrowdNav_HEIGHT"
    assert preflight["status"] == "skipped"
    assert preflight["policy"] == "skip-with-warning"
    assert "missing upstream prereq" in str(preflight["error"])


def test_build_socnav_config_and_seed_loading(tmp_path: Path) -> None:
    """Verify SocNav config ignores unknown keys but preserves known ones."""
    cfg = _build_socnav_config(
        {
            "invalid_key": 123,
            "predictive_goal_weight": 7.25,
            "predictive_allow_reverse_candidates": True,
        }
    )
    assert hasattr(cfg, "social_force_desired_speed")
    assert cfg.predictive_goal_weight == 7.25
    assert cfg.predictive_allow_reverse_candidates is True

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
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
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
    assert record["metrics"]["success"] == 0.0
    algo_md = record["algorithm_metadata"]
    assert algo_md["baseline_category"] == "classical"
    assert algo_md["planner_kinematics"]["robot_kinematics"] in {"unknown", "differential_drive"}
    feasibility = algo_md.get("kinematics_feasibility")
    assert isinstance(feasibility, dict)
    assert "projection_rate" in feasibility
    assert "infeasible_rate" in feasibility


def test_run_map_episode_calls_planner_reset_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Episode start should reset stateful planner adapters before stepping."""

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

    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})
    reset_calls: list[int] = []

    def _build_policy_stub(*args, **kwargs):
        _ = args, kwargs

        def _policy(obs: dict[str, object]) -> tuple[float, float]:
            _ = obs
            return 0.0, 0.0

        _policy._planner_reset = lambda seed=None: reset_calls.append(int(seed))
        return _policy, {"status": "ok", "planner_kinematics": {"robot_kinematics": "unknown"}}

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(_minimal_map_def()),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _build_policy_stub)
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    record = _run_map_episode(
        {"name": "s1", "simulation_config": {"max_episode_steps": 1}},
        seed=7,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )

    assert reset_calls == [7]
    assert record["scenario_id"] == "s1"


def test_run_map_episode_merges_planner_runtime_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Episode records should include optional planner runtime diagnostics when available."""

    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.zeros((0, 2), dtype=float)
            self.goal_pos = [np.array([1.0, 0.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((0, 2), dtype=float)

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)

        def reset(self, seed: int | None = None):
            _ = seed
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 0.0]},
            }
            return obs, {}

        def step(self, action):
            _ = action
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 0.0]},
            }
            return obs, 0.0, True, False, {"success": False}

        def close(self) -> None:
            return None

    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})

    def _build_policy_stub(*args, **kwargs):
        _ = args, kwargs

        def _policy(obs: dict[str, object]) -> tuple[float, float]:
            _ = obs
            return 0.0, 0.0

        _policy._planner_stats = lambda: {"solver_failures": 2, "fallback_stop_count": 2}
        return _policy, {"status": "ok", "planner_kinematics": {"robot_kinematics": "unknown"}}

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(_minimal_map_def()),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _build_policy_stub)
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    record = _run_map_episode(
        {"name": "s1", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )

    assert record["algorithm_metadata"]["planner_runtime"] == {
        "solver_failures": 2,
        "fallback_stop_count": 2,
    }


def test_run_map_episode_snapshots_planner_runtime_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner runtime stats should be captured before planner teardown mutates state."""

    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.zeros((0, 2), dtype=float)
            self.goal_pos = [np.array([1.0, 0.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((0, 2), dtype=float)

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)

        def reset(self, seed: int | None = None):
            _ = seed
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 0.0]},
            }
            return obs, {}

        def step(self, action):
            _ = action
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 0.0]},
            }
            return obs, 0.0, True, False, {"success": False}

        def close(self) -> None:
            return None

    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})

    def _build_policy_stub(*args, **kwargs):
        _ = args, kwargs
        closed = {"value": False}

        def _policy(obs: dict[str, object]) -> tuple[float, float]:
            _ = obs
            return 0.0, 0.0

        _policy._planner_stats = lambda: (
            {"solver_failures": 0} if closed["value"] else {"solver_failures": 3}
        )
        _policy._planner_close = lambda: closed.__setitem__("value", True)
        return _policy, {"status": "ok", "planner_kinematics": {"robot_kinematics": "unknown"}}

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(_minimal_map_def()),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner._build_policy", _build_policy_stub)
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    record = _run_map_episode(
        {"name": "s1", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )

    assert record["algorithm_metadata"]["planner_runtime"] == {"solver_failures": 3}


def test_run_map_episode_does_not_stop_on_waypoint_only_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waypoint-only success flags must not trigger map-runner early success stop."""

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
            self.step_calls = 0

        def reset(self, seed: int | None = None):
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            return obs, {}

        def step(self, action):
            self.step_calls += 1
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            info = {
                "success": True,
                "meta": {"is_waypoint_complete": True, "is_route_complete": False},
            }
            # Keep env alive unless map-runner incorrectly performs early success break.
            return obs, 0.0, False, False, info

        def close(self) -> None:
            return None

    map_def = _minimal_map_def()
    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})
    dummy_env = _DummyEnv(map_def)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: dummy_env,
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    scenario = {"name": "s1", "simulation_config": {"max_episode_steps": 999}}
    record = _run_map_episode(
        scenario,
        seed=1,
        horizon=50,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )
    assert dummy_env.step_calls == 50
    assert record["steps"] == 50
    assert record["termination_reason"] == "max_steps"


def test_run_map_episode_stops_immediately_on_route_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route completion should trigger early success stop immediately."""

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
            self.step_calls = 0

        def reset(self, seed: int | None = None):
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            return obs, {}

        def step(self, action):
            self.step_calls += 1
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            info = {"meta": {"is_waypoint_complete": True, "is_route_complete": True}}
            return obs, 0.0, False, False, info

        def close(self) -> None:
            return None

    map_def = _minimal_map_def()
    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})
    dummy_env = _DummyEnv(map_def)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: dummy_env,
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

    scenario = {"name": "s1", "simulation_config": {"max_episode_steps": 999}}
    record = _run_map_episode(
        scenario,
        seed=1,
        horizon=50,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )
    assert dummy_env.step_calls == 1
    assert record["steps"] == 1
    assert record["termination_reason"] == "success"


def test_run_map_episode_collision_wins_over_route_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collision + route-complete in same terminal step must resolve as collision."""

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
            self.step_calls = 0

        def reset(self, seed: int | None = None):
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            return obs, {}

        def step(self, action):
            self.step_calls += 1
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0]},
                "goal": {"current": [1.0, 1.0]},
            }
            info = {
                "meta": {
                    "is_waypoint_complete": True,
                    "is_route_complete": True,
                    "is_obstacle_collision": True,
                }
            }
            return obs, 0.0, True, False, info

        def close(self) -> None:
            return None

    map_def = _minimal_map_def()
    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})
    dummy_env = _DummyEnv(map_def)

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: dummy_env,
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 1.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    scenario = {"name": "s1", "simulation_config": {"max_episode_steps": 999}}
    record = _run_map_episode(
        scenario,
        seed=1,
        horizon=50,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        algo_config_path=None,
        scenario_path=Path("."),
    )
    assert record["termination_reason"] == "collision"
    assert record["outcome"]["collision_event"] is True
    assert record["outcome"]["route_complete"] is False


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


def test_run_map_batch_parallel_writes_results_in_job_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel map batch execution should preserve job order in output files."""
    scenario_one = {"name": "slow", "metadata": {"supported": True}}
    scenario_two = {"name": "fast", "metadata": {"supported": True}}
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.validate_scenario_list", lambda scenarios: []
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.load_schema", lambda path: {})

    def fake_run(job):
        scenario, seed, _ = job
        if scenario["name"] == "slow":
            time.sleep(0.05)
        return {"episode_id": f"{scenario['name']}-{seed}"}

    def fake_write(out_path_arg, schema, record):
        out_path_arg.parent.mkdir(parents=True, exist_ok=True)
        with out_path_arg.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.ProcessPoolExecutor",
        ThreadPoolExecutor,
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner._run_map_job_worker", fake_run)
    monkeypatch.setattr("robot_sf.benchmark.map_runner._write_validated", fake_write)

    result = run_map_batch(
        [scenario_one, scenario_two],
        out_path,
        schema_path=tmp_path / "schema.json",
        workers=2,
        resume=False,
    )

    assert result["written"] == 2
    lines = out_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records[0]["episode_id"].startswith("slow-")
    assert records[1]["episode_id"].startswith("fast-")


def test_run_map_batch_parallel_write_failure_prefers_top_level_scenario_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel write failures should report the record-level scenario_id when present."""
    scenarios = [
        {"name": "s1", "metadata": {"supported": True}},
        {"name": "s2", "metadata": {"supported": True}},
    ]
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr("robot_sf.benchmark.map_runner.validate_scenario_list", lambda _: [])
    monkeypatch.setattr("robot_sf.benchmark.map_runner.load_schema", lambda _: {})
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.ProcessPoolExecutor",
        ThreadPoolExecutor,
    )

    def fake_run(job):
        scenario, seed, _ = job
        return {
            "scenario_id": f"record-{scenario['name']}",
            "seed": seed,
            "episode_id": f"{scenario['name']}-{seed}",
        }

    def fake_write(*_args, **_kwargs):
        raise RuntimeError("forced write failure")

    monkeypatch.setattr("robot_sf.benchmark.map_runner._run_map_job_worker", fake_run)
    monkeypatch.setattr("robot_sf.benchmark.map_runner._write_validated", fake_write)

    result = run_map_batch(
        scenarios,
        out_path,
        schema_path=tmp_path / "schema.json",
        workers=2,
        resume=False,
    )

    assert result["written"] == 0
    assert result["failed_jobs"] == 2
    assert {failure["scenario_id"] for failure in result["failures"]} == {
        "record-s1",
        "record-s2",
    }


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


def test_run_map_batch_skips_incompatible_kinematics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incompatible planner/kinematics combinations should be skipped with explicit reason."""
    scenario = {
        "name": "s1",
        "metadata": {"supported": True},
        "robot_config": {"type": "holonomic"},
    }
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.validate_scenario_list", lambda scenarios: []
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.load_schema", lambda path: {})
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._planner_kinematics_compatibility",
        lambda **kwargs: (False, "mock incompatible combo"),
    )
    result = run_map_batch(
        [scenario],
        tmp_path / "out.jsonl",
        schema_path=tmp_path / "schema.json",
        algo="goal",
        workers=1,
        resume=False,
    )
    assert result["written"] == 0
    assert result["preflight"]["status"] == "skipped"
    assert "compatibility_reason" in result["preflight"]


def test_run_map_batch_preserves_runtime_planner_contract_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch summary should retain runtime-resolved holonomic command contract details."""
    scenario = {
        "name": "s1",
        "metadata": {"supported": True},
        "robot_config": {"type": "holonomic", "command_mode": "vx_vy"},
    }
    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.validate_scenario_list", lambda scenarios: []
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.load_schema", lambda path: {})
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._write_validated",
        lambda *args, **kwargs: None,
    )

    def _fake_worker(job):
        del job
        return {
            "episode_id": "ep1",
            "algorithm_metadata": {
                "planner_kinematics": {
                    "robot_kinematics": "holonomic",
                    "execution_mode": "adapter",
                    "planner_command_space": "holonomic_vxy_world",
                    "benchmark_command_space": "holonomic_vxy_world",
                    "projection_policy": "world_velocity_passthrough",
                    "execution_detail": "direct_holonomic_world_velocity",
                },
                "upstream_reference": {
                    "adapter_boundary": "runtime direct world velocity passthrough"
                },
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.map_runner._run_map_job_worker", _fake_worker)

    result = run_map_batch(
        [scenario],
        out_path,
        schema_path=tmp_path / "schema.json",
        algo="orca",
        workers=1,
        resume=False,
    )

    planner_meta = result["algorithm_metadata_contract"]["planner_kinematics"]
    assert planner_meta["planner_command_space"] == "holonomic_vxy_world"
    assert planner_meta["benchmark_command_space"] == "holonomic_vxy_world"
    assert planner_meta["projection_policy"] == "world_velocity_passthrough"
    assert planner_meta["execution_detail"] == "direct_holonomic_world_velocity"
    assert (
        result["algorithm_metadata_contract"]["upstream_reference"]["adapter_boundary"]
        == "runtime direct world velocity passthrough"
    )


def test_run_map_batch_hrvo_smoke_writes_episode_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HRVO should run through the batch runner and emit an episodes.jsonl record."""

    monkeypatch.setattr("robot_sf.benchmark.map_runner.validate_scenario_list", lambda _: [])

    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.array([[1.2, 0.0]], dtype=float)
            self.goal_pos = [np.array([2.0, 0.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((1, 2), dtype=float)

        def iter_obstacle_segments(self):
            return [((0.5, -0.5), (0.5, 0.5))]

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)
            self.action_space = None

        def reset(self, seed: int | None = None):
            _ = seed
            obs = {
                "robot": {
                    "position": np.array([0.0, 0.0], dtype=np.float32),
                    "heading": np.array([0.0], dtype=np.float32),
                    "speed": np.array([0.0, 0.0], dtype=np.float32),
                    "radius": np.array([0.5], dtype=np.float32),
                },
                "goal": {
                    "current": np.array([2.0, 0.0], dtype=np.float32),
                    "next": np.array([0.0, 0.0], dtype=np.float32),
                },
                "pedestrians": {
                    "positions": np.array([[1.2, 0.0]], dtype=np.float32),
                    "velocities": np.zeros((1, 2), dtype=np.float32),
                    "radius": np.array([0.4], dtype=np.float32),
                    "count": np.array([1.0], dtype=np.float32),
                },
                "map": {"size": np.array([5.0, 4.0], dtype=np.float32)},
                "sim": {"timestep": np.array([0.1], dtype=np.float32)},
            }
            return obs, {}

        def step(self, action):
            _ = action
            obs, _ = self.reset()
            return obs, 0.0, True, False, {"meta": {"is_route_complete": True}}

        def close(self) -> None:
            return None

    dummy_config = type(
        "Cfg",
        (),
        {
            "sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})(),
            "robot_config": HolonomicDriveSettings(
                max_speed=1.0,
                max_angular_speed=1.0,
                command_mode="vx_vy",
            ),
        },
    )
    scenario = {
        "name": "hrvo_smoke",
        "metadata": {"supported": True},
        "robot_config": {"type": "holonomic", "command_mode": "vx_vy"},
        "simulation_config": {"max_episode_steps": 1},
        "seeds": [1],
    }
    out_path = tmp_path / "episodes.jsonl"
    schema_path = tmp_path / "episode.schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(_minimal_map_def()),
    )
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
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.sample_obstacle_points",
        lambda segments, spacing: np.array([[0.5, 0.0], [0.5, 0.25]], dtype=float),
    )

    result = run_map_batch(
        [scenario],
        out_path,
        schema_path=schema_path,
        algo="hrvo",
        algo_config_path="configs/algos/hrvo_camera_ready.yaml",
        benchmark_profile="experimental",
        workers=1,
        resume=False,
    )

    assert result["written"] == 1
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "hrvo_smoke" in lines[0]


def _normalize_episode_record(record: dict[str, object]) -> dict[str, object]:
    normalized = dict(record)
    normalized.pop("timestamps", None)
    normalized.pop("wall_time_sec", None)
    normalized.pop("timing", None)
    return normalized


def test_run_map_batch_repeated_runs_produce_stable_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeat a minimal map-run batch and assert stable episode contents ignoring runtime metadata."""

    monkeypatch.setattr("robot_sf.benchmark.map_runner.validate_scenario_list", lambda _: [])

    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.array([[1.2, 0.0]], dtype=float)
            self.goal_pos = [np.array([2.0, 0.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((1, 2), dtype=float)

        def iter_obstacle_segments(self):
            return [((0.5, -0.5), (0.5, 0.5))]

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)
            self.action_space = None

        def reset(self, seed: int | None = None):
            _ = seed
            obs = {
                "robot": {
                    "position": np.array([0.0, 0.0], dtype=np.float32),
                    "heading": np.array([0.0], dtype=np.float32),
                    "speed": np.array([0.0, 0.0], dtype=np.float32),
                    "radius": np.array([0.5], dtype=np.float32),
                },
                "goal": {
                    "current": np.array([2.0, 0.0], dtype=np.float32),
                    "next": np.array([0.0, 0.0], dtype=np.float32),
                },
                "pedestrians": {
                    "positions": np.array([[1.2, 0.0]], dtype=np.float32),
                    "velocities": np.zeros((1, 2), dtype=np.float32),
                    "radius": np.array([0.4], dtype=np.float32),
                    "count": np.array([1.0], dtype=np.float32),
                },
                "map": {"size": np.array([5.0, 4.0], dtype=np.float32)},
                "sim": {"timestep": np.array([0.1], dtype=np.float32)},
            }
            return obs, {}

        def step(self, action):
            _ = action
            obs, _ = self.reset()
            return obs, 0.0, True, False, {"meta": {"is_route_complete": True}}

        def close(self) -> None:
            return None

    dummy_config = type(
        "Cfg",
        (),
        {
            "sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})(),
            "robot_config": HolonomicDriveSettings(
                max_speed=1.0,
                max_angular_speed=1.0,
                command_mode="vx_vy",
            ),
        },
    )
    scenario = {
        "name": "hrvo_repeat",
        "metadata": {"supported": True},
        "robot_config": {"type": "holonomic", "command_mode": "vx_vy"},
        "simulation_config": {"max_episode_steps": 1},
        "seeds": [1],
    }
    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    schema_path = tmp_path / "episode.schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(_minimal_map_def()),
    )
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
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.sample_obstacle_points",
        lambda segments, spacing: np.array([[0.5, 0.0], [0.5, 0.25]], dtype=float),
    )

    run_map_batch(
        [scenario],
        out1,
        schema_path=schema_path,
        algo="hrvo",
        algo_config_path="configs/algos/hrvo_camera_ready.yaml",
        benchmark_profile="experimental",
        workers=1,
        resume=False,
    )
    run_map_batch(
        [scenario],
        out2,
        schema_path=schema_path,
        algo="hrvo",
        algo_config_path="configs/algos/hrvo_camera_ready.yaml",
        benchmark_profile="experimental",
        workers=1,
        resume=False,
    )

    # Parse both outputs
    lines1 = out1.read_text(encoding="utf-8").splitlines()
    lines2 = out2.read_text(encoding="utf-8").splitlines()
    assert len(lines1) == len(lines2), "Line count differs between runs"

    # For each record, verify key order is consistent (deterministic serialization)
    recs1 = []
    recs2 = []
    for i, (line1, line2) in enumerate(zip(lines1, lines2, strict=True)):
        rec1 = json.loads(line1)
        rec2 = json.loads(line2)
        recs1.append(rec1)
        recs2.append(rec2)

        # Verify top-level key ordering is consistent
        keys1 = list(rec1.keys())
        keys2 = list(rec2.keys())
        assert keys1 == keys2, (
            f"Top-level key order differs at line {i} "
            f"(JSON serialization may be non-deterministic):\n"
            f"Run 1 keys: {keys1}\n"
            f"Run 2 keys: {keys2}"
        )

    # Then verify semantic equality (ignoring runtime metadata)
    for i, (rec1, rec2) in enumerate(zip(recs1, recs2, strict=True)):
        norm1 = _normalize_episode_record(rec1)
        norm2 = _normalize_episode_record(rec2)
        assert norm1 == norm2, f"Episode {i} records differ: {norm1} vs {norm2}"


def test_policy_command_to_env_action_holonomic_vx_vy_uses_midpoint_heading() -> None:
    """Holonomic vx/vy conversion should include angular intent via midpoint heading."""

    class HolonomicConfig:
        command_mode = "vx_vy"

    robot = SimpleNamespace(pose=((0.0, 0.0), 0.0))
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[robot]), action_space=None)
    config = SimpleNamespace(
        robot_config=HolonomicConfig(),
        sim_config=SimpleNamespace(time_per_step_in_secs=0.2),
    )
    action = _policy_command_to_env_action(env=env, config=config, command=(1.0, 2.0))

    expected_heading = 0.2
    np.testing.assert_allclose(
        action,
        np.array([np.cos(expected_heading), np.sin(expected_heading)], dtype=float),
    )


def test_default_robot_command_space_prefers_runtime_command_mode() -> None:
    """Runtime command mode should override algo-config command mode for holonomic metadata."""
    assert (
        _default_robot_command_space(
            "holonomic",
            {"command_mode": "unicycle_vw"},
            robot_command_mode="vx_vy",
        )
        == "holonomic_vxy_world"
    )


# ---------------------------------------------------------------------------
# Issue #697 — holonomic social-force diagnosis config contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/algos/social_force_holonomic_tuned_tau_high.yaml",
        "configs/algos/social_force_holonomic_tuned_tau_low.yaml",
        "configs/algos/social_force_holonomic_tuned_repulsion_low.yaml",
    ],
)
def test_social_force_holonomic_sweep_configs_produce_valid_socnav_config(
    config_path: str,
) -> None:
    """Each issue-697 parameter-sweep YAML should parse into a valid SocNavPlannerConfig."""
    from robot_sf.planner.socnav import SocNavPlannerConfig

    cfg = _parse_algo_config(config_path)
    socnav_cfg = _build_socnav_config(cfg)
    assert isinstance(socnav_cfg, SocNavPlannerConfig)
    assert socnav_cfg.max_linear_speed == pytest.approx(1.0)
    assert socnav_cfg.social_force_tau > 0.0, "tau must be positive"
    assert socnav_cfg.social_force_repulsion_weight >= 0.0, "repulsion_weight must be non-negative"


@pytest.mark.parametrize(
    ("config_rel_path", "param_name", "direction"),
    [
        (
            "../../configs/algos/social_force_holonomic_tuned_tau_high.yaml",
            "social_force_tau",
            "higher",
        ),
        (
            "../../configs/algos/social_force_holonomic_tuned_tau_low.yaml",
            "social_force_tau",
            "lower",
        ),
        (
            "../../configs/algos/social_force_holonomic_tuned_repulsion_low.yaml",
            "social_force_repulsion_weight",
            "lower",
        ),
    ],
)
def test_social_force_holonomic_sweep_applies_correct_change(
    config_rel_path: str,
    param_name: str,
    direction: str,
) -> None:
    """Each sweep config must move the target parameter in the stated direction vs. default."""
    from robot_sf.planner.socnav import SocNavPlannerConfig

    config_path = Path(__file__).parent / config_rel_path
    cfg = _parse_algo_config(str(config_path))
    socnav_cfg = _build_socnav_config(cfg)
    default_value = getattr(SocNavPlannerConfig(), param_name)
    tuned_value = getattr(socnav_cfg, param_name)
    if direction == "higher":
        assert tuned_value > default_value, (
            f"{param_name}: tuned {tuned_value} must exceed default {default_value}"
        )
    else:
        assert tuned_value < default_value, (
            f"{param_name}: tuned {tuned_value} must be below default {default_value}"
        )


def test_holonomic_social_force_diagnosis_config_contains_expected_planners() -> None:
    """The diagnosis benchmark config must list local social_force and the upstream wrapper."""
    import yaml

    config_path = (
        Path(__file__).parent / "../../configs/benchmarks/holonomic_social_force_diagnosis.yaml"
    )
    assert config_path.exists(), f"Diagnosis config not found: {config_path}"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    planner_keys = {p["key"] for p in data.get("planners", [])}
    assert "social_force" in planner_keys, "Diagnosis config must include local social_force"
    assert "social_navigation_pyenvs_socialforce" in planner_keys, (
        "Diagnosis config must include the upstream socialforce wrapper"
    )
    assert data.get("holonomic_command_mode") == "vx_vy", (
        "Diagnosis config must use holonomic vx_vy command mode"
    )
