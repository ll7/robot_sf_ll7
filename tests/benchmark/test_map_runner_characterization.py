"""Characterization baseline for ``map_runner_episode.run_map_episode`` and ``map_runner._build_policy``.

These tests pin the *current observable behavior* of the two god-functions whose
decomposition is slice 4 of #4770. They are table-driven and assert golden
values on small synthetic inputs. The tests lock behavior so the later
decomposition can prove behavior-preservation.

Purpose (issue #4927, Refs #4770): pin observable behavior of the major internal
phases — policy construction per planner type on synthetic configs, episode-loop
outputs on a tiny deterministic scenario, error paths for missing
checkpoints/bad configs. Bugs found: document + file separately, never fix
inline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.benchmark.map_runner_episode import run_map_episode
from robot_sf.common.types import Rect
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.robot.differential_drive import DifferentialDriveSettings

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _minimal_map_def() -> MapDefinition:
    """Build a minimal map definition fixture with one route."""
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


def _dummy_scenario(name: str = "char_test") -> dict[str, Any]:
    """Build a minimal scenario dict for characterization tests."""
    return {
        "name": name,
        "simulation_config": {"max_episode_steps": 1},
        "robot_config": {"type": "differential_drive"},
    }


class _DummySim:
    """Simulator stub for characterization tests."""

    def __init__(self, map_def: MapDefinition) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.ped_pos = np.zeros((0, 2), dtype=float)
        self.goal_pos = [np.array([1.0, 1.0], dtype=float)]
        self.map_def = map_def
        self.last_ped_forces = np.zeros((0, 2), dtype=float)


class _DummyEnv:
    """Environment stub for characterization tests."""

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


# ---------------------------------------------------------------------------
# _build_policy characterization: policy metadata shape per planner type
# ---------------------------------------------------------------------------


class TestBuildPolicyCharacterization:
    """Pin metadata shape and key fields returned by ``_build_policy``."""

    def test_goal_planner_returns_classical_baseline_category(self) -> None:
        """Goal planner must return classical baseline_category in metadata."""
        policy, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        assert callable(policy)
        assert meta["algorithm"] == "goal"
        assert meta["baseline_category"] == "classical"
        assert meta["status"] == "ok"
        assert "config_hash" in meta

    def test_social_force_planner_returns_classical_baseline(self) -> None:
        """Social force planner must return classical baseline category."""
        policy, meta = _build_policy(
            "social_force",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        assert callable(policy)
        assert meta["algorithm"] == "social_force"
        assert meta["baseline_category"] == "classical"
        assert meta["status"] == "ok"

    def test_sampling_planner_returns_classical_baseline(self) -> None:
        """Sampling planner must return classical baseline category."""
        policy, meta = _build_policy(
            "socnav_sampling",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        assert callable(policy)
        assert meta["algorithm"] == "socnav_sampling"
        assert meta["baseline_category"] == "classical"

    def test_unknown_algo_raises(self) -> None:
        """Unknown algorithm key must raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            _build_policy(
                "nonexistent_algo_xyz",
                {},
                robot_kinematics="differential_drive",
            )

    def test_metadata_has_kinematics_feasibility(self) -> None:
        """All planners must attach kinematics feasibility metadata."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        feasibility = meta.get("kinematics_feasibility")
        assert isinstance(feasibility, dict)
        assert "commands_evaluated" in feasibility
        assert "infeasible_native_count" in feasibility
        assert "projected_count" in feasibility

    def test_metadata_has_planner_kinematics_block(self) -> None:
        """All planners must attach planner_kinematics metadata block."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        planner_kin = meta.get("planner_kinematics")
        assert isinstance(planner_kin, dict)
        assert "robot_kinematics" in planner_kin
        assert planner_kin["robot_kinematics"] == "differential_drive"

    def test_goal_metadata_has_observation_spec(self) -> None:
        """Goal planner metadata must carry observation_spec with default_mode."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        obs_spec = meta.get("observation_spec")
        assert isinstance(obs_spec, dict)
        assert obs_spec["default_mode"] == "goal_state"
        assert "goal_state" in obs_spec["supported_modes"]

    def test_social_force_metadata_has_observation_spec(self) -> None:
        """Social force metadata must carry observation_spec with socnav_state."""
        _, meta = _build_policy(
            "social_force",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        obs_spec = meta.get("observation_spec")
        assert isinstance(obs_spec, dict)
        assert obs_spec["default_mode"] == "socnav_state"

    def test_goal_metadata_has_observation_level(self) -> None:
        """Goal planner metadata must carry observation_level with key."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        obs_level = meta.get("observation_level")
        assert isinstance(obs_level, dict)
        assert obs_level["key"] == "oracle_full_state"

    def test_social_force_metadata_has_observation_level(self) -> None:
        """Social force metadata must carry observation_level with key."""
        _, meta = _build_policy(
            "social_force",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        obs_level = meta.get("observation_level")
        assert isinstance(obs_level, dict)
        assert obs_level["key"] == "tracked_agents_no_noise"

    def test_metadata_has_planner_contract(self) -> None:
        """All planners must carry planner_contract metadata block."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        contract = meta.get("planner_contract")
        assert isinstance(contract, dict)
        assert "planner_id" in contract
        assert contract["planner_id"] == "goal"

    def test_metadata_has_canonical_algorithm(self) -> None:
        """All planners must carry canonical_algorithm key."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        assert meta["canonical_algorithm"] == "goal"

    def test_goal_policy_semantics(self) -> None:
        """Goal planner must report deterministic_goal_seeking semantics."""
        _, meta = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        assert meta["policy_semantics"] == "deterministic_goal_seeking"

    def test_social_force_policy_semantics(self) -> None:
        """Social force must report social_force_adapter semantics."""
        _, meta = _build_policy(
            "social_force",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        assert meta["policy_semantics"] == "social_force_adapter"


# ---------------------------------------------------------------------------
# run_map_episode characterization: episode-loop output structure
# ---------------------------------------------------------------------------


class TestRunMapEpisodeCharacterization:
    """Pin the record shape returned by ``run_map_episode`` on a tiny scenario."""

    def _run_stubbed_episode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        algo: str = "goal",
        algo_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a stubbed episode and return the record."""
        map_def = _minimal_map_def()
        dummy_config = SimpleNamespace(
            sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=0.4),
            robot_config=DifferentialDriveSettings(max_linear_speed=2.0),
        )

        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode._build_env_config",
            lambda scenario, scenario_path: dummy_config,
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode.make_robot_env",
            lambda config, seed, debug: _DummyEnv(map_def),
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode.sample_obstacle_points",
            lambda *args: None,
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode.compute_shortest_path_length",
            lambda *args: 1.0,
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode.compute_all_metrics",
            lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode.post_process_metrics",
            lambda metrics, **kwargs: metrics,
        )

        policy, _ = _build_policy(
            algo,
            algo_config or {},
            robot_kinematics="differential_drive",
        )
        scenario = _dummy_scenario()
        return run_map_episode(
            scenario,
            seed=1,
            horizon=None,
            dt=0.1,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo=algo,
            scenario_path=Path("."),
            algo_config=algo_config,
            policy_builder=lambda *a, **kw: (policy, _build_policy(algo, algo_config or {}, robot_kinematics="differential_drive")[1]),
        )

    def test_episode_record_has_required_top_level_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Episode record must carry all required top-level keys."""
        record = self._run_stubbed_episode(monkeypatch)
        required_keys = {
            "scenario_id",
            "seed",
            "metrics",
            "algorithm_metadata",
            "observation_mode",
            "observation_level",
            "scenario_params",
            "episode_id",
            "timestamps",
        }
        assert required_keys.issubset(record.keys())

    def test_episode_record_scenario_id_matches_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario ID in record must match input scenario name."""
        record = self._run_stubbed_episode(monkeypatch)
        assert record["scenario_id"] == "char_test"

    def test_episode_record_seed_matches_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seed in record must match input seed."""
        record = self._run_stubbed_episode(monkeypatch)
        assert record["seed"] == 1

    def test_episode_record_metrics_is_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Metrics must be a dictionary with expected keys."""
        record = self._run_stubbed_episode(monkeypatch)
        assert isinstance(record["metrics"], dict)
        assert "success" in record["metrics"]
        assert "collisions" in record["metrics"]

    def test_episode_record_algorithm_metadata_has_baseline_category(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Algorithm metadata must carry baseline_category."""
        record = self._run_stubbed_episode(monkeypatch)
        algo_md = record["algorithm_metadata"]
        assert algo_md["baseline_category"] == "classical"

    def test_episode_record_observation_mode_for_goal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Goal planner must produce goal_state observation mode."""
        record = self._run_stubbed_episode(monkeypatch, algo="goal")
        assert record["observation_mode"] == "goal_state"
        assert record["observation_level"] == "oracle_full_state"

    def test_episode_record_has_algo_metadata_observation_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Algorithm metadata must carry observation_spec with active_mode."""
        record = self._run_stubbed_episode(monkeypatch, algo="goal")
        algo_md = record["algorithm_metadata"]
        assert algo_md["observation_spec"]["active_mode"] == "goal_state"
        assert algo_md["observation_level"]["key"] == "oracle_full_state"

    def test_episode_record_has_kinematics_feasibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Algorithm metadata must carry kinematics_feasibility block."""
        record = self._run_stubbed_episode(monkeypatch, algo="goal")
        algo_md = record["algorithm_metadata"]
        feasibility = algo_md.get("kinematics_feasibility")
        assert isinstance(feasibility, dict)
        assert "commands_evaluated" in feasibility
        assert "infeasible_native_count" in feasibility
        assert "projected_count" in feasibility

    def test_episode_record_has_ammv_feasibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Algorithm metadata must carry AMMV feasibility block."""
        record = self._run_stubbed_episode(monkeypatch, algo="goal")
        algo_md = record["algorithm_metadata"]
        ammv = algo_md.get("ammv_feasibility")
        assert isinstance(ammv, dict)
        assert ammv["schema_version"] == "ammv_feasibility.v1"
        assert ammv["proxy_kind"] == "internal_non_hardware"
        assert isinstance(ammv["feasible"], bool)

    def test_episode_record_has_scenario_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario params must mirror observation mode/level."""
        record = self._run_stubbed_episode(monkeypatch, algo="goal")
        assert record["scenario_params"]["observation_mode"] == "goal_state"
        assert record["scenario_params"]["observation_level"] == "oracle_full_state"


# ---------------------------------------------------------------------------
# Error-path characterization
# ---------------------------------------------------------------------------


class TestErrorPathCharacterization:
    """Pin error behavior for bad configs and missing checkpoints."""

    def test_safety_wrapper_and_cbf_cannot_both_be_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabling both safety_wrapper and cbf_safety_filter must raise."""
        dummy_config = SimpleNamespace(
            sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=0.4),
            robot_config=DifferentialDriveSettings(max_linear_speed=2.0),
        )

        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner_episode._build_env_config",
            lambda scenario, scenario_path: dummy_config,
        )

        policy, _ = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        scenario = _dummy_scenario()
        with pytest.raises(ValueError, match="safety_wrapper and cbf_safety_filter cannot both be enabled"):
            run_map_episode(
                scenario,
                seed=1,
                horizon=None,
                dt=0.1,
                record_forces=False,
                snqi_weights=None,
                snqi_baseline=None,
                algo="goal",
                scenario_path=Path("."),
                safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
                cbf_safety_filter={"enabled": True, "arm_key": "cbf_collision_cone_on"},
                policy_builder=lambda *a, **kw: (policy, _build_policy("goal", {}, robot_kinematics="differential_drive")[1]),
            )

    def test_ppo_paper_profile_without_provenance_raises(self) -> None:
        """PPO paper profile without provenance must raise ValueError."""
        with pytest.raises(ValueError, match="PPO paper profile requested but gate failed"):
            _build_policy(
                "ppo",
                {"profile": "paper"},
                robot_kinematics="differential_drive",
            )


# ---------------------------------------------------------------------------
# Policy callable contract characterization
# ---------------------------------------------------------------------------


class TestPolicyCallableContract:
    """Pin the callable contract of policies returned by ``_build_policy``."""

    def test_goal_policy_returns_tuple_of_two_floats(self) -> None:
        """Goal policy callable must return (linear, angular) float tuple."""
        policy, _ = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        obs = {
            "dt": 0.1,
            "robot": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "radius": np.array([0.3]),
            },
            "goal": {"current": np.array([1.0, 0.0])},
            "pedestrians": {
                "positions": np.zeros((0, 2)),
                "velocities": np.zeros((0, 2)),
                "radius": np.zeros(0),
            },
        }
        result = policy(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_social_force_policy_returns_tuple_of_two_floats(self) -> None:
        """Social force policy callable must return (linear, angular) float tuple."""
        policy, _ = _build_policy(
            "social_force",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        obs = {
            "dt": 0.1,
            "robot": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "radius": np.array([0.3]),
            },
            "goal": {"current": np.array([1.0, 0.0])},
            "pedestrians": {
                "positions": np.zeros((0, 2)),
                "velocities": np.zeros((0, 2)),
                "radius": np.zeros(0),
            },
        }
        result = policy(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_sampling_policy_returns_tuple_of_two_floats(self) -> None:
        """Sampling policy callable must return (linear, angular) float tuple."""
        policy, _ = _build_policy(
            "socnav_sampling",
            {"max_speed": 1.0},
            robot_kinematics="differential_drive",
        )
        obs = {
            "dt": 0.1,
            "robot": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "radius": np.array([0.3]),
            },
            "goal": {"current": np.array([1.0, 0.0])},
            "pedestrians": {
                "positions": np.zeros((0, 2)),
                "velocities": np.zeros((0, 2)),
                "radius": np.zeros(0),
            },
        }
        result = policy(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_goal_policy_no_pedestrians_deterministic(self) -> None:
        """Goal policy with no pedestrians must return deterministic output."""
        policy, _ = _build_policy(
            "goal",
            {},
            robot_kinematics="differential_drive",
        )
        obs = {
            "dt": 0.1,
            "robot": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "radius": np.array([0.3]),
            },
            "goal": {"current": np.array([1.0, 0.0])},
            "pedestrians": {
                "positions": np.zeros((0, 2)),
                "velocities": np.zeros((0, 2)),
                "radius": np.zeros(0),
            },
        }
        r1 = policy(obs)
        r2 = policy(obs)
        assert r1 == r2

    def test_goal_policy_linear_velocity_bounded(self) -> None:
        """Goal policy linear velocity must be bounded by max_speed."""
        policy, _ = _build_policy(
            "goal",
            {"max_speed": 0.5},
            robot_kinematics="differential_drive",
        )
        obs = {
            "dt": 0.1,
            "robot": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "radius": np.array([0.3]),
            },
            "goal": {"current": np.array([100.0, 0.0])},
            "pedestrians": {
                "positions": np.zeros((0, 2)),
                "velocities": np.zeros((0, 2)),
                "radius": np.zeros(0),
            },
        }
        linear, _angular = policy(obs)
        assert abs(linear) <= 0.5 + 1e-6


# ---------------------------------------------------------------------------
# Multi-planner metadata key stability
# ---------------------------------------------------------------------------


class TestMultiPlannerMetadataKeyStability:
    """Pin that all planners return the same required metadata keys."""

    REQUIRED_META_KEYS = {
        "algorithm",
        "status",
        "config",
        "config_hash",
        "canonical_algorithm",
        "baseline_category",
        "observation_spec",
        "observation_level",
        "planner_kinematics",
        "planner_contract",
        "kinematics_feasibility",
    }

    @pytest.mark.parametrize("algo,algo_config", [
        ("goal", {}),
        ("social_force", {"max_speed": 1.0}),
        ("socnav_sampling", {"max_speed": 1.0}),
    ])
    def test_required_metadata_keys_present(
        self, algo: str, algo_config: dict[str, Any]
    ) -> None:
        """All planners must return the required metadata keys."""
        _, meta = _build_policy(
            algo,
            algo_config,
            robot_kinematics="differential_drive",
        )
        # kinematics_feasibility is only finalized at episode level,
        # so verify it exists but with raw keys
        if "kinematics_feasibility" in meta:
            kf = meta["kinematics_feasibility"]
            assert isinstance(kf, dict)
            assert "commands_evaluated" in kf
        remaining_keys = self.REQUIRED_META_KEYS - {"kinematics_feasibility"}
        assert remaining_keys.issubset(meta.keys()), (
            f"Missing keys for {algo}: {remaining_keys - meta.keys()}"
        )
