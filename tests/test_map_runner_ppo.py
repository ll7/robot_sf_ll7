"""Unit tests for PPO support in map_runner policy bridge."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from robot_sf.benchmark import map_runner
from robot_sf.gym_env.unified_config import RobotSimulationConfig


class _DummyPPOPlanner:
    """Test double for PPO planner integration in map runner."""

    def __init__(self, config, *, seed=None):
        self.config = dict(config)
        self.seed = seed
        self.closed = False
        self.last_obs = None
        self.bound_envs = []

    def step(self, _obs):
        """Return the configured action and retain the received observation."""
        self.last_obs = _obs
        return self.config.get("test_action", {"v": 0.5, "omega": 0.0})

    def close(self):
        """Mark the dummy planner as closed."""
        self.closed = True

    def bind_env(self, env):
        """Record runtime env binding."""
        self.bound_envs.append(env)

    def get_metadata(self):
        """Return map-runner metadata for the dummy planner.

        Returns:
            Metadata dictionary mirroring the planner contract.
        """
        return {"algorithm": "ppo", "status": "ok", "config": dict(self.config)}


class _DummyDrlVoPlanner:
    """Test double for DRL-VO planner integration in map runner."""

    def __init__(self, config, *, seed=None):
        self.config = dict(config)
        self.seed = seed
        self.closed = False
        self.reset_called = False
        self.last_obs = None

    def step(self, obs):
        """Return the configured action and retain the received observation."""
        self.last_obs = obs
        return self.config.get("test_action", {"v": 0.5, "omega": 0.0})

    def close(self):
        """Mark the dummy planner as closed."""
        self.closed = True

    def reset(self):
        """Record that the dummy planner reset hook was invoked."""
        self.reset_called = True

    def get_metadata(self):
        """Return map-runner metadata for the dummy planner.

        Returns:
            Metadata dictionary mirroring the planner contract.
        """
        return {"algorithm": "drl_vo", "status": "ok", "config": dict(self.config)}


def _sample_obs(heading: float = 0.0) -> dict:
    """Build a minimal SocNav observation used by policy bridge tests.

    Returns:
        Structured observation dictionary.
    """
    return {
        "dt": 0.1,
        "robot": {
            "position": np.array([1.0, 1.0], dtype=float),
            "velocity": np.array([0.0, 0.0], dtype=float),
            "heading": np.array([heading], dtype=float),
            "radius": np.array([0.3], dtype=float),
        },
        "goal": {"current": np.array([4.0, 1.0], dtype=float)},
        "pedestrians": {
            "positions": np.array([[2.0, 2.0]], dtype=float),
            "velocities": np.array([[0.0, 0.0]], dtype=float),
            "radius": np.array([0.35], dtype=float),
        },
    }


def test_build_policy_ppo_accepts_unicycle_action(monkeypatch):
    """Ensure PPO map policy accepts native unicycle outputs."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, meta = map_runner._build_policy(
        "ppo",
        {"test_action": {"v": 0.7, "omega": -0.2}},
    )

    action_v, action_w = policy(_sample_obs())
    assert action_v == pytest.approx(0.7)
    assert action_w == pytest.approx(-0.2)
    assert meta["algorithm"] == "ppo"
    assert meta["status"] == "ok"
    assert callable(getattr(policy, "_planner_close", None))


def test_build_policy_ppo_converts_velocity_to_unicycle(monkeypatch):
    """Ensure velocity-vector PPO outputs are converted to unicycle commands."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, _ = map_runner._build_policy(
        "ppo",
        {
            "test_action": {"vx": 1.0, "vy": 0.0},
            "v_max": 0.8,
            "omega_max": 0.5,
        },
    )

    action_v, action_w = policy(_sample_obs(heading=np.pi / 2))
    assert action_v == pytest.approx(0.8)
    assert action_w == pytest.approx(-0.5)


def test_build_policy_ppo_rejects_unknown_action_payload(monkeypatch):
    """Reject malformed PPO action payloads lacking known action keys."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, _ = map_runner._build_policy(
        "ppo",
        {"test_action": {"foo": 1.0}},
    )

    with pytest.raises(ValueError, match="Unsupported PPO action payload"):
        policy(_sample_obs())


def test_build_policy_ppo_adapter_impact_updates_metadata(monkeypatch):
    """PPO adapter-impact counters should mutate the returned metadata in-place."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, meta = map_runner._build_policy(
        "ppo",
        {"test_action": {"v": 0.4, "omega": 0.1}},
        adapter_impact_eval=True,
    )

    policy(_sample_obs())
    impact = meta.get("adapter_impact")
    assert isinstance(impact, dict)
    assert impact["requested"] is True
    assert impact["native_steps"] == 1
    assert impact["adapted_steps"] == 0
    assert impact["status"] == "collecting"


def test_build_policy_drl_vo_adapter_impact_updates_metadata(monkeypatch):
    """DRL-VO adapter-impact counters should mutate returned metadata in-place."""
    monkeypatch.setattr(map_runner, "DrlVoPlanner", _DummyDrlVoPlanner)
    policy, meta = map_runner._build_policy(
        "drl_vo",
        {"test_action": {"vx": 1.0, "vy": 0.0}, "v_max": 0.8},
        adapter_impact_eval=True,
    )

    action_v, action_w = policy(_sample_obs(heading=np.pi / 2))
    impact = meta.get("adapter_impact")
    assert action_v == pytest.approx(0.8)
    assert action_w < 0.0
    assert isinstance(impact, dict)
    assert impact["requested"] is True
    assert impact["native_steps"] == 0
    assert impact["adapted_steps"] == 1
    assert impact["status"] == "collecting"
    assert callable(getattr(policy, "_planner_close", None))
    assert callable(getattr(policy, "_planner_reset", None))


def test_obs_to_ppo_format_preserves_heading_for_unicycle_fallback():
    """DRL-VO unicycle fallback needs robot heading in the converted observation."""
    obs = _sample_obs(heading=np.pi / 2)

    formatted = map_runner._obs_to_ppo_format(obs)
    assert formatted["robot"]["heading"] == pytest.approx(np.pi / 2)


def test_build_policy_ppo_dict_mode_passes_raw_observation(monkeypatch):
    """Dict obs mode should pass the raw SocNav structured observation through to PPO."""
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    policy, _ = map_runner._build_policy(
        "ppo",
        {
            "obs_mode": "dict",
            "test_action": {"v": 0.3, "omega": 0.0},
        },
    )

    obs = _sample_obs()
    obs["robot_position"] = np.array([1.0, 1.0], dtype=float)
    obs["goal_current"] = np.array([4.0, 1.0], dtype=float)
    action_v, action_w = policy(obs)
    assert action_v == pytest.approx(0.3)
    assert action_w == pytest.approx(0.0)
    # Validate passthrough by rebuilding policy and checking last_obs on dummy planner.
    dummy = _DummyPPOPlanner({"obs_mode": "dict", "test_action": {"v": 0.3, "omega": 0.0}})
    monkeypatch.setattr(map_runner, "PPOPlanner", lambda *_args, **_kwargs: dummy)
    policy2, _ = map_runner._build_policy(
        "ppo",
        {"obs_mode": "dict", "test_action": {"v": 0.3, "omega": 0.0}},
    )
    policy2(obs)
    assert dummy.last_obs is obs


class _DummyGuardedPPOAdapter:
    """Test double for guarded PPO arbitration."""

    instances: ClassVar[list[_DummyGuardedPPOAdapter]] = []
    residual_clipped: ClassVar[bool] = False

    def __init__(self, config=None, *, fallback_adapter=None, prior_adapter=None):
        self.config = config
        self.fallback_adapter = fallback_adapter
        self.prior_adapter = prior_adapter
        self.last_command = None
        self.bound_envs = []
        self.reset_seeds = []
        self.closed = False
        self.__class__.instances.append(self)

    def choose_command_decision(self, obs, ppo_command):
        """Return a structured guard decision with residual adaptation metadata."""
        self.last_command = (obs, ppo_command)
        return map_runner.ShieldDecision(
            proposed_action=(float(ppo_command[0]), float(ppo_command[1])),
            filtered_action=(0.1, -0.2),
            decision_label="fallback_safe",
            intervention_reason="test_guard_decision",
            fallback_controller_state={
                "action_adaptation": {
                    "mode": "prior_residual",
                    "residual_clipped": bool(self.__class__.residual_clipped),
                }
            },
        )

    def choose_command(self, obs, ppo_command):
        """Return a fallback command while recording arbitration inputs.

        Returns:
            Guarded command and guard status.
        """
        self.last_command = (obs, ppo_command)
        return (0.1, -0.2), "fallback_safe"

    def bind_env(self, env):
        """Record the environment passed to the guard adapter."""
        self.bound_envs.append(env)

    def reset(self, *, seed=None):
        """Record guard reset seed values."""
        self.reset_seeds.append(seed)

    def close(self):
        """Mark the guard adapter as closed."""
        self.closed = True


def test_build_policy_guarded_ppo_arbitrates_and_tracks_guard_stats(monkeypatch):
    """Guarded PPO should route PPO output through the guard and record intervention counts."""
    _DummyGuardedPPOAdapter.instances = []
    monkeypatch.setattr(_DummyGuardedPPOAdapter, "residual_clipped", True)
    monkeypatch.setattr(map_runner, "PPOPlanner", _DummyPPOPlanner)
    monkeypatch.setattr(map_runner, "GuardedPPOAdapter", _DummyGuardedPPOAdapter)
    monkeypatch.setattr(map_runner, "build_guarded_ppo_fallback", lambda cfg: object())
    policy, meta = map_runner._build_policy(
        "guarded_ppo",
        {
            "obs_mode": "dict",
            "test_action": {"v": 0.7, "omega": 0.3},
        },
    )

    action_v, action_w = policy(_sample_obs())
    assert action_v == pytest.approx(0.1)
    assert action_w == pytest.approx(-0.2)
    guard_stats = meta.get("guard_stats")
    assert isinstance(guard_stats, dict)
    assert guard_stats["fallback_safe"] == 1
    residual_stats = meta.get("residual_clipping_stats")
    assert isinstance(residual_stats, dict)
    assert residual_stats["decision_count"] == 1
    assert residual_stats["clipped_count"] == 1

    guard = _DummyGuardedPPOAdapter.instances[-1]
    env = object()
    policy._planner_bind_env(env)
    policy._planner_reset(seed=5)
    policy._planner_close()
    assert guard.bound_envs == [env]
    assert guard.reset_seeds == [5]
    assert guard.closed


def test_build_policy_guarded_ppo_binds_ppo_and_guard_env(monkeypatch):
    """Guarded PPO should bind both PPO obs adapter and guard map context."""
    _DummyGuardedPPOAdapter.instances = []
    ppo_instances: list[_DummyPPOPlanner] = []

    def _make_ppo(*args, **kwargs):
        planner = _DummyPPOPlanner(*args, **kwargs)
        ppo_instances.append(planner)
        return planner

    monkeypatch.setattr(map_runner, "PPOPlanner", _make_ppo)
    monkeypatch.setattr(map_runner, "GuardedPPOAdapter", _DummyGuardedPPOAdapter)
    monkeypatch.setattr(map_runner, "build_guarded_ppo_fallback", lambda cfg: object())

    policy, _ = map_runner._build_policy(
        "guarded_ppo",
        {"obs_mode": "dict", "test_action": {"v": 0.7, "omega": 0.3}},
    )

    env = object()
    policy._planner_bind_env(env)

    assert ppo_instances[-1].bound_envs == [env]
    assert _DummyGuardedPPOAdapter.instances[-1].bound_envs == [env]


def test_policy_env_observation_overrides_apply_predictive_contract() -> None:
    """Policy-search candidate env_overrides should restore BC observation features."""
    config = RobotSimulationConfig()

    map_runner._apply_policy_env_observation_overrides(
        config,
        {
            "env_overrides": {
                "predictive_foresight_enabled": True,
                "predictive_foresight_device": "cpu",
                "predictive_foresight_max_agents": 16,
                "predictive_foresight_horizon_steps": 8,
            }
        },
    )

    assert config.predictive_foresight_enabled is True
    assert config.predictive_foresight_device == "cpu"
    assert config.predictive_foresight_max_agents == 16
    assert config.predictive_foresight_horizon_steps == 8


def test_obs_to_ppo_format_uses_ped_count_and_sim_timestep():
    """Ensure padded pedestrian channels are sliced by count and dt comes from sim metadata."""
    obs = _sample_obs()
    obs["sim"] = {"timestep": np.array([0.25], dtype=float)}
    obs["pedestrians"] = {
        "positions": np.array([[2.0, 2.0], [5.0, 5.0], [0.0, 0.0]], dtype=float),
        "velocities": np.array([[0.1, 0.2]], dtype=float),
        "count": np.array([2], dtype=float),
        "radius": np.array([0.35], dtype=float),
    }

    formatted = map_runner._obs_to_ppo_format(obs)
    assert formatted["dt"] == pytest.approx(0.25)
    assert len(formatted["agents"]) == 2
    assert formatted["agents"][0]["position"] == [2.0, 2.0]
    assert formatted["agents"][0]["velocity"] == [0.1, 0.2]
    assert formatted["agents"][1]["position"] == [5.0, 5.0]
    assert formatted["agents"][1]["velocity"] == [0.0, 0.0]


def test_obs_to_ppo_format_handles_malformed_flat_ped_arrays():
    """Malformed odd-length flat pedestrian arrays should not produce ghost agents."""
    obs = _sample_obs()
    obs["pedestrians"] = {
        "positions": np.array([1.0, 2.0, 3.0], dtype=float),
        "velocities": np.array([0.1], dtype=float),
        "count": np.array([3], dtype=float),
        "radius": np.array([0.35], dtype=float),
    }

    formatted = map_runner._obs_to_ppo_format(obs)
    assert formatted["agents"] == []
