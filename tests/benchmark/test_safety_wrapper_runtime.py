"""Tests for opt-in benchmark runtime safety-wrapper binding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import map_runner_episode
from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.safety_wrapper_runtime import (
    SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA,
    SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA,
    apply_runtime_safety_wrapper,
    runtime_config_from_mapping,
    summarize_safety_wrapper_trace,
)
from robot_sf.robot.safety_wrapper import INTERVENTION_HARD_STOP


class _Robot:
    pose = np.array([0.0, 0.0, 0.0], dtype=float)


class _Simulator:
    def __init__(self) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.ped_pos = np.array([[0.2, 0.0]], dtype=float)
        self.robots = [_Robot()]


class _Env:
    def __init__(self) -> None:
        self.simulator = _Simulator()


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            robot_radius=0.1,
            ped_radius=0.1,
        )
    )


def _policy_builder(*_args, **_kwargs):
    def policy(_obs):
        return (1.0, 0.25)

    return policy, {"algorithm": "unit"}


class _EpisodeSim:
    def __init__(self) -> None:
        self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
        self.goal_pos = [np.array([1.0, 0.0], dtype=float)]
        self.ped_pos = np.array([[0.2, 0.0]], dtype=float)
        self.last_ped_forces = np.zeros((1, 2), dtype=float)
        self.map_def = SimpleNamespace(obstacles=[], bounds=(0.0, 0.0, 2.0, 2.0))


class _EpisodeEnv:
    def __init__(self) -> None:
        self.simulator = _EpisodeSim()
        self.action_space = None

    def reset(self, seed=None):
        return {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {"positions": self.simulator.ped_pos},
        }, {}

    def step(self, _action):
        return self.reset()[0], 0.0, True, False, {"meta": {"is_route_complete": True}}

    def close(self) -> None:
        return None


def _patch_episode_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        map_runner_episode,
        "_build_env_config",
        lambda _scenario, scenario_path: _config(),
    )
    monkeypatch.setattr(
        map_runner_episode,
        "make_robot_env",
        lambda config, seed, debug: _EpisodeEnv(),
    )
    monkeypatch.setattr(map_runner_episode, "sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(map_runner_episode, "compute_shortest_path_length", lambda *args: 1.0)
    monkeypatch.setattr(
        map_runner_episode,
        "compute_all_metrics",
        lambda *args, **kwargs: {"success": 1.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        map_runner_episode,
        "post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )


def _run_episode_with_policy(monkeypatch, policy_builder, *, safety_wrapper):
    _patch_episode_runtime(monkeypatch)
    return map_runner_episode.run_map_episode(
        {"name": "wrapper-runtime", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=Path(__file__),
        safety_wrapper=safety_wrapper,
        policy_builder=policy_builder,
    )


def test_runtime_config_disabled_by_default_preserves_off_state() -> None:
    """Missing runtime config keeps wrapper disabled and therefore opt-in only."""

    runtime = runtime_config_from_mapping(None)

    assert runtime.enabled is False
    assert runtime.arm_key == "wrapper_off"


def test_wrapper_on_emits_schema_tagged_intervention_record_and_summary() -> None:
    """A close pre-step pedestrian produces well-formed wrapper evidence."""

    runtime = runtime_config_from_mapping({"enabled": True, "arm_key": "wrapper_on"})
    corrected, record = apply_runtime_safety_wrapper(
        command=(1.0, 0.25),
        env=_Env(),
        config=_config(),
        runtime=runtime,
        previous_ped_positions=None,
        step_idx=3,
    )

    assert corrected == (0.0, 0.25)
    assert record["schema_version"] == SAFETY_WRAPPER_RUNTIME_STEP_SCHEMA
    assert record["arm_key"] == "wrapper_on"
    assert record["enabled"] is True
    assert record["eligible_for_wrapper"] is True
    assert record["context_source"] == "simulator_state_pre_step"
    assert record["clearance_sources"] == ["pedestrians"]
    assert record["intervention"] == INTERVENTION_HARD_STOP
    assert record["intervened"] is True

    summary = summarize_safety_wrapper_trace([record], runtime=runtime)
    assert summary["schema_version"] == SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA
    assert summary["intervened_step_count"] == 1
    assert summary["intervention_rate"] == 1.0
    assert summary["first_hard_stop_step"] == 3
    assert summary["min_context_clearance_m"] <= 0.0


def test_run_map_episode_records_wrapper_metadata_when_enabled(monkeypatch) -> None:
    """Runtime binding emits episode and ledger-ready evidence only when enabled."""

    record = _run_episode_with_policy(
        monkeypatch,
        _policy_builder,
        safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
    )

    summary = record["algorithm_metadata"]["safety_wrapper"]
    assert summary["schema_version"] == SAFETY_WRAPPER_EPISODE_SUMMARY_SCHEMA
    assert summary["intervened_step_count"] == 1
    assert record["metrics"]["wrapper_intervention_rate"] == 1.0

    ledger = build_event_ledger(record)
    assert ledger["provenance"]["safety_wrapper"] == summary


def test_run_map_episode_fails_closed_for_native_action_when_wrapper_enabled(
    monkeypatch,
) -> None:
    """Native env actions must not be silently transformed by wrapper-on runs."""

    def native_policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return np.array([0.1, 0.0], dtype=float)

        policy._last_step_native = True
        return policy, {"algorithm": "unit"}

    with pytest.raises(ValueError, match="native environment actions"):
        _run_episode_with_policy(
            monkeypatch,
            native_policy_builder,
            safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
        )


def test_run_map_episode_fails_closed_for_unsupported_command_when_wrapper_enabled(
    monkeypatch,
) -> None:
    """Unsupported structured commands fail closed under wrapper-on by default."""

    def unsupported_policy_builder(*_args, **_kwargs):
        def policy(_obs):
            return {"command_kind": "unsupported"}

        return policy, {"algorithm": "unit"}

    with pytest.raises(TypeError, match="expects commands shaped"):
        _run_episode_with_policy(
            monkeypatch,
            unsupported_policy_builder,
            safety_wrapper={"enabled": True, "arm_key": "wrapper_on"},
        )
