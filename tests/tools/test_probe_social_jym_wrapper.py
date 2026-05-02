"""Tests for the social-jym wrapper feasibility probe."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.tools import probe_social_jym_wrapper as probe


def test_build_social_jym_policy_inputs_from_nested_socnav_observation() -> None:
    """Nested Robot SF SocNav observations should map to the upstream SARL observation shape."""
    social_obs, social_info, keys = probe.build_social_jym_policy_inputs(
        {
            "robot": {
                "position": [1.0, 2.0],
                "heading": [0.25],
                "speed": [0.2, 0.1],
                "radius": [0.4],
            },
            "goal": {"current": [3.0, 4.0]},
            "pedestrians": {
                "positions": [[5.0, 6.0]],
                "velocities": [[0.7, 0.8]],
                "count": [1],
                "radius": [0.3],
            },
        },
        max_humans=1,
    )

    assert social_obs.shape == (2, 6)
    assert social_obs[0].tolist() == pytest.approx([5.0, 6.0, 0.7, 0.8, 0.3, 0.0])
    assert social_obs[1].tolist() == pytest.approx([1.0, 2.0, 0.2, 0.1, 0.4, 0.25])
    assert social_info["robot_goal"].tolist() == pytest.approx([3.0, 4.0])
    assert keys == ["goal", "pedestrians", "robot"]


def test_build_social_jym_policy_inputs_fails_fast_on_missing_goal() -> None:
    """Missing required fields should fail closed instead of silently zero-filling."""
    with pytest.raises(ValueError, match="goal.current"):
        probe.build_social_jym_policy_inputs(
            {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0, 0.0]},
                "goal": {},
                "pedestrians": {},
            }
        )


def test_project_holonomic_action_to_unicycle_heading_safe() -> None:
    """Projection should preserve forward actions and turn toward lateral actions."""
    linear, angular, heading_error = probe.project_holonomic_action_to_unicycle(
        np.array([1.0, 0.0]),
        robot_heading=0.0,
        dt=0.25,
        max_linear_speed=1.0,
        max_angular_speed=2.0,
    )
    assert linear == pytest.approx(1.0)
    assert angular == pytest.approx(0.0)
    assert heading_error == pytest.approx(0.0)

    linear, angular, heading_error = probe.project_holonomic_action_to_unicycle(
        np.array([0.0, 1.0]),
        robot_heading=0.0,
        dt=0.25,
        max_linear_speed=1.0,
        max_angular_speed=2.0,
    )
    assert linear == pytest.approx(0.0, abs=1e-6)
    assert angular == pytest.approx(2.0)
    assert heading_error == pytest.approx(np.pi / 2)


class _FakeRobot:
    pass


class _FakeSimulator:
    def __init__(self) -> None:
        self.robots = [_FakeRobot()]


class _FakeEnv:
    def __init__(self) -> None:
        self.simulator = _FakeSimulator()
        self.action_space = object()
        self.env_config = type(
            "Cfg", (), {"sim_config": type("Sim", (), {"time_per_step_in_secs": 0.25})()}
        )()
        self.latest_action: np.ndarray | None = None
        self.obs = {
            "robot": {
                "position": np.array([0.0, 0.0]),
                "heading": np.array([0.0]),
                "speed": np.array([0.0, 0.0]),
                "radius": np.array([0.3]),
            },
            "goal": {"current": np.array([1.0, 0.0])},
            "pedestrians": {
                "positions": np.array([[1.0, 1.0]]),
                "velocities": np.array([[0.0, 0.0]]),
                "count": np.array([1]),
                "radius": np.array([0.3]),
            },
        }

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, int | None]]:
        return self.obs, {"seed": seed}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.latest_action = np.asarray(action, dtype=float)
        return self.obs, 0.0, False, False, {"action": self.latest_action}

    def close(self) -> None:
        return None


class _FakePlannerActionAdapter:
    def __init__(self, robot: object, action_space: object, time_step: float) -> None:
        self.robot = robot
        self.action_space = action_space
        self.time_step = time_step

    def from_velocity_command(self, command: tuple[float, float]) -> np.ndarray:
        return np.asarray(command, dtype=float)


class _FakeWrapper:
    def __init__(
        self, *, repo_root: object | None = None, max_humans: int = 1, seed: int = 0
    ) -> None:
        self.repo_root = repo_root
        self.max_humans = max_humans
        self.seed = seed

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        return np.asarray([1.0, 0.0], dtype=float), {
            "observation_keys": sorted(obs.keys()),
            "source_action_xy": [1.0, 0.0],
        }


def test_run_probe_executes_one_robot_sf_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe should execute a real Robot SF loop boundary when dependencies are available."""
    captured: dict[str, Any] = {}

    def fake_make_robot_env(config: object, debug: bool = False) -> _FakeEnv:
        captured["config"] = config
        captured["debug"] = debug
        return _FakeEnv()

    monkeypatch.setattr(probe, "make_robot_env", fake_make_robot_env)
    monkeypatch.setattr(probe, "PlannerActionAdapter", _FakePlannerActionAdapter)
    monkeypatch.setattr(probe, "SocialJymSARLWrapper", _FakeWrapper)

    report = probe.run_probe(repo_root=None, seed=5, max_steps=1)

    assert captured["config"].observation_mode == probe.ObservationMode.SOCNAV_STRUCT
    assert report.verdict == "wrapper prototype viable"
    assert report.steps_executed == 1
    assert report.latest_source_action_xy == [1.0, 0.0]
    assert report.latest_robot_command_vw == [1.0, 0.0]
    assert report.projection_policy == "heading_safe_holonomic_xy_to_unicycle_vw"


def test_render_markdown_preserves_benchmark_boundary() -> None:
    """Markdown output should keep the benchmark-support caveat visible."""
    report = probe.SocialJymWrapperReport(
        issue=905,
        verdict="wrapper prototype viable",
        wrapper_boundary="test boundary",
        source_environment="env",
        source_policy="policy",
        source_scenario="scenario",
        source_humans_policy="hsfm",
        projection_policy="projection",
        steps_executed=1,
        latest_source_action_xy=[1.0, 0.0],
        latest_robot_command_vw=[1.0, 0.0],
        latest_robot_action=[1.0, 0.0],
        latest_heading_error_rad=0.0,
        observation_keys=["robot", "goal", "pedestrians"],
    )

    markdown = probe._render_markdown(report)

    assert "wrapper prototype viable" in markdown
    assert "does not add JAX or social-jym" in markdown
    assert "must not be reported as benchmark success" in markdown
