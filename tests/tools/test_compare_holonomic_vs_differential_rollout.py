"""Tests for paired holonomic vs differential rollout comparison."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from scripts.tools.compare_holonomic_vs_differential_rollout import (
    _compare_policy_inputs,
    _compare_ppo_inputs,
    _flatten_orca_input,
    _flatten_ppo_input,
    run_pairwise_rollout,
)

if TYPE_CHECKING:
    from pathlib import Path


def _obs(
    *,
    position: tuple[float, float],
    velocity: tuple[float, float],
    heading: float,
    goal: tuple[float, float] = (5.0, 0.0),
    agents: list[dict] | None = None,
) -> dict:
    return {
        "dt": 0.1,
        "robot": {
            "position": list(position),
            "velocity": list(velocity),
            "heading": [heading],
            "goal": list(goal),
            "radius": [0.3],
        },
        "goal": {"current": list(goal)},
        "agents": agents or [],
    }


def test_flatten_ppo_input_preserves_expected_contract() -> None:
    """Flattening should keep the PPO input contract stable and numeric."""
    vec, stats = _flatten_ppo_input(
        _obs(
            position=(1.0, 2.0),
            velocity=(0.5, -0.25),
            heading=0.1,
            agents=[{"position": [2.0, 3.0], "velocity": [0.1, 0.2], "radius": 0.3}],
        ),
    )
    assert vec.shape[0] == 1 + 2 + 2 + 2 + 1 + 1 + 5
    assert tuple(stats["robot_position"]) == pytest.approx((1.0, 2.0))
    assert tuple(stats["robot_velocity"]) == pytest.approx((0.5, -0.25))
    assert stats["agent_count"] == pytest.approx(1.0)


def test_compare_ppo_inputs_detects_divergence() -> None:
    """Direct comparison should surface a nonzero input delta when observations drift."""
    l2, summary = _compare_ppo_inputs(
        _obs(position=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.0),
        _obs(position=(0.1, 0.0), velocity=(0.5, 0.0), heading=0.2),
    )
    assert l2 > 0.0
    assert summary["robot_position_l2"] > 0.0
    assert summary["robot_velocity_l2"] > 0.0


def test_flatten_orca_input_preserves_expected_contract() -> None:
    """Flattening should keep the ORCA input contract stable and numeric."""
    vec, stats = _flatten_orca_input(
        _obs(
            position=(1.0, 2.0),
            velocity=(0.5, -0.25),
            heading=0.1,
            goal=(4.0, 3.0),
            agents=[{"position": [2.0, 3.0], "velocity": [0.1, 0.2], "radius": 0.3}],
        ),
    )
    assert vec.shape[0] == 15
    assert tuple(stats["robot_position"]) == pytest.approx((1.0, 2.0))
    assert tuple(stats["robot_velocity"]) == pytest.approx((0.5, -0.25))
    assert stats["agent_count"] == pytest.approx(1.0)


def test_compare_policy_inputs_detects_divergence_for_orca() -> None:
    """ORCA-specific comparison should surface drift in the flattened policy inputs."""
    l2, summary = _compare_policy_inputs(
        _obs(position=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.0),
        _obs(position=(0.1, 0.0), velocity=(0.5, 0.0), heading=0.2),
        algo="orca",
    )
    assert l2 > 0.0
    assert summary["robot_position_l2"] > 0.0
    assert summary["robot_velocity_l2"] > 0.0


@dataclass
class _StubRobotCfg:
    radius: float = 0.3
    max_linear_speed: float = 2.0
    max_angular_speed: float = 1.0
    allow_backwards: bool = False


class _StubEnv:
    def __init__(self, observations: list[dict[str, object]]) -> None:
        self._observations = observations
        self._index = 0

    def reset(self, seed: int | None = None):
        self._index = 0
        return self._observations[0], {}

    def step(self, action):
        self._index += 1
        obs = self._observations[min(self._index, len(self._observations) - 1)]
        done = self._index >= len(self._observations) - 1
        return obs, 0.0, done, False, {}

    def close(self) -> None:
        return None


def test_run_pairwise_rollout_writes_artifacts_and_flags_first_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired rollout should write reports and flag the first step where inputs diverge."""

    scenario = {
        "name": "stub_scenario",
        "robot_config": {"type": "differential_drive"},
        "simulation_config": {"max_episode_steps": 5},
    }

    diff_obs = [
        _obs(position=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.0),
        _obs(position=(0.1, 0.0), velocity=(1.0, 0.0), heading=0.0),
        _obs(position=(0.2, 0.0), velocity=(1.0, 0.0), heading=0.0),
    ]
    holo_obs = [
        _obs(position=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.0),
        _obs(position=(0.05, 0.0), velocity=(0.9, 0.0), heading=0.1),
        _obs(position=(0.15, 0.0), velocity=(0.8, 0.0), heading=0.2),
    ]

    stub_envs = {
        "differential_drive": _StubEnv(diff_obs),
        "holonomic": _StubEnv(holo_obs),
    }

    def _fake_build_env_and_policy(*, scenario, scenario_path, kinematics, algo, config_path):
        del scenario, scenario_path, algo, config_path
        return (
            stub_envs[kinematics],
            (lambda _obs: (0.0, 0.0)),
            {"status": "ok"},
            SimpleNamespace(robot_config=_StubRobotCfg()),
        )

    monkeypatch.setattr(
        "scripts.tools.compare_holonomic_vs_differential_rollout._build_env_and_policy",
        _fake_build_env_and_policy,
    )
    monkeypatch.setattr(
        "scripts.tools.compare_holonomic_vs_differential_rollout.load_scenarios",
        lambda *_args, **_kwargs: [scenario],
    )
    monkeypatch.setattr(
        "scripts.tools.compare_holonomic_vs_differential_rollout.select_scenario",
        lambda scenarios, scenario_id: scenarios[0],
    )
    monkeypatch.setattr(
        "scripts.tools.compare_holonomic_vs_differential_rollout._build_env_config",
        lambda *_args, **_kwargs: SimpleNamespace(robot_config=_StubRobotCfg()),
    )

    output_dir = tmp_path / "report"
    payload = run_pairwise_rollout(
        algo="orca",
        scenario_file=tmp_path / "scenario.yaml",
        scenario_id=None,
        seed=111,
        max_steps=4,
        tolerance=1e-12,
        output_dir=output_dir,
    )

    assert payload["scenario_id"] == "stub_scenario"
    assert payload["first_divergence_step"] == 1
    assert payload["rows"][0]["policy_input_l2"] == pytest.approx(0.0)
    assert payload["rows"][1]["policy_input_l2"] > 0.0
    assert (output_dir / "rollout_diff.json").exists()
    assert (output_dir / "rollout_diff.md").exists()
    assert (output_dir / "rollout_diff_series.csv").exists()
    if (output_dir / "rollout_diff_plot.png").exists():
        assert (output_dir / "rollout_diff_plot.png").stat().st_size > 0
