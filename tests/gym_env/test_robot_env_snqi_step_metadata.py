"""Tests for step-level SNQI proxy metadata plumbing in RobotEnv."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.snqi_proxy import StepSNQIProxy, compute_snqi_step_proxies
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridConfig


class _FakeSimulator:
    """Minimal simulator stub for SNQI step proxy tests."""

    def __init__(self) -> None:
        self.robot_poses = [((0.0, 0.0), 0.0)]
        self.ped_pos = np.array([[0.30, 0.0], [4.0, 4.0]], dtype=float)
        self.last_ped_forces = np.array([[3.0, 0.0], [0.5, 0.0]], dtype=float)


def test_compute_snqi_step_proxies_emits_non_default_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy computation should emit near-miss, force, and running jerk signals."""
    d_coll = 0.2
    d_near = 0.6
    comfort_threshold = 1.4
    near_distance = (d_coll + d_near) / 2.0
    sim = _FakeSimulator()
    sim.ped_pos = np.array([[near_distance, 0.0], [4.0, 4.0]], dtype=float)
    sim.last_ped_forces = np.array(
        [[comfort_threshold + 0.8, 0.0], [comfort_threshold - 0.8, 0.0]],
        dtype=float,
    )
    monkeypatch.setattr(
        "robot_sf.gym_env.snqi_proxy.resolve_snqi_thresholds",
        lambda: (d_coll, d_near, comfort_threshold),
    )
    proxy = StepSNQIProxy()

    first = compute_snqi_step_proxies(simulator=sim, dt=0.1, proxy_state=proxy.state)
    assert first["near_misses"] == 1.0
    assert first["force_exceed_events"] == 1.0
    assert first["comfort_exposure"] == 0.5
    assert first["jerk_mean"] == 0.0

    # Drive a non-linear position profile so finite-difference jerk becomes non-zero.
    sim.robot_poses = [((0.1, 0.0), 0.0)]
    compute_snqi_step_proxies(simulator=sim, dt=0.1, proxy_state=proxy.state)
    sim.robot_poses = [((0.3, 0.0), 0.0)]
    compute_snqi_step_proxies(simulator=sim, dt=0.1, proxy_state=proxy.state)
    sim.robot_poses = [((0.5, 0.0), 0.0)]
    fourth = compute_snqi_step_proxies(simulator=sim, dt=0.1, proxy_state=proxy.state)
    assert fourth["jerk_mean"] > 0.0


def test_robot_env_step_info_includes_snqi_proxy_fields() -> None:
    """Step info metadata should include SNQI proxy fields in standard env rollouts."""
    env = make_robot_env(reward_name="snqi_step")
    try:
        env.reset(seed=7)
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([0.5, 0.1], dtype=np.float32)
        )
        if terminated or truncated:
            env.reset(seed=8)
            _obs, _reward, _terminated, _truncated, info = env.step(
                np.array([0.4, -0.1], dtype=np.float32)
            )
    finally:
        env.close()

    meta = info["meta"]
    for key in ("near_misses", "force_exceed_events", "comfort_exposure", "jerk_mean"):
        assert key in meta
        assert np.isfinite(float(meta[key]))


def test_robot_env_reuses_occupancy_grid_ped_positions_for_snqi_proxy() -> None:
    """RobotEnv.step should pass its occupancy-grid pedestrian snapshot to SNQI metrics."""
    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        grid_config=GridConfig(width=4.0, height=4.0, resolution=0.5),
        include_grid_in_observation=False,
    )
    env = make_robot_env(config=config, reward_name="snqi_step")
    captured: dict[str, object] = {}

    def capture_compute_step_metrics(
        simulator,
        *,
        dt: float,
        ped_positions_override=None,
    ) -> dict[str, float]:
        captured["simulator"] = simulator
        captured["dt"] = dt
        captured["ped_positions_override"] = ped_positions_override
        return {
            "near_misses": 0.0,
            "force_exceed_events": 0.0,
            "comfort_exposure": 0.0,
            "jerk_mean": 0.0,
        }

    try:
        env.reset(seed=11)
        env._snqi_proxy.compute_step_metrics = capture_compute_step_metrics  # type: ignore[method-assign]

        env.step(np.array([0.2, 0.0], dtype=np.float32))
    finally:
        env.close()

    assert captured["simulator"] is env.simulator
    assert captured["ped_positions_override"] is not None
    assert np.asarray(captured["ped_positions_override"]).shape[-1] >= 2
