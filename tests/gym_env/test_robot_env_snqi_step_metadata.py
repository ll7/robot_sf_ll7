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
    near_distance = 1.0 + 0.4 + (d_near / 2.0)
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
    assert first["min_clearance"] == pytest.approx(d_near / 2.0)
    assert first["center_distance_near_miss_diagnostic"] == 0.0
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


def test_robot_env_passes_occupancy_grid_ped_positions_to_telemetry() -> None:
    """RobotEnv.step should pass its occupancy-grid pedestrian snapshot to telemetry."""
    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        grid_config=GridConfig(width=4.0, height=4.0, resolution=0.5),
        include_grid_in_observation=False,
    )
    env = make_robot_env(config=config, reward_name="snqi_step", telemetry_record=True)
    captured: dict[str, object] = {}
    original_emit_telemetry = env._emit_telemetry

    def capture_emit_telemetry(
        reward: float,
        terminated: bool,
        truncated: bool,
        action,
        meta=None,
        ped_positions_override=None,
    ) -> None:
        captured["ped_positions_override"] = ped_positions_override
        original_emit_telemetry(
            reward,
            terminated,
            truncated,
            action,
            meta,
            ped_positions_override=ped_positions_override,
        )

    try:
        env.reset(seed=13)
        env._emit_telemetry = capture_emit_telemetry  # type: ignore[method-assign]

        env.step(np.array([0.2, 0.0], dtype=np.float32))
    finally:
        env.close()

    assert captured["ped_positions_override"] is not None
    assert np.asarray(captured["ped_positions_override"]).shape[-1] >= 2


def test_robot_env_passes_ped_positions_to_telemetry_without_occupancy_grid() -> None:
    """RobotEnv.step should still share one pedestrian snapshot without an occupancy grid."""
    env = make_robot_env(reward_name="snqi_step", telemetry_record=True)
    captured: dict[str, object] = {}
    original_emit_telemetry = env._emit_telemetry

    def capture_emit_telemetry(
        reward: float,
        terminated: bool,
        truncated: bool,
        action,
        meta=None,
        ped_positions_override=None,
    ) -> None:
        captured["ped_positions_override"] = ped_positions_override
        original_emit_telemetry(
            reward,
            terminated,
            truncated,
            action,
            meta,
            ped_positions_override=ped_positions_override,
        )

    try:
        env.reset(seed=15)
        env._emit_telemetry = capture_emit_telemetry  # type: ignore[method-assign]

        env.step(np.array([0.2, 0.0], dtype=np.float32))
    finally:
        env.close()

    assert captured["ped_positions_override"] is not None
    assert np.asarray(captured["ped_positions_override"]).shape[-1] >= 2


def test_emit_telemetry_uses_ped_position_override_without_reading_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry min-pedestrian distance should use the step-local snapshot when provided."""

    class CapturingTelemetrySession:
        """Capture appended telemetry payloads."""

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def append(self, payload: dict) -> None:
            self.payloads.append(payload)

    def guarded_ped_pos(_self):
        raise AssertionError("telemetry should not read simulator.ped_pos with an override")

    env = make_robot_env(reward_name="snqi_step", telemetry_record=True)
    telemetry_session = CapturingTelemetrySession()
    try:
        env.reset(seed=17)
        env._telemetry_session = telemetry_session
        robot_pos = np.asarray(env.simulator.robot_poses[0][0], dtype=float)
        monkeypatch.setattr(type(env.simulator), "ped_pos", property(guarded_ped_pos))

        env._emit_telemetry(
            reward=1.0,
            terminated=False,
            truncated=False,
            action=np.array([0.0, 0.0], dtype=np.float32),
            ped_positions_override=robot_pos + np.array([[0.3, 0.4], [3.0, 4.0]], dtype=float),
        )
    finally:
        env.close()

    payload = telemetry_session.payloads[-1]
    metrics = payload["metrics"]
    assert metrics["min_ped_distance"] == pytest.approx(0.5)


def test_emit_telemetry_falls_back_to_simulator_ped_positions() -> None:
    """Telemetry should preserve the simulator-read fallback when no snapshot is provided."""

    class CapturingTelemetrySession:
        """Capture appended telemetry payloads."""

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def append(self, payload: dict) -> None:
            self.payloads.append(payload)

    env = make_robot_env(reward_name="snqi_step", telemetry_record=True)
    telemetry_session = CapturingTelemetrySession()
    try:
        env.reset(seed=19)
        env._telemetry_session = telemetry_session
        ped_positions = np.asarray(env.simulator.ped_pos)
        robot_pos = np.asarray(env.simulator.robot_poses[0][0], dtype=float)
        expected_min_distance = None
        if ped_positions.size > 0:
            expected_min_distance = float(np.min(np.linalg.norm(ped_positions - robot_pos, axis=1)))

        env._emit_telemetry(
            reward=1.0,
            terminated=False,
            truncated=False,
            action=np.array([0.0, 0.0], dtype=np.float32),
        )
    finally:
        env.close()

    payload = telemetry_session.payloads[-1]
    metrics = payload["metrics"]
    if expected_min_distance is None:
        assert metrics["min_ped_distance"] is None
    else:
        assert metrics["min_ped_distance"] == pytest.approx(expected_min_distance)
