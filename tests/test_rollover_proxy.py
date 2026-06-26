"""Tests for the internal-proxy rollover stability margin (issue #3479)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.metrics import evaluate_stability_margin
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.robot.rollover_proxy import (
    PROXY_SCHEMA_VERSION,
    RolloverProxyParams,
    critical_lateral_acceleration,
    is_rollover_critical,
    lateral_acceleration,
    rollover_proxy_telemetry,
    stability_margin,
)

# a_y,crit = g * (t_w / (2 h_c)) * (a / L) for the default proxy geometry, aligned with the
# benchmark-surface source of truth metrics.evaluate_stability_margin (t_w=0.8, L=1.2, h_c=0.6, a=0.5).
_DEFAULT_CRIT = 9.81 * (0.80 / (2 * 0.60)) * (0.50 / 1.20)


def test_critical_lateral_acceleration_matches_closed_form() -> None:
    """The critical lateral acceleration must match the issue's closed form."""
    assert critical_lateral_acceleration(RolloverProxyParams()) == pytest.approx(_DEFAULT_CRIT)


def test_lateral_acceleration_is_v_times_omega() -> None:
    """Proxy lateral acceleration must be ``v · ω``."""
    assert lateral_acceleration(1.5, 2.0) == pytest.approx(3.0)


def test_feasible_command_stays_stable() -> None:
    """A low-speed, low-yaw command must stay well within the proxy threshold."""
    margin = stability_margin(0.5, 0.5)  # a_y = 0.25 << a_y,crit ~ 2.73

    assert margin == pytest.approx(1.0 - 0.25 / _DEFAULT_CRIT)
    assert not is_rollover_critical(margin)


def test_over_yaw_command_trips_rollover_critical() -> None:
    """An over-yaw command exceeding the critical accel must trip ROLLOVER_CRITICAL."""
    margin = stability_margin(2.0, 2.0)  # a_y = 4.0 > a_y,crit ~ 2.73

    assert margin == 0.0
    assert is_rollover_critical(margin)


def test_margin_is_clamped_to_unit_interval() -> None:
    """Zero motion yields full margin; gross over-demand clamps to zero."""
    assert stability_margin(0.0, 0.0) == 1.0
    assert stability_margin(100.0, 100.0) == 0.0


def test_margin_at_exact_threshold_is_critical() -> None:
    """Demanding exactly the critical lateral acceleration yields a zero margin."""
    crit = critical_lateral_acceleration(RolloverProxyParams())
    # Pick v, omega whose product equals the critical lateral acceleration.
    margin = stability_margin(crit, 1.0)

    assert margin == pytest.approx(0.0, abs=1e-12)
    assert is_rollover_critical(margin)


def test_margin_decreases_monotonically_with_demand() -> None:
    """Higher |v · ω| must never increase the stability margin."""
    margins = [stability_margin(v, 1.0) for v in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert margins == sorted(margins, reverse=True)


def test_margin_depends_on_magnitude_not_sign() -> None:
    """Opposite yaw signs of equal magnitude must give the same margin."""
    assert stability_margin(1.0, 1.5) == pytest.approx(stability_margin(1.0, -1.5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_width_m": 0.0},
        {"cog_height_m": -0.1},
        {"front_axle_to_cog_m": 0.0},
        {"wheelbase_m": -1.0},
        {"gravity_m_s2": 0.0},
        {"front_axle_to_cog_m": 0.7, "wheelbase_m": 0.6},  # a > L
    ],
)
def test_invalid_params_are_rejected(kwargs: dict[str, float]) -> None:
    """Non-physical proxy geometry must fail closed at construction."""
    with pytest.raises(ValueError):
        RolloverProxyParams(**kwargs)


def test_telemetry_is_schema_tagged_and_labels_non_hardware() -> None:
    """Telemetry must carry the schema version and the internal-proxy provenance label."""
    record = rollover_proxy_telemetry(2.0, 2.0)

    assert record["schema_version"] == PROXY_SCHEMA_VERSION
    assert record["proxy_kind"] == "internal_non_hardware"
    assert record["rollover_critical"] is True
    assert record["stability_margin"] == 0.0
    assert record["lateral_acceleration"] == pytest.approx(4.0)
    assert record["critical_lateral_acceleration"] == pytest.approx(_DEFAULT_CRIT)


@pytest.mark.parametrize(
    ("v", "omega"),
    [(0.0, 0.0), (0.5, 0.5), (1.0, 1.5), (2.0, 2.0), (3.0, 1.0), (1.0, -2.0)],
)
def test_proxy_agrees_with_benchmark_surface_source_of_truth(v: float, omega: float) -> None:
    """The runtime proxy must match metrics.evaluate_stability_margin (issue #3587).

    Both implement the same closed form; with the default geometry now aligned to the
    benchmark-surface values, the two must produce identical stability margins so the runtime
    diagnostic and the benchmark column cannot diverge.
    """
    params = RolloverProxyParams()
    proxy_margin = stability_margin(v, omega, params)
    benchmark_margin = evaluate_stability_margin(
        v,
        omega,
        t_w=params.track_width_m,
        L=params.wheelbase_m,
        h_c=params.cog_height_m,
        a=params.front_axle_to_cog_m,
    )

    assert proxy_margin == pytest.approx(benchmark_margin)


class _StepRobot:
    """Minimal robot double exposing executed ``current_speed``."""

    def __init__(self) -> None:
        self.current_speed = (0.0, 0.0)

    def parse_action(self, action: np.ndarray) -> tuple[float, float]:
        """Return action as differential-drive ``(v, omega)`` command."""
        return (float(action[0]), float(action[1]))


class _StepSimulator:
    """Minimal simulator double for ``RobotEnv.step`` rollover fixtures."""

    def __init__(self) -> None:
        self.robots = [_StepRobot()]
        self.robot_poses = [((0.0, 0.0), 0.0)]
        self.ped_pos = np.zeros((0, 2), dtype=float)

    def step_once(self, actions: list[tuple[float, float]]) -> None:
        """Record the executed command as the robot's current speed."""
        self.robots[0].current_speed = actions[0]


class _StepState:
    """Minimal state double for ``RobotEnv.step`` rollover fixtures."""

    d_t = 0.25
    is_terminal = False

    def step(self) -> np.ndarray:
        """Return a deterministic observation."""
        return np.zeros(1, dtype=np.float32)

    def meta_dict(self) -> dict:
        """Return nonterminal state metadata."""
        return {
            "step": 1,
            "episode": 1,
            "step_of_episode": 1,
            "is_pedestrian_collision": False,
            "is_robot_collision": False,
            "is_obstacle_collision": False,
            "is_waypoint_complete": False,
            "is_route_complete": False,
            "distance_to_goal": 1.0,
            "prev_distance_to_goal": 1.2,
            "is_timesteps_exceeded": False,
            "max_sim_steps": 100,
        }


def _make_step_env(
    *,
    rollover_enabled: bool = True,
    penalty: float = -4.0,
    env_config: object | None = None,
) -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.env_config = env_config or EnvSettings(
        rollover_proxy_enabled=rollover_enabled,
        rollover_proxy_penalty=penalty,
    )
    env.config = env.env_config
    env.debug_without_robot_movement = False
    env.simulator = _StepSimulator()
    env.state = _StepState()
    env.occupancy_grid = None
    env._asymmetric_critic_enabled = False
    env._snqi_proxy = SimpleNamespace(compute_step_metrics=lambda *args, **kwargs: {})
    env.action_space = SimpleNamespace()
    env.last_action = None
    env.reward_func = lambda _meta: 1.0
    env._telemetry_session = None
    env.recording_enabled = False
    return env


def test_runtime_rollover_proxy_keeps_feasible_command_nonterminal() -> None:
    """Opt-in runtime proxy surfaces stable telemetry without terminating."""
    env = _make_step_env()

    _obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]))

    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert truncated is False
    assert info["rollover_critical"] is False
    assert info["rollover_proxy"]["stability_margin"] > 0.0
    assert "termination_reason" not in info


def test_runtime_rollover_proxy_over_yaw_trips_terminal_penalty() -> None:
    """Opt-in runtime proxy trips ``ROLLOVER_CRITICAL`` on over-yaw command."""
    env = _make_step_env(penalty=-4.0)

    _obs, reward, terminated, truncated, info = env.step(np.array([2.0, 2.0]))

    assert reward == pytest.approx(-3.0)
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "ROLLOVER_CRITICAL"
    assert info["rollover_critical"] is True
    assert info["rollover_proxy"]["rollover_critical"] is True
    assert info["meta"]["reward_terms"]["rollover_proxy_penalty"] == pytest.approx(-4.0)


def test_runtime_rollover_proxy_disabled_by_default_keeps_step_semantics() -> None:
    """Disabled proxy leaves even over-yaw toy command nonterminal."""
    env = _make_step_env(rollover_enabled=False, penalty=-4.0)

    _obs, reward, terminated, truncated, info = env.step(np.array([2.0, 2.0]))

    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert truncated is False
    assert "rollover_proxy" not in info
    assert "termination_reason" not in info


def test_robot_simulation_config_rollover_proxy_default_keeps_step_semantics() -> None:
    """Unified robot configs keep the rollover proxy disabled by default."""
    env = _make_step_env(env_config=RobotSimulationConfig())

    _obs, reward, terminated, truncated, info = env.step(np.array([2.0, 2.0]))

    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert truncated is False
    assert "rollover_proxy" not in info
    assert "termination_reason" not in info


def test_missing_rollover_config_attributes_keep_proxy_disabled() -> None:
    """Older/minimal config objects without rollover attributes stay compatible."""
    env = _make_step_env(env_config=SimpleNamespace())

    _obs, reward, terminated, truncated, info = env.step(np.array([2.0, 2.0]))

    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert truncated is False
    assert "rollover_proxy" not in info
    assert "termination_reason" not in info


def test_rollover_proxy_penalty_must_not_be_positive() -> None:
    """Configured rollover penalty must not accidentally reward rollover."""
    with pytest.raises(ValueError, match="rollover_proxy_penalty"):
        EnvSettings(rollover_proxy_penalty=0.1)
