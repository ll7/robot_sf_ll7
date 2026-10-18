"""Physical speed contract for the synthetic runner's delta PPO observations."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.baselines.ppo import PPOPlannerConfig
from robot_sf.baselines.social_force import Observation
from robot_sf.benchmark import runner
from robot_sf.robot.dynamics import RobotDynamicsState, UnicycleDynamics


def test_delta_ppo_observation_tracks_current_drive_speed_after_steps() -> None:
    """PPO sees actual signed speed and yaw rate, including after turning and clipping."""
    observations = []
    commands = [(0.8, 0.4), (1.2, -0.6), (-0.3, 0.2), (3.0, 2.0), (0.0, 0.0)]

    def step(obs: Observation) -> dict[str, float]:
        """Capture the real runner observation and issue a physical velocity command."""
        observations.append(obs)
        v, omega = commands[len(observations) - 1]
        return {"v": v, "omega": omega}

    config = PPOPlannerConfig(action_space="unicycle", action_semantics="velocity_delta")
    policy = runner._build_baseline_policy_fn(
        algo="ppo",
        planner=SimpleNamespace(config=config),
        observation_cls=Observation,
        step_runner=SimpleNamespace(step=step),
        timeout_metadata={},
        metadata={"action_semantics": "velocity_delta"},
        retry_budget=0,
        robot_radius=0.3,
        ped_radius=0.35,
    )
    drive = UnicycleDynamics(max_linear_speed=config.v_max, max_angular_speed=config.omega_max)
    state = RobotDynamicsState(x=2.0, y=3.0)
    pos = np.array([state.x, state.y])
    velocity = np.zeros(2)
    dt = 0.1

    for command in commands:
        velocity = policy(pos, velocity, np.array([9.0, 8.0]), np.empty((0, 2)), dt)
        speed = observations[-1].robot["speed"]
        assert speed.dtype == np.float32
        assert speed.shape == (2,)
        np.testing.assert_array_equal(
            speed, np.asarray([state.linear_speed, state.angular_speed], dtype=np.float32)
        )
        np.testing.assert_allclose(observations[-1].robot["heading"], state.heading)
        state = drive.step(state, command, dt)
        pos = pos + velocity * dt
        np.testing.assert_allclose(pos, [state.x, state.y])

    # A turn and reverse motion must survive; neither can be recovered from |velocity_xy|.
    np.testing.assert_array_equal(observations[3].robot["speed"], np.array([-0.3, 0.2], np.float32))
    np.testing.assert_array_equal(observations[4].robot["speed"], np.array([2.0, 1.0], np.float32))


def test_delta_drive_worker_fallback_is_physical_stop() -> None:
    """Worker timeout fallback stops a moving, turning drive without stale velocity."""
    state = RobotDynamicsState(x=2.0, y=3.0, heading=0.4, linear_speed=0.8, angular_speed=0.6)
    updated, velocity = runner._advance_delta_ppo_drive(
        UnicycleDynamics(), state, {"vx": 0.0, "vy": 0.0}, 0.1
    )
    assert (updated.x, updated.y, updated.heading) == pytest.approx((2.0, 3.0, 0.4))
    assert (updated.linear_speed, updated.angular_speed) == (0.0, 0.0)
    assert velocity == (0.0, 0.0)


@pytest.mark.parametrize(
    ("state", "action", "error", "message"),
    [
        (None, {"v": 0.2, "omega": 0.1}, RuntimeError, "not initialized"),
        (RobotDynamicsState(), {"vx": 0.2, "vy": 0.0}, ValueError, "unicycle velocity command"),
    ],
)
def test_delta_drive_rejects_uninitialized_or_cartesian_command(state, action, error, message):
    """A delta policy cannot use an absent physical state or a non-stop Cartesian command."""
    with pytest.raises(error, match=message):
        runner._advance_delta_ppo_drive(UnicycleDynamics(), state, action, 0.1)
