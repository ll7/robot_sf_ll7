"""Physical speed contract for the synthetic runner's delta PPO observations."""

from types import SimpleNamespace

import numpy as np

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
