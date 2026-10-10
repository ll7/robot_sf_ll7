"""Legacy settings retain native acceleration actions without PPO configuration."""

import numpy as np
import pytest

from robot_sf.gym_env.env_config import EnvSettings, RobotEnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage


@pytest.mark.parametrize(
    ("settings_type", "env_type"),
    [(EnvSettings, RobotEnv), (RobotEnvSettings, RobotEnvWithImage)],
)
def test_legacy_settings_default_to_acceleration(settings_type, env_type):
    """Construct and step both legacy settings families with physical speed oracles."""
    settings = settings_type()
    settings.sim_config.ped_density = 0.0
    settings.sim_config.pedestrian_seed = 1001
    settings.sim_config.time_per_step_in_secs = 0.1
    env = env_type(env_config=settings, debug=False)
    try:
        env.reset(seed=1001)
        robot = env.simulator.robots[0]
        robot.state.velocity = (0.6, 0.2)
        env.step(np.array([-0.05, -0.04]))
        # Native acceleration: (0.6, 0.2) + (-0.05, -0.04) * 0.1.
        np.testing.assert_allclose(robot.current_speed, [0.595, 0.196], atol=1e-8, rtol=0)
    finally:
        env.close()
