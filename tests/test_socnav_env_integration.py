"""Integration smoke: SocNav structured obs + planner policy with RobotEnv."""

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.socnav import SocNavPlannerPolicy


def test_socnav_policy_runs_single_step():
    """Ensure policy and structured observation mode run a step without errors."""
    env = RobotEnv(env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT))
    obs, _ = env.reset()
    policy = SocNavPlannerPolicy()
    action = policy.act(obs)
    new_obs, _, _, _, info = env.step(action)
    assert env.observation_space.contains(new_obs)
    assert "success" in info
