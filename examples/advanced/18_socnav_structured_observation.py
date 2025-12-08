"""
Example: run RobotEnv with SocNavBench-style structured observations and a simple planner.
"""

from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.socnav import SocNavPlannerPolicy


def main():
    """Run a short rollout using SocNav structured observations and a simple planner."""

    config = RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT)
    env = make_robot_env(config=config, debug=False)
    policy = SocNavPlannerPolicy()

    obs, _ = env.reset()
    done = False
    step = 0
    while not done and step < 100:
        action = policy.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated or info.get("success", False)
        step += 1

    env.close()
    logger.info(f"Episode finished after {step} steps, success={info.get('success', False)}")


if __name__ == "__main__":
    main()
