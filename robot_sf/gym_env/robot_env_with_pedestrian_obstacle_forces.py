"""
Create a robot environment with pedestrian obstacle forces
"""

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.sim.simulator import init_simulators

class RobotEnvWithPedestrianObstacleForces(RobotEnv):
    def __init__(self):
        super().__init__()
        self.simulator = init_simulators(
            env_config=self.env_config,
            map_def=self.map_def,
            peds_have_obstacle_forces=True
        )
