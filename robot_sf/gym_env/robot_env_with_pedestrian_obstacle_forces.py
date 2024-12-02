"""
Create a robot environment with pedestrian obstacle forces
"""

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.sim.simulator import init_simulators


class RobotEnvWithPedestrianObstacleForces(RobotEnv):
    """
    Robot environment with pedestrian obstacle forces
    This increases the simulation time by roughly 40%
    """

    def __init__(self):
        """
        Initialize the Robot Environment with pedestrian obstacle forces
        """
        super().__init__()
        # init_simulators returns a list, so we need to get the first simulator
        self.simulator = init_simulators(
            env_config=self.env_config,
            map_def=self.map_def,
            peds_have_obstacle_forces=True,
        )[
            0
        ]  # Take the first simulator from the list
