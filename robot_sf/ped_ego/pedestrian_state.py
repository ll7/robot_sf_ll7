"""
`PedestrianState`: A data class representing the state of a pedestrian in the simulation environment
It includes information about occupancy (for collision detection),
sensor fusion, and simulation time. It also tracks various conditions such as collision states,
timeout condition, simulation time elapsed, and timestep count.
"""

from dataclasses import dataclass, field
from math import ceil

import numpy as np

from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.sensor.sensor_fusion import SensorFusion


@dataclass
class PedestrianState:
    """
    Represents the state of a pedestrian in a simulated environment,
    including information about
    occupancy, sensor fusion, and simulation time.

    Attributes:
        robot_occupancy (ContinuousOccupancy): Object tracking spatial occupation,
            for collision detection.
        egp_ped_occupancy (EgoPedContinuousOccupancy): Object tracking spatial occupation,
            for collision detection.
        sensors (SensorFusion): Object for managing sensor data and fusion.
        d_t (float): The simulation timestep duration.
        sim_time_limit (float): The maximum allowed simulation time.

    Additional attributes with default values initialized to False or 0 representing various
    conditions such as collision states, timeout condition, simulation time elapsed,
    and timestep count.
    These are updated during the simulation.
    """

    robot_occupancy: ContinuousOccupancy
    ego_ped_occupancy: EgoPedContinuousOccupancy
    sensors: SensorFusion
    d_t: float
    sim_time_limit: float
    episode: int = field(init=False, default=0)
    is_collision_with_ped: bool = field(init=False, default=False)
    is_collision_with_obst: bool = field(init=False, default=False)
    is_collision_with_robot: bool = field(init=False, default=False)
    is_robot_at_goal: bool = field(init=False, default=False)
    is_collision_robot_with_obstacle: bool = field(init=False, default=False)
    is_collision_robot_with_pedestrian: bool = field(init=False, default=False)
    is_timeout: bool = field(init=False, default=False)
    distance_to_robot: float = field(init=False, default=0.0)
    sim_time_elapsed: float = field(init=False, default=0.0)
    timestep: int = field(init=False, default=0)

    @property
    def max_sim_steps(self) -> int:
        """Calculates the maximum number of simulation steps based on time limit."""
        return ceil(self.sim_time_limit / self.d_t)

    @property
    def is_terminal(self) -> bool:
        """
        Checks if the current state is terminal, i.e., if the robot has reached its goal,
        timed out, or collided with any object or other robots.
        """
        return (
            self.is_timeout
            or self.is_collision_with_robot
            or self.is_collision_with_ped
            or self.is_collision_with_obst
            or self.is_robot_at_goal
            or self.is_collision_robot_with_obstacle
            or self.is_collision_robot_with_pedestrian
        )

    def reset(self):
        """
        Resets the pedestrians state for a new simulation episode, incrementing the episode counter,
        resetting the timestep and elapsed time, clearing collision and goal flags, and refreshing
        sensor data for the initial observation.

        Returns:
            object: The initial observation produced by the sensor fusion after reset.
        """
        self.episode += 1
        self.timestep = 0
        self.sim_time_elapsed = 0.0
        self.is_collision_with_ped = False
        self.is_collision_with_obst = False
        self.is_collision_with_robot = False
        self.is_robot_at_goal = False
        self.is_collision_robot_with_obstacle = False
        self.is_collision_robot_with_pedestrian = False
        self.is_timeout = False
        self.distance_to_robot = np.inf
        self.sensors.reset_cache()
        return self.sensors.next_obs()

    def step(self):
        """
        Advances the pedestrian's state by one simulation timestep, updating the elapsed time,
        checking for collisions, goal achievement, and timeout. Returns the next observation
        from sensors.

        Returns:
            object: The next observation produced by the sensor fusion after advancing one step.
        """
        self.timestep += 1
        self.sim_time_elapsed += self.d_t
        self.is_collision_with_ped = self.ego_ped_occupancy.is_pedestrian_collision
        self.is_collision_with_obst = self.ego_ped_occupancy.is_obstacle_collision
        self.is_collision_with_robot = self.ego_ped_occupancy.is_agent_agent_collision
        self.is_robot_at_goal = self.robot_occupancy.is_robot_at_goal
        self.is_collision_robot_with_obstacle = self.robot_occupancy.is_obstacle_collision
        self.is_collision_robot_with_pedestrian = self.robot_occupancy.is_pedestrian_collision
        self.distance_to_robot = self.ego_ped_occupancy.distance_to_robot
        self.is_timeout = self.sim_time_elapsed > self.sim_time_limit
        return self.sensors.next_obs()

    def meta_dict(self) -> dict:
        """
        Compiles a dictionary of metadata about the pedestrian's state for logging or
        monitoring purposes.
        Includes information such as episode number, current timestep, collision status,
        goal achievement.

        Returns:
            dict: A snapshot of the current state and flags for diagnostics/telemetry.
        """
        return {
            "step": self.episode * self.max_sim_steps,
            "episode": self.episode,
            "step_of_episode": self.timestep,
            "is_pedestrian_collision": self.is_collision_with_ped,
            "is_robot_collision": self.is_collision_with_robot,
            "is_obstacle_collision": self.is_collision_with_obst,
            "distance_to_robot": self.distance_to_robot,
            "is_robot_at_goal": self.is_robot_at_goal,
            "is_robot_obstacle_collision": self.is_collision_robot_with_obstacle,
            "is_robot_pedestrian_collision": self.is_collision_robot_with_pedestrian,
            "is_timesteps_exceeded": self.is_timeout,
            "max_sim_steps": self.max_sim_steps,
        }
