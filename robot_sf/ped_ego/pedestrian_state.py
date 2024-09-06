"""
`PedestrianState`: A data class that represents the state of a pedestrian in the simulation environment.
It includes information about occupancy (for collision detection),
sensor fusion, and simulation time. It also tracks various conditions such as collision states,
timeout condition, simulation time elapsed, and timestep count.
"""

from math import ceil
from dataclasses import dataclass, field
import numpy as np

from robot_sf.sensor.sensor_fusion import SensorFusion
from robot_sf.nav.occupancy import ContinuousOccupancy

@dataclass
class PedestrianState:
    """
    Represents the state of a pedestrian in a simulated environment,
    including information about
    occupancy, sensor fusion, and simulation time.

    Attributes:
        occupancy (ContinuousOccupancy): Object tracking spatial occupation, 
            for collision detection.
        sensors (SensorFusion): Object for managing sensor data and fusion.
        d_t (float): The simulation timestep duration.
        sim_time_limit (float): The maximum allowed simulation time.

    Additional attributes with default values initialized to False or 0 representing various
    conditions such as collision states, timeout condition, simulation time elapsed,
    and timestep count.
    These are updated during the simulation.
    """
    occupancy: ContinuousOccupancy
    sensors: SensorFusion
    d_t: float
    sim_time_limit: float
    episode: int = field(init=False, default=0)
    is_at_goal: bool = field(init=False, default=False)
    is_collision_with_ped: bool = field(init=False, default=False)
    is_collision_with_obst: bool = field(init=False, default=False)
    is_collision_with_robot: bool = field(init=False, default=False)
    is_timeout: bool = field(init=False, default=False)
    distance_to_robot: float = field(init=False, default=0.0)
    sim_time_elapsed: float = field(init=False, default=0.0)
    timestep: int = field(init=False, default=0)

    @property
    def max_sim_steps(self) -> int:
        """Calculates the maximum number of simulation steps based on time limit."""
        return int(ceil(self.sim_time_limit / self.d_t))

    @property
    def is_terminal(self) -> bool:
        """
        Checks if the current state is terminal, i.e., if the robot has reached its goal,
        timed out, or collided with any object or other robots.
        """
        return (self.is_timeout or self.is_collision_with_robot or
                self.is_collision_with_ped or self.is_collision_with_obst) # or self.is_at_goal
    
    @property
    def max_distance(self) -> int:
        """Calculates the maximum number of simulation steps based on time limit."""
        w, h = self.occupancy.width, self.occupancy.height
        return np.sqrt(w**2 + h**2)

    def reset(self):
        """
        Resets the pedestrian's state for a new simulation episode, incrementing the episode counter,
        resetting the timestep and elapsed time, clearing collision and goal flags, and refreshing
        sensor data for the initial observation.
        """
        self.episode += 1
        self.timestep = 0
        self.sim_time_elapsed = 0.0
        self.is_collision_with_ped = False
        self.is_collision_with_obst = False
        self.is_collision_with_robot = False
        self.is_at_goal = False
        self.is_timeout = False
        self.distance_to_robot = np.inf
        self.sensors.reset_cache()
        return self.sensors.next_obs()

    def step(self):
        """
        Advances the pedestrian's state by one simulation timestep, updating the elapsed time,
        checking for collisions, goal achievement, and timeout. Returns the next observation
        from sensors.
        """
        self.timestep += 1
        self.sim_time_elapsed += self.d_t
        self.is_collision_with_ped = self.occupancy.is_pedestrian_collision
        self.is_collision_with_obst = self.occupancy.is_obstacle_collision
        self.is_collision_with_robot = self.occupancy.is_agent_agent_collision
        self.is_at_goal = self.occupancy.is_robot_at_goal
        self.distance_to_robot = self.occupancy.distance_to_robot
        self.is_timeout = self.sim_time_elapsed > self.sim_time_limit
        return self.sensors.next_obs()

    def meta_dict(self) -> dict:
        """
        Compiles a dictionary of metadata about the pedestrian's state for logging or
        monitoring purposes.
        Includes information such as episode number, current timestep, collision status,
        goal achievement.
        """
        return {
            "step": self.episode * self.max_sim_steps,
            "episode": self.episode,
            "step_of_episode": self.timestep,
            "is_pedestrian_collision": self.is_collision_with_ped,
            "is_robot_collision": self.is_collision_with_robot,
            "is_obstacle_collision": self.is_collision_with_obst,
            "distance_to_robot": self.distance_to_robot,
            "is_timesteps_exceeded": self.is_timeout,
            "max_sim_steps": self.max_sim_steps,
            "max_distance": self.max_distance,
        }
