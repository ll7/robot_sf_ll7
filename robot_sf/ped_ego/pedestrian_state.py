"""
`PedestrianState`: A data class representing the state of a pedestrian in the simulation environment
It includes information about occupancy (for collision detection),
sensor fusion, and simulation time. It also tracks various conditions such as collision states,
timeout condition, simulation time elapsed, and timestep count.
"""

import math
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
    ego_ped_speed: float = field(init=False, default=0.0)
    collision_impact_angle_rad: float = field(init=False, default=0.0)
    robot_ped_collision_zone: str = field(init=False, default="none")
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
        self.ego_ped_speed = 0.0
        self.collision_impact_angle_rad = 0.0
        self.robot_ped_collision_zone = "none"
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
        self.ego_ped_speed = self._extract_linear_speed(self.sensors.robot_speed_sensor())
        self.collision_impact_angle_rad = 0.0
        self.robot_ped_collision_zone = "none"
        if self.is_collision_with_robot:
            impact_angle, zone = self._compute_robot_ped_impact_metrics()
            self.collision_impact_angle_rad = impact_angle
            self.robot_ped_collision_zone = zone
        return self.sensors.next_obs()

    @staticmethod
    def _extract_linear_speed(
        speed_like: float | tuple[float, ...] | list[float] | np.ndarray,
    ) -> float:
        """Return the translational speed component from sensor output."""
        if isinstance(speed_like, np.ndarray):
            flat = speed_like.reshape(-1)
            return float(abs(flat[0])) if flat.size else 0.0
        if isinstance(speed_like, (tuple, list)):
            return float(abs(speed_like[0])) if speed_like else 0.0
        return float(abs(speed_like))

    def _compute_robot_ped_impact_metrics(
        self,
    ) -> tuple[float, str]:
        """Compute impact metrics for robot-pedestrian collisions only.

        Returns:
            tuple[float, str]:
                ``(abs_relative_angle_rad, zone)``
                where zone is one of ``front``, ``side``, ``back``, ``unknown``, ``none``.
        """
        if not self.is_collision_with_robot:
            return 0.0, "none"

        robot_x, robot_y = self.robot_occupancy.get_agent_coords()
        ego_x, ego_y = self.ego_ped_occupancy.get_agent_coords()
        rel_x = float(ego_x - robot_x)
        rel_y = float(ego_y - robot_y)
        bearing_world = float(math.atan2(rel_y, rel_x))

        robot_heading = self.robot_occupancy.agent_heading
        if robot_heading is None:
            return 0.0, "unknown"

        rel_angle = float(
            math.atan2(
                math.sin(bearing_world - robot_heading),
                math.cos(bearing_world - robot_heading),
            )
        )
        abs_angle = abs(rel_angle)
        if abs_angle <= (math.pi / 4):
            zone = "front"
        elif abs_angle >= (3 * math.pi / 4):
            zone = "back"
        else:
            zone = "side"
        return abs_angle, zone

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
            "ego_ped_speed": self.ego_ped_speed,
            "collision_impact_angle_rad": self.collision_impact_angle_rad,
            "collision_impact_angle_deg": math.degrees(self.collision_impact_angle_rad),
            "robot_ped_collision_zone": self.robot_ped_collision_zone,
            # For pedestrian-side observations, robot goal completion is a single-stage flag.
            # Emit route/waypoint aliases so downstream consumers can use canonical keys.
            "is_waypoint_complete": self.is_robot_at_goal,
            "is_route_complete": self.is_robot_at_goal,
            "is_robot_at_goal": self.is_robot_at_goal,
            "is_robot_obstacle_collision": self.is_collision_robot_with_obstacle,
            "is_robot_pedestrian_collision": self.is_collision_robot_with_pedestrian,
            "is_timesteps_exceeded": self.is_timeout,
            "max_sim_steps": self.max_sim_steps,
        }
