from math import ceil
from typing import List
from dataclasses import dataclass, field

from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig


@dataclass
class SimulationSettings:
    """
    Configuration settings for the simulation.
    """

    # Simulation time in seconds
    sim_time_in_secs: float = 200.0
    # Time per step in seconds
    time_per_step_in_secs: float = 0.1
    # Pedestrian speed multiplier
    peds_speed_mult: float = 1.3
    # Difficulty level
    difficulty: int = 0
    # Maximum number of pedestrians per group
    max_peds_per_group: int = 6
    # Pedestrian radius
    ped_radius: float = 0.4
    # Goal radius
    goal_radius: float = 1.0
    # Number of steps to stack in observations
    # TODO move "stack_steps" from SimulationSettings to ?
    stack_steps: int = 3
    # Whether to use the next goal in the path as the current goal
    use_next_goal: bool = True
    # Pedestrian-robot force configuration
    prf_config: PedRobotForceConfig = field(default_factory=PedRobotForceConfig)
    # Pedestrian density by difficulty level
    ped_density_by_difficulty: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.04, 0.08]
    )

    def __post_init__(self):
        """
        Validate the simulation settings.

        This method is called after the object is initialized. It checks that all the
        settings are valid and raises a ValueError if any of them are not.
        """
        # Check that the simulation time is positive
        if self.sim_time_in_secs <= 0:
            raise ValueError(
                "Simulation length for episodes mustn't be negative or zero!"
            )
        # Check that the time per step is positive
        if self.time_per_step_in_secs <= 0:
            raise ValueError("Step time mustn't be negative or zero!")
        # Check that the pedestrian speed multiplier is positive
        if self.peds_speed_mult <= 0:
            raise ValueError("Pedestrian speed mustn't be negative or zero!")
        # Check that the maximum number of pedestrians per group is positive
        if self.max_peds_per_group <= 0:
            raise ValueError(
                "Maximum pedestrians per group mustn't be negative or zero!"
            )
        # Check that the pedestrian radius is positive
        if self.ped_radius <= 0:
            raise ValueError("Pedestrian radius mustn't be negative or zero!")
        # Check that the goal radius is positive
        if self.goal_radius <= 0:
            raise ValueError("Goal radius mustn't be negative or zero!")
        # Check that the difficulty level is within the valid range
        if not 0 <= self.difficulty < len(self.ped_density_by_difficulty):
            raise ValueError(
                "No pedestrian density registered for selected difficulty level!"
            )
        # Check that the pedestrian-robot force configuration is specified
        if not self.prf_config:
            raise ValueError("Pedestrian-Robot-Force settings need to be specified!")

    @property
    def max_sim_steps(self) -> int:
        return ceil(self.sim_time_in_secs / self.time_per_step_in_secs)

    @property
    def peds_per_area_m2(self) -> float:
        return self.ped_density_by_difficulty[self.difficulty]
