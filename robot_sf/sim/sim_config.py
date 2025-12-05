"""TODO docstring. Document this module."""

from dataclasses import dataclass, field
from math import ceil

from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig


@dataclass
class SimulationSettings:
    """
    Configuration settings for the simulation.
    """

    sim_time_in_secs: float = 200.0
    """Simulation time in seconds"""

    time_per_step_in_secs: float = 0.1
    """Time per step in seconds"""

    peds_speed_mult: float = 1.3
    """Pedestrian speed multiplier"""

    difficulty: int = 0
    """Difficulty level"""

    max_peds_per_group: int = 6
    """Maximum number of pedestrians per group"""

    ped_radius: float = 0.4
    """Pedestrian radius"""

    goal_radius: float = 1.0
    """Goal radius"""

    stack_steps: int = 3
    """Number of steps to stack in observations
    TODO move "stack_steps" from SimulationSettings to ?"""

    use_next_goal: bool = True
    """Whether to use the next goal in the path as the current goal"""

    prf_config: PedRobotForceConfig = field(default_factory=PedRobotForceConfig)
    """Pedestrian-robot force configuration"""

    ped_density_by_difficulty: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.04, 0.08])
    """Pedestrian density by difficulty level"""
    max_total_pedestrians: int | None = None
    """Optional upper bound for pedestrians used to size SocNav structured observations."""

    def __post_init__(self):
        """
        Validate the simulation settings.

        This method is called after the object is initialized. It checks that all the
        settings are valid and raises a ValueError if any of them are not.
        """
        # Check that the simulation time is positive
        if self.sim_time_in_secs <= 0:
            raise ValueError("Simulation length for episodes mustn't be negative or zero!")
        # Check that the time per step is positive
        if self.time_per_step_in_secs <= 0:
            raise ValueError("Step time mustn't be negative or zero!")
        # Check that the pedestrian speed multiplier is positive
        if self.peds_speed_mult <= 0:
            raise ValueError("Pedestrian speed mustn't be negative or zero!")
        # Check that the maximum number of pedestrians per group is positive
        if self.max_peds_per_group <= 0:
            raise ValueError("Maximum pedestrians per group mustn't be negative or zero!")
        # Check that the pedestrian radius is positive
        if self.ped_radius <= 0:
            raise ValueError("Pedestrian radius mustn't be negative or zero!")
        # Check that the goal radius is positive
        if self.goal_radius <= 0:
            raise ValueError("Goal radius mustn't be negative or zero!")
        # Check that the difficulty level is within the valid range
        if not 0 <= self.difficulty < len(self.ped_density_by_difficulty):
            raise ValueError("No pedestrian density registered for selected difficulty level!")
        # Check that the pedestrian-robot force configuration is specified
        if not self.prf_config:
            raise ValueError("Pedestrian-Robot-Force settings need to be specified!")

    @property
    def max_sim_steps(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return ceil(self.sim_time_in_secs / self.time_per_step_in_secs)

    @property
    def peds_per_area_m2(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.ped_density_by_difficulty[self.difficulty]
