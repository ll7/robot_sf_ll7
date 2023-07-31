from math import ceil
from typing import List
from dataclasses import dataclass, field

from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig


@dataclass
class SimulationSettings:
    sim_time_in_secs: float = 200.0
    time_per_step_in_secs: float = 0.1
    peds_speed_mult: float = 1.3
    difficulty: int = 0
    max_peds_per_group: int = 6
    ped_radius: float = 0.4
    goal_radius: float = 1.0
    stack_steps: int = 3
    use_next_goal: bool = True
    prf_config: PedRobotForceConfig = PedRobotForceConfig(is_active=True)
    ped_density_by_difficulty: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.04, 0.08])

    def __post_init__(self):
        if self.sim_time_in_secs <= 0:
            raise ValueError("Simulation length for episodes mustn't be negative or zero!")
        if self.time_per_step_in_secs <= 0:
            raise ValueError("Step time mustn't be negative or zero!")
        if self.peds_speed_mult <= 0:
            raise ValueError("Pedestrian speed mustn't be negative or zero!")
        if self.max_peds_per_group <= 0:
            raise ValueError("Maximum pedestrians per group mustn't be negative or zero!")
        if self.ped_radius <= 0:
            raise ValueError("Pedestrian radius mustn't be negative or zero!")
        if self.goal_radius <= 0:
            raise ValueError("Goal radius mustn't be negative or zero!")
        if not 0 <= self.difficulty < len(self.ped_density_by_difficulty):
            raise ValueError("No pedestrian density registered for selected difficulty level!")
        if not self.prf_config:
            raise ValueError("Pedestrian-Robot-Force settings need to be specified!")

    @property
    def max_sim_steps(self) -> int:
        return ceil(self.sim_time_in_secs / self.time_per_step_in_secs)

    @property
    def peds_per_area_m2(self) -> float:
        return self.ped_density_by_difficulty[self.difficulty]
