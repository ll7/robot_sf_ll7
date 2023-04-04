from math import ceil
from typing import List
from dataclasses import dataclass, field


from robot_sf.map_config import MapDefinitionPool
from robot_sf.range_sensor import LidarScannerSettings
from robot_sf.robot import RobotSettings
from robot_sf.ped_robot_force import PedRobotForceConfig


@dataclass
class SimulationSettings:
    sim_length_in_secs: float = 200.0
    step_time_in_secs: float = 0.1
    peds_speed_mult: float = 1.3
    difficulty: int = 2
    max_peds_per_group: int = 6
    ped_radius: float = 0.4
    goal_radius: float = 1.0
    prf_config: PedRobotForceConfig = PedRobotForceConfig(is_active=True)
    ped_density_by_difficulty: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.04, 0.06])

    def __post_init__(self):
        if self.sim_length_in_secs <= 0:
            raise ValueError("Simulation length for episodes mustn't be negative or zero!")
        if self.step_time_in_secs <= 0:
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
        return ceil(self.sim_length_in_secs / self.step_time_in_secs)

    @property
    def peds_per_area_m2(self) -> float:
        return self.ped_density_by_difficulty[self.difficulty]


@dataclass
class EnvSettings:
    sim_config: SimulationSettings = SimulationSettings()
    lidar_config: LidarScannerSettings = LidarScannerSettings()
    robot_config: RobotSettings = RobotSettings()
    map_pool: MapDefinitionPool = MapDefinitionPool()

    def __post_init__(self):
        if not self.sim_config or not self.lidar_config \
                or not self.robot_config or not self.map_pool:
            raise ValueError('Please make sure all properties are initialized!')
