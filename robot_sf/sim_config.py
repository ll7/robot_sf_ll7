from typing import Union
from dataclasses import dataclass

from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings, DifferentialDriveRobot
from robot_sf.robot.bicycle_drive import BicycleDriveSettings, BicycleDriveRobot
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class EnvSettings:
    sim_config: SimulationSettings = SimulationSettings()
    lidar_config: LidarScannerSettings = LidarScannerSettings()
    robot_config: Union[DifferentialDriveSettings, BicycleDriveSettings] = BicycleDriveSettings()
    map_pool: MapDefinitionPool = MapDefinitionPool()

    def __post_init__(self):
        if not self.sim_config or not self.lidar_config \
                or not self.robot_config or not self.map_pool:
            raise ValueError('Please make sure all properties are initialized!')

    def robot_factory(self) -> Union[DifferentialDriveRobot, BicycleDriveRobot]:
        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(f"unsupported robot type {type(self.robot_config)}!")
