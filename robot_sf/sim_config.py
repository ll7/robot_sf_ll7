from typing import Union
from dataclasses import dataclass

from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings, DifferentialDriveRobot
from robot_sf.robot.bicycle_drive import BicycleDriveSettings, BicycleDriveRobot
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class EnvSettings:
    """
    Data class to hold environment settings for a simulation.
    """
    sim_config: SimulationSettings = SimulationSettings()
    lidar_config: LidarScannerSettings = LidarScannerSettings()
    robot_config: Union[DifferentialDriveSettings, BicycleDriveSettings] = \
        DifferentialDriveSettings()
    map_pool: MapDefinitionPool = MapDefinitionPool()

    def __post_init__(self):
        """
        Check if any of the properties are not initialized (None) and raise an error if so.
        """
        if not self.sim_config or not self.lidar_config \
                or not self.robot_config or not self.map_pool:
            raise ValueError('Please make sure all properties are initialized!')

    def robot_factory(self) -> Union[DifferentialDriveRobot, BicycleDriveRobot]:
        """
        Factory method to create a robot instance based on the type of robot configuration provided.
        :return: robot instance.
        """
        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(f"unsupported robot type {type(self.robot_config)}!")
