"""
The `sim_config.py` file defines `EnvSettings` and `PedEnvSettings` for simulation settings. 

These settings include:
- `sim_config`: Simulation configuration
- `lidar_config`: LiDAR scanner settings
- `robot_config`: Robot configuration (differential drive or bicycle drive)
- `map_pool`: A pool of map definitions
For PedEnvSettings, it also includes:
- `ego_ped_config`: Ego pedestrian configuration (unicycle drive)

The `__post_init__` method checks if all properties are initialized, raising an error if not.

The `robot_factory` method creates a robot instance based on the robot configuration.
It supports `DifferentialDriveRobot` and `BicycleDriveRobot`.
If the robot configuration is unsupported, it raises a `NotImplementedError`.

The `pedestrian_factory` method creates a pedestrian instance based on the configuration.
"""

from typing import Union
from dataclasses import dataclass

from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.robot.differential_drive import (
    DifferentialDriveSettings,
    DifferentialDriveRobot)
from robot_sf.ped_ego.unicycle_drive import UnicycleDriveSettings, UnicycleDrivePedestrian
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
        Check if any of the properties are not initialized (None) and raise an
        error if so.
        """
        if not self.sim_config or not self.lidar_config \
                or not self.robot_config or not self.map_pool:
            raise ValueError('Please make sure all properties are initialized!')

    def robot_factory(self) -> Union[DifferentialDriveRobot, BicycleDriveRobot]:
        """
        Factory method to create a robot instance based on the type of robot
        configuration provided.
        :return: robot instance.
        """

        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(
                f"unsupported robot type {type(self.robot_config)}!")


@dataclass
class PedEnvSettings(EnvSettings):
    """
    Data class to hold environment settings for a simulation that includes an ego pedestrian.
    """
    ego_ped_config: UnicycleDriveSettings = UnicycleDriveSettings()

    def __post_init__(self):
        """
        Check if any of the properties are not initialized (None) and raise an
        error if so.
        """
        super().__post_init__()
        if not self.ego_ped_config:
            raise ValueError('Please ensure ego_ped_config is initialized!')

        # Comment following line to allow different radius for ego pedestrian
        self.ego_ped_config.radius = self.sim_config.ped_radius # Ensure radius consistency

    def pedestrian_factory(self) -> UnicycleDrivePedestrian:
        """
        Factory method to create a pedestrian instance based on the type of pedestrian
        configuration provided.
        :return: pedestrian instance.
        """

        if isinstance(self.ego_ped_config, UnicycleDriveSettings):
            return UnicycleDrivePedestrian(self.ego_ped_config)
        else:
            raise NotImplementedError(
                f"unsupported pedestrian type {type(self.ego_ped_config)}!")
