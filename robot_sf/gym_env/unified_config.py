"""
Consolidated configuration classes for environments.

This module provides a unified configuration hierarchy that eliminates
duplication and provides clear separation of concerns.
"""

from dataclasses import dataclass, field
from typing import Union

from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.ped_ego.unicycle_drive import UnicycleDriveSettings
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings
from robot_sf.sensor.image_sensor import ImageSensorSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class BaseSimulationConfig:
    """
    Core simulation configuration shared by all environments.

    This replaces the multiple overlapping configuration classes with
    a single, consistent base configuration.
    """

    sim_config: SimulationSettings = field(default_factory=SimulationSettings)
    map_pool: MapDefinitionPool = field(default_factory=MapDefinitionPool)
    lidar_config: LidarScannerSettings = field(default_factory=LidarScannerSettings)

    def __post_init__(self):
        """Validate that all required fields are initialized."""
        if not self.sim_config or not self.map_pool or not self.lidar_config:
            raise ValueError("All configuration fields must be initialized!")


@dataclass
class RobotSimulationConfig(BaseSimulationConfig):
    """
    Configuration for robot-based environments.

    Extends base configuration with robot-specific settings.
    """

    robot_config: Union[DifferentialDriveSettings, BicycleDriveSettings] = field(
        default_factory=DifferentialDriveSettings
    )
    # Environment behavior flags
    use_image_obs: bool = field(default=False)
    peds_have_obstacle_forces: bool = field(default=False)

    def __post_init__(self):
        """Validate robot-specific configuration."""
        super().__post_init__()
        if not self.robot_config:
            raise ValueError("Robot configuration must be initialized!")

    def robot_factory(self) -> Union[DifferentialDriveRobot, BicycleDriveRobot]:
        """Create a robot instance based on configuration."""
        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(f"Unsupported robot type: {type(self.robot_config)}")


@dataclass
class ImageRobotConfig(RobotSimulationConfig):
    """
    Configuration for robot environments with image observations.

    Automatically enables image observations and includes image sensor settings.
    """

    image_config: ImageSensorSettings = field(default_factory=ImageSensorSettings)
    use_image_obs: bool = field(default=True)  # Override default to True

    def __post_init__(self):
        """Validate image-specific configuration."""
        super().__post_init__()
        if not self.image_config:
            raise ValueError(
                "Image configuration must be initialized when using image observations!"
            )


@dataclass
class PedestrianSimulationConfig(RobotSimulationConfig):
    """
    Configuration for pedestrian environments.

    Includes both robot and ego pedestrian configurations for adversarial training.
    """

    ego_ped_config: UnicycleDriveSettings = field(default_factory=UnicycleDriveSettings)

    def __post_init__(self):
        """Validate pedestrian-specific configuration."""
        super().__post_init__()
        if not self.ego_ped_config:
            raise ValueError("Ego pedestrian configuration must be initialized!")

        # Ensure radius consistency between ego pedestrian and simulation
        self.ego_ped_config.radius = self.sim_config.ped_radius

    def pedestrian_factory(self):
        """Create a pedestrian instance based on configuration."""
        from robot_sf.ped_ego.unicycle_drive import UnicycleDrivePedestrian

        if isinstance(self.ego_ped_config, UnicycleDriveSettings):
            return UnicycleDrivePedestrian(self.ego_ped_config)
        else:
            raise NotImplementedError(f"Unsupported pedestrian type: {type(self.ego_ped_config)}")


@dataclass
class MultiRobotConfig(RobotSimulationConfig):
    """
    Configuration for multi-robot environments.
    """

    num_robots: int = field(default=1)

    def __post_init__(self):
        """Validate multi-robot configuration."""
        super().__post_init__()
        if self.num_robots < 1:
            raise ValueError("Number of robots must be at least 1!")


# Backward compatibility - these can be deprecated gradually
@dataclass
class EnvSettings(RobotSimulationConfig):
    """Deprecated: Use RobotSimulationConfig instead."""

    pass


@dataclass
class RobotEnvSettings(ImageRobotConfig):
    """Deprecated: Use ImageRobotConfig instead."""

    pass


@dataclass
class PedEnvSettings(PedestrianSimulationConfig):
    """Deprecated: Use PedestrianSimulationConfig instead."""

    pass
