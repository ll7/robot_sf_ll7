"""
The `env_config.py` file defines `EnvSettings` and `PedEnvSettings` for simulation settings.

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

from dataclasses import dataclass, field

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.ped_ego.unicycle_drive import (
    UnicycleDrivePedestrian,
    UnicycleDriveSettings,
)
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.robot.differential_drive import (
    DifferentialDriveRobot,
    DifferentialDriveSettings,
)
from robot_sf.sensor.image_sensor import ImageSensorSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class BaseEnvSettings:
    """Base environment settings."""

    sim_config: SimulationSettings = field(default_factory=SimulationSettings)
    map_pool: MapDefinitionPool = field(default_factory=MapDefinitionPool)
    # Optional UI/render scaling factor for SimulationView; when None, defaults apply.
    render_scaling: int | None = None
    observation_mode: ObservationMode = ObservationMode.DEFAULT_GYM

    def __post_init__(self):
        """
        Check if any of the properties are not initialized (None) and raise an
        error if so.
        """
        if not self.sim_config or not self.map_pool:
            raise ValueError("Please make sure all properties are initialized!")


@dataclass
class RobotEnvSettings(BaseEnvSettings):
    """Robot-specific environment settings."""

    lidar_config: LidarScannerSettings = field(default_factory=LidarScannerSettings)
    robot_config: DifferentialDriveSettings | BicycleDriveSettings = field(
        default_factory=DifferentialDriveSettings,
    )
    # Image observation settings
    image_config: ImageSensorSettings = field(default_factory=ImageSensorSettings)
    use_image_obs: bool = field(default=False)  # Enable/disable image observations

    def __post_init__(self):
        """
        Check if any of the properties are not initialized (None) and raise an
        error if so.
        """
        if (
            not self.sim_config
            or not self.lidar_config
            or not self.robot_config
            or not self.map_pool
        ):
            raise ValueError("Please make sure all properties are initialized!")

    def robot_factory(self) -> DifferentialDriveRobot | BicycleDriveRobot:
        """
        Factory method to create a robot instance based on the type of robot
        configuration provided.
        :return: robot instance.

        Returns:
            Robot instance (DifferentialDriveRobot or BicycleDriveRobot) created from config.
        """

        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(f"unsupported robot type {type(self.robot_config)}!")


@dataclass
class EnvSettings:
    """
    Data class to hold environment settings for a simulation.
    """

    sim_config: SimulationSettings = field(default_factory=SimulationSettings)
    lidar_config: LidarScannerSettings = field(default_factory=LidarScannerSettings)
    robot_config: DifferentialDriveSettings | BicycleDriveSettings = field(
        default_factory=DifferentialDriveSettings,
    )
    map_pool: MapDefinitionPool = field(default_factory=MapDefinitionPool)
    observation_mode: ObservationMode = ObservationMode.DEFAULT_GYM

    def __post_init__(self):
        """
        Check if any of the properties are not initialized (None) and raise an
        error if so.
        """
        if (
            not self.sim_config
            or not self.lidar_config
            or not self.robot_config
            or not self.map_pool
        ):
            raise ValueError("Please make sure all properties are initialized!")

    def robot_factory(self) -> DifferentialDriveRobot | BicycleDriveRobot:
        """
        Factory method to create a robot instance based on the type of robot
        configuration provided.
        :return: robot instance.

        Returns:
            Robot instance (DifferentialDriveRobot or BicycleDriveRobot) created from config.
        """

        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(f"unsupported robot type {type(self.robot_config)}!")


@dataclass
class PedEnvSettings(EnvSettings):
    """
    Data class to hold environment settings for a simulation that includes an ego pedestrian.
    """

    ego_ped_config: UnicycleDriveSettings = field(default_factory=UnicycleDriveSettings)

    def __post_init__(self):
        """
        Check if any of the properties are not initialized (None) and raise an
        error if so.
        """
        super().__post_init__()
        if not self.ego_ped_config:
            raise ValueError("Please ensure ego_ped_config is initialized!")

        # Comment following line to allow different radius for ego pedestrian
        self.ego_ped_config.radius = self.sim_config.ped_radius  # Ensure radius consistency

    def pedestrian_factory(self) -> UnicycleDrivePedestrian:
        """
        Factory method to create a pedestrian instance based on the type of pedestrian
        configuration provided.
        :return: pedestrian instance.

        Returns:
            UnicycleDrivePedestrian instance created from config.
        """

        if isinstance(self.ego_ped_config, UnicycleDriveSettings):
            return UnicycleDrivePedestrian(self.ego_ped_config)
        else:
            raise NotImplementedError(f"unsupported pedestrian type {type(self.ego_ped_config)}!")


# Backward compatibility imports - these provide access to the new unified configuration
# while maintaining the old interface
try:
    from robot_sf.gym_env.unified_config import (
        BaseSimulationConfig,
    )
    from robot_sf.gym_env.unified_config import (
        ImageRobotConfig as RobotEnvSettingsNew,
    )
    from robot_sf.gym_env.unified_config import (
        PedestrianSimulationConfig as PedEnvSettingsNew,
    )
    from robot_sf.gym_env.unified_config import (
        RobotSimulationConfig as EnvSettingsNew,
    )

    # These can be used as drop-in replacements for gradual migration
    __all__ = [
        "BaseEnvSettings",
        # New unified config classes for forward compatibility
        "BaseSimulationConfig",
        "EnvSettings",
        "EnvSettingsNew",
        "PedEnvSettings",
        "PedEnvSettingsNew",
        "RobotEnvSettings",
        "RobotEnvSettingsNew",
    ]
except ImportError:
    # If unified_config is not available, just export the original classes
    __all__ = ["BaseEnvSettings", "EnvSettings", "PedEnvSettings", "RobotEnvSettings"]
