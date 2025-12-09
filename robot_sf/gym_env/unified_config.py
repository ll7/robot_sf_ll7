"""
Consolidated configuration classes for environments.

This module provides a unified configuration hierarchy that eliminates
duplication and provides clear separation of concerns.
"""

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_sf.ped_ego.unicycle_drive import UnicycleDrivePedestrian

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.telemetry_config import TelemetryConfigMixin
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.ped_ego.unicycle_drive import UnicycleDriveSettings
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.robot.differential_drive import (
    DifferentialDriveRobot,
    DifferentialDriveSettings,
)
from robot_sf.sensor.image_sensor import ImageSensorSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class BaseSimulationConfig(TelemetryConfigMixin):
    """
    Core simulation configuration shared by all environments.

    This replaces the multiple overlapping configuration classes with
    a single, consistent base configuration.
    """

    sim_config: SimulationSettings = field(default_factory=SimulationSettings)
    map_pool: MapDefinitionPool = field(default_factory=MapDefinitionPool)
    lidar_config: LidarScannerSettings = field(default_factory=LidarScannerSettings)
    # Optional UI/render scaling factor for SimulationView; when None, defaults apply.
    render_scaling: int | None = None
    # Backend selection and sensor wiring via registries
    backend: str = "fast-pysf"
    sensors: list[dict] = field(default_factory=list)
    observation_mode: ObservationMode = ObservationMode.DEFAULT_GYM

    def __post_init__(self):
        """Validate that all required fields are initialized."""
        if not self.sim_config or not self.map_pool or not self.lidar_config:
            raise ValueError("All configuration fields must be initialized!")
        self._validate_telemetry()


@dataclass
class RobotSimulationConfig(BaseSimulationConfig):
    """
    Configuration for robot-based environments.

    Extends base configuration with robot-specific settings.
    """

    robot_config: DifferentialDriveSettings | BicycleDriveSettings = field(
        default_factory=DifferentialDriveSettings,
    )
    # Environment behavior flags
    use_image_obs: bool = field(default=False)
    peds_have_obstacle_forces: bool = field(default=False)

    # Occupancy grid configuration
    grid_config: GridConfig | None = field(default=None)
    use_occupancy_grid: bool = field(default=False)
    # Grid observation flag - when True, includes occupancy grid in observation space
    include_grid_in_observation: bool = field(default=False)
    # Grid visualization configuration
    show_occupancy_grid: bool = field(
        default=False, metadata={"doc": "Show occupancy grid overlay in pygame visualization"}
    )
    grid_visualization_alpha: float = field(
        default=0.5,
        metadata={"doc": "Alpha blending for grid overlay (0.0=transparent, 1.0=opaque)"},
    )

    def __post_init__(self):
        """Validate robot-specific configuration."""
        super().__post_init__()
        if not self.robot_config:
            raise ValueError("Robot configuration must be initialized!")

        self._init_grid_config()
        self._validate_grid_observation()
        self._validate_grid_visualization()

    def _init_grid_config(self) -> None:
        """Initialize and validate the occupancy grid configuration."""
        if self.use_occupancy_grid and self.grid_config is None:
            self.grid_config = GridConfig()

        if self.grid_config is not None and not isinstance(self.grid_config, GridConfig):
            raise ValueError(
                f"grid_config must be GridConfig instance, got {type(self.grid_config)}"
            )

    def _validate_grid_observation(self) -> None:
        """Validate observation-related grid settings."""
        if not self.include_grid_in_observation:
            return

        if not self.use_occupancy_grid:
            raise ValueError("include_grid_in_observation=True requires use_occupancy_grid=True")

        if self.grid_config is None:
            raise ValueError("include_grid_in_observation=True requires valid grid_config")

    def _validate_grid_visualization(self) -> None:
        """Validate grid visualization settings."""
        if not self.show_occupancy_grid:
            return

        if not self.use_occupancy_grid:
            raise ValueError("show_occupancy_grid=True requires use_occupancy_grid=True")

        if not 0.0 <= self.grid_visualization_alpha <= 1.0:
            raise ValueError("grid_visualization_alpha must be between 0.0 and 1.0")

    def robot_factory(self) -> DifferentialDriveRobot | BicycleDriveRobot:
        """Create a robot instance based on configuration.

        Returns:
            Robot instance (DifferentialDriveRobot or BicycleDriveRobot) from config.
        """
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
                "Image configuration must be initialized when using image observations!",
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

    def pedestrian_factory(self) -> "UnicycleDrivePedestrian":
        """Create a pedestrian instance based on configuration.

        Returns:
            UnicycleDrivePedestrian instance created from ego_ped_config.
        """
        module = importlib.import_module("robot_sf.ped_ego.unicycle_drive")
        UnicycleDrivePedestrian = module.UnicycleDrivePedestrian

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


@dataclass
class RobotEnvSettings(ImageRobotConfig):
    """Deprecated: Use ImageRobotConfig instead."""


@dataclass
class PedEnvSettings(PedestrianSimulationConfig):
    """Deprecated: Use PedestrianSimulationConfig instead."""
