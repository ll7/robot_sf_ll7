"""Environment configuration dataclasses for robot and pedestrian simulations.

This module defines the configuration dataclass hierarchy for Robot SF environments:

- `BaseEnvSettings`: Core simulation settings shared across all environment types,
  including simulation parameters, map pools, observation modes, and telemetry config.
- `EnvSettings`: Robot-specific settings extending BaseEnvSettings with lidar and
  robot drivetrain configurations. Provides factory methods for robot instantiation.
- `RobotEnvSettings`: Extended robot settings with support for image-based observations.
- `PedEnvSettings`: Settings for ego pedestrian environments, supporting both robot and
  pedestrian control with unified physics via SocialForce.

Backward compatibility is maintained through optional imports of unified configuration
classes from `robot_sf.gym_env.unified_config`, which serve as forward-compatible
alternatives for gradual migration.

All configuration classes validate required fields during initialization and inherit
telemetry configuration support from `TelemetryConfigMixin`."""

from dataclasses import dataclass, field

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.telemetry_config import TelemetryConfigMixin
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.occupancy_grid import GridConfig
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
class BaseEnvSettings(TelemetryConfigMixin):
    """Base environment settings shared across all simulation types.

    This dataclass provides core configuration for Robot SF simulations, including
    physics settings, map definitions, observation modes, and optional occupancy grid
    configuration. All environment configurations inherit from this base class.

    Attributes:
        sim_config: Physics and dynamics configuration for the simulation.
        map_pool: Collection of available map definitions for scenario selection.
        render_scaling: Optional UI scaling factor for SimulationView; defaults apply if None.
        observation_mode: Specifies how observations are formatted (e.g., DEFAULT_GYM).
        use_occupancy_grid: Enable occupancy grid computation (legacy compatibility).
        grid_config: Occupancy grid parameters; required if use_occupancy_grid is True.
        include_grid_in_observation: Include occupancy grid data in environment observations.
        show_occupancy_grid: Render occupancy grid visualization during playback.

    Raises:
        ValueError: If sim_config or map_pool are not initialized after __post_init__.
    """

    sim_config: SimulationSettings = field(default_factory=SimulationSettings)
    map_pool: MapDefinitionPool = field(default_factory=MapDefinitionPool)
    # Optional UI/render scaling factor for SimulationView; when None, defaults apply.
    render_scaling: int | None = None
    observation_mode: ObservationMode = ObservationMode.DEFAULT_GYM
    # Occupancy grid toggles (legacy settings for compatibility)
    use_occupancy_grid: bool = False
    grid_config: GridConfig | None = None
    include_grid_in_observation: bool = False
    show_occupancy_grid: bool = False

    def __post_init__(self):
        """Validate required configuration fields and telemetry settings.

        Ensures that critical simulation components (sim_config, map_pool) are
        initialized and invokes telemetry validation from the mixin.

        Raises:
            ValueError: If sim_config or map_pool are not initialized.
        """
        if not self.sim_config or not self.map_pool:
            raise ValueError("Please make sure all properties are initialized!")
        self._validate_telemetry()


@dataclass
class EnvSettings(BaseEnvSettings):
    """Environment settings for robot-based simulations with differential or bicycle drivetrain.

    This configuration class extends BaseEnvSettings to support robot-specific sensors
    and actuation models. It provides factory methods for instantiating robot instances
    based on the configured drivetrain type.

    Attributes:
        lidar_config: Lidar scanner configuration for range-based observations.
        robot_config: Robot drivetrain settings (DifferentialDriveSettings or
            BicycleDriveSettings); determines the kinematic model used in simulation.

    Raises:
        ValueError: If lidar_config or robot_config are not initialized after __post_init__.
    """

    lidar_config: LidarScannerSettings = field(default_factory=LidarScannerSettings)
    robot_config: DifferentialDriveSettings | BicycleDriveSettings = field(
        default_factory=DifferentialDriveSettings,
    )

    def __post_init__(self):
        """Validate robot-specific configuration fields.

        Calls parent validation and ensures lidar and robot drivetrain configs
        are properly initialized.

        Raises:
            ValueError: If lidar_config or robot_config are not initialized.
        """
        super().__post_init__()
        if not self.lidar_config or not self.robot_config:
            raise ValueError("Please make sure all properties are initialized!")

    def robot_factory(self) -> DifferentialDriveRobot | BicycleDriveRobot:
        """Create a robot instance matching the configured drivetrain type.

        Instantiates the appropriate robot class (DifferentialDriveRobot or
        BicycleDriveRobot) based on the type of robot_config.

        Returns:
            DifferentialDriveRobot: If robot_config is DifferentialDriveSettings.
            BicycleDriveRobot: If robot_config is BicycleDriveSettings.

        Raises:
            NotImplementedError: If robot_config type is not recognized.
        """

        if isinstance(self.robot_config, DifferentialDriveSettings):
            return DifferentialDriveRobot(self.robot_config)
        elif isinstance(self.robot_config, BicycleDriveSettings):
            return BicycleDriveRobot(self.robot_config)
        else:
            raise NotImplementedError(f"unsupported robot type {type(self.robot_config)}!")


@dataclass
class RobotEnvSettings(EnvSettings):
    """Extended robot environment settings with image-based observation support.

    This configuration enables visual observations from a simulated camera sensor
    in addition to range-based lidar data. Useful for training agents with
    convolutional observation models or multi-modal sensor fusion.

    Attributes:
        image_config: Camera sensor configuration for image observations.
        use_image_obs: Enable/disable image observation collection and inclusion
            in environment step returns.
    """

    # Image observation settings
    image_config: ImageSensorSettings = field(default_factory=ImageSensorSettings)
    use_image_obs: bool = field(default=False)  # Enable/disable image observations

    def __post_init__(self):
        """Validate configuration after initialization.

        Calls parent validation to ensure all robot and base settings are valid.
        """
        super().__post_init__()


@dataclass
class PedEnvSettings(EnvSettings):
    """Environment settings for ego pedestrian simulations with social navigation.

    This configuration supports an ego pedestrian agent navigating among crowds
    using unicycle kinematics. The environment simulates social force interactions
    between the ego pedestrian and other agents in the scene.

    Attributes:
        ego_ped_config: Control and dynamics settings for the ego pedestrian,
            using unicycle model (forward velocity and angular velocity).

    Note:
        The ego pedestrian's collision radius is automatically synchronized with
        the global pedestrian radius from sim_config to ensure physics consistency.
        Modify the __post_init__ implementation if different radii are required.

    Raises:
        ValueError: If ego_ped_config is not initialized after __post_init__.
    """

    ego_ped_config: UnicycleDriveSettings = field(default_factory=UnicycleDriveSettings)

    def __post_init__(self):
        """Validate pedestrian configuration and ensure radius consistency.

        Calls parent validation and synchronizes the ego pedestrian's collision
        radius with the global pedestrian radius for physics consistency.

        Raises:
            ValueError: If ego_ped_config is not initialized.
        """
        super().__post_init__()
        if not self.ego_ped_config:
            raise ValueError("Please ensure ego_ped_config is initialized!")

        # Comment following line to allow different radius for ego pedestrian
        self.ego_ped_config.radius = self.sim_config.ped_radius  # Ensure radius consistency

    def pedestrian_factory(self) -> UnicycleDrivePedestrian:
        """Create an ego pedestrian instance with configured dynamics.

        Instantiates a UnicycleDrivePedestrian agent from the ego_ped_config
        settings. The ego pedestrian uses unicycle kinematics (forward and
        angular velocity control) and participates in social force interactions.

        Returns:
            UnicycleDrivePedestrian: Ego pedestrian instance ready for control and
                physics simulation.

        Raises:
            NotImplementedError: If ego_ped_config type is not UnicycleDriveSettings.
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
