"""Config"""

from dataclasses import dataclass, field

from pysocialforce.ped_population import PedSpawnConfig


@dataclass
class SceneConfig:
    """SceneConfig class."""

    enable_group: bool = True
    agent_radius: float = 0.35
    dt_secs: float = 0.1
    max_speed_multiplier: float = 1.3
    tau: float = 0.5
    resolution: float = 10


@dataclass
class GroupCoherenceForceConfig:
    """GroupCoherenceForceConfig class."""

    factor: float = 3.0


@dataclass
class GroupReplusiveForceConfig:
    """GroupReplusiveForceConfig class."""

    factor: float = 1.0
    threshold: float = 0.55


@dataclass
class GroupGazeForceConfig:
    """GroupGazeForceConfig class."""

    factor: float = 4.0
    fov_phi: float = 90.0


@dataclass
class DesiredForceConfig:
    """DesiredForceConfig class."""

    factor: float = 1.0
    relaxation_time: float = 0.5
    goal_threshold: float = 0.2


@dataclass
class SocialForceConfig:
    """SocialForceConfig class."""

    factor: float = 5.1
    lambda_importance: float = 2.0
    gamma: float = 0.35
    n: int = 2
    n_prime: int = 3
    activation_threshold: float = 20.0


@dataclass
class ObstacleForceConfig:
    """ObstacleForceConfig class."""

    factor: float = 10.0
    sigma: float = 0.0
    threshold: float = -0.57


@dataclass
class SimulatorConfig:
    """SimulatorConfig class."""

    scene_config: SceneConfig = field(default_factory=SceneConfig)
    group_coherence_force_config: GroupCoherenceForceConfig = field(
        default_factory=GroupCoherenceForceConfig
    )
    group_repulsive_force_config: GroupReplusiveForceConfig = field(
        default_factory=GroupReplusiveForceConfig
    )
    group_gaze_force_config: GroupGazeForceConfig = field(default_factory=GroupGazeForceConfig)
    desired_force_config: DesiredForceConfig = field(default_factory=DesiredForceConfig)
    social_force_config: SocialForceConfig = field(default_factory=SocialForceConfig)
    obstacle_force_config: ObstacleForceConfig = field(default_factory=ObstacleForceConfig)
    ped_spawn_config: PedSpawnConfig = field(default_factory=PedSpawnConfig)
