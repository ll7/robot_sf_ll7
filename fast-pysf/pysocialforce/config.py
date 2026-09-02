"""Configuration dataclasses and version metadata for the fast-pysf simulator."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pysocialforce.ped_population import PedSpawnConfig

LEGACY_SHIFTED_GRADIENT_V1 = "legacy_shifted_gradient_v1"
SURFACE_DISTANCE_UNIT_NORMAL_V2 = "surface_distance_unit_normal_v2"
DEFAULT_OBSTACLE_FORCE_LAW = LEGACY_SHIFTED_GRADIENT_V1
OBSTACLE_FORCE_LAW_VERSIONS = frozenset(
    {LEGACY_SHIFTED_GRADIENT_V1, SURFACE_DISTANCE_UNIT_NORMAL_V2}
)
OBSTACLE_FORCE_LAW_METADATA_SCHEMA = "obstacle_force_law_metadata.v1"


def resolve_obstacle_force_law(value: Any = None) -> str:
    """Resolve an obstacle-force law identifier with legacy compatibility.

    Missing, empty, and unversioned metadata intentionally resolve to the historical
    law.  Unknown explicit identifiers fail closed so a caller cannot silently run a
    different force law than the one recorded in its configuration.

    Args:
        value: Law identifier or a metadata mapping containing ``law_version`` (or
            one of the compatibility keys ``obstacle_force_law_version``,
            ``law_id``, ``obstacle_force_law``, ``social_force_obstacle_law``, or
            ``law``).

    Returns:
        Canonical obstacle-force law identifier.

    Raises:
        TypeError: If an explicit value is not a string.
        ValueError: If an explicit string is not a supported law identifier.
    """
    if isinstance(value, Mapping):
        selected = None
        for key in (
            "law_version",
            "obstacle_force_law_version",
            "law_id",
            "obstacle_force_law",
            "social_force_obstacle_law",
            "law",
        ):
            if key in value:
                selected = value[key]
                break
        value = selected

    if value is None:
        return DEFAULT_OBSTACLE_FORCE_LAW
    if not isinstance(value, str):
        raise TypeError("obstacle-force law version must be a string or None")

    resolved = value.strip()
    if not resolved:
        return DEFAULT_OBSTACLE_FORCE_LAW
    if resolved not in OBSTACLE_FORCE_LAW_VERSIONS:
        supported = ", ".join(sorted(OBSTACLE_FORCE_LAW_VERSIONS))
        raise ValueError(
            f"unsupported obstacle-force law {resolved!r}; expected one of {supported}"
        )
    return resolved


def obstacle_force_law_metadata(
    law_version: Any = None,
    *,
    site: str,
    geometry_convention: str,
    radius_convention: str,
    enabled: bool = True,
    applied: bool | None = None,
) -> dict[str, Any]:
    """Build explicit, JSON-safe metadata for one obstacle-force runtime site.

    The metadata describes implementation dispatch and compatibility only.  It is
    not a benchmark, physical, safety, or social-behavior evidence declaration.

    Args:
        law_version: Explicit law identifier or legacy-compatible metadata input.
        site: Stable runtime site identifier.
        geometry_convention: Site-specific obstacle geometry convention.
        radius_convention: Site-specific radius/offset convention.
        enabled: Whether this runtime site is configured to contribute obstacle force.
        applied: Whether the site evaluated obstacle force for the current runtime.
            When omitted, this follows ``enabled`` for configuration-only callers.

    Returns:
        Versioned metadata mapping suitable for configuration or episode diagnostics.
    """
    resolved = resolve_obstacle_force_law(law_version)
    return {
        "schema_version": OBSTACLE_FORCE_LAW_METADATA_SCHEMA,
        "law_version": resolved,
        "site": site,
        "geometry_convention": geometry_convention,
        "radius_convention": radius_convention,
        "compatibility_mode": (
            "legacy_compatible" if resolved == LEGACY_SHIFTED_GRADIENT_V1 else "corrected_opt_in"
        ),
        "enabled": bool(enabled),
        "applied": bool(enabled if applied is None else applied),
    }


@dataclass
class SceneConfig:
    """Global simulation parameters shared across all force terms.

    Attributes:
        enable_group: Enable group-related forces (coherence, repulsion, gaze).
        agent_radius: Pedestrian radius used for collision/interaction geometry (meters).
        dt_secs: Integration step size in seconds.
        max_speed_multiplier: Upper bound multiplier for desired speeds. Used to
            derive desired speeds from the spawn speed when ``desired_speed_mean``
            is not set (legacy behavior, ~0.65 m/s for default spawning).
        tau: Relaxation time constant used by force models (seconds).
        resolution: Spatial resolution used for obstacle preprocessing.
        integration_scheme: Pedestrian position-update scheme. ``semi_implicit_euler``
            advances position with the newly integrated velocity; ``explicit_euler``
            advances with the pre-step velocity.
        desired_speed_mean: Optional per-pedestrian desired (preferred) walking
            speed mean in m/s, decoupled from the spawn speed (issue #4972). When
            set, each pedestrian's goal-driving speed is drawn from a truncated
            normal distribution ``N(desired_speed_mean, desired_speed_std)`` clipped
            to ``[0, desired_speed_high]`` instead of ``max_speed_multiplier *
            initial_speed``. ``None`` preserves the legacy spawn-coupled default.
        desired_speed_std: Optional standard deviation (m/s) of the desired-speed
            distribution. Ignored unless ``desired_speed_mean`` is set; defaults to
            a small spread when ``desired_speed_mean`` is set without an explicit
            ``desired_speed_std``.
        desired_speed_high: Inclusive upper bound (m/s) for the truncated desired-
            speed distribution. Defaults to a high ceiling so only the non-negative
            side is truncated in practice.
        desired_speed_seed: Optional RNG seed for deterministic desired-speed
            sampling. When ``None`` the global NumPy RNG is used.
    """

    enable_group: bool = True
    agent_radius: float = 0.35
    dt_secs: float = 0.1
    max_speed_multiplier: float = 1.3
    tau: float = 0.5
    resolution: float = 10
    integration_scheme: str = "semi_implicit_euler"
    desired_speed_mean: float | None = None
    desired_speed_std: float | None = None
    desired_speed_high: float = 3.0
    desired_speed_seed: int | None = None


@dataclass
class GroupCoherenceForceConfig:
    """Parameters for attraction that keeps pedestrians within a group together.

    Attributes:
        factor: Scaling factor for group coherence force magnitude.
    """

    factor: float = 3.0


@dataclass
class GroupReplusiveForceConfig:
    """Parameters for short-range repulsion between members of the same group.

    Attributes:
        factor: Scaling factor for intra-group repulsive force.
        threshold: Distance threshold where repulsion becomes active (meters).
    """

    factor: float = 1.0
    threshold: float = 0.55


@dataclass
class GroupGazeForceConfig:
    """Parameters for gaze-alignment force encouraging shared heading.

    Attributes:
        factor: Scaling factor for group gaze force magnitude.
        fov_phi: Field-of-view angle used by gaze interaction logic (degrees).
    """

    factor: float = 4.0
    fov_phi: float = 90.0


@dataclass
class DesiredForceConfig:
    """Parameters for goal-directed acceleration toward target waypoints.

    Attributes:
        factor: Scaling factor for desired force magnitude.
        relaxation_time: Time to relax toward desired velocity (seconds).
        goal_threshold: Distance considered "arrived at goal" (meters).
    """

    factor: float = 1.0
    relaxation_time: float = 0.5
    goal_threshold: float = 0.2


@dataclass
class SocialForceConfig:
    """Parameters for pedestrian-pedestrian interaction (social repulsion).

    Attributes:
        factor: Global scaling factor for social interaction force.
        lambda_importance: Relative weight between velocity and distance terms.
        gamma: Interaction range/smoothing parameter from the SFM formulation.
        n: Exponent shaping angular dependency.
        n_prime: Exponent shaping directional weighting.
        activation_threshold: Max interaction distance for social force (meters).
    """

    factor: float = 5.1
    lambda_importance: float = 2.0
    gamma: float = 0.35
    n: int = 2
    n_prime: int = 3
    activation_threshold: float = 20.0


@dataclass
class ObstacleForceConfig:
    """Parameters for repulsion from static obstacles and map boundaries.

    Attributes:
        factor: Scaling factor for obstacle force magnitude.
        sigma: Additional radius inflation term for obstacle interaction.
        threshold: Additive distance offset (m), subtracted like a radius from
            the pedestrian-obstacle distance. Negative values inflate the
            effective distance and soften near-wall repulsion.
        law_version: Versioned obstacle-force law. Missing historical configuration
            resolves to ``legacy_shifted_gradient_v1``; the corrected law is opt-in.
    """

    factor: float = 10.0
    sigma: float = 0.0
    threshold: float = -0.57
    law_version: str = DEFAULT_OBSTACLE_FORCE_LAW

    def __post_init__(self) -> None:
        """Resolve explicit and unversioned configuration to a supported law."""
        self.law_version = resolve_obstacle_force_law(self.law_version)

    @property
    def obstacle_force_law_version(self) -> str:
        """Return the canonical law identifier using the descriptive alias."""
        return self.law_version

    @obstacle_force_law_version.setter
    def obstacle_force_law_version(self, value: Any) -> None:
        """Set the law through the descriptive compatibility alias."""
        self.law_version = resolve_obstacle_force_law(value)


@dataclass
class SimulatorConfig:
    """Top-level container aggregating all simulator and force configurations.

    This dataclass is passed to the simulator factory and forwarded to
    scene setup, force construction, and pedestrian spawn initialization.
    """

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
