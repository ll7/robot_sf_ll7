"""Configuration dataclasses for simulator timing and pedestrian behavior."""

from dataclasses import dataclass, field, replace
from math import ceil, isfinite

from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_TTC_PREDICTIVE_V1,
    normalize_pedestrian_model,
)


@dataclass(frozen=True)
class TtcPredictiveForceConfig:
    """Opt-in TTC predictive pedestrian force parameters."""

    enabled: bool = False
    tau0_s: float = 1.0
    horizon_s: float = 3.0
    force_scale: float = 1.0
    max_force: float = 5.0
    include_ped_ped: bool = True
    include_robot_proxy: bool = False

    def __post_init__(self) -> None:
        """Validate finite positive predictive-force parameters."""
        if not isfinite(self.tau0_s) or self.tau0_s <= 0:
            raise ValueError("ttc_predictive_force.tau0_s must be > 0")
        if not isfinite(self.horizon_s) or self.horizon_s <= 0:
            raise ValueError("ttc_predictive_force.horizon_s must be > 0")
        if not isfinite(self.force_scale) or self.force_scale < 0:
            raise ValueError("ttc_predictive_force.force_scale must be >= 0")
        if not isfinite(self.max_force) or self.max_force <= 0:
            raise ValueError("ttc_predictive_force.max_force must be > 0")


def _normalize_ttc_predictive_force_config(
    value: TtcPredictiveForceConfig | dict,
) -> TtcPredictiveForceConfig:
    """Return a validated TTC predictive force config."""
    if isinstance(value, TtcPredictiveForceConfig):
        return value
    if isinstance(value, dict):
        return TtcPredictiveForceConfig(**value)
    raise ValueError("ttc_predictive_force must be a TtcPredictiveForceConfig")


def _pedestrian_model_ttc_config(
    pedestrian_model: str,
    ttc_config: TtcPredictiveForceConfig,
) -> TtcPredictiveForceConfig:
    """Enable TTC provenance when the predictive pedestrian-model selector is active.

    Returns:
        The original or selector-adjusted TTC predictive force config.
    """
    if pedestrian_model == HSFM_TTC_PREDICTIVE_V1 and not ttc_config.enabled:
        return replace(ttc_config, enabled=True)
    return ttc_config


@dataclass
class SimulationSettings:
    """
    Configuration settings for the simulation.
    """

    sim_time_in_secs: float = 200.0
    """Simulation time in seconds"""

    time_per_step_in_secs: float = 0.1
    """Time per step in seconds"""

    peds_speed_mult: float = 1.3
    """Pedestrian speed multiplier"""

    pedestrian_model: str = "social_force_default"
    """Pedestrian dynamics model selector."""

    ttc_predictive_force: TtcPredictiveForceConfig = field(default_factory=TtcPredictiveForceConfig)
    """TTC predictive force settings used by ``hsfm_ttc_predictive_v1``."""

    difficulty: int = 0
    """Difficulty level"""

    max_peds_per_group: int = 3
    """Maximum number of pedestrians per group"""

    ped_radius: float = 0.4
    """Pedestrian radius"""

    pedestrian_uncertainty_envelope_enabled: bool = False
    """Whether planner configs should opt into horizon-dependent pedestrian inflation."""

    pedestrian_uncertainty_alpha_mps: float = 0.0
    """Linear pedestrian-envelope inflation rate in metres per second."""

    goal_radius: float = 1.0
    """Goal radius"""

    stack_steps: int = 3
    """Deprecated alias for observation history depth.

    Prefer ``observation_stack.stack_steps`` on environment configuration objects.
    """

    use_next_goal: bool = True
    """Whether to use the next goal in the path as the current goal"""

    prf_config: PedRobotForceConfig = field(default_factory=PedRobotForceConfig)
    """Pedestrian-robot force configuration"""

    apf_config: AdversarialPedForceConfig = field(default_factory=AdversarialPedForceConfig)
    """Adversarial pedestrian force configuration"""

    ped_density_by_difficulty: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.04, 0.08])
    """Pedestrian density by difficulty level"""
    max_total_pedestrians: int | None = None
    """Optional upper bound for pedestrians used to size SocNav structured observations."""
    route_spawn_distribution: str = "cluster"
    """Route pedestrian spawn distribution: "cluster" (default) or "spread"."""
    route_spawn_jitter_frac: float = 0.0
    """Fraction of spacing used as jitter when route_spawn_distribution='spread'."""
    route_spawn_seed: int | None = None
    """Optional RNG seed for route spawn placement/jitter."""

    archetype_composition: dict[str, float] | None = None
    """Optional pedestrian behavior-archetype composition for route/crowd spawning."""
    archetype_speed_factors: dict[str, float] | None = None
    """Optional archetype desired-speed factors used with ``archetype_composition``."""
    archetype_seed: int | None = None
    """Optional RNG seed for deterministic archetype assignment."""

    peds_reset_follow_route_at_start: bool = False
    """Whether pedestrians following routes should reset to the start of their routes"""
    debug_without_robot_movement: bool = False
    """Whether to disable robot movement in the simulator for debugging purposes"""

    def __post_init__(self):
        """
        Validate the simulation settings.

        This method is called after the object is initialized. It checks that all the
        settings are valid and raises a ValueError if any of them are not.
        """
        # Check that the simulation time is positive
        if self.sim_time_in_secs <= 0:
            raise ValueError("Simulation length for episodes mustn't be negative or zero!")
        # Check that the time per step is positive
        if self.time_per_step_in_secs <= 0:
            raise ValueError("Step time mustn't be negative or zero!")
        # Check that the pedestrian speed multiplier is positive
        if self.peds_speed_mult <= 0:
            raise ValueError("Pedestrian speed mustn't be negative or zero!")
        self.pedestrian_model = normalize_pedestrian_model(self.pedestrian_model)
        self.ttc_predictive_force = _normalize_ttc_predictive_force_config(
            self.ttc_predictive_force
        )
        self.ttc_predictive_force = _pedestrian_model_ttc_config(
            self.pedestrian_model,
            self.ttc_predictive_force,
        )
        # Check that the maximum number of pedestrians per group is positive
        if self.max_peds_per_group <= 0:
            raise ValueError("Maximum pedestrians per group mustn't be negative or zero!")
        # Check that the pedestrian radius is positive
        if self.ped_radius <= 0:
            raise ValueError("Pedestrian radius mustn't be negative or zero!")
        self._validate_pedestrian_uncertainty_envelope_config()
        # Check that the goal radius is positive
        if self.goal_radius <= 0:
            raise ValueError("Goal radius mustn't be negative or zero!")
        # Check that the difficulty level is within the valid range
        if not 0 <= self.difficulty < len(self.ped_density_by_difficulty):
            raise ValueError("No pedestrian density registered for selected difficulty level!")
        # Check that the pedestrian-robot force configuration is specified
        if not self.prf_config:
            raise ValueError("Pedestrian-Robot-Force settings need to be specified!")
        if self.apf_config is None or not isinstance(self.apf_config, AdversarialPedForceConfig):
            raise ValueError("Adversarial-ped-force settings need to be specified!")
        self._validate_route_spawn_config()

    def _validate_pedestrian_uncertainty_envelope_config(self) -> None:
        """Validate planner-facing uncertainty-envelope simulation settings."""
        if self.pedestrian_uncertainty_alpha_mps < 0:
            raise ValueError("pedestrian_uncertainty_alpha_mps must be >= 0")

    def _validate_route_spawn_config(self) -> None:
        """Validate route spawn configuration flags."""
        if self.route_spawn_distribution not in {"cluster", "spread"}:
            raise ValueError(
                "route_spawn_distribution must be 'cluster' or 'spread' (got "
                f"{self.route_spawn_distribution!r})"
            )
        if self.route_spawn_jitter_frac < 0:
            raise ValueError("route_spawn_jitter_frac must be >= 0")

    @property
    def max_sim_steps(self) -> int:
        """Return the maximum number of fixed-time simulation steps.


        Returns:
            Ceiling of episode duration divided by step duration.
        """
        return ceil(self.sim_time_in_secs / self.time_per_step_in_secs)

    @property
    def peds_per_area_m2(self) -> float:
        """Return the pedestrian density for the configured difficulty.


        Returns:
            Pedestrians per square meter selected from ``ped_density_by_difficulty``.
        """
        return self.ped_density_by_difficulty[self.difficulty]
