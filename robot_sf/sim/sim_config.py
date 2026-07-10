"""Configuration dataclasses for simulator timing and pedestrian behavior."""

from dataclasses import dataclass, field, replace
from math import ceil, isfinite
from typing import Any

from pysocialforce.scene import normalize_integration_scheme

from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ALIGNMENT_TORQUE_V1,
    HSFM_ANISOTROPIC_FOV_V1,
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


@dataclass(frozen=True)
class AnisotropicFovConfig:
    """Opt-in anisotropic field-of-view attenuation parameters."""

    enabled: bool = False
    cone_half_angle_rad: float = 1.5707963267948966
    rear_weight: float = 0.1

    def __post_init__(self) -> None:
        """Validate finite bounded FoV parameters."""
        if (
            not isfinite(self.cone_half_angle_rad)
            or not 0 <= self.cone_half_angle_rad <= 3.141592653589793
        ):
            raise ValueError("anisotropic_fov.cone_half_angle_rad must be within [0, pi]")
        if not isfinite(self.rear_weight) or not 0 <= self.rear_weight <= 1:
            raise ValueError("anisotropic_fov.rear_weight must be within [0, 1]")


def _normalize_anisotropic_fov_config(
    value: AnisotropicFovConfig | dict,
) -> AnisotropicFovConfig:
    """Return validated anisotropic FoV config."""
    if isinstance(value, AnisotropicFovConfig):
        return value
    if isinstance(value, dict):
        return AnisotropicFovConfig(**value)
    raise ValueError("anisotropic_fov must be an AnisotropicFovConfig or dict")


def _pedestrian_model_anisotropic_fov_config(
    pedestrian_model: str,
    fov_config: AnisotropicFovConfig,
) -> AnisotropicFovConfig:
    """Enable anisotropic FoV provenance when selected via pedestrian model.

    Returns:
        Updated FoV config with ``enabled=True`` when the selector activates it.
    """
    if pedestrian_model == HSFM_ANISOTROPIC_FOV_V1 and not fov_config.enabled:
        return replace(fov_config, enabled=True)
    return fov_config


@dataclass(frozen=True)
class AlignmentTorqueConfig:
    """Opt-in HSFM body-orientation alignment-torque parameters.

    The alignment torque decouples pedestrian body orientation ``phi`` from the instantaneous
    force/velocity direction: instead of snapping to the desired direction each step, ``phi``
    is driven toward it by a damped second-order torque (``k_theta`` stiffness, ``k_omega``
    damping) with a bounded angular speed. Critical damping is ``k_omega = 2 * sqrt(k_theta)``.
    """

    enabled: bool = False
    k_theta: float = 4.0
    k_omega: float = 4.0
    max_angular_speed_rad_s: float = 3.141592653589793

    def __post_init__(self) -> None:
        """Validate finite positive alignment-torque parameters."""
        if not isfinite(self.k_theta) or self.k_theta <= 0:
            raise ValueError("alignment_torque.k_theta must be > 0")
        if not isfinite(self.k_omega) or self.k_omega < 0:
            raise ValueError("alignment_torque.k_omega must be >= 0")
        if not isfinite(self.max_angular_speed_rad_s) or self.max_angular_speed_rad_s <= 0:
            raise ValueError("alignment_torque.max_angular_speed_rad_s must be > 0")


def _normalize_alignment_torque_config(
    value: AlignmentTorqueConfig | dict,
) -> AlignmentTorqueConfig:
    """Return a validated alignment-torque config."""
    if isinstance(value, AlignmentTorqueConfig):
        return value
    if isinstance(value, dict):
        return AlignmentTorqueConfig(**value)
    raise ValueError("alignment_torque must be an AlignmentTorqueConfig or dict")


def _pedestrian_model_alignment_torque_config(
    pedestrian_model: str,
    alignment_config: AlignmentTorqueConfig,
) -> AlignmentTorqueConfig:
    """Enable alignment-torque provenance when the selector is active.

    Returns:
        Updated config with ``enabled=True`` when the selector activates it.
    """
    if pedestrian_model == HSFM_ALIGNMENT_TORQUE_V1 and not alignment_config.enabled:
        return replace(alignment_config, enabled=True)
    return alignment_config


@dataclass
class SimulationSettings:
    """
    Configuration settings for the simulation.
    """

    sim_time_in_secs: float = 200.0
    """Simulation time in seconds"""

    time_per_step_in_secs: float = 0.1
    """Time per step in seconds"""

    action_latency_steps: int = 0
    """Discrete control-to-actuation delay; zero preserves immediate action execution."""

    action_latency_ms: float | None = None
    """Optional requested control-to-actuation delay in milliseconds.

    The simulator executes only whole steps. When this field is set, it takes
    precedence over ``action_latency_steps`` and is rounded up to the first
    whole step that does not understate the requested delay.
    """

    pedestrian_integration_scheme: str = "semi_implicit_euler"
    """Pedestrian position-update scheme; defaults to the historical semi-implicit update."""

    peds_speed_mult: float = 1.3
    """Pedestrian speed multiplier"""

    pedestrian_model: str = "social_force_default"
    """Pedestrian dynamics model selector."""

    ttc_predictive_force: TtcPredictiveForceConfig = field(default_factory=TtcPredictiveForceConfig)
    """TTC predictive force settings used by ``hsfm_ttc_predictive_v1``."""

    anisotropic_fov: AnisotropicFovConfig = field(default_factory=AnisotropicFovConfig)
    """Anisotropic field-of-view settings used by ``hsfm_anisotropic_fov_v1``."""

    alignment_torque: AlignmentTorqueConfig = field(default_factory=AlignmentTorqueConfig)
    """Body-orientation alignment-torque settings used by ``hsfm_alignment_torque_v1``."""

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
    response_law_composition: dict[str, float] | None = None
    """Optional pedestrian response-law composition for route/crowd spawning."""
    response_law_seed: int | None = None
    """Optional RNG seed for deterministic response-law assignment."""
    population_size: int | None = None
    """Optional override to force exact population size spawned (issue #3574)."""
    pedestrian_control_trace_labels: list[dict[str, Any]] | None = None
    """Optional per-pedestrian response-law labels (issue #3574).

    Each entry maps a ``simulator_index`` to a ``response_law`` (e.g. ``non_yielding``)
    consumed by the simulator to build ``PedRobotForce`` response multipliers. Declared
    as a first-class field (issue #4618 R3) instead of being attached dynamically so the
    attribute is type-checked and discoverable on ``SimulationSettings``.
    """

    non_reactive_response_multiplier: float = 0.0
    """Response multiplier for non-reactive/non-yielding pedestrians (issue #4850).

    This knob controls the strength of pedestrian-robot repulsion for pedestrians
    configured with response_law='non_reactive' or 'non_yielding'. The default value
    of 0.0 preserves the existing semantics where such pedestrians do not respond
    to the robot at all.
    """

    peds_reset_follow_route_at_start: bool = False
    """Whether pedestrians following routes should reset to the start of their routes"""
    debug_without_robot_movement: bool = False
    """Whether to disable robot movement in the simulator for debugging purposes"""

    @property
    def resolved_action_latency_steps(self) -> int:
        """Return the whole-step delay enforced by the environment action queue."""
        if self.action_latency_ms is None:
            return int(self.action_latency_steps)
        step_ms = float(self.time_per_step_in_secs) * 1000.0
        return ceil(float(self.action_latency_ms) / step_ms)

    def action_latency_metadata(self) -> dict[str, int | float | None]:
        """Return JSON-safe requested and effective action-latency settings."""
        effective_steps = self.resolved_action_latency_steps
        return {
            "configured_steps": int(self.action_latency_steps),
            "configured_ms": (
                None if self.action_latency_ms is None else float(self.action_latency_ms)
            ),
            "effective_steps": effective_steps,
            "effective_ms": round(
                float(effective_steps * self.time_per_step_in_secs * 1000.0),
                12,
            ),
        }

    def _validate_action_latency_config(self) -> None:
        """Reject ambiguous or non-realizable action-latency configuration values."""
        if isinstance(self.action_latency_steps, bool) or not isinstance(
            self.action_latency_steps,
            int,
        ):
            raise TypeError("action_latency_steps must be an integer")
        if self.action_latency_steps < 0:
            raise ValueError("action_latency_steps must be >= 0")
        if self.action_latency_ms is None:
            return
        if isinstance(self.action_latency_ms, bool) or not isinstance(
            self.action_latency_ms,
            int | float,
        ):
            raise TypeError("action_latency_ms must be a number when set")
        if not isfinite(float(self.action_latency_ms)) or self.action_latency_ms < 0:
            raise ValueError("action_latency_ms must be finite and >= 0 when set")
        if self.action_latency_steps != 0:
            raise ValueError("action_latency_steps and action_latency_ms cannot both be configured")

    def __post_init__(self):  # noqa: C901
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
        self._validate_action_latency_config()
        self.pedestrian_integration_scheme = normalize_integration_scheme(
            self.pedestrian_integration_scheme
        )
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
        self.anisotropic_fov = _normalize_anisotropic_fov_config(self.anisotropic_fov)
        self.anisotropic_fov = _pedestrian_model_anisotropic_fov_config(
            self.pedestrian_model,
            self.anisotropic_fov,
        )
        self.alignment_torque = _normalize_alignment_torque_config(self.alignment_torque)
        self.alignment_torque = _pedestrian_model_alignment_torque_config(
            self.pedestrian_model,
            self.alignment_torque,
        )
        # Check that the maximum number of pedestrians per group is positive
        if self.max_peds_per_group <= 0:
            raise ValueError("Maximum pedestrians per group mustn't be negative or zero!")
        # Check that the pedestrian radius is positive
        if self.ped_radius <= 0:
            raise ValueError("Pedestrian radius mustn't be negative or zero!")
        self._validate_pedestrian_uncertainty_envelope_config()
        # Check that the non-reactive response multiplier is finite and >= 0
        if (
            not isfinite(self.non_reactive_response_multiplier)
            or self.non_reactive_response_multiplier < 0
        ):
            raise ValueError("non_reactive_response_multiplier must be a finite value >= 0!")
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
