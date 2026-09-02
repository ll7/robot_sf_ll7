"""Robot and pedestrian simulation management.

This module provides the core simulation infrastructure for managing robots,
pedestrians, and their interactions in a shared environment. It integrates the
PySocialForce physics engine for pedestrian dynamics and supports both robot-only
and pedestrian-robot interaction scenarios.

Key Components:
    - Simulator: Base simulation engine managing robots, pedestrian physics,
      and navigation waypoints.
    - PedSimulator: Extended simulator with ego pedestrian (robot-as-pedestrian)
      for pedestrian-centric environments.
    - init_simulators: Factory for creating robot-only simulator instances.
    - init_ped_simulators: Factory for creating pedestrian simulator instances.

Example:
    >>> from robot_sf.gym_env.unified_config import RobotSimulationConfig
    >>> from robot_sf.nav.svg_map_parser import load_svg_maps
    >>> config = RobotSimulationConfig()
    >>> maps = load_svg_maps("maps/svg_maps/")
    >>> sims = init_simulators(config, maps["hallway"], num_robots=2)
    >>> for sim in sims:
    ...     sim.step_once([action1, action2])"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import atan2, ceil, cos, isfinite, pi, sin
from random import sample, uniform
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from pysocialforce import Simulator as PySFSimulator
from pysocialforce.config import SimulatorConfig as PySFSimConfig
from pysocialforce.config import obstacle_force_law_metadata
from pysocialforce.force_trace import (
    ForceComputationResult,
    annotate_force_component,
)
from pysocialforce.forces import Force as PySFForce
from pysocialforce.forces import ObstacleForce, SocialForce
from pysocialforce.simulator import make_forces as pysf_make_forces

if TYPE_CHECKING:
    from collections.abc import Callable

    from pysocialforce.scene import PedestrianStepDiagnostics

    from robot_sf.common.types import Line2D, PedPose, RobotAction, RobotPose, Vec2D
    from robot_sf.gym_env.env_config import (
        EnvSettings,
        PedEnvSettings,
        SimulationSettings,
    )
    from robot_sf.gym_env.unified_config import (
        PedestrianSimulationConfig,
        RobotSimulationConfig,
    )
    from robot_sf.ped_ego.unicycle_drive import UnicycleAction, UnicycleDrivePedestrian
    from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
    from robot_sf.robot.robot_state import Robot

from robot_sf.nav.map_config import MapDefinition, SocialGroupDefinition
from robot_sf.nav.navigation import RouteNavigator, get_prepared_obstacles, sample_route
from robot_sf.nav.occupancy import circle_collides_any_lines
from robot_sf.ped_npc.adversial_ped_force import (
    AdversarialPedForce,
    AdversarialPedForceConfig,
)
from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels
from robot_sf.ped_npc.ped_behavior import (
    FollowRouteBehavior,
    PedestrianBehavior,
    SinglePedestrianBehavior,
)
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation
from robot_sf.ped_npc.ped_robot_force import PedRobotForce, PedRobotForceConfig
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.ped_npc.residual_adversary import (
    BoundedResidualAdversary,
    ResidualAdversaryBehaviorSummary,
    ResidualAdversaryConfig,
    build_default_residual_adversary,
)
from robot_sf.prediction.oracle_transition_trace import (
    ORACLE_TRANSITION_TRACE_SCHEMA_VERSION,
    ControllerMutationFlags,
    DynamicsParameters,
    ExactInverseReason,
    ForceComponentOperationKind,
    ForceComponentRecord,
    ForceComponents,
    ForceOperationKind,
    ForceStageResult,
    ForceTimeRobotState,
    GoalChangeKind,
    OracleTransitionTraceV1,
    RobotForceState,
    SpeedCap,
    SpeedCapStatus,
    TransitionBoundary,
    TransitionBoundaryKind,
)
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ALIGNMENT_TORQUE_V1,
    HSFM_ANISOTROPIC_FOV_V1,
    HSFM_TOTAL_FORCE_V1,
    HSFM_TTC_PREDICTIVE_V1,
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
    PYSF_POSITION_SLICE,
    PYSF_VELOCITY_SLICE,
    fov_attenuated_total_force,
    normalize_pedestrian_model,
    pairwise_social_force_contributions,
    step_alignment_torque_heading,
    step_hsfm_total_force,
    ttc_predictive_repulsion,
    zanlungo_collision_prediction_repulsion,
)
from robot_sf.sim.pedestrian_speed_tiers import sample_desired_pedestrian_speeds

PYSF_TAU_INDEX = 6
MIN_HEADING_SPEED_MPS = 1e-6


@dataclass(frozen=True, slots=True)
class _ForceStageArrays:
    """Internal per-pedestrian force-stage arrays captured for one transition."""

    operation_kind: ForceOperationKind
    operation: str | None
    delta_force_xy: np.ndarray | None
    result_force_xy: np.ndarray | None


def _row_xy(values: np.ndarray, row: int) -> tuple[float, float]:
    """Return one finite two-dimensional row as a typed coordinate tuple.

    Returns:
        The selected row's ``(x, y)`` values.
    """
    return (float(values[row, 0]), float(values[row, 1]))


def _heading_from_velocity(velocity_xy: np.ndarray, fallback_heading: float) -> float:
    """Derive a heading from a PySF velocity vector or keep the fallback heading.

    Returns:
        float: Heading angle in radians.
    """
    if float(np.linalg.norm(velocity_xy)) <= MIN_HEADING_SPEED_MPS:
        return fallback_heading
    return float(atan2(float(velocity_xy[1]), float(velocity_xy[0])))


def _apply_ped_desired_speed_config(
    pysf_config: PySFSimConfig, settings: SimulationSettings
) -> None:
    """Propagate decoupled desired-speed settings onto the PySF scene config.

    Forwards the optional ``desired_speed_mean``/``desired_speed_std``/``desired_speed_seed``
    from :class:`SimulationSettings` (which may have been derived from ``ped_speed_tier``)
    onto ``pysf_config.scene_config`` so :class:`pysocialforce.scene.PedState` samples a
    decoupled preferred walking speed instead of ``peds_speed_mult * initial_speed``
    (issue #4972). A ``None`` mean preserves the legacy spawn-coupled default.
    """
    pysf_config.scene_config.desired_speed_mean = settings.desired_speed_mean
    pysf_config.scene_config.desired_speed_std = settings.desired_speed_std
    pysf_config.scene_config.desired_speed_seed = settings.desired_speed_seed


def _enforce_ped_desired_speeds(peds, settings: SimulationSettings) -> None:
    """Apply decoupled desired speeds directly to a PedState after simulator creation.

    ``_apply_ped_desired_speed_config`` propagates settings into the PySF config so
    that ``PedState.__init__`` can sample them. However, older installed pysf versions
    (< the fast-pysf update in #5042) do not read ``desired_speed_mean`` from the scene
    config and silently fall back to the spawn-coupled ``max_speed_multiplier *
    initial_speed`` = 0.65 m/s default (issue #5217).

    This function re-applies the desired speeds directly on the ``PedState`` object
    after ``PySFSimulator`` is constructed, which restores the tier contract for stale
    pysf installs. It is a no-op when the new pysf already applied the speeds via
    ``assign_desired_speeds`` (the explicit-desired-speeds path uses ``_explicit_desired_speeds``
    and ignores the ``initial_speeds`` override we write here).

    Calling this after ``max_speed_multiplier`` is set is intentional: we encode the
    desired speeds into ``initial_speeds`` so the legacy recomputation
    ``max_speeds = max_speed_multiplier * initial_speeds`` also yields the correct values
    on every subsequent ``_update_state`` call.
    """
    if settings.desired_speed_mean is None:
        return
    desired_speeds = sample_desired_pedestrian_speeds(
        peds.size(),
        mean=settings.desired_speed_mean,
        std=settings.desired_speed_std,
        seed=settings.desired_speed_seed,
    )
    # Direct assignment: works with both new pysf (``assign_desired_speeds`` already ran,
    # this overwrites with identical values) and old pysf (no explicit-speed support).
    peds.max_speeds = desired_speeds.copy()
    # Compat: encode into initial_speeds so legacy _update_state recomputation preserves them.
    initial_speeds = getattr(peds, "initial_speeds", None)
    if initial_speeds is not None and peds.max_speed_multiplier > 0:
        peds.initial_speeds = desired_speeds / peds.max_speed_multiplier


def _make_ped_forces(
    sim: PySFSimulator,
    config: PySFSimConfig,
    robots: list[Robot],
    peds_have_obstacle_forces: bool,
    prf_config: PedRobotForceConfig,
    apf_config: AdversarialPedForceConfig,
    pedestrian_response_multipliers: np.ndarray | None = None,
) -> list[PySFForce]:
    """Configure pedestrian forces for the physics engine.

    Creates default SocialForce forces, optionally filters obstacle forces,
    and adds pedestrian-robot interaction forces if enabled.

    Args:
        sim: PySocialForce simulator instance.
        config: PySocialForce configuration object.
        robots: Active robots used for interaction forces.
        peds_have_obstacle_forces: Whether to keep obstacle forces.
        prf_config: Ped-robot force config (repulsion).
        apf_config: Adversarial ped force config.
        pedestrian_response_multipliers: Optional array of response multipliers.

    Returns:
        List of Force objects including social, goal attraction, obstacle
        (conditional), and pedestrian-robot interaction (conditional) forces.
    """
    forces = pysf_make_forces(sim, config)

    if peds_have_obstacle_forces is False:
        logger.info("Peds have no obstacle forces.")
        forces = [f for f in forces if not isinstance(f, ObstacleForce)]

    if prf_config.is_active:
        for robot_index, robot in enumerate(robots):
            robot_prf_config = replace(prf_config, robot_radius=robot.config.radius)

            def robot_position_provider(current_robot: Robot = robot) -> Vec2D:
                """Return the current position for one robot force callback."""
                return current_robot.pos

            def response_multiplier_provider() -> np.ndarray | None:
                """Return the current optional response-law multipliers."""
                return pedestrian_response_multipliers

            ped_robot_force = PedRobotForce(
                robot_prf_config,
                sim.peds,
                robot_position_provider,
                get_ped_response_multipliers=response_multiplier_provider,
            )
            forces.append(
                annotate_force_component(
                    ped_robot_force,
                    component_id=f"ped_robot:robot_{robot_index}",
                    component_type="pedestrian_robot",
                    source_entity=f"robot:{robot_index}",
                )
            )

    if apf_config.is_active:
        for robot_index, robot in enumerate(robots):
            robot_apf_config = replace(apf_config, robot_radius=robot.config.radius)

            def robot_pose_provider(current_robot: Robot = robot) -> RobotPose:
                """Return the current pose for one adversarial force callback."""
                return current_robot.pose

            adversarial_force = AdversarialPedForce(
                robot_apf_config,
                sim.peds,
                robot_pose_provider,
            )
            forces.append(
                annotate_force_component(
                    adversarial_force,
                    component_id=f"adversarial:robot_{robot_index}",
                    component_type="adversarial",
                    source_entity=f"robot:{robot_index}",
                )
            )

    return forces


def _compute_pedestrian_response_multipliers(
    config: SimulationSettings, num_peds: int
) -> np.ndarray:
    """Compute per-pedestrian response multipliers for PedRobotForce (issue #3574).

    Extracted from ``Simulator.__post_init__`` (issue #6465) so the shared pysf
    initialization factory can own the multiplier wiring without duplicating the
    branching logic. Returns an all-ones vector when neither control-trace labels
    nor a response-law composition are configured, which is byte-identical to the
    prior homogeneous default.

    Args:
        config: Simulation settings carrying the optional control-trace labels and
            response-law ablation knobs (``non_reactive_response_multiplier``,
            ``hesitating_response_multiplier``).
        num_peds: Number of pedestrian rows in the PySocialForce state.

    Returns:
        Per-pedestrian scaling factors consumed by :class:`PedRobotForce`.
    """
    multipliers = np.ones(num_peds, dtype=float)

    labels = getattr(config, "pedestrian_control_trace_labels", None)
    if labels and num_peds > 0:
        for label in labels:
            sim_idx = label.get("simulator_index")
            resp_law = label.get("response_law")
            if sim_idx is not None and 0 <= sim_idx < num_peds:
                if resp_law in ("non_reactive", "non_yielding"):
                    multipliers[sim_idx] = config.non_reactive_response_multiplier
                elif resp_law == "hesitating":
                    multipliers[sim_idx] = config.hesitating_response_multiplier
    elif getattr(config, "response_law_composition", None) and num_peds > 0:
        response_laws = assign_archetype_labels(
            num_peds,
            config.response_law_composition,
            seed=config.response_law_seed,
        )
        for idx, law in enumerate(response_laws):
            if law in ("non_reactive", "non_yielding"):
                multipliers[idx] = config.non_reactive_response_multiplier
            elif law == "hesitating":
                multipliers[idx] = config.hesitating_response_multiplier

    return multipliers


def _build_pysf_simulation(  # noqa: PLR0913
    *,
    config: SimulationSettings,
    map_def: MapDefinition,
    robots: list[Robot],
    robot_pose_provider: Callable[[], list[RobotPose]],
    peds_have_obstacle_forces: bool,
    add_ego_state: bool = False,
    include_response_law_multipliers: bool = True,
    response_law_composition: dict[str, float] | None = None,
    response_law_seed: int | None = None,
    force_population_size: int | None = None,
) -> tuple[
    PySFSimulator,
    PedestrianStates,
    PedestrianGroupings,
    list[PedestrianBehavior],
    np.ndarray | None,
]:
    """Build the shared PySocialForce simulator and pedestrian state (issue #6465).

    Owns the PySocialForce initialization previously duplicated between
    ``Simulator.__post_init__`` and ``PedSimulator.__post_init__``: the PySF scene
    config, the :class:`PedSpawnConfig`, the :func:`populate_simulation` call, the
    single-pedestrian robot-pose-provider wiring, the per-pedestrian response
    multipliers, and the :class:`PySFSimulator` construction (including the
    max-speed and desired-speed propagation).

    The behavior-preserving divergence (issue #4618 R2) is kept explicit at the call
    site rather than unified:

    - :class:`Simulator` forwards its ``response_law_composition`` /
      ``response_law_seed`` / ``force_population_size`` into :class:`PedSpawnConfig`
      and requests computed response multipliers
      (``include_response_law_multipliers=True``). The
      ``peds_have_obstacle_forces is None`` warning-and-default guard runs in
      ``Simulator.__post_init__`` before this factory is called, so callers must pass
      a resolved boolean.
    - :class:`PedSimulator` intentionally OMITS the three response-law spawn fields
      (they default to ``None``) and sets ``include_response_law_multipliers=False``
      so the per-pedestrian multiplier vector is ``None`` and :class:`PedRobotForce`
      falls back to unscaled robot repulsion. The heterogeneous-population ablation
      targets the robot-only benchmark simulator; the appended ego-pedestrian row
      would otherwise misalign the per-pedestrian multiplier vector.

    Args:
        config: Simulation settings (timestep, density, forces, response-law knobs).
        map_def: Map definition with obstacles, spawn zones, and routes.
        robots: Active robots used for interaction forces and reserved-zone sizing.
        robot_pose_provider: Dynamic provider for the caller's current robot poses.
            Keeping this callback separate from ``robots`` preserves the original
            ``lambda: self.robot_poses`` behavior if the public robot collection is
            replaced after construction.
        peds_have_obstacle_forces: Resolved flag controlling whether pedestrians
            experience obstacle collision forces. Callers must run the
            ``None``-warning guard (:class:`Simulator`) before calling.
        add_ego_state: When True, append an ego-pedestrian state row
            (:class:`PedSimulator` only).
        include_response_law_multipliers: When True, compute per-pedestrian response
            multipliers (:class:`Simulator`); when False, return ``None``
            (:class:`PedSimulator`, issue #4618 R2).
        response_law_composition: Optional spawn-config response-law composition,
            forwarded to :class:`PedSpawnConfig` only when provided.
        response_law_seed: Optional spawn-config response-law seed.
        force_population_size: Optional exact pedestrian count for spawn config.

    Returns:
        Tuple of ``(pysf_sim, pysf_state, groups, peds_behaviors,
        pedestrian_response_multipliers)`` for the caller to assign to its instance.
    """
    pysf_config = PySFSimConfig()
    pysf_config.scene_config.dt_secs = config.time_per_step_in_secs
    pysf_config.scene_config.integration_scheme = config.pedestrian_integration_scheme
    pysf_config.obstacle_force_config.law_version = getattr(config, "obstacle_force_law", None)
    _apply_ped_desired_speed_config(pysf_config, config)
    spawn_config = PedSpawnConfig(
        config.peds_per_area_m2,
        config.max_peds_per_group,
        route_spawn_distribution=config.route_spawn_distribution,
        route_spawn_jitter_frac=config.route_spawn_jitter_frac,
        route_spawn_seed=config.route_spawn_seed,
        reset_follow_route_at_start=config.peds_reset_follow_route_at_start,
        archetype_composition=config.archetype_composition,
        archetype_speed_factors=config.archetype_speed_factors,
        archetype_seed=config.archetype_seed,
        response_law_composition=response_law_composition,
        response_law_seed=response_law_seed,
        force_population_size=force_population_size,
    )
    pysf_state, groups, peds_behaviors = populate_simulation(
        pysf_config.scene_config.tau,
        spawn_config,
        map_def.ped_routes,
        map_def.ped_crowded_zones,
        obstacle_polygons=get_prepared_obstacles(map_def),
        single_pedestrians=map_def.single_pedestrians,
        time_step_s=config.time_per_step_in_secs,
        single_ped_goal_threshold=pysf_config.desired_force_config.goal_threshold,
        add_ego_state=add_ego_state,
        map_bounds=map_def.get_map_bounds(),
        reserved_zones=[*map_def.robot_spawn_zones, *map_def.robot_goal_zones],
        ped_radius=config.ped_radius,
        reserved_zone_radius=max(
            (float(robot.config.radius) for robot in robots),
            default=0.0,
        ),
    )
    for behavior in peds_behaviors:
        if isinstance(behavior, SinglePedestrianBehavior):
            behavior.set_robot_pose_provider(robot_pose_provider)

    if include_response_law_multipliers:
        num_peds = pysf_state.pysf_states().shape[0]
        pedestrian_response_multipliers = _compute_pedestrian_response_multipliers(config, num_peds)
    else:
        # issue #4618 R2: PedSimulator keeps an unscaled PedRobotForce (None multipliers).
        pedestrian_response_multipliers = None

    pysf_sim = PySFSimulator(
        pysf_state.pysf_states(),
        groups.groups_as_lists,
        map_def.obstacles_pysf,
        config=pysf_config,
        make_forces=lambda sim, sf_config: _make_ped_forces(
            sim,
            sf_config,
            robots,
            peds_have_obstacle_forces,
            config.prf_config,
            config.apf_config,
            pedestrian_response_multipliers,
        ),
    )
    pysf_sim.peds.max_speed_multiplier = config.peds_speed_mult
    _enforce_ped_desired_speeds(pysf_sim.peds, config)

    return pysf_sim, pysf_state, groups, peds_behaviors, pedestrian_response_multipliers


@dataclass
class Simulator:
    """Manages robot and pedestrian simulation in a shared environment.

    Coordinates robot navigation, pedestrian dynamics via PySocialForce,
    collision detection, and timestep synchronization. Automatically initializes
    pedestrian spawn locations, behaviors, and navigation routes on creation.

    Attributes:
        config: Simulation settings (timestep, pedestrian density, forces).
        map_def: Map definition with obstacles, spawn zones, routes.
        robots: List of Robot instances in the environment.
        goal_proximity_threshold: Distance threshold for waypoint arrival (robot radius + goal radius).
        random_start_pos: If True, robots spawn at random valid positions; else, assigned positions.
        robot_navs: (init=False) RouteNavigator instances tracking waypoints.
        pysf_sim: (init=False) PySocialForce simulator managing pedestrian physics.
        pysf_state: (init=False) Pedestrian state snapshots (positions, velocities).
        groups: (init=False) Pedestrian group assignments for crowd behavior.
        peds_behaviors: (init=False) Behavior instances (goal selection, group dynamics).
        peds_have_obstacle_forces: Enable pedestrian-obstacle collision forces.
            Note: Activating increases simulation duration by ~40%.
        last_ped_forces: (init=False, repr=False) Last computed pedestrian forces (K, 2).
        last_force_computation: (init=False, repr=False) Opt-in typed force registry result.
        last_step_diagnostics: (init=False, repr=False) Opt-in cap/integration snapshot.
        last_oracle_transition_traces: (init=False, repr=False) Opt-in per-pedestrian traces.
    """

    config: SimulationSettings
    map_def: MapDefinition
    robots: list[Robot]
    goal_proximity_threshold: float
    random_start_pos: bool
    robot_navs: list[RouteNavigator] = field(init=False)
    pysf_sim: PySFSimulator = field(init=False)
    pysf_state: PedestrianStates = field(init=False)
    groups: PedestrianGroupings = field(init=False)
    peds_behaviors: list[PedestrianBehavior] = field(init=False)
    peds_have_obstacle_forces: bool
    # Last pedestrian force vectors used to step the simulation (K,2)
    last_ped_forces: np.ndarray = field(init=False, repr=False)
    _initial_pysf_states: np.ndarray = field(init=False, repr=False)
    ped_headings: np.ndarray = field(init=False, repr=False)
    _initial_ped_headings: np.ndarray = field(init=False, repr=False)
    ped_angular_velocities: np.ndarray = field(init=False, repr=False)
    pedestrian_model: str = field(init=False)
    _cached_social_force: SocialForce | None = field(init=False, repr=False, default=None)
    _residual_adversary: BoundedResidualAdversary | None = field(
        init=False, repr=False, default=None
    )
    last_force_computation: ForceComputationResult | None = field(
        init=False, repr=False, default=None
    )
    last_step_diagnostics: PedestrianStepDiagnostics | None = field(
        init=False, repr=False, default=None
    )
    last_oracle_transition_traces: tuple[OracleTransitionTraceV1, ...] | None = field(
        init=False, repr=False, default=None
    )
    _oracle_pre_behavior_state: np.ndarray | None = field(init=False, repr=False, default=None)
    _oracle_post_behavior_state: np.ndarray | None = field(init=False, repr=False, default=None)
    _oracle_pre_route_indices: tuple[int | None, ...] | None = field(
        init=False, repr=False, default=None
    )
    _oracle_pre_groups: tuple[tuple[int, ...], ...] | None = field(
        init=False, repr=False, default=None
    )
    _oracle_episode_index: int = field(init=False, repr=False, default=0)
    _oracle_transition_step_index: int = field(init=False, repr=False, default=0)
    _oracle_episode_id: str = field(init=False, repr=False, default="simulator-episode-0")
    _last_residual_delta: np.ndarray | None = field(init=False, repr=False, default=None)
    _last_model_variant_stage: _ForceStageArrays | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        """Initialize simulator components after dataclass construction.

        Sets up pedestrian spawn locations, groups, and behaviors; configures
        PySocialForce physics engine with optional obstacle/interaction forces;
        initializes robot navigation paths; and resets all agents to start state.
        Route spawning honors SimulationSettings route spawn options when provided.
        """
        # The ``peds_have_obstacle_forces is None`` warning-and-default-to-False guard
        # stays in Simulator (issue #6465): it mutates ``self`` before the shared
        # :func:`_build_pysf_simulation` factory reads the resolved value.
        if self.peds_have_obstacle_forces is None:
            logger.warning(
                "The peds_have_obstacle_forces attribute is not set. "
                "This may lead to unexpected behavior."
                "Setting it to False by default.",
            )
            self.peds_have_obstacle_forces = False

        (
            self.pysf_sim,
            self.pysf_state,
            self.groups,
            self.peds_behaviors,
            self.pedestrian_response_multipliers,
        ) = _build_pysf_simulation(
            config=self.config,
            map_def=self.map_def,
            robots=self.robots,
            robot_pose_provider=lambda: self.robot_poses,
            peds_have_obstacle_forces=self.peds_have_obstacle_forces,
            add_ego_state=False,
            include_response_law_multipliers=True,
            response_law_composition=self.config.response_law_composition,
            response_law_seed=self.config.response_law_seed,
            force_population_size=self.config.population_size,
        )

        # Cache the SocialForce component once instead of scanning forces every step (#6493)
        self._cached_social_force = next(
            (f for f in self.pysf_sim.forces if isinstance(f, SocialForce)), None
        )

        self.robot_navs = [
            RouteNavigator(proximity_threshold=self.goal_proximity_threshold) for _ in self.robots
        ]

        self.last_ped_forces = np.zeros((0, 2), dtype=float)
        self.pedestrian_model = normalize_pedestrian_model(self.config.pedestrian_model)
        self.ped_headings = self._headings_from_current_ped_velocities()
        self._initial_ped_headings = self.ped_headings.copy()
        self.ped_angular_velocities = np.zeros_like(self.ped_headings)
        self._initial_pysf_states = self.pysf_state.pysf_states().copy()
        self.reset_state()

    def _headings_from_current_ped_velocities(self) -> np.ndarray:
        """Initialize pedestrian headings from current velocities, falling back to zero radians.

        Returns:
            One heading angle in radians per pedestrian row.
        """
        velocities = np.asarray(self.pysf_state.ped_velocities, dtype=float)
        if velocities.size == 0:
            return np.empty((0,), dtype=float)
        speeds = np.linalg.norm(velocities, axis=-1)
        velocity_headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        return np.where(speeds <= MIN_HEADING_SPEED_MPS, 0.0, velocity_headings)

    def _oracle_trace_enabled(self) -> bool:
        """Return whether privileged force-transition capture is explicitly enabled."""
        return bool(getattr(self.config, "oracle_force_trace_enabled", False))

    def obstacle_force_law_metadata(self) -> dict[str, Any]:
        """Return the active fast-pysf obstacle-law metadata for this simulator."""
        metadata_fn = getattr(self.pysf_sim, "obstacle_force_law_metadata", None)
        if not callable(metadata_fn):
            return obstacle_force_law_metadata(
                getattr(self.config, "obstacle_force_law", None),
                site="fast_pysf",
                geometry_convention="map_line_endpoints_orthogonal_vector",
                radius_convention="threshold_plus_agent_radius_sigma",
            )
        return dict(metadata_fn())

    def _oracle_route_indices(self) -> tuple[int | None, ...]:
        """Return best-effort route indices aligned with simulator pedestrian rows."""
        count = int(self.pysf_state.num_peds)
        indices: list[int | None] = [None] * count
        for behavior in self.peds_behaviors:
            if isinstance(behavior, FollowRouteBehavior):
                for group_id, pedestrian_ids in behavior.groups.groups.items():
                    navigator = behavior.navigators.get(group_id)
                    if navigator is None:
                        continue
                    for pedestrian_id in pedestrian_ids:
                        global_pedestrian_id = behavior.global_ped_offset + pedestrian_id
                        if 0 <= global_pedestrian_id < count:
                            indices[global_pedestrian_id] = int(navigator.waypoint_id)
            elif isinstance(behavior, SinglePedestrianBehavior):
                for runtime in behavior._runtimes:
                    if 0 <= runtime.ped_id < count:
                        indices[runtime.ped_id] = int(runtime.waypoint_index)
        return tuple(indices)

    def _oracle_group_snapshot(self) -> tuple[tuple[int, ...], ...]:
        """Return a deterministic group-membership snapshot for mutation checks."""
        groups = getattr(self.groups, "groups", {})
        return tuple(
            sorted(
                tuple(sorted(int(pedestrian_id) for pedestrian_id in members))
                for members in groups.values()
            )
        )

    def _begin_oracle_transition(self) -> None:
        """Snapshot the pre-behavior boundary for an opt-in transition."""
        if not self._oracle_trace_enabled():
            return
        self._oracle_pre_behavior_state = np.array(
            self.pysf_state.pysf_states(),
            dtype=float,
            copy=True,
        )
        self._oracle_pre_route_indices = self._oracle_route_indices()
        self._oracle_pre_groups = self._oracle_group_snapshot()
        self.last_oracle_transition_traces = None

    def _oracle_force_time_robot_state(self) -> ForceTimeRobotState:
        """Capture every robot pose used by robot-aware force callbacks.

        Returns:
            The force-time robot state record for all configured robots.
        """
        states: list[RobotForceState] = []
        for robot_index, robot in enumerate(self.robots):
            position, heading = robot.pose
            velocity_xy: tuple[float, float] | None = None
            angular_velocity: float | None = None
            current_speed = getattr(robot, "current_speed", None)
            if current_speed is not None:
                try:
                    linear_speed = float(current_speed[0])
                    angular_velocity = float(current_speed[1])
                    velocity_xy = (
                        linear_speed * cos(float(heading)),
                        linear_speed * sin(float(heading)),
                    )
                except (TypeError, IndexError, ValueError):
                    velocity_xy = None
                    angular_velocity = None
            states.append(
                RobotForceState(
                    robot_index=robot_index,
                    position_xy=(float(position[0]), float(position[1])),
                    heading_rad=float(heading),
                    velocity_xy=velocity_xy,
                    angular_velocity_rad_s=angular_velocity,
                )
            )
        return ForceTimeRobotState(tuple(states))

    @staticmethod
    def _oracle_mutation_flags(
        pre_state: np.ndarray,
        post_behavior_state: np.ndarray,
        pre_groups: tuple[tuple[int, ...], ...],
        post_groups: tuple[tuple[int, ...], ...],
        behaviors: list[PedestrianBehavior],
    ) -> ControllerMutationFlags:
        """Classify controller-side mutations between state and force evaluation.

        Returns:
            Conservative mutation flags for the transition.
        """
        population_changed = pre_state.shape != post_behavior_state.shape
        if population_changed:
            position_changed = velocity_changed = goal_redirected = True
        else:
            position_changed = not np.array_equal(pre_state[:, 0:2], post_behavior_state[:, 0:2])
            velocity_changed = not np.array_equal(pre_state[:, 2:4], post_behavior_state[:, 2:4])
            goal_redirected = not np.array_equal(pre_state[:, 4:6], post_behavior_state[:, 4:6])
        hold_wait_active = any(
            bool(
                getattr(runtime, "waiting_for_advance", False)
                or getattr(runtime, "proximity_hold_engaged", False)
            )
            for behavior in behaviors
            for runtime in getattr(behavior, "_runtimes", ())
        )
        hold_velocity_reset = velocity_changed and (
            not population_changed
            and bool(np.any(np.linalg.norm(post_behavior_state[:, 2:4], axis=1) <= 1e-12))
        )
        return ControllerMutationFlags(
            goal_redirected=goal_redirected,
            hold_velocity_reset=hold_velocity_reset,
            respawn_reposition=position_changed,
            population_changed=population_changed,
            controller_jump_modelled=False,
            position_changed=position_changed,
            velocity_changed=velocity_changed,
            group_changed=pre_groups != post_groups,
            hold_wait_active=hold_wait_active,
            role_changed=False,
        )

    @staticmethod
    def _oracle_goal_change_kind(
        pre_state: np.ndarray,
        post_behavior_state: np.ndarray,
        pre_route_index: int | None,
        post_route_index: int | None,
        mutation_flags: ControllerMutationFlags,
        row: int,
    ) -> GoalChangeKind:
        """Classify a single pedestrian's behavior-time goal transition.

        Returns:
            The typed goal or route change classification.
        """
        if mutation_flags.respawn_reposition and not np.array_equal(
            pre_state[row, 0:2], post_behavior_state[row, 0:2]
        ):
            return GoalChangeKind.RESPAWN
        if np.array_equal(pre_state[row, 4:6], post_behavior_state[row, 4:6]):
            return GoalChangeKind.NONE
        if (
            pre_route_index is not None
            and post_route_index is not None
            and post_route_index == pre_route_index + 1
        ):
            return GoalChangeKind.WAYPOINT_ADVANCE
        return GoalChangeKind.REDIRECT

    def _oracle_stage_for_row(
        self,
        stage: _ForceStageArrays | None,
        row: int,
    ) -> ForceStageResult:
        """Convert a captured vector stage into the canonical per-ped record.

        Returns:
            The selected pedestrian's canonical force-stage record.
        """
        if stage is None:
            return ForceStageResult()
        result_force = (
            None if stage.result_force_xy is None else _row_xy(stage.result_force_xy, row)
        )
        if stage.operation_kind is ForceOperationKind.ADDITIVE:
            if stage.delta_force_xy is None or result_force is None:
                return ForceStageResult(operation_kind=ForceOperationKind.UNKNOWN)
            return ForceStageResult(
                operation_kind=stage.operation_kind,
                operation=stage.operation,
                delta_force_xy=_row_xy(stage.delta_force_xy, row),
                result_force_xy=result_force,
            )
        return ForceStageResult(
            operation_kind=stage.operation_kind,
            operation=stage.operation,
            result_force_xy=result_force,
        )

    @staticmethod
    def _oracle_force_component_records(
        result: ForceComputationResult,
        row: int,
    ) -> tuple[ForceComponentRecord, ...]:
        """Convert exact low-level arrays into canonical per-ped component records.

        Returns:
            Ordered component records for one pedestrian.
        """
        records: list[ForceComponentRecord] = []
        for component in result.components:
            if component.values is None:
                records.append(
                    ForceComponentRecord(
                        component_id=component.component_id,
                        component_type=component.component_type,
                        implementation_module=component.implementation_module,
                        implementation_class=component.implementation_class,
                        source_entity=component.source_entity,
                        force_xy=None,
                        enabled=component.enabled,
                        config_hash=component.config_hash,
                        evaluation_order=component.evaluation_order,
                        operation_kind=ForceComponentOperationKind.UNAVAILABLE,
                        operation="unavailable",
                        actor_observable=component.actor_observable,
                        unavailable_reason="low-level force output unavailable",
                    )
                )
                continue
            records.append(
                ForceComponentRecord(
                    component_id=component.component_id,
                    component_type=component.component_type,
                    implementation_module=component.implementation_module,
                    implementation_class=component.implementation_class,
                    source_entity=component.source_entity,
                    force_xy=_row_xy(component.values, row),
                    enabled=component.enabled,
                    config_hash=component.config_hash,
                    evaluation_order=component.evaluation_order,
                    operation_kind=ForceComponentOperationKind(component.operation.value),
                    operation=component.operation.value,
                    actor_observable=component.actor_observable,
                )
            )
        return tuple(records)

    def _oracle_force_components_for_row(
        self,
        result: ForceComputationResult,
        row: int,
        final_force: np.ndarray,
        diagnostics: PedestrianStepDiagnostics | None,
    ) -> tuple[ForceComponents, SpeedCap, DynamicsParameters]:
        """Build canonical force, cap, and dynamics records for one pedestrian.

        Returns:
            Typed force, speed-cap, and dynamics records.
        """
        values_by_type: dict[str, list[np.ndarray]] = {}
        for component in result.components:
            if component.values is not None:
                values_by_type.setdefault(component.component_type, []).append(
                    component.values[row]
                )

        def aggregate(*component_types: str) -> tuple[float, float] | None:
            values = [
                value
                for component_type in component_types
                for value in values_by_type.get(component_type, [])
            ]
            if not values:
                return None
            total = np.sum(np.asarray(values, dtype=float), axis=0)
            return (float(total[0]), float(total[1]))

        base_total = result.base_total[row]
        residual_stage = None
        if self._last_residual_delta is not None:
            residual_stage = _ForceStageArrays(
                operation_kind=ForceOperationKind.ADDITIVE,
                operation="additive_delta",
                delta_force_xy=self._last_residual_delta,
                result_force_xy=np.asarray(result.base_total, dtype=float)
                + self._last_residual_delta,
            )
        force_components = ForceComponents(
            social_force_xy=aggregate("social"),
            goal_force_xy=aggregate("desired"),
            obstacle_force_xy=aggregate("obstacle"),
            pedestrian_robot_force_xy=aggregate("pedestrian_robot"),
            adversarial_force_xy=aggregate("adversarial"),
            group_force_xy=aggregate("group_coherence", "group_repulsive", "group_gaze"),
            registry_total_force_xy=(float(base_total[0]), float(base_total[1])),
            residual_operation=self._oracle_stage_for_row(residual_stage, row),
            model_variant_operation=self._oracle_stage_for_row(
                self._last_model_variant_stage,
                row,
            ),
            final_pre_cap_force_xy=(float(final_force[0]), float(final_force[1])),
            uncapped_velocity_xy=(
                None if diagnostics is None else _row_xy(diagnostics.uncapped_velocity, row)
            ),
            applied_velocity_xy=(
                None if diagnostics is None else _row_xy(diagnostics.applied_velocity, row)
            ),
            component_records=self._oracle_force_component_records(result, row),
        )
        if diagnostics is None:
            speed_cap = SpeedCap(SpeedCapStatus.UNKNOWN)
        else:
            status = (
                SpeedCapStatus.APPLIED
                if bool(diagnostics.cap_active[row])
                else SpeedCapStatus.NOT_APPLIED
            )
            speed_cap = SpeedCap(
                status=status,
                max_speed_mps=float(diagnostics.max_speed_mps[row]),
                uncapped_speed_mps=float(diagnostics.uncapped_speed_mps[row]),
                applied_speed_mps=float(np.linalg.norm(diagnostics.applied_velocity[row])),
            )
        desired_force = next(
            (force for force in self.pysf_sim.forces if type(force).__name__ == "DesiredForce"),
            None,
        )
        desired_config = getattr(desired_force, "config", None)
        max_speeds = getattr(self.pysf_sim.peds, "max_speeds", None)
        preferred_speed = (
            None if max_speeds is None else float(np.asarray(max_speeds, dtype=float)[row])
        )
        dynamics = DynamicsParameters(
            preferred_speed_mps=preferred_speed,
            relaxation_time_s=getattr(desired_config, "relaxation_time", None),
            desired_force_factor=getattr(desired_config, "factor", None),
            goal_threshold_m=getattr(desired_config, "goal_threshold", None),
            goal_threshold_reached=None,
        )
        return force_components, speed_cap, dynamics

    def _record_oracle_transition(  # noqa: C901
        self, post_behavior_state: np.ndarray
    ) -> None:
        """Finalize one opt-in canonical oracle trace after pedestrian integration."""
        if not self._oracle_trace_enabled():
            return
        pre_state = self._oracle_pre_behavior_state
        pre_route_indices = self._oracle_pre_route_indices
        pre_groups = self._oracle_pre_groups
        force_result = self.last_force_computation
        if pre_state is None or pre_route_indices is None or pre_groups is None:
            raise RuntimeError("oracle transition started without a pre-behavior snapshot")
        if force_result is None:
            raise RuntimeError("oracle transition has no captured force computation")
        post_state = np.asarray(self.pysf_state.pysf_states(), dtype=float)
        if post_state.shape != post_behavior_state.shape:
            raise RuntimeError("pedestrian state changed after integration capture")
        post_route_indices = self._oracle_route_indices()
        post_groups = self._oracle_group_snapshot()
        mutation_flags = self._oracle_mutation_flags(
            pre_state,
            post_behavior_state,
            pre_groups,
            post_groups,
            self.peds_behaviors,
        )
        force_time_robot_state = self._oracle_force_time_robot_state()
        diagnostics = self.last_step_diagnostics
        final_force_array = np.asarray(self.last_ped_forces, dtype=float)
        if self._last_model_variant_stage is not None:
            if self._last_model_variant_stage.result_force_xy is None:
                raise RuntimeError("oracle model-variant stage has no final force")
            final_force_array = self._last_model_variant_stage.result_force_xy
        if final_force_array.shape != post_state[:, 0:2].shape:
            raise RuntimeError("oracle final force shape does not match pedestrian state")
        traces: list[OracleTransitionTraceV1] = []
        dt = float(self.config.time_per_step_in_secs)
        step_index = self._oracle_transition_step_index
        for row in range(post_state.shape[0]):
            goal_change_kind = self._oracle_goal_change_kind(
                pre_state,
                post_behavior_state,
                pre_route_indices[row] if row < len(pre_route_indices) else None,
                post_route_indices[row] if row < len(post_route_indices) else None,
                mutation_flags,
                row,
            )
            force_components, speed_cap, dynamics = self._oracle_force_components_for_row(
                force_result,
                row,
                final_force_array[row],
                diagnostics,
            )
            reasons: set[ExactInverseReason] = set()
            if diagnostics is None:
                reasons.add(ExactInverseReason.SPEED_CAP_UNKNOWN)
            elif bool(diagnostics.cap_active[row]):
                reasons.add(ExactInverseReason.SPEED_CAP_ACTIVE)
            if diagnostics is not None and (
                np.linalg.norm(diagnostics.previous_velocity[row]) <= 1e-12
                and np.linalg.norm(diagnostics.uncapped_velocity[row]) <= 1e-12
            ):
                reasons.add(ExactInverseReason.STATIONARY_EDGE_CASE)
            if mutation_flags.hold_velocity_reset:
                reasons.add(ExactInverseReason.HOLD_VELOCITY_RESET)
            if mutation_flags.respawn_reposition:
                reasons.add(ExactInverseReason.RESPAWN_REPOSITION)
            if mutation_flags.population_changed:
                reasons.add(ExactInverseReason.POPULATION_CHANGE)
            if any(
                (
                    mutation_flags.position_changed,
                    mutation_flags.velocity_changed,
                    mutation_flags.group_changed,
                    mutation_flags.hold_wait_active,
                    mutation_flags.role_changed,
                )
            ):
                reasons.add(ExactInverseReason.UNMODELED_CONTROLLER_MUTATION)
            pre_boundary = TransitionBoundary(
                boundary=TransitionBoundaryKind.PRE_BEHAVIOR,
                timestamp_s=step_index * dt,
                step_index=step_index,
                position_xy=_row_xy(pre_state[:, 0:2], row),
                velocity_xy=_row_xy(pre_state[:, 2:4], row),
                active_goal_xy=_row_xy(pre_state[:, 4:6], row),
                route_waypoint_index=pre_route_indices[row],
                goal_threshold_reached=None,
            )
            post_behavior_boundary = TransitionBoundary(
                boundary=TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE,
                timestamp_s=step_index * dt,
                step_index=step_index,
                position_xy=_row_xy(post_behavior_state[:, 0:2], row),
                velocity_xy=_row_xy(post_behavior_state[:, 2:4], row),
                active_goal_xy=_row_xy(post_behavior_state[:, 4:6], row),
                route_waypoint_index=post_route_indices[row],
                goal_threshold_reached=None,
                force_time_robot_state=force_time_robot_state,
                mutation_flags=mutation_flags,
            )
            post_integration_boundary = TransitionBoundary(
                boundary=TransitionBoundaryKind.POST_INTEGRATION,
                timestamp_s=(step_index + 1) * dt,
                step_index=step_index + 1,
                position_xy=_row_xy(post_state[:, 0:2], row),
                velocity_xy=_row_xy(post_state[:, 2:4], row),
                active_goal_xy=_row_xy(post_state[:, 4:6], row),
                route_waypoint_index=post_route_indices[row],
                goal_threshold_reached=None,
            )
            traces.append(
                OracleTransitionTraceV1(
                    episode_id=self._oracle_episode_id,
                    transition_id=f"{self._oracle_episode_id}:t{step_index}",
                    transition_step_index=step_index,
                    simulator_pedestrian_id=f"pysf-{row}",
                    actor_track_id=None,
                    actor_tracking_epoch_id=None,
                    backend="pysocialforce",
                    pre_behavior=pre_boundary,
                    post_behavior_pre_force=post_behavior_boundary,
                    post_integration=post_integration_boundary,
                    force_components=force_components,
                    dynamics=dynamics,
                    speed_cap=speed_cap,
                    goal_change_kind=goal_change_kind,
                    exact_inverse_eligible=not reasons,
                    exact_inverse_reasons=tuple(reasons),
                )
            )
        self.last_oracle_transition_traces = tuple(traces)
        self._oracle_transition_step_index += 1

    def _build_residual_adversary(self) -> BoundedResidualAdversary | None:
        """Construct the bounded residual adversary from config.

        The adversary is strictly additive: :meth:`_apply_residual_adversary` adds its
        output to the nominal pedestrian forces, so the Social Force base law is
        preserved and only perturbed. Route polylines, obstacle segments, and map
        bounds are forwarded when available so the walkable-space and route-deviation
        bounds can fire; otherwise those bounds degrade to no-ops while the kinematic
        bounds and inter-agent separation remain enforced.

        Returns
        -------
        BoundedResidualAdversary | None
            A ready-to-step adversary, or ``None`` when the config is inactive.
        """
        num_peds = self.pysf_state.num_peds
        config = self._residual_adversary_config(num_peds)
        return build_default_residual_adversary(
            config,
            self.config.time_per_step_in_secs,
            num_peds,
            route_polylines=self._collect_residual_route_polylines(),
            obstacle_segments=self._collect_residual_obstacle_segments(),
            bounds=self._collect_residual_map_bounds(),
            ped_radius=float(self.config.ped_radius),
        )

    def _residual_adversary_config(self, num_peds: int) -> ResidualAdversaryConfig:
        """Return residual-adversary config for rows controlled by this simulator."""
        del num_peds
        return self.config.residual_adversary

    def _collect_residual_route_polylines(self) -> dict[int, np.ndarray] | None:
        """Return actual route polylines keyed by global pedestrian index.

        ``FollowRouteBehavior`` owns the group-to-route assignments and its
        ``global_ped_offset`` identifies the corresponding simulator rows. This
        avoids coupling the residual controller to route-population or target-mask
        ordering. Pedestrians without a route assignment are intentionally absent,
        so their route-deviation bound is a no-op.
        """
        route_polylines: dict[int, np.ndarray] = {}
        for behavior in self.peds_behaviors:
            if not isinstance(behavior, FollowRouteBehavior):
                continue
            for group_id, route in behavior.route_assignments.items():
                polyline = np.asarray(route.waypoints, dtype=float)
                for local_ped_id in behavior.groups.groups.get(group_id, set()):
                    route_polylines[behavior.global_ped_offset + local_ped_id] = polyline
        return route_polylines or None

    def _collect_residual_obstacle_segments(self) -> np.ndarray | None:
        """Return standard ``[x1, y1, x2, y2]`` obstacle segments, or ``None``.

        ``MapDefinition.obstacles_pysf`` stores its legacy fast-pysf ordering as
        ``[x1, x2, y1, y2]``. The residual-adversary geometry helpers use the
        conventional endpoint ordering so their segment projection is unambiguous.
        """
        obstacles = getattr(self.map_def, "obstacles_pysf", None)
        if obstacles is None or len(obstacles) == 0:
            return None
        obstacle_array = np.asarray(obstacles, dtype=float)
        if obstacle_array.ndim != 2 or obstacle_array.shape[1] != 4:
            raise ValueError("MapDefinition.obstacles_pysf must have shape (S, 4)")
        return obstacle_array[:, [0, 2, 1, 3]]

    def _collect_residual_map_bounds(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return finite map bounds for the walkable-space clamp, or ``None``."""
        min_x, max_x, min_y, max_y = self.map_def.get_map_bounds()
        coords = (min_x, max_x, min_y, max_y)
        if not all(isinstance(value, int | float) and isfinite(float(value)) for value in coords):
            return None
        return (float(min_x), float(max_x)), (float(min_y), float(max_y))

    def _apply_residual_adversary(self, ped_forces: np.ndarray) -> np.ndarray:
        """Return pedestrian forces with the bounded residual acceleration added.

        When the adversary is inactive (the default) the input forces are returned
        unchanged. When active, the additive residual perturbs (never replaces) the
        nominal Social Force contribution already present in ``ped_forces``. An empty
        crowd short-circuits with no residual.
        """
        trace_enabled = bool(getattr(self.config, "oracle_force_trace_enabled", False))
        if trace_enabled:
            self._last_residual_delta = None
        if not self.config.residual_adversary.is_active:
            return ped_forces
        forces_array = np.asarray(ped_forces, dtype=float)
        if forces_array.shape[0] == 0:
            return forces_array
        if self._residual_adversary is None:
            self._residual_adversary = self._build_residual_adversary()
        adversary = self._residual_adversary
        assert adversary is not None  # built above because the config is active
        positions = np.asarray(self.pysf_state.ped_positions, dtype=float)
        velocities = np.asarray(self.pysf_state.ped_velocities, dtype=float)
        max_speeds = np.asarray(self.pysf_sim.peds.max_speeds, dtype=float)
        robot_poses = self.robot_poses
        if not robot_poses:
            raise ValueError("active residual adversary requires at least one robot pose")
        # The capability-only interface carries one robot pose. Multi-robot target
        # selection is deliberately deferred; the first simulator robot is the
        # reactive reference for this slice.
        robot_pose = robot_poses[0]
        residual = adversary.step_residual(positions, velocities, max_speeds, robot_pose)
        if trace_enabled:
            self._last_residual_delta = np.array(residual, dtype=float, copy=True)
        return forces_array + residual

    @property
    def residual_adversary_behavior_summary(self) -> ResidualAdversaryBehaviorSummary | None:
        """Return an immutable diagnostic summary after lazy adversary allocation.

        The property is ``None`` while the adversary is inactive or has not yet
        processed an active non-empty crowd. The summary is diagnostic-only and is
        not included in benchmark episode schemas.
        """
        if not self.config.residual_adversary.is_active:
            return None
        adversary = self._residual_adversary
        return adversary.behavior_summary if adversary is not None else None

    @property
    def oracle_force_trace_payload(self) -> dict[str, Any] | None:
        """Return the latest privileged trace in the canonical transition shape."""
        traces = self.last_oracle_transition_traces
        if traces is None:
            return None
        return {
            "schema_version": ORACLE_TRANSITION_TRACE_SCHEMA_VERSION,
            "transitions": [trace.to_dict() for trace in traces],
        }

    def _step_pedestrians(  # noqa: C901, PLR0912
        self,
        ped_forces: np.ndarray,
        groups: list[list[int]],
        *,
        capture_diagnostics: bool = False,
    ) -> None:
        """Advance pedestrians through the configured pedestrian-model implementation."""
        if not capture_diagnostics:
            self.last_step_diagnostics = None
        if self.pedestrian_model not in {
            HSFM_TOTAL_FORCE_V1,
            HSFM_TTC_PREDICTIVE_V1,
            HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
            HSFM_ANISOTROPIC_FOV_V1,
            HSFM_ALIGNMENT_TORQUE_V1,
        }:
            if capture_diagnostics:
                self.pysf_sim.peds.step(
                    ped_forces,
                    groups,
                    capture_diagnostics=True,
                )
            else:
                self.pysf_sim.peds.step(ped_forces, groups)
            if capture_diagnostics:
                self.last_step_diagnostics = self.pysf_sim.peds.last_step_diagnostics
            self.ped_headings = self._headings_from_current_ped_velocities()
            return

        max_speeds = self.pysf_sim.peds.max_speeds
        if max_speeds is None:
            raise RuntimeError("PySocialForce max_speeds are unavailable for HSFM total-force step")
        current_state = self.pysf_sim.peds.state
        if self.pedestrian_model == HSFM_TTC_PREDICTIVE_V1:
            ttc_config = self.config.ttc_predictive_force
            if ttc_config.include_robot_proxy:
                raise RuntimeError(
                    "TTC robot proxy coupling is not implemented for pedestrian stepping"
                )
            if ttc_config.include_ped_ped:
                radii = np.full(current_state.shape[0], self.config.ped_radius, dtype=float)
                predictive_delta = ttc_predictive_repulsion(
                    current_state[:, PYSF_POSITION_SLICE],
                    current_state[:, PYSF_VELOCITY_SLICE],
                    radii,
                    tau0_s=ttc_config.tau0_s,
                    horizon_s=ttc_config.horizon_s,
                    force_scale=ttc_config.force_scale,
                    max_force=ttc_config.max_force,
                )
                ped_forces = np.asarray(ped_forces, dtype=float) + predictive_delta
                self._set_model_variant_stage(
                    ForceOperationKind.ADDITIVE,
                    "additive_delta",
                    predictive_delta,
                    ped_forces,
                )
            else:
                self._set_model_variant_stage(
                    ForceOperationKind.TRANSFORMED,
                    "dedicated_integrator",
                    None,
                    ped_forces,
                )
        elif self.pedestrian_model == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1:
            zanlungo_config = self.config.zanlungo_collision_prediction
            if zanlungo_config.include_ped_ped:
                pairwise_social = self._pairwise_social_force_contributions(current_state)
                collision_prediction = zanlungo_collision_prediction_repulsion(
                    current_state[:, PYSF_POSITION_SLICE],
                    current_state[:, PYSF_VELOCITY_SLICE],
                    interaction_strength=zanlungo_config.interaction_strength,
                    interaction_range_m=zanlungo_config.interaction_range_m,
                    anisotropy_lambda=zanlungo_config.anisotropy_lambda,
                    angle_threshold_rad=zanlungo_config.angle_threshold_rad,
                    max_force=zanlungo_config.max_force,
                )
                transformed_forces = (
                    np.asarray(ped_forces, dtype=float)
                    - pairwise_social.sum(axis=1)
                    + collision_prediction
                )
                ped_forces = transformed_forces
                self._set_model_variant_stage(
                    ForceOperationKind.TRANSFORMED,
                    "transform_total",
                    None,
                    ped_forces,
                )
            else:
                self._set_model_variant_stage(
                    ForceOperationKind.TRANSFORMED,
                    "dedicated_integrator",
                    None,
                    ped_forces,
                )
        elif self.pedestrian_model == HSFM_ANISOTROPIC_FOV_V1:
            fov_config = self.config.anisotropic_fov
            # Consume per-pair pedestrian-pedestrian contributions instead of the coarse
            # ``np.min`` aggregate (issue #3481): isolate the social term, attenuate each
            # neighbor's push by its own field-of-view weight, and leave the actor's
            # goal/obstacle drive untouched.
            pairwise_social = self._pairwise_social_force_contributions(current_state)
            ped_forces = fov_attenuated_total_force(
                np.asarray(ped_forces, dtype=float),
                pairwise_social,
                current_state[:, PYSF_POSITION_SLICE],
                self.ped_headings,
                cone_half_angle_rad=fov_config.cone_half_angle_rad,
                rear_weight=fov_config.rear_weight,
            )
            self._set_model_variant_stage(
                ForceOperationKind.TRANSFORMED,
                "transform_total",
                None,
                ped_forces,
            )
        else:
            # ``HSFM_TOTAL_FORCE_V1`` and the alignment-torque variant preserve the
            # force values but use a dedicated integration/orientation stage.
            self._set_model_variant_stage(
                ForceOperationKind.TRANSFORMED,
                "dedicated_integrator",
                None,
                ped_forces,
            )
        diagnostics = (
            self.pysf_sim.peds.compute_step_diagnostics(
                ped_forces,
                cap_epsilon=1e-9,
            )
            if capture_diagnostics
            else None
        )
        next_state, target_headings = step_hsfm_total_force(
            current_state,
            ped_forces,
            self.ped_headings,
            dt=self.config.time_per_step_in_secs,
            max_speeds=max_speeds,
            integration_scheme=getattr(
                self.config, "pedestrian_integration_scheme", "semi_implicit_euler"
            ),
        )
        if self.pedestrian_model == HSFM_ALIGNMENT_TORQUE_V1:
            # Decouple body orientation from the instantaneous force direction (issue #3481):
            # treat the total-force heading as the desired orientation and relax toward it
            # with a bounded damped torque instead of snapping to it each step.
            torque_config = self.config.alignment_torque
            self.ped_headings, self.ped_angular_velocities = step_alignment_torque_heading(
                self.ped_headings,
                self.ped_angular_velocities,
                target_headings,
                dt=self.config.time_per_step_in_secs,
                k_theta=torque_config.k_theta,
                k_omega=torque_config.k_omega,
                max_angular_speed=torque_config.max_angular_speed_rad_s,
            )
        else:
            self.ped_headings = target_headings
        current_state[...] = next_state
        self.pysf_sim.peds.update(current_state, groups)
        if diagnostics is not None:
            self.last_step_diagnostics = diagnostics
            self.pysf_sim.peds.record_step_diagnostics(diagnostics)

    def _set_model_variant_stage(
        self,
        operation_kind: ForceOperationKind,
        operation: str,
        delta_force_xy: np.ndarray | None,
        result_force_xy: np.ndarray,
    ) -> None:
        """Record the exact model-variant operation used by the current step."""
        self._last_model_variant_stage = _ForceStageArrays(
            operation_kind=operation_kind,
            operation=operation,
            delta_force_xy=(
                None if delta_force_xy is None else np.array(delta_force_xy, dtype=float, copy=True)
            ),
            result_force_xy=np.array(result_force_xy, dtype=float, copy=True),
        )

    def _social_force_component(self) -> SocialForce:
        """Return the active PySocialForce ped-ped ``SocialForce`` component.

        Pairwise replacement models need the social force's parameters
        (activation threshold, factor, and interaction exponents) so its per-pair
        reconstruction exactly matches the aggregate PySocialForce already sums into the
        total force. Fail closed if the component is missing, mirroring the ``max_speeds``
        guard, so the opt-in model never silently degrades to a different force law.

        The instance is cached during ``__post_init__`` to avoid an ``isinstance``
        scan of ``self.pysf_sim.forces`` on every simulation step (issue #6493).
        Simulators built without ``__post_init__`` (for example, seams that inject
        ``pysf_sim`` directly) lazily scan once and cache the result on first use,
        preserving the call-time lookup contract.

        Returns:
            The ``SocialForce`` instance from the physics engine's force list.
        """
        cached = self._cached_social_force
        if cached is None:
            cached = next((f for f in self.pysf_sim.forces if isinstance(f, SocialForce)), None)
            self._cached_social_force = cached
        if cached is not None:
            return cached
        raise RuntimeError(
            "PySocialForce SocialForce component is unavailable for the pairwise pedestrian model"
        )

    def _pairwise_social_force_contributions(self, state: np.ndarray) -> np.ndarray:
        """Build the per-pair ped-ped social-force matrix for the current state.

        Args:
            state: PySocialForce state buffer whose columns expose positions and
                velocities via ``PYSF_POSITION_SLICE`` / ``PYSF_VELOCITY_SLICE``.

        Returns:
            Per-pair contributions with shape ``(N, N, 2)`` matching the aggregate
            ``SocialForce`` output when summed over neighbors.
        """
        social_config = self._social_force_component().config
        return pairwise_social_force_contributions(
            state[:, PYSF_POSITION_SLICE],
            state[:, PYSF_VELOCITY_SLICE],
            activation_threshold=social_config.activation_threshold,
            n=social_config.n,
            n_prime=social_config.n_prime,
            lambda_importance=social_config.lambda_importance,
            gamma=social_config.gamma,
            factor=social_config.factor,
        )

    def _reset_social_force_state(self) -> None:
        """Restore pedestrian physics state for a fresh deterministic episode reset."""
        residual_adversary = getattr(self, "_residual_adversary", None)
        if residual_adversary is not None:
            residual_adversary.reset()
        initial_states = getattr(self, "_initial_pysf_states", None)
        if initial_states is not None:
            self.pysf_state.pysf_states()[...] = initial_states
        initial_headings = getattr(self, "_initial_ped_headings", None)
        if initial_headings is not None:
            self.ped_headings = initial_headings.copy()
        # Only (re)initialize angular velocities when headings exist. ``_reset_social_force_state``
        # runs on the shared reset path, including partially-constructed ``PedSimulator`` stubs that
        # never set ``ped_headings``; an unconditional ``zeros_like(self.ped_headings)`` regressed
        # that path with an AttributeError.
        existing_headings = getattr(self, "ped_headings", None)
        if existing_headings is not None:
            self.ped_angular_velocities = np.zeros_like(existing_headings)
        self.last_ped_forces = np.zeros((0, 2), dtype=float)
        self.last_force_computation = None
        self.last_step_diagnostics = None
        self.last_oracle_transition_traces = None
        self._oracle_pre_behavior_state = None
        self._oracle_post_behavior_state = None
        self._oracle_pre_route_indices = None
        self._oracle_pre_groups = None
        self._last_residual_delta = None
        self._last_model_variant_stage = None
        for behavior in getattr(self, "peds_behaviors", ()):
            behavior.reset()

    @property
    def goal_pos(self) -> list[Vec2D]:
        """Current goal waypoint for each robot navigator."""
        return [n.current_waypoint for n in self.robot_navs]

    @property
    def next_goal_pos(self) -> list[Vec2D | None]:
        """Next waypoint for each robot navigator (None if at route end)."""
        return [n.next_waypoint for n in self.robot_navs]

    @property
    def robot_poses(self) -> list[RobotPose]:
        """Current poses (position + orientation) of all robots."""
        return [r.pose for r in self.robots]

    @property
    def robot_pos(self) -> list[Vec2D]:
        """Current (x, y) positions of all robots."""
        return [r.pose[0] for r in self.robots]

    @property
    def social_groups(self) -> list[SocialGroupDefinition]:
        """Declared social pedestrian groups exposed to group-aware consumers.

        Read-only view of the runtime map's ``social_groups`` (issue #3972);
        empty when the scenario declares none. Consumed by group-space metrics
        and, in a later slice, a group-avoidance planner wrapper.
        """
        return list(getattr(self.map_def, "social_groups", ()) or ())

    @property
    def ped_pos(self) -> np.ndarray:
        """Current (x, y) positions of all pedestrians."""
        return self.pysf_state.ped_positions

    @property
    def ped_vel(self) -> np.ndarray:
        """Current (vx, vy) velocities of all pedestrians."""
        return self.pysf_state.ped_velocities

    def _validate_robot_action_count(self, actions: list[RobotAction]) -> None:
        """Raise when a simulator step receives the wrong number of robot actions."""

        expected = len(self.robots)
        actual = len(actions)
        if actual != expected:
            action_word = "action" if expected == 1 else "actions"
            raise ValueError(
                f"{type(self).__name__}.step_once expected {expected} robot "
                f"{action_word}, got {actual}."
            )

    def reset_state(self) -> None:
        """Reset robot navigation and spawn positions.

        Reassigns routes and respawns robots when they collide or reach
        their destination goal. Updates are necessary for episodic reset
        or continuous replay scenarios.
        """
        self._reset_social_force_state()
        self._oracle_episode_index += 1
        self._oracle_episode_id = f"simulator-episode-{self._oracle_episode_index}"
        self._oracle_transition_step_index = 0
        for i, (robot, nav) in enumerate(zip(self.robots, self.robot_navs, strict=False)):
            collision = not nav.reached_waypoint
            is_at_final_goal = nav.reached_destination
            if collision or is_at_final_goal:
                waypoints = sample_route(self.map_def, None if self.random_start_pos else i)
                nav.new_route(waypoints[1:], start_pos=waypoints[0])
                robot.reset_state((waypoints[0], nav.initial_orientation))

    def step_once(self, actions: list[RobotAction]) -> None:
        """Advance simulation by one timestep.

        Updates pedestrian behaviors and physics (via PySocialForce), applies
        robot actions, and updates navigation state. Called once per episode
        timestep.

        Args:
            actions: Control actions for each robot (velocity, angular velocity, etc.).
        """
        self._validate_robot_action_count(actions)
        trace_enabled = bool(
            getattr(getattr(self, "config", None), "oracle_force_trace_enabled", False)
        )
        if trace_enabled:
            self._begin_oracle_transition()
        for behavior in self.peds_behaviors:
            behavior.step()
        if trace_enabled:
            self._oracle_post_behavior_state = np.array(
                self.pysf_state.pysf_states(),
                dtype=float,
                copy=True,
            )
        if trace_enabled:
            compute_components = getattr(self.pysf_sim, "compute_force_components", None)
            if not callable(compute_components):
                raise RuntimeError(
                    "oracle_force_trace_enabled requires a PySocialForce backend with "
                    "compute_force_components()"
                )
            self.last_force_computation = compute_components()
            ped_forces = np.array(self.last_force_computation.base_total, copy=True)
        else:
            self.last_force_computation = None
            ped_forces = self.pysf_sim.compute_forces()
        ped_forces = self._apply_residual_adversary(ped_forces)
        self.last_ped_forces = np.asarray(ped_forces, dtype=float)
        groups = self.groups.groups_as_lists
        self._last_model_variant_stage = None
        if trace_enabled:
            self._step_pedestrians(
                self.last_ped_forces,
                groups,
                capture_diagnostics=True,
            )
        else:
            self._step_pedestrians(self.last_ped_forces, groups)
        if trace_enabled:
            post_behavior_state = self._oracle_post_behavior_state
            if post_behavior_state is None:
                raise RuntimeError("oracle transition missing post-behavior capture")
            self._record_oracle_transition(post_behavior_state)
        for robot, nav, action in zip(self.robots, self.robot_navs, actions, strict=True):
            robot.apply_action(action, self.config.time_per_step_in_secs)
            nav.update_position(robot.pos)

    def get_obstacle_lines(self) -> np.ndarray:
        """Return obstacle line segments for collision/occupancy queries.

        Returns:
            np.ndarray: Array of shape (N, 4) with columns
                [start_x, start_y, end_x, end_y] for each obstacle segment.
        """
        return self.pysf_sim.env.obstacles_raw[:, :4]

    def iter_obstacle_segments(self) -> list[Line2D]:
        """Return obstacle line segments as typed Line2D tuples.

        Returns:
            list[Line2D]: List of ((x1, y1), (x2, y2)) tuples for each segment.
        """
        return [
            ((float(sx), float(sy)), (float(ex), float(ey)))
            for sx, sy, ex, ey in self.get_obstacle_lines()
        ]


def init_simulators(
    env_config: EnvSettings | RobotSimulationConfig,
    map_def: MapDefinition,
    num_robots: int = 1,
    random_start_pos: bool = True,
    peds_have_obstacle_forces: bool = True,
) -> list[Simulator]:
    """Initialize one or more simulator instances for the robot environment.

    Args:
        env_config: Environment configuration containing simulator/robot settings.
        map_def: Map definition describing start positions, goals, and obstacles.
        num_robots: Total number of robots to simulate across instances.
        random_start_pos: Whether robots start at random spawn positions.
        peds_have_obstacle_forces: Whether pedestrians experience obstacle forces.

    Returns:
        list[Simulator]: Simulator instances sized to cover ``num_robots`` robots.
    """
    if not isinstance(map_def, MapDefinition):
        raise TypeError(f"map_def should be of type MapDefinition, got {type(map_def)}")

    # Calculate the number of simulators needed based on the number of robots and start positions
    if map_def.num_start_pos <= 0:
        # Defensive guard: division-by-zero would occur and route sampling will fail later anyway.
        raise ValueError(
            "Cannot initialize simulators: map definition provides zero robot start positions "
            "(no robot routes detected). Ensure the map JSON/SVG conversion produced robot_routes "
            "and that spawn/goal zones plus routes are present.",
        )

    num_sims = ceil(num_robots / map_def.num_start_pos)

    # Calculate the proximity to the goal based on the robot radius and goal radius
    goal_proximity = env_config.robot_config.radius + env_config.sim_config.goal_radius

    # Initialize an empty list to hold the simulators
    sims: list[Simulator] = []

    # Create the required number of simulators
    for i in range(num_sims):
        # Determine the number of robots for this simulator
        n = (
            map_def.num_start_pos
            if i < num_sims - 1
            else max(1, num_robots % map_def.num_start_pos)
        )

        # Create the robots for this simulator
        sim_robots = [env_config.robot_factory() for _ in range(n)]

        # Create the simulator with the robots and add it to the list
        sim = Simulator(
            config=env_config.sim_config,
            map_def=map_def,
            robots=sim_robots,
            goal_proximity_threshold=goal_proximity,
            random_start_pos=random_start_pos,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )
        sims.append(sim)

    return sims


@dataclass
class PedSimulator(Simulator):
    """Extended simulator with ego pedestrian in a multi-agent scenario.

    Inherits robot and NPC pedestrian management from Simulator, adding a
    controllable ego pedestrian (e.g., human surrogate or trained robot-as-ped).
    Supports pedestrian-centric observations and action spaces.

    Attributes:
        ego_ped: Controllable pedestrian instance (typically UnicycleDrivePedestrian).
        spawn_near_robot: If True, ego pedestrian spawns near the first robot; else, random.
    """

    ego_ped: UnicycleDrivePedestrian

    def _residual_adversary_config(self, num_peds: int) -> ResidualAdversaryConfig:
        """Exclude the externally controlled ego pedestrian from policy targets.

        The ego row remains in controller state as a stationary separation
        constraint, so targeted non-player pedestrians cannot be nudged through it.

        Returns
        -------
        ResidualAdversaryConfig
            Config whose target set excludes the final ego-pedestrian row.
        """
        config = self.config.residual_adversary
        if num_peds == 0:
            return config
        target_mask = config.resolve_target_mask(num_peds)
        target_mask[-1] = False
        return replace(config, target_ped_idx=np.flatnonzero(target_mask).tolist())

    @staticmethod
    def _validate_ego_ped_action_count(ego_ped_actions: list[UnicycleAction]) -> None:
        """Raise when a pedestrian simulator step receives the wrong ego-ped action count."""

        actual = len(ego_ped_actions)
        if actual != 1:
            raise ValueError(
                f"PedSimulator.step_once expected 1 ego pedestrian action, got {actual}."
            )

    spawn_near_robot: bool = True

    def __post_init__(self):
        """Initialize pedestrian simulator with ego pedestrian and physics engine.

        Sets up PySocialForce configuration, populates pedestrians including the
        ego pedestrian state, initializes the physics simulator with pedestrian
        forces and robot interactions, and prepares robot navigation paths.
        """
        # NOTE (issue #4618 R2): the pedestrian-centric simulator intentionally
        # diverges from Simulator's heterogeneous-population wiring, and the
        # divergence is preserved (not unified) per issue #6465. It OMITS the
        # response_law_* / force_population_size spawn fields and requests
        # ``include_response_law_multipliers=False`` so
        # ``self.pedestrian_response_multipliers`` stays ``None`` and
        # :class:`PedRobotForce` falls back to unscaled robot repulsion. The
        # heterogeneous-population ablation targets the robot-only benchmark
        # simulator; the appended ego-pedestrian row would otherwise misalign the
        # per-pedestrian multiplier vector.
        (
            self.pysf_sim,
            self.pysf_state,
            self.groups,
            self.peds_behaviors,
            self.pedestrian_response_multipliers,
        ) = _build_pysf_simulation(
            config=self.config,
            map_def=self.map_def,
            robots=self.robots,
            robot_pose_provider=lambda: self.robot_poses,
            peds_have_obstacle_forces=self.peds_have_obstacle_forces,
            add_ego_state=True,
            include_response_law_multipliers=False,
        )

        # Cache the SocialForce component once instead of scanning forces every step (#6493)
        self._cached_social_force = next(
            (f for f in self.pysf_sim.forces if isinstance(f, SocialForce)), None
        )

        self.robot_navs = [
            RouteNavigator(proximity_threshold=self.goal_proximity_threshold) for _ in self.robots
        ]

        self.last_ped_forces = np.zeros((0, 2), dtype=float)
        self.pedestrian_model = normalize_pedestrian_model(self.config.pedestrian_model)
        self.ped_headings = self._headings_from_current_ped_velocities()
        self._initial_ped_headings = self.ped_headings.copy()
        self.ped_angular_velocities = np.zeros_like(self.ped_headings)
        self._initial_pysf_states = self.pysf_state.pysf_states().copy()

        self.reset_state()

    @property
    def ped_pos(self) -> np.ndarray:
        """
        Returns the current positions of all pedestrians.
        """
        return self.pysf_state.ped_positions[:-1]  # Exclude the ego pedestrian

    @property
    def ped_and_ego_pos(self) -> np.ndarray:
        """Return current NPC and ego pedestrian positions as one PySF-backed view."""
        return self.pysf_state.ped_positions

    @property
    def ped_vel(self) -> np.ndarray:
        """Return current velocities for NPC pedestrians only."""
        return self.pysf_state.ped_velocities[:-1]

    @property
    def ego_ped_pos(self) -> Vec2D:
        """Return the current 2D position of the ego pedestrian.

        Returns:
            Vec2D: The (x, y) coordinates of the ego pedestrian.
        """
        return self.ego_ped.pos

    @property
    def ego_ped_pose(self) -> PedPose:
        """Return the current pose of the ego pedestrian.

        Returns:
            PedPose: The full pose including position and orientation of the ego pedestrian.
        """
        return self.ego_ped.pose

    @property
    def ego_ped_goal_pos(self) -> Vec2D:
        """Return the goal position for the ego pedestrian (robot position).

        Returns:
            Vec2D: The (x, y) coordinates of the first robot, which serves as the goal.
        """
        return self.robots[0].pos

    @property
    def ego_ped_next_goal_pos(self) -> Vec2D | None:
        """Return the route target after the ego pedestrian's current target.

        The ego pedestrian's current target is the robot position. The following point is therefore
        the robot's current route goal, which lets the target sensor expose the robot's route
        direction without changing the existing robot-chasing goal contract.
        """
        return self.goal_pos[0] if self.goal_pos else None

    def _sync_ego_ped_social_force_state(self) -> None:
        """Synchronize the appended ego-pedestrian row in the PySF state array."""
        pysf_states = self.pysf_state.pysf_states()
        ego_speed, ego_heading = self.ego_ped.current_speed
        pysf_states[-1, PYSF_POSITION_SLICE] = self.ego_ped.pos
        ego_velocity = pysf_states[-1, PYSF_VELOCITY_SLICE]
        ego_velocity[0] = ego_speed * cos(ego_heading)
        ego_velocity[1] = ego_speed * sin(ego_heading)

    def reset_state(self) -> None:
        """Reset robot and ego pedestrian state.

        Calls parent reset_state() to reassign robot routes, then spawns
        the ego pedestrian at a random valid location 10-15 units away
        from the first robot.
        """
        self._reset_social_force_state()
        self._oracle_episode_index += 1
        self._oracle_episode_id = f"simulator-episode-{self._oracle_episode_index}"
        self._oracle_transition_step_index = 0
        for i, (robot, nav) in enumerate(zip(self.robots, self.robot_navs, strict=False)):
            collision = not nav.reached_waypoint
            is_at_final_goal = nav.reached_destination
            if collision or is_at_final_goal:
                waypoints = sample_route(self.map_def, None if self.random_start_pos else i)
                nav.new_route(waypoints[1:], start_pos=waypoints[0])
                robot.reset_state((waypoints[0], nav.initial_orientation))
        # Ego_pedestrian reset
        if self.spawn_near_robot:
            robot_spawn = self.robot_pos[0]
            ped_spawn = self.get_proximity_point(robot_spawn, 10, 15)
            self.ego_ped.reset_state((ped_spawn, self.ego_ped.pose[1]))
        else:
            # Spawn ego pedestrian randomly in one of the pedestrian spawn zones
            if not self.map_def.ped_spawn_zones:
                raise ValueError(
                    "spawn_near_robot=False requires at least one pedestrian spawn zone.",
                )
            ped_spawn_zone = sample(self.map_def.ped_spawn_zones, k=1)[0]
            ped_spawn = sample_zone(ped_spawn_zone, 1)[0]
            npc_orient = self.ego_ped.pose[1]
            if self.pysf_state.num_peds > 1:
                npc_velocity = self.pysf_state.pysf_states()[0, PYSF_VELOCITY_SLICE]
                npc_orient = _heading_from_velocity(
                    npc_velocity,
                    fallback_heading=npc_orient,
                )
            self.ego_ped.reset_state((ped_spawn, npc_orient))
        self._sync_ego_ped_social_force_state()

    def step_once(self, actions: list[RobotAction], ego_ped_actions: list[UnicycleAction]) -> None:
        """Advance simulation with robot and ego pedestrian actions.

        Updates pedestrian behaviors and physics, applies robot actions,
        applies ego pedestrian actions, and updates navigation.

        Args:
            actions: Control actions for each robot.
            ego_ped_actions: Control actions for the ego pedestrian.
        """
        self._validate_robot_action_count(actions)
        self._validate_ego_ped_action_count(ego_ped_actions)
        trace_enabled = bool(
            getattr(getattr(self, "config", None), "oracle_force_trace_enabled", False)
        )
        if trace_enabled:
            self._begin_oracle_transition()
        for behavior in self.peds_behaviors:
            behavior.step()
        if trace_enabled:
            self._oracle_post_behavior_state = np.array(
                self.pysf_state.pysf_states(),
                dtype=float,
                copy=True,
            )
            compute_components = getattr(self.pysf_sim, "compute_force_components", None)
            if not callable(compute_components):
                raise RuntimeError(
                    "oracle_force_trace_enabled requires a PySocialForce backend with "
                    "compute_force_components()"
                )
            self.last_force_computation = compute_components()
            ped_forces = np.array(self.last_force_computation.base_total, copy=True)
        else:
            self.last_force_computation = None
            ped_forces = self.pysf_sim.compute_forces()
        ped_forces = self._apply_residual_adversary(ped_forces)
        self.last_ped_forces = np.asarray(ped_forces, dtype=float)
        groups = self.groups.groups_as_lists
        self._last_model_variant_stage = None
        if trace_enabled:
            self._step_pedestrians(
                self.last_ped_forces,
                groups,
                capture_diagnostics=True,
            )
        else:
            self._step_pedestrians(self.last_ped_forces, groups)
        if trace_enabled:
            post_behavior_state = self._oracle_post_behavior_state
            if post_behavior_state is None:
                raise RuntimeError("oracle transition missing post-behavior capture")
            self._record_oracle_transition(post_behavior_state)
        for robot, nav, action in zip(self.robots, self.robot_navs, actions, strict=True):
            robot.apply_action(action, self.config.time_per_step_in_secs)
            nav.update_position(robot.pos)

        self.ego_ped.apply_action(ego_ped_actions[0], self.config.time_per_step_in_secs)
        self._sync_ego_ped_social_force_state()

    def get_proximity_point(
        self,
        fixed_point: tuple[float, float],
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[float, float]:
        """Sample a collision-free point at a given distance from a reference point.

        Attempts up to 10 times to find a valid point within the distance bounds,
        checking for obstacle collisions and map bounds. Falls back to random
        pedestrian spawn zone if unsuccessful.

        Args:
            fixed_point: Reference (x, y) coordinates.
            lower_bound: Minimum distance from fixed_point.
            upper_bound: Maximum distance from fixed_point.

        Returns:
            Tuple of (x, y) for collision-free point, or fallback spawn location.
        """
        x, y = fixed_point
        for _ in range(10):
            angle = uniform(0, 2 * pi)
            distance = uniform(lower_bound, upper_bound)

            new_x = x + distance * cos(angle)
            new_y = y + distance * sin(angle)
            if not self.is_obstacle_collision(new_x, new_y):
                return new_x, new_y

        logger.warning("Could not find a valid proximity point: {point}.", point=f"{fixed_point}")
        spawn_id = sample(self.map_def.ped_spawn_zones, k=1)[0]  # Spawn in pedestrian spawn_zone
        initial_spawn = sample_zone(spawn_id, 1)[0]
        return initial_spawn

    def is_obstacle_collision(self, x: float, y: float) -> bool:
        """Check if a position collides with obstacles or is outside map bounds.

        Validates both map boundary containment and obstacle collision using
        circle-line intersection with ego pedestrian radius. Adapted from
        occupancy.py for spawn validation.

        Args:
            x: X coordinate to check.
            y: Y coordinate to check.

        Returns:
            True if position is out of bounds or collides with an obstacle,
            False if position is collision-free and within bounds.
        """
        if not (0 <= x <= self.map_def.width and 0 <= y <= self.map_def.height):
            return True

        collision_distance = self.ego_ped.config.radius
        circle_agent = ((x, y), collision_distance)
        return circle_collides_any_lines(circle_agent, self.get_obstacle_lines())


def init_ped_simulators(
    env_config: PedEnvSettings | PedestrianSimulationConfig,
    map_def: MapDefinition,
    random_start_pos: bool = False,
    peds_have_obstacle_forces: bool = True,
) -> list[PedSimulator]:
    """Create a pedestrian-centric simulator instance.

    Factory function for initializing a PedSimulator with one robot and one
    controllable ego pedestrian. Validates map definition and initializes
    navigation, physics, and spawn configurations.

    Args:
        env_config: Pedestrian environment settings (robot/ped config, physics).
        map_def: Map with obstacles, spawn zones, routes.
        random_start_pos: If False, use deterministic spawn; if True, randomize.
        peds_have_obstacle_forces: Enable pedestrian-obstacle collision forces.

    Returns:
        Single-element list containing initialized PedSimulator instance.
    """

    # Calculate the proximity to the goal based on the robot radius and goal radius
    goal_proximity = env_config.robot_config.radius + env_config.sim_config.goal_radius

    # Create the robots for this simulator
    sim_robot = env_config.robot_factory()

    # Create the pedestrian for this simulator
    sim_ped = env_config.pedestrian_factory()

    # Create the simulator with the robots and add it to the list
    sim = PedSimulator(
        env_config.sim_config,
        map_def,
        [sim_robot],
        goal_proximity,
        random_start_pos,
        ego_ped=sim_ped,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        spawn_near_robot=env_config.spawn_near_robot,
    )

    return [sim]


__all__ = [
    "PYSF_POSITION_SLICE",
    "PYSF_TAU_INDEX",
    "PYSF_VELOCITY_SLICE",
    "PedSimulator",
    "Simulator",
    "init_ped_simulators",
    "init_simulators",
]
