"""Runtime radius-binding canary for the #6600 collision-envelope campaign (Gate 1).

Issue #6600 quantifies whether planner rankings and scenario-family conclusions change
when the configured robot collision-envelope radius changes. Before any production
SLURM sweep (Gate 2), Gate 1 must prove on at least one geometry-sensitive scenario
that a declared envelope radius propagates consistently to every surface that consumes
it. This module is that binding canary.

The canary declares a target radius on a scenario through the same surface the campaign
uses (:func:`robot_sf.scenario_certification.feasibility_oracle.make_envelope_scenario`,
which writes ``robot_config.radius``), then reads the *effective* radius each binding
surface binds through that surface's own real code path:

1. ``simulator_collision_geometry`` -- the robot collision envelope the initialized
   simulator builds (``Simulator.robots[0].config.radius``, which also sizes the
   pedestrian reserved zone).
2. ``obstacle_pedestrian_contact_logic`` -- the initialized ``ContinuousOccupancy``
   radius fields and collision properties, plus the radius-aware contact boundary
   (``robot_radius + ped_radius``) used by the benchmark clearance/contact regime.
3. ``feasibility_oracle`` -- the planner-free oracle's envelope injection and geometric
   inflation (``envelope_radius_m`` / ``envelope_diameter_m``).
4. ``metric_metadata_and_output_rows`` -- the radius the benchmark records in metric
   metadata and output rows (runner row extraction plus the production ``EpisodeData``
   builder).
5. ``planner_inputs`` -- the radius injected into the initialized planner/force inputs
   that consume it (active ped-robot and adversarial-ped force objects).

Semantics are fail-closed: any surface that binds a radius differing from the declared
target by more than the tolerance, or that cannot be observed, is a no-go. A single
no-go surface stops the campaign. The canary does not change the frozen ``0.0.3.post1``
metric semantics; it only observes which radius each surface binds.

Claim boundary: this is a within-simulator radius-binding canary. It is not a
physical-footprint validation, a realism result, or a safety guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.scenario_certification.feasibility_oracle import (
    FeasibilityOracleConfig,
    make_envelope_scenario,
    run_feasibility_oracle,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.scenario_certification.v1 import ScenarioCertificate

#: Machine-readable schema for the emitted canary verdict.
CANARY_SCHEMA = "radius_binding_canary.v1"

#: Default radius comparison tolerance (metres). Radius binding is exact in the code
#: paths under test, so any non-trivial delta signals a divergent default or a silently
#: ignored binding.
DEFAULT_TOLERANCE_M = 1e-9

#: Claim boundary recorded on every verdict.
DIAGNOSTIC_CLAIM_BOUNDARY = (
    "within-simulator radius-binding canary; not a physical-footprint validation, "
    "a realism result, or a safety guarantee"
)

# Binding-surface identifiers (issue #6641 / #6600 Gate 1).
SURFACE_SIM_COLLISION_GEOMETRY = "simulator_collision_geometry"
SURFACE_CONTACT_LOGIC = "obstacle_pedestrian_contact_logic"
SURFACE_FEASIBILITY_ORACLE = "feasibility_oracle"
SURFACE_METRIC_METADATA = "metric_metadata_and_output_rows"
SURFACE_PLANNER_INPUTS = "planner_inputs"

#: Canonical ordering of the five binding surfaces.
BINDING_SURFACES: tuple[str, ...] = (
    SURFACE_SIM_COLLISION_GEOMETRY,
    SURFACE_CONTACT_LOGIC,
    SURFACE_FEASIBILITY_ORACLE,
    SURFACE_METRIC_METADATA,
    SURFACE_PLANNER_INPUTS,
)

# Probe/setup failures that should become machine-readable no-go verdicts. Keep this
# explicit so the canary's fail-closed boundary does not add an unreviewed broad catch.
_CANARY_PROBE_ERRORS: tuple[type[Exception], ...] = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    StopIteration,
    TypeError,
    ValueError,
)

#: Fixed radius treatment from the #6600 campaign (metres): 0.5, 0.8, and the 1.0 m
#: release baseline.
CAMPAIGN_ENVELOPE_RADII_M: tuple[float, ...] = (0.5, 0.8, 1.0)


@dataclass(frozen=True, slots=True)
class _RuntimeBinding:
    """Production runtime objects used by the simulator-facing probes."""

    simulator: Any
    occupancy: Any
    force_radii_m: dict[str, tuple[float, ...]]


def _build_runtime_binding(config: RobotSimulationConfig) -> _RuntimeBinding:
    """Initialize the simulator surfaces that consume the configured radius.

    Returns:
        Runtime simulator, occupancy, and active force-radius observations.
    """
    from robot_sf.gym_env.env_util import (  # noqa: PLC0415
        init_collision_and_sensors,
        init_spaces,
    )
    from robot_sf.ped_npc.adversial_ped_force import (  # noqa: PLC0415
        AdversarialPedForce,
    )
    from robot_sf.ped_npc.ped_robot_force import PedRobotForce  # noqa: PLC0415
    from robot_sf.sim.simulator import init_simulators  # noqa: PLC0415

    map_pool = getattr(config, "map_pool", None)
    map_defs = getattr(map_pool, "map_defs", None)
    if not map_defs:
        raise ValueError("radius canary requires a configured map pool")
    map_def = next(iter(map_defs.values()))
    simulators = init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=bool(getattr(config, "peds_have_static_obstacle_forces", True)),
    )
    if not simulators:
        raise RuntimeError("radius canary simulator initialization returned no simulator")
    simulator = simulators[0]
    _, _, orig_obs_space = init_spaces(config, map_def)
    occupancies, _ = init_collision_and_sensors(simulator, config, orig_obs_space)
    if not occupancies:
        raise RuntimeError("radius canary collision initialization returned no occupancy")

    force_radii: dict[str, list[float]] = {}
    for force in simulator.pysf_sim.forces:
        if not isinstance(force, (PedRobotForce, AdversarialPedForce)):
            continue
        radius = getattr(getattr(force, "config", None), "robot_radius", None)
        if radius is not None:
            force_radii.setdefault(type(force).__name__, []).append(float(radius))

    return _RuntimeBinding(
        simulator=simulator,
        occupancy=occupancies[0],
        force_radii_m={name: tuple(radii) for name, radii in force_radii.items()},
    )


@dataclass(frozen=True, slots=True)
class SurfaceVerdict:
    """Radius-binding verdict for one binding surface.

    Attributes:
        surface: Binding-surface identifier (one of :data:`BINDING_SURFACES`).
        expected_radius_m: Declared target radius (metres).
        observed_radius_m: Effective radius the surface bound (metres), or ``None`` when
            the surface could not be observed.
        bound: Whether the surface bound the declared radius within tolerance.
        tolerance_m: Comparison tolerance used (metres).
        evidence: Surface-specific binding evidence (real code path + read values).
        note: Human-readable note; populated on no-go with the failure reason.
    """

    surface: str
    expected_radius_m: float
    observed_radius_m: float | None
    bound: bool
    tolerance_m: float
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True, slots=True)
class RadiusBindingCanaryVerdict:
    """Machine-readable go/no-go canary verdict for one scenario at one target radius.

    Attributes:
        schema: Verdict schema identifier (:data:`CANARY_SCHEMA`).
        scenario_id: Scenario identifier the canary ran on.
        scenario_path: Scenario manifest path used to resolve maps/routes/robot config.
        target_radius_m: Declared envelope radius (metres).
        go: Overall go/no-go; ``True`` only when every binding surface is bound.
        surfaces: Per-surface verdicts in :data:`BINDING_SURFACES` order.
        generated_at: ISO-8601 UTC generation timestamp.
        claim_boundary: Claim boundary for the verdict.
        tolerance_m: Comparison tolerance used (metres).
    """

    schema: str
    scenario_id: str
    scenario_path: str
    target_radius_m: float
    go: bool
    surfaces: tuple[SurfaceVerdict, ...]
    generated_at: str
    claim_boundary: str
    tolerance_m: float


def validate_tolerance_m(tolerance_m: float) -> float:
    """Validate and normalize a radius comparison tolerance.

    Returns:
        Finite, non-negative tolerance in metres.
    """
    try:
        value = float(tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance_m must be finite and non-negative") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative")
    return value


def _radius_matches(observed: float | None, expected: float, tolerance_m: float) -> bool:
    """Return whether an observed radius matches the declared target within tolerance."""
    tolerance = validate_tolerance_m(tolerance_m)
    return (
        observed is not None
        and math.isfinite(float(observed))
        and abs(float(observed) - float(expected)) <= tolerance
    )


def _failed_verdict(
    surface: str,
    target_radius_m: float,
    tolerance_m: float,
    exc: Exception,
) -> SurfaceVerdict:
    """Return a fail-closed verdict when a surface probe raises."""
    return SurfaceVerdict(
        surface=surface,
        expected_radius_m=float(target_radius_m),
        observed_radius_m=None,
        bound=False,
        tolerance_m=float(tolerance_m),
        evidence={"error": f"{type(exc).__name__}: {exc}"},
        note=f"binding surface could not be observed ({type(exc).__name__}); fail-closed no-go",
    )


def _failed_surfaces(
    surfaces: tuple[str, ...], target_radius_m: float, tolerance_m: float, exc: Exception
) -> tuple[SurfaceVerdict, ...]:
    """Return fail-closed verdicts for a shared initialization failure."""
    return tuple(
        _failed_verdict(surface, target_radius_m, tolerance_m, exc) for surface in surfaces
    )


def _robot_config(
    declared: Mapping[str, Any],
    scenario_path: Path,
    prebuilt: RobotSimulationConfig | None,
) -> RobotSimulationConfig:
    """Return a prebuilt robot config or build one from the declared scenario."""
    if prebuilt is not None:
        return prebuilt
    return build_robot_config_from_scenario(dict(declared), scenario_path=scenario_path)


def probe_sim_collision_geometry(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    robot_config: RobotSimulationConfig | None = None,
    runtime_binding: _RuntimeBinding | None = None,
    require_runtime: bool = False,
) -> SurfaceVerdict:
    """Probe the simulator collision-geometry binding surface.

    The simulator's robot collision circle is ``robot.config.radius``; the same value
    sizes the pedestrian reserved zone. The top-level canary reads the radius back from
    an initialized ``Simulator`` so a config-only default cannot produce a false pass.

    Returns:
        Verdict with ``observed_radius_m`` = the robot collision envelope radius.
    """
    tolerance_m = validate_tolerance_m(tolerance_m)
    surface = SURFACE_SIM_COLLISION_GEOMETRY
    try:
        if require_runtime and runtime_binding is None:
            raise RuntimeError("initialized simulator binding is unavailable")
        cfg = _robot_config(declared, scenario_path, robot_config)
        config_radius = float(cfg.robot_config.radius)
        if runtime_binding is None:
            observed = config_radius
            runtime_evidence = {
                "binding": "robot.config.radius -> simulator collision circle + reserved zone",
            }
        else:
            simulator = runtime_binding.simulator
            if not simulator.robots:
                raise RuntimeError("initialized simulator has no robot")
            observed = float(simulator.robots[0].config.radius)
            runtime_evidence = {
                "runtime_component": "Simulator.robots[0].config.radius",
                "simulator_goal_proximity_threshold_m": float(simulator.goal_proximity_threshold),
                "reserved_zone_radius_input_m": max(observed, 0.0),
                "binding": (
                    "init_simulators -> Simulator.robots[0].config.radius -> "
                    "_build_pysf_simulation reserved_zone_radius"
                ),
            }
        bound = _radius_matches(config_radius, target_radius_m, tolerance_m) and _radius_matches(
            observed, target_radius_m, tolerance_m
        )
        return SurfaceVerdict(
            surface=surface,
            expected_radius_m=float(target_radius_m),
            observed_radius_m=observed,
            bound=bound,
            tolerance_m=float(tolerance_m),
            evidence={
                "robot_config_type": type(cfg.robot_config).__name__,
                "robot_config_radius_m": config_radius,
                "runtime_robot_radius_m": observed,
                **runtime_evidence,
            },
            note="" if bound else "simulator collision geometry did not bind the declared radius",
        )
    except _CANARY_PROBE_ERRORS as exc:
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def probe_contact_logic(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    robot_config: RobotSimulationConfig | None = None,
    runtime_binding: _RuntimeBinding | None = None,
    require_runtime: bool = False,
) -> SurfaceVerdict:
    """Probe the obstacle/pedestrian contact-logic binding surface.

    The runtime contact surface is ``ContinuousOccupancy``. This probe reads its robot
    and pedestrian radii, evaluates both runtime collision properties, then forms the
    benchmark contact boundary and verifies the clearance classifier flips at that
    boundary -- proving the contact logic consumes the declared radius.

    Returns:
        Verdict with ``observed_radius_m`` = the robot radius bound into the contact
        boundary.
    """
    tolerance_m = validate_tolerance_m(tolerance_m)
    surface = SURFACE_CONTACT_LOGIC
    try:
        if require_runtime and runtime_binding is None:
            raise RuntimeError("initialized contact binding is unavailable")
        from robot_sf.benchmark.collision_definition_inventory import (  # noqa: PLC0415
            LABEL_COLLISION,
            LABEL_NEAR_MISS,
            classify_clearance_regime,
        )
        from robot_sf.benchmark.constants import NEAR_MISS_DIST  # noqa: PLC0415

        cfg = _robot_config(declared, scenario_path, robot_config)
        if runtime_binding is None:
            robot_radius = float(cfg.robot_config.radius)
            ped_radius = float(cfg.sim_config.ped_radius)
            runtime_evidence: dict[str, Any] = {}
            runtime_boundary_binds = True
        else:
            import numpy as np  # noqa: PLC0415

            occupancy = runtime_binding.occupancy
            robot_radius = float(occupancy.agent_radius)
            ped_radius = float(occupancy.ped_radius)
            center = (float(occupancy.width) / 2.0, float(occupancy.height) / 2.0)
            boundary_eps = 1e-6

            def obstacle_segments(offset: float) -> np.ndarray:
                obstacle_x = center[0] + float(offset)
                return np.asarray(
                    [[obstacle_x, center[1] - 1.0, obstacle_x, center[1] + 1.0]],
                    dtype=float,
                )

            def pedestrian_coords(offset: float) -> np.ndarray:
                return np.asarray([[center[0] + float(offset), center[1]]], dtype=float)

            obstacle_inside = bool(
                replace(
                    occupancy,
                    get_agent_coords=lambda: center,
                    get_obstacle_coords=lambda: obstacle_segments(robot_radius - boundary_eps),
                    get_pedestrian_coords=lambda: np.empty((0, 2), dtype=float),
                ).is_obstacle_collision
            )
            obstacle_outside = bool(
                replace(
                    occupancy,
                    get_agent_coords=lambda: center,
                    get_obstacle_coords=lambda: obstacle_segments(robot_radius + boundary_eps),
                    get_pedestrian_coords=lambda: np.empty((0, 2), dtype=float),
                ).is_obstacle_collision
            )
            pedestrian_inside = bool(
                replace(
                    occupancy,
                    get_agent_coords=lambda: center,
                    get_obstacle_coords=lambda: np.empty((0, 4), dtype=float),
                    get_pedestrian_coords=lambda: pedestrian_coords(
                        robot_radius + ped_radius - boundary_eps
                    ),
                ).is_pedestrian_collision
            )
            pedestrian_outside = bool(
                replace(
                    occupancy,
                    get_agent_coords=lambda: center,
                    get_obstacle_coords=lambda: np.empty((0, 4), dtype=float),
                    get_pedestrian_coords=lambda: pedestrian_coords(
                        robot_radius + ped_radius + boundary_eps
                    ),
                ).is_pedestrian_collision
            )
            runtime_boundary_binds = (
                obstacle_inside
                and not obstacle_outside
                and pedestrian_inside
                and not pedestrian_outside
            )
            runtime_evidence = {
                "runtime_component": (
                    "ContinuousOccupancy.agent_radius/ped_radius + "
                    "is_obstacle_collision/is_pedestrian_collision"
                ),
                "runtime_obstacle_collision": bool(occupancy.is_obstacle_collision),
                "runtime_pedestrian_collision": bool(occupancy.is_pedestrian_collision),
                "runtime_obstacle_boundary_inside": obstacle_inside,
                "runtime_obstacle_boundary_outside": obstacle_outside,
                "runtime_pedestrian_boundary_inside": pedestrian_inside,
                "runtime_pedestrian_boundary_outside": pedestrian_outside,
            }
        radius_sum = robot_radius + ped_radius

        # The contact boundary must flip exactly at center_distance == radius_sum:
        # just inside -> collision; just outside the near-miss band -> clear.
        eps = 1e-6
        inside_label = classify_clearance_regime(
            radius_sum - eps, radius_sum=radius_sum, near_miss_dist=NEAR_MISS_DIST
        )
        outside_label = classify_clearance_regime(
            radius_sum + float(NEAR_MISS_DIST) + eps,
            radius_sum=radius_sum,
            near_miss_dist=NEAR_MISS_DIST,
        )
        boundary_binds = inside_label == LABEL_COLLISION and outside_label != LABEL_COLLISION
        bound = (
            _radius_matches(robot_radius, target_radius_m, tolerance_m)
            and boundary_binds
            and runtime_boundary_binds
        )
        note = ""
        if not _radius_matches(robot_radius, target_radius_m, tolerance_m):
            note = "contact logic robot radius did not bind the declared radius"
        elif not boundary_binds:
            note = "contact boundary did not flip at robot_radius + ped_radius"
        elif not runtime_boundary_binds:
            note = "runtime occupancy collision boundaries did not consume the bound radii"
        return SurfaceVerdict(
            surface=surface,
            expected_radius_m=float(target_radius_m),
            observed_radius_m=robot_radius,
            bound=bound,
            tolerance_m=float(tolerance_m),
            evidence={
                "robot_radius_m": robot_radius,
                "ped_radius_m": ped_radius,
                "contact_boundary_radius_sum_m": radius_sum,
                "inside_label": inside_label,
                "outside_label": outside_label,
                "near_miss_dist_m": float(NEAR_MISS_DIST),
                "binding": "clearance = center_distance - (robot_radius + ped_radius)",
                "expected_inside_label": LABEL_COLLISION,
                "expected_outside_label_not": LABEL_COLLISION,
                "near_miss_label": LABEL_NEAR_MISS,
                **runtime_evidence,
            },
            note=note,
        )
    except _CANARY_PROBE_ERRORS as exc:
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def probe_feasibility_oracle(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
    episode_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> SurfaceVerdict:
    """Probe the feasibility/oracle binding surface.

    Runs the planner-free feasibility oracle at the declared radius and verifies the
    envelope radius propagates into the verdict (``envelope_radius_m``) and the geometric
    inflation (``envelope_diameter_m == 2 * radius``), and that the oracle's scenario
    injection wrote ``robot_config.radius``. The radius-binding observation is independent
    of feasibility classification: a geometrically infeasible cell still binds the radius.

    Args:
        declared: Radius-declared scenario mapping (already envelope-overridden).
        target_radius_m: Declared envelope radius (metres).
        scenario_path: Scenario manifest path for map/route/robot-config resolution.
        tolerance_m: Comparison tolerance (metres).
        certifier: Optional injected route certifier (defaults to the canonical certifier).
        episode_runner: Optional injected scripted rollout runner. A trivial runner keeps
            the probe bounded; the radius-binding evidence comes from the geometric margin.

    Returns:
        Verdict with ``observed_radius_m`` = the oracle geometric envelope radius.
    """
    tolerance_m = validate_tolerance_m(tolerance_m)
    surface = SURFACE_FEASIBILITY_ORACLE
    try:
        # The campaign declares the radius through make_envelope_scenario, which writes
        # robot_config.radius. Run the oracle at that declared radius (the same value the
        # envelope-sensitivity sweep forwards) and verify the geometric inflation binds it.
        injected_radius = float((declared.get("robot_config") or {}).get("radius", math.nan))
        if not math.isfinite(injected_radius) or injected_radius <= 0.0:
            return SurfaceVerdict(
                surface=surface,
                expected_radius_m=float(target_radius_m),
                observed_radius_m=None,
                bound=False,
                tolerance_m=float(tolerance_m),
                evidence={"injected_robot_config_radius_m": injected_radius},
                note="oracle scenario injection missing a finite positive robot_config.radius",
            )
        config = FeasibilityOracleConfig(
            scenario_path=scenario_path,
            envelope_radii_m=(injected_radius,),
        )
        runner = episode_runner if episode_runner is not None else (lambda *args, **kwargs: {})
        verdict = run_feasibility_oracle(
            declared,
            config=config,
            envelope_radius_m=injected_radius,
            certifier=certifier,
            episode_runner=runner,
        )
        observed = float(verdict.geometric.envelope_radius_m)
        diameter = float(verdict.geometric.envelope_diameter_m)
        bound = (
            _radius_matches(injected_radius, target_radius_m, tolerance_m)
            and _radius_matches(observed, target_radius_m, tolerance_m)
            and _radius_matches(verdict.envelope_radius_m, target_radius_m, tolerance_m)
            and _radius_matches(diameter, 2.0 * float(target_radius_m), tolerance_m)
        )
        note = "" if bound else "feasibility oracle did not bind the declared envelope radius"
        return SurfaceVerdict(
            surface=surface,
            expected_radius_m=float(target_radius_m),
            observed_radius_m=observed,
            bound=bound,
            tolerance_m=float(tolerance_m),
            evidence={
                "injected_robot_config_radius_m": injected_radius,
                "verdict_envelope_radius_m": float(verdict.envelope_radius_m),
                "geometric_envelope_radius_m": observed,
                "geometric_envelope_diameter_m": diameter,
                "geometric_classification": verdict.geometric.classification,
                "route_geometrically_feasible": verdict.geometric.route_geometrically_feasible,
                "min_corridor_width_m": verdict.geometric.min_corridor_width_m,
                "oracle_status": verdict.status,
                "binding": "make_envelope_scenario -> oracle envelope_radius_m / diameter",
            },
            note=note,
        )
    except _CANARY_PROBE_ERRORS as exc:
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def probe_metric_metadata(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    robot_config: RobotSimulationConfig | None = None,
) -> SurfaceVerdict:
    """Probe the metric-metadata / output-row binding surface.

    The benchmark records the robot radius in metric metadata and output rows through the
    runner's row-level scenario radius extraction (``_scenario_robot_radius_m``) and the
    production ``_build_episode_data`` path that stores ``EpisodeData.robot_radius``.
    The resolved simulation config is checked as the upstream source for both paths.

    Returns:
        Verdict with ``observed_radius_m`` = the runner-recorded output-row radius.
    """
    tolerance_m = validate_tolerance_m(tolerance_m)
    surface = SURFACE_METRIC_METADATA
    try:
        import numpy as np  # noqa: PLC0415

        from robot_sf.benchmark.runner import (  # noqa: PLC0415
            _build_episode_data,
            _scenario_robot_radius_m,
        )

        runner_radius = float(_scenario_robot_radius_m(dict(declared)))
        cfg = _robot_config(declared, scenario_path, robot_config)
        config_radius = float(getattr(cfg.robot_config, "radius", math.nan))
        ped_radius = float(getattr(cfg.sim_config, "ped_radius", math.nan))
        episode = _build_episode_data(
            [np.zeros((1, 2), dtype=float)],
            [np.zeros((1, 2), dtype=float)],
            [np.zeros((1, 2), dtype=float)],
            [np.empty((0, 2), dtype=float)],
            [np.empty((0, 2), dtype=float)],
            None,
            np.ones(2, dtype=float),
            0.1,
            None,
            robot_radius=runner_radius,
            ped_radius=ped_radius,
        )
        episode_radius = float(episode.robot_radius)
        bound = all(
            _radius_matches(observed, target_radius_m, tolerance_m)
            for observed in (runner_radius, episode_radius, config_radius)
        )
        note = "" if bound else "metric metadata / output rows did not record the declared radius"
        return SurfaceVerdict(
            surface=surface,
            expected_radius_m=float(target_radius_m),
            observed_radius_m=runner_radius,
            bound=bound,
            tolerance_m=float(tolerance_m),
            evidence={
                "runner_row_robot_radius_m": runner_radius,
                "episode_data_robot_radius_m": episode_radius,
                "simulation_config_robot_radius_m": config_radius,
                "output_row_key": "robot_radius",
                "runtime_component": "runner._build_episode_data -> EpisodeData.robot_radius",
                "binding": "robot_config.radius -> metric metadata + output rows",
            },
            note=note,
        )
    except _CANARY_PROBE_ERRORS as exc:
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def probe_planner_inputs(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    robot_config: RobotSimulationConfig | None = None,
    runtime_binding: _RuntimeBinding | None = None,
    require_runtime: bool = False,
) -> SurfaceVerdict:
    """Probe the planner-input binding surface.

    The simulator injects ``robot.config.radius`` into the active force inputs that consume
    the radius: the ped-robot force (``PedRobotForceConfig.robot_radius``) and the
    adversarial-ped force (``AdversarialPedForceConfig.robot_radius``). The top-level canary
    reads the initialized force objects, so merely reproducing the ``replace`` expression
    cannot produce a false pass. Direct callers without a runtime binding retain the small
    config-level probe for focused unit tests.

    Returns:
        Verdict with ``observed_radius_m`` = the radius bound into the planner force inputs.
    """
    tolerance_m = validate_tolerance_m(tolerance_m)
    surface = SURFACE_PLANNER_INPUTS
    try:
        if require_runtime and runtime_binding is None:
            raise RuntimeError("initialized planner binding is unavailable")
        from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig  # noqa: PLC0415
        from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig  # noqa: PLC0415

        cfg = _robot_config(declared, scenario_path, robot_config)
        if runtime_binding is None:
            robot_radius = float(cfg.robot_config.radius)
            prf_radius = float(
                replace(PedRobotForceConfig(), robot_radius=robot_radius).robot_radius
            )
            apf_radius = float(
                replace(AdversarialPedForceConfig(), robot_radius=robot_radius).robot_radius
            )
            active_force_types: list[str] = []
            inactive_force_configs: list[str] = []
            force_bound = _radius_matches(
                prf_radius, target_radius_m, tolerance_m
            ) and _radius_matches(apf_radius, target_radius_m, tolerance_m)
            runtime_evidence: dict[str, Any] = {}
        else:
            simulator = runtime_binding.simulator
            if not simulator.robots:
                raise RuntimeError("initialized simulator has no robot")
            robot_radius = float(simulator.robots[0].config.radius)
            force_radii = runtime_binding.force_radii_m
            prf_values = force_radii.get("PedRobotForce", ())
            apf_values = force_radii.get("AdversarialPedForce", ())
            prf_radius = float(prf_values[0]) if prf_values else None
            apf_radius = float(apf_values[0]) if apf_values else None
            active_force_types = sorted(force_radii)
            configured_force_types = {
                "PedRobotForce": bool(cfg.sim_config.prf_config.is_active),
                "AdversarialPedForce": bool(cfg.sim_config.apf_config.is_active),
            }
            inactive_force_configs = sorted(
                name for name, is_active in configured_force_types.items() if not is_active
            )
            required_force_types = [
                name for name, is_active in configured_force_types.items() if is_active
            ]
            force_bound = bool(required_force_types) and all(
                name in force_radii
                and all(
                    _radius_matches(value, target_radius_m, tolerance_m)
                    for value in force_radii[name]
                )
                for name in required_force_types
            )
            runtime_evidence = {
                "runtime_component": "Simulator.pysf_sim.forces[].config.robot_radius",
                "active_force_types": active_force_types,
                "inactive_force_configs": inactive_force_configs,
                "configured_force_types": configured_force_types,
            }
        bound = _radius_matches(robot_radius, target_radius_m, tolerance_m) and force_bound
        note = "" if bound else "planner inputs did not consume the declared radius"
        if runtime_binding is not None and not active_force_types:
            note = "no active radius-consuming planner force was initialized"
        return SurfaceVerdict(
            surface=surface,
            expected_radius_m=float(target_radius_m),
            observed_radius_m=robot_radius,
            bound=bound,
            tolerance_m=float(tolerance_m),
            evidence={
                "robot_config_radius_m": robot_radius,
                "ped_robot_force_robot_radius_m": prf_radius,
                "adversarial_ped_force_robot_radius_m": apf_radius,
                "binding": "Simulator._make_ped_forces -> force.config.robot_radius",
                **runtime_evidence,
            },
            note=note,
        )
    except _CANARY_PROBE_ERRORS as exc:
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def run_radius_binding_canary(
    scenario: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
    episode_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> RadiusBindingCanaryVerdict:
    """Run the radius-binding canary for one scenario at one declared radius.

    Declares the target radius via :func:`make_envelope_scenario`, probes all five binding
    surfaces against the declared scenario, and emits a fail-closed go/no-go verdict.

    Args:
        scenario: Source scenario mapping (e.g. one entry from ``load_scenarios``).
        target_radius_m: Declared envelope radius (metres).
        scenario_path: Scenario manifest path for map/route/robot-config resolution.
        tolerance_m: Radius comparison tolerance (metres).
        certifier: Optional injected route certifier for the oracle probe.
        episode_runner: Optional injected scripted rollout runner for the oracle probe.

    Returns:
        Machine-readable canary verdict; ``go`` is ``True`` only when all five surfaces
        bind the declared radius.
    """
    tolerance_m = validate_tolerance_m(tolerance_m)
    if not math.isfinite(float(target_radius_m)) or float(target_radius_m) <= 0.0:
        raise ValueError("target_radius_m must be finite and positive")
    declared = make_envelope_scenario(scenario, envelope_radius_m=float(target_radius_m))

    # Build the robot config once and share it across the probes that consume it, avoiding
    # repeated map parsing. The oracle probe rebuilds internally via run_feasibility_oracle.
    # Any shared initialization failure must become a machine-readable no-go rather than an
    # exception that lets the runner report a usage error.
    try:
        shared_config = build_robot_config_from_scenario(
            dict(declared), scenario_path=scenario_path
        )
        runtime_binding = _build_runtime_binding(shared_config)
    except _CANARY_PROBE_ERRORS as exc:
        # The simulator-facing surfaces share this initialization. If it is not observable,
        # report every surface as no-go instead of letting the runner emit a usage error.
        surfaces = _failed_surfaces(BINDING_SURFACES, target_radius_m, tolerance_m, exc)
    else:
        surfaces = (
            probe_sim_collision_geometry(
                declared,
                target_radius_m,
                scenario_path=scenario_path,
                tolerance_m=tolerance_m,
                robot_config=shared_config,
                runtime_binding=runtime_binding,
                require_runtime=True,
            ),
            probe_contact_logic(
                declared,
                target_radius_m,
                scenario_path=scenario_path,
                tolerance_m=tolerance_m,
                robot_config=shared_config,
                runtime_binding=runtime_binding,
                require_runtime=True,
            ),
            probe_feasibility_oracle(
                declared,
                target_radius_m,
                scenario_path=scenario_path,
                tolerance_m=tolerance_m,
                certifier=certifier,
                episode_runner=episode_runner,
            ),
            probe_metric_metadata(
                declared,
                target_radius_m,
                scenario_path=scenario_path,
                tolerance_m=tolerance_m,
                robot_config=shared_config,
            ),
            probe_planner_inputs(
                declared,
                target_radius_m,
                scenario_path=scenario_path,
                tolerance_m=tolerance_m,
                robot_config=shared_config,
                runtime_binding=runtime_binding,
                require_runtime=True,
            ),
        )
    go = all(verdict.bound for verdict in surfaces)
    scenario_id = str(
        scenario.get("name")
        or (scenario.get("metadata") or {}).get("scenario_id")
        or scenario_path.stem
    )
    verdict = RadiusBindingCanaryVerdict(
        schema=CANARY_SCHEMA,
        scenario_id=scenario_id,
        scenario_path=str(scenario_path),
        target_radius_m=float(target_radius_m),
        go=go,
        surfaces=surfaces,
        generated_at=datetime.now(UTC).isoformat(),
        claim_boundary=DIAGNOSTIC_CLAIM_BOUNDARY,
        tolerance_m=float(tolerance_m),
    )
    logger.info(
        "radius_binding_canary scenario={scenario_id} radius={radius} go={go} unbound={unbound}",
        scenario_id=scenario_id,
        radius=float(target_radius_m),
        go=go,
        unbound=[v.surface for v in surfaces if not v.bound],
    )
    return verdict


def surface_verdict_to_dict(verdict: SurfaceVerdict) -> dict[str, Any]:
    """Return a JSON-serializable view of one surface verdict."""
    return {
        "surface": verdict.surface,
        "expected_radius_m": verdict.expected_radius_m,
        "observed_radius_m": verdict.observed_radius_m,
        "bound": verdict.bound,
        "tolerance_m": verdict.tolerance_m,
        "evidence": dict(verdict.evidence),
        "note": verdict.note,
    }


def canary_verdict_to_dict(verdict: RadiusBindingCanaryVerdict) -> dict[str, Any]:
    """Return a JSON-serializable view of a canary verdict (schema ``radius_binding_canary.v1``)."""
    return {
        "schema": verdict.schema,
        "scenario_id": verdict.scenario_id,
        "scenario_path": verdict.scenario_path,
        "target_radius_m": verdict.target_radius_m,
        "go": verdict.go,
        "surfaces": [surface_verdict_to_dict(surface) for surface in verdict.surfaces],
        "generated_at": verdict.generated_at,
        "claim_boundary": verdict.claim_boundary,
        "tolerance_m": verdict.tolerance_m,
    }


__all__ = [
    "BINDING_SURFACES",
    "CAMPAIGN_ENVELOPE_RADII_M",
    "CANARY_SCHEMA",
    "DEFAULT_TOLERANCE_M",
    "DIAGNOSTIC_CLAIM_BOUNDARY",
    "SURFACE_CONTACT_LOGIC",
    "SURFACE_FEASIBILITY_ORACLE",
    "SURFACE_METRIC_METADATA",
    "SURFACE_PLANNER_INPUTS",
    "SURFACE_SIM_COLLISION_GEOMETRY",
    "RadiusBindingCanaryVerdict",
    "SurfaceVerdict",
    "canary_verdict_to_dict",
    "probe_contact_logic",
    "probe_feasibility_oracle",
    "probe_metric_metadata",
    "probe_planner_inputs",
    "probe_sim_collision_geometry",
    "run_radius_binding_canary",
    "surface_verdict_to_dict",
    "validate_tolerance_m",
]
