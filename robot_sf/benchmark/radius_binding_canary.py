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

1. ``simulator_collision_geometry`` -- the robot collision envelope the simulator builds
   (``robot.config.radius``, which also sizes the pedestrian reserved zone).
2. ``obstacle_pedestrian_contact_logic`` -- the radius-aware contact boundary
   (``robot_radius + ped_radius``) used by the benchmark clearance/contact regime.
3. ``feasibility_oracle`` -- the planner-free oracle's envelope injection and geometric
   inflation (``envelope_radius_m`` / ``envelope_diameter_m``).
4. ``metric_metadata_and_output_rows`` -- the radius the benchmark records in metric
   metadata and output rows (runner row extraction + orchestrator metric-data binding).
5. ``planner_inputs`` -- the radius injected into planner/force inputs that consume it
   (ped-robot and adversarial-ped force configs, mirroring the simulator wiring).

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

#: Fixed radius treatment from the #6600 campaign (metres): 0.5, 0.8, and the 1.0 m
#: release baseline.
CAMPAIGN_ENVELOPE_RADII_M: tuple[float, ...] = (0.5, 0.8, 1.0)


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


def _radius_matches(observed: float | None, expected: float, tolerance_m: float) -> bool:
    """Return whether an observed radius matches the declared target within tolerance."""
    return (
        observed is not None
        and math.isfinite(float(observed))
        and abs(float(observed) - float(expected)) <= tolerance_m
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
) -> SurfaceVerdict:
    """Probe the simulator collision-geometry binding surface.

    The simulator's robot collision circle is ``robot.config.radius``; the same value
    sizes the pedestrian reserved zone (``max(robot.config.radius)``). This probe builds
    the robot config the simulator builds and reads the radius back.

    Returns:
        Verdict with ``observed_radius_m`` = the robot collision envelope radius.
    """
    surface = SURFACE_SIM_COLLISION_GEOMETRY
    try:
        cfg = _robot_config(declared, scenario_path, robot_config)
        observed = float(cfg.robot_config.radius)
        bound = _radius_matches(observed, target_radius_m, tolerance_m)
        return SurfaceVerdict(
            surface=surface,
            expected_radius_m=float(target_radius_m),
            observed_radius_m=observed,
            bound=bound,
            tolerance_m=float(tolerance_m),
            evidence={
                "robot_config_type": type(cfg.robot_config).__name__,
                "robot_config_radius_m": observed,
                "reserved_zone_radius_m": max(observed, 0.0),
                "binding": "robot.config.radius -> simulator collision circle + reserved zone",
            },
            note="" if bound else "simulator collision geometry did not bind the declared radius",
        )
    except Exception as exc:  # noqa: BLE001 - canary must fail closed on probe errors.
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def probe_contact_logic(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    robot_config: RobotSimulationConfig | None = None,
) -> SurfaceVerdict:
    """Probe the obstacle/pedestrian contact-logic binding surface.

    The benchmark contact regime treats robot-pedestrian contact as a collision when
    ``clearance = center_distance - (robot_radius + ped_radius) < 0``. This probe reads
    the robot envelope radius the simulator carries (the contact geometry), forms the
    radius-aware contact boundary, and verifies the clearance classifier flips at that
    boundary -- proving the contact logic consumes the declared radius.

    Returns:
        Verdict with ``observed_radius_m`` = the robot radius bound into the contact
        boundary.
    """
    surface = SURFACE_CONTACT_LOGIC
    try:
        from robot_sf.benchmark.collision_definition_inventory import (  # noqa: PLC0415
            LABEL_COLLISION,
            LABEL_NEAR_MISS,
            classify_clearance_regime,
        )
        from robot_sf.benchmark.constants import NEAR_MISS_DIST  # noqa: PLC0415

        cfg = _robot_config(declared, scenario_path, robot_config)
        robot_radius = float(cfg.robot_config.radius)
        ped_radius = float(cfg.sim_config.ped_radius)
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
        bound = _radius_matches(robot_radius, target_radius_m, tolerance_m) and boundary_binds
        note = ""
        if not _radius_matches(robot_radius, target_radius_m, tolerance_m):
            note = "contact logic robot radius did not bind the declared radius"
        elif not boundary_binds:
            note = "contact boundary did not flip at robot_radius + ped_radius"
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
                "reserved_zone_radius_m": max(robot_radius, 0.0),
                "binding": "clearance = center_distance - (robot_radius + ped_radius)",
                "expected_inside_label": LABEL_COLLISION,
                "expected_outside_label_not": LABEL_COLLISION,
                "near_miss_label": LABEL_NEAR_MISS,
            },
            note=note,
        )
    except Exception as exc:  # noqa: BLE001 - canary must fail closed on probe errors.
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
    except Exception as exc:  # noqa: BLE001 - canary must fail closed on probe errors.
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

    The benchmark records the robot radius in metric metadata and output rows through two
    real extraction paths: the runner's row-level scenario radius extraction
    (``_scenario_robot_radius_m``) and the orchestrator's metric-data binding
    (``getattr(robot_cfg, "radius", 1.0)``). This probe reads both and verifies each
    records the declared radius.

    Returns:
        Verdict with ``observed_radius_m`` = the runner-recorded output-row radius.
    """
    surface = SURFACE_METRIC_METADATA
    try:
        from robot_sf.benchmark.runner import _scenario_robot_radius_m  # noqa: PLC0415

        runner_radius = float(_scenario_robot_radius_m(dict(declared)))
        cfg = _robot_config(declared, scenario_path, robot_config)
        orchestrator_radius = float(getattr(cfg.robot_config, "radius", 1.0))
        bound = _radius_matches(runner_radius, target_radius_m, tolerance_m) and _radius_matches(
            orchestrator_radius, target_radius_m, tolerance_m
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
                "orchestrator_metric_robot_radius_m": orchestrator_radius,
                "output_row_key": "robot_radius",
                "binding": "robot_config.radius -> metric metadata + output rows",
            },
            note=note,
        )
    except Exception as exc:  # noqa: BLE001 - canary must fail closed on probe errors.
        return _failed_verdict(surface, target_radius_m, tolerance_m, exc)


def probe_planner_inputs(
    declared: Mapping[str, Any],
    target_radius_m: float,
    *,
    scenario_path: Path,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    robot_config: RobotSimulationConfig | None = None,
) -> SurfaceVerdict:
    """Probe the planner-input binding surface.

    The simulator injects ``robot.config.radius`` into the planner/force inputs that
    consume the radius: the ped-robot force (``PedRobotForceConfig.robot_radius``) and the
    adversarial-ped force (``AdversarialPedForceConfig.robot_radius``), via
    ``replace(config, robot_radius=robot.config.radius)``. This probe reproduces that
    binding from the declared robot config and verifies the planner inputs receive the
    declared radius.

    Returns:
        Verdict with ``observed_radius_m`` = the radius bound into the planner force inputs.
    """
    surface = SURFACE_PLANNER_INPUTS
    try:
        from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig  # noqa: PLC0415
        from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig  # noqa: PLC0415

        cfg = _robot_config(declared, scenario_path, robot_config)
        robot_radius = float(cfg.robot_config.radius)
        prf_radius = float(replace(PedRobotForceConfig(), robot_radius=robot_radius).robot_radius)
        apf_radius = float(
            replace(AdversarialPedForceConfig(), robot_radius=robot_radius).robot_radius
        )
        bound = (
            _radius_matches(robot_radius, target_radius_m, tolerance_m)
            and _radius_matches(prf_radius, target_radius_m, tolerance_m)
            and _radius_matches(apf_radius, target_radius_m, tolerance_m)
        )
        note = "" if bound else "planner inputs did not consume the declared radius"
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
                "binding": "replace(force_config, robot_radius=robot.config.radius)",
            },
            note=note,
        )
    except Exception as exc:  # noqa: BLE001 - canary must fail closed on probe errors.
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
    if not math.isfinite(float(target_radius_m)) or float(target_radius_m) <= 0.0:
        raise ValueError("target_radius_m must be finite and positive")
    declared = make_envelope_scenario(scenario, envelope_radius_m=float(target_radius_m))

    # Build the robot config once and share it across the probes that consume it, avoiding
    # repeated map parsing. The oracle probe rebuilds internally via run_feasibility_oracle.
    shared_config = build_robot_config_from_scenario(dict(declared), scenario_path=scenario_path)
    surfaces = (
        probe_sim_collision_geometry(
            declared,
            target_radius_m,
            scenario_path=scenario_path,
            tolerance_m=tolerance_m,
            robot_config=shared_config,
        ),
        probe_contact_logic(
            declared,
            target_radius_m,
            scenario_path=scenario_path,
            tolerance_m=tolerance_m,
            robot_config=shared_config,
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
]
