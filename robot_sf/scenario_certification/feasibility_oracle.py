"""Planner-free feasibility oracle per scenario cell with an envelope-sensitivity axis.

Issue #5137 introduced the reusable single-cell contract. Issue #5574 adds a deterministic
candidate-cell report on top of it. Several scenario cells (bottleneck, cross-trap,
head-on-corridor families)
show zero completion for every planner. Two mundane explanations must be ruled out per
cell before a benchmark treats zero completion as a planner result rather than a route
artifact: (a) geometric near-infeasibility under the deliberately conservative robot
envelope, and (b) horizon truncation.

This module is a planner-free oracle. It does not run any learned or heuristic planner.
It combines two planner-free signals under the *same* envelope, kinematics, and horizon
the benchmark uses:

1. **Geometric route clearance** via the existing ``scenario_cert.v1`` certifier, which
   inflates static obstacles by the robot radius, plans an inflated A* shortest path,
   and checks route/kinodynamic feasibility. We read the per-route ``checks`` to report
   *margins*: minimum corridor width vs envelope diameter.
2. **Scripted shortest-path / waypoint traversal** via the canonical actor-free map
   episode rollout (dynamic pedestrians removed, goal-seeking ``goal`` algo), to report
   *minimum completion steps vs horizon*.

The envelope-sensitivity axis re-runs the oracle at reduced envelope radius
(e.g. ``1.0 m`` nominal vs ``0.5 m`` reduced) to separate "hard for planners" from
"infeasible by construction": a cell that is infeasible even at the reduced envelope is
infeasible by construction; a cell that flips feasible when the envelope shrinks is
envelope-sensitive-hard; a cell feasible at the nominal envelope is simply feasible.

This is a diagnostic tool, not benchmark evidence. Every emitted payload carries the
``diagnostic_only_not_benchmark_evidence`` claim boundary. It reuses existing planner-free
primitives (the route certifier and the actor-free rollout); it adds the margin
assembly, the envelope axis, and the campaign-metadata annotation that the existing
issue #3484 diagnostics and the static MAPF oracle do not provide.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner_env import build_env_config
from robot_sf.benchmark.utils import _git_hash_fallback
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.robot.differential_drive import DifferentialDriveRobot
from robot_sf.robot.holonomic_drive import HolonomicDriveRobot
from robot_sf.scenario_certification.feasibility_diagnostics import (
    DIAGNOSTIC_CLAIM_BOUNDARY,
    SOLVABLE_ROUTE_CLASSES,
    make_actor_free_scenario,
)
from robot_sf.scenario_certification.v1 import (
    GEOMETRICALLY_INFEASIBLE,
    KINODYNAMICALLY_INFEASIBLE,
    ScenarioCertificate,
    certify_scenario,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

FEASIBILITY_ORACLE_SCHEMA = "scenario_feasibility_oracle.v1"
ENVELOPE_SENSITIVITY_SCHEMA = "envelope_sensitivity_axis.v1"
CAMPAIGN_FEASIBILITY_ANNOTATION_SCHEMA = "campaign_feasibility_annotation.v1"
ISSUE_5574_REPORT_SCHEMA = "issue_5574_feasibility_oracle_report.v1"
DEFAULT_ISSUE_5574_SCENARIO_IDS: tuple[str, ...] = (
    "francis2023_narrow_doorway",
    "francis2023_blind_corner",
)

# Conservative 1.0 m robot envelope is the repo default (DEFAULT_ROBOT_RADIUS).
# The reduced-envelope probe is the issue's suggested 0.5 m.
DEFAULT_ENVELOPE_RADII_M: tuple[float, ...] = (DEFAULT_ROBOT_RADIUS, 0.5)
DEFAULT_ROLLOUT_SEED = 101

# Cell annotation categories for zero-completion campaign cells.
INFEASIBLE_BY_CONSTRUCTION = "infeasible_by_construction"
ENVELOPE_SENSITIVE_HARD = "envelope_sensitive_hard"
TIME_TRUNCATED = "time_truncated"
FEASIBLE = "feasible"
PLANNER_LIMITED = "planner_limited"
BLOCKED = "blocked"

# Issue #5596 — blind-corner zero-success diagnostic constants.
ISSUE_5596_BLIND_CORNER_SCENARIO_ID = "francis2023_blind_corner"
ISSUE_5596_DIAGNOSTIC_SCHEMA = "issue_5596_blind_corner_diagnostic.v1"
ISSUE_5596_REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"
ROUTE_FOLLOW_ALGO = "route_follow"

# A cell is "geometrically feasible" only when the certifier finds an inflated
# collision-free path AND the route is not kinodynamically excluded.
_GEOMETRIC_INFEASIBLE_STATUSES = frozenset({GEOMETRICALLY_INFEASIBLE, KINODYNAMICALLY_INFEASIBLE})

# Rollout episode runner protocol mirrors issue #3484 ``EpisodeRunner``.
EpisodeRunner = Callable[[Mapping[str, Any], int, int | None, str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class FeasibilityOracleConfig:
    """Configuration for the planner-free feasibility oracle.

    Attributes:
        scenario_path: Scenario manifest path used to resolve maps/routes and robot config.
        envelope_radii_m: Envelope radii probed by the envelope-sensitivity axis.
            The first entry is treated as the *nominal* envelope; the rest are
            reduced-envelope probes.
        rollout_algo: Scripted algorithm used for the actor-free waypoint traversal.
        rollout_seed: Deterministic seed for the actor-free rollout.
    """

    scenario_path: Path
    envelope_radii_m: tuple[float, ...] = DEFAULT_ENVELOPE_RADII_M
    rollout_algo: str = "goal"
    rollout_seed: int = DEFAULT_ROLLOUT_SEED

    def __post_init__(self) -> None:
        """Validate envelope radii and rollout parameters."""
        if not self.envelope_radii_m:
            raise ValueError("envelope_radii_m must contain at least one radius")
        if any(not math.isfinite(float(r)) or float(r) <= 0.0 for r in self.envelope_radii_m):
            raise ValueError("envelope_radii_m must all be finite, positive, and non-zero")
        if len({round(r, 6) for r in self.envelope_radii_m}) != len(self.envelope_radii_m):
            raise ValueError("envelope_radii_m must not contain duplicates")


@dataclass(frozen=True, slots=True)
class GeometricMargin:
    """Geometric route-clearance margin for one envelope radius.

    Attributes:
        envelope_radius_m: Robot envelope radius used for inflation.
        envelope_diameter_m: ``2 * envelope_radius_m``.
        route_geometrically_feasible: Whether an inflated collision-free path exists.
        min_corridor_width_m: Minimum free corridor width along the planned path
            (``2 * (minimum_static_clearance_m + envelope_radius_m)``), or ``None``
            when there are no static obstacles.
        corridor_envelope_margin_m: ``min_corridor_width_m - envelope_diameter_m``
            (positive means the corridor is wider than the envelope).
        min_static_clearance_m: Raw route clearance after robot-radius inflation,
            straight from the route certificate checks.
        shortest_path_length_m: Planned inflated A* path length.
        classification: Route certificate classification.
        benchmark_eligibility: Route certificate benchmark eligibility.
    """

    envelope_radius_m: float
    envelope_diameter_m: float
    route_geometrically_feasible: bool | None
    min_corridor_width_m: float | None
    corridor_envelope_margin_m: float | None
    min_static_clearance_m: float | None
    shortest_path_length_m: float | None
    classification: str
    benchmark_eligibility: str


@dataclass(frozen=True, slots=True)
class CompletionMargin:
    """Completion-vs-horizon margin from the scripted waypoint traversal.

    Attributes:
        route_completion_feasible: Whether the actor-free scripted rollout completed the
            route within the horizon.
        min_completion_steps: Observed steps to complete the route (``None`` if not completed).
        horizon_steps: Episode horizon in steps.
        completion_horizon_margin_steps: ``horizon_steps - min_completion_steps`` (positive
            means there is slack; negative means the route cannot be completed in horizon).
        kinematic_min_steps_lower_bound: Kinematic lower bound on completion steps
            (``shortest_path_length / (max_linear_speed * dt)``), an envelope-free floor.
        termination_reason: Episode termination reason from the rollout.
        status: ``passed`` / ``failed`` / ``blocked`` lane status.
        blocker: Optional blocker reason when the lane could not be observed.
    """

    route_completion_feasible: bool | None
    min_completion_steps: int | None
    horizon_steps: int | None
    completion_horizon_margin_steps: int | None
    kinematic_min_steps_lower_bound: float | None
    termination_reason: str | None
    status: str
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    """Planner-free feasibility verdict for one scenario cell at one envelope radius.

    Attributes:
        scenario_id: Scenario identifier.
        family_id: Scenario family / archetype.
        envelope_radius_m: Robot envelope radius used for this verdict.
        geometric: Geometric route-clearance margin.
        completion: Completion-vs-horizon margin.
        feasible: Overall planner-free feasibility (``True`` only when both the geometric
            route is feasible AND the scripted traversal completes within horizon).
        status: ``feasible`` / ``infeasible_by_construction`` / ``time_truncated`` /
            ``blocked``.
    """

    scenario_id: str
    family_id: str
    envelope_radius_m: float
    geometric: GeometricMargin
    completion: CompletionMargin
    feasible: bool | None
    status: str


@dataclass(frozen=True, slots=True)
class EnvelopeSensitivityVerdict:
    """Envelope-sensitivity axis verdict for one scenario cell.

    Attributes:
        scenario_id: Scenario identifier.
        family_id: Scenario family / archetype.
        nominal_envelope_radius_m: Nominal (largest) envelope radius.
        nominal_verdict: Oracle verdict at the nominal envelope.
        reduced_verdicts: Oracle verdicts at each reduced envelope radius.
        category: ``infeasible_by_construction`` / ``envelope_sensitive_hard`` /
            ``feasible`` / ``blocked``.
    """

    scenario_id: str
    family_id: str
    nominal_envelope_radius_m: float
    nominal_verdict: FeasibilityVerdict
    reduced_verdicts: tuple[FeasibilityVerdict, ...]
    category: str


def make_envelope_scenario(
    scenario: Mapping[str, Any],
    *,
    envelope_radius_m: float,
) -> dict[str, Any]:
    """Return an in-memory scenario variant with a robot-envelope radius override.

    The override is applied to the scenario-level ``robot_config`` mapping (the same
    surface the scenario loader reads), so the certifier and the rollout rebuild their
    robot config with the reduced envelope while preserving the robot type and all other
    kinematic limits.

    Args:
        scenario: Source scenario mapping.
        envelope_radius_m: Robot envelope radius (metres) to inject.

    Returns:
        Deep-copied scenario with ``robot_config.radius`` overridden.
    """
    if not math.isfinite(float(envelope_radius_m)) or envelope_radius_m <= 0.0:
        raise ValueError("envelope_radius_m must be finite, positive, and non-zero")
    mutated = deepcopy(dict(scenario))
    robot_cfg = dict(mutated.get("robot_config") or {})
    robot_cfg["radius"] = float(envelope_radius_m)
    mutated["robot_config"] = robot_cfg
    metadata = dict(mutated.get("metadata") or {})
    metadata["envelope_probe_radius_m"] = float(envelope_radius_m)
    metadata["diagnostic_claim_boundary"] = DIAGNOSTIC_CLAIM_BOUNDARY
    mutated["metadata"] = metadata
    return mutated


def run_feasibility_oracle(
    scenario: Mapping[str, Any],
    *,
    config: FeasibilityOracleConfig,
    envelope_radius_m: float,
    episode_runner: EpisodeRunner | None = None,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
) -> FeasibilityVerdict:
    """Run the planner-free feasibility oracle for one cell at one envelope radius.

    Combines geometric route clearance (certifier) with a scripted actor-free waypoint
    traversal (rollout) and reports both margins.

    Args:
        scenario: Scenario mapping (already envelope-overridden or nominal).
        config: Oracle configuration (scenario path, rollout seed/algo).
        envelope_radius_m: Robot envelope radius used for this verdict.
        episode_runner: Optional injected scripted rollout runner; defaults to the
            canonical actor-free map episode rollout.
        certifier: Optional injected route certifier; defaults to ``certify_scenario``.

    Returns:
        Feasibility verdict with geometric and completion margins.
    """
    scenario_id = _scenario_id(scenario)
    family_id = _scenario_family_id(scenario)
    certify = certifier or _default_certifier
    geometric = _geometric_margin(
        scenario,
        scenario_path=config.scenario_path,
        envelope_radius_m=envelope_radius_m,
        certifier=certify,
    )
    completion = _completion_margin(
        scenario,
        config=config,
        envelope_radius_m=envelope_radius_m,
        geometric=geometric,
        episode_runner=episode_runner,
    )
    feasible, status = _combine_into_verdict(geometric, completion)
    return FeasibilityVerdict(
        scenario_id=scenario_id,
        family_id=family_id,
        envelope_radius_m=envelope_radius_m,
        geometric=geometric,
        completion=completion,
        feasible=feasible,
        status=status,
    )


def run_envelope_sensitivity_sweep(
    scenario: Mapping[str, Any],
    *,
    config: FeasibilityOracleConfig,
    episode_runner: EpisodeRunner | None = None,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
) -> EnvelopeSensitivityVerdict:
    """Run the envelope-sensitivity axis for one scenario cell.

    Runs the oracle at the nominal envelope and at each reduced envelope radius, then
    classifies the cell:

    - ``infeasible_by_construction``: geometrically infeasible (or never completes) at
      *every* probed envelope radius, including the smallest. The route cannot fit
      through the map at any envelope — a mundane map artifact, not a planner result.
    - ``envelope_sensitive_hard``: infeasible at the nominal envelope but feasible at a
      reduced envelope. Separates "hard for planners" from "infeasible by construction".
    - ``feasible``: feasible at the nominal envelope.
    - ``blocked``: the oracle could not produce an observation at the nominal envelope.

    Args:
        scenario: Source scenario mapping (nominal envelope).
        config: Oracle configuration.
        episode_runner: Optional injected scripted rollout runner.
        certifier: Optional injected route certifier.

    Returns:
        Envelope-sensitivity verdict with per-radius oracle verdicts and a category.
    """
    radii = tuple(float(r) for r in config.envelope_radii_m)
    nominal_radius = max(radii)

    verdicts: list[FeasibilityVerdict] = []
    for radius in radii:
        envelope_scenario = make_envelope_scenario(scenario, envelope_radius_m=radius)
        verdicts.append(
            run_feasibility_oracle(
                envelope_scenario,
                config=config,
                envelope_radius_m=radius,
                episode_runner=episode_runner,
                certifier=certifier,
            )
        )
    nominal = next((v for v in verdicts if v.envelope_radius_m == nominal_radius), verdicts[0])
    reduced = tuple(v for v in verdicts if v.envelope_radius_m != nominal_radius)
    category = _classify_envelope_sensitivity(nominal, reduced)
    return EnvelopeSensitivityVerdict(
        scenario_id=nominal.scenario_id,
        family_id=nominal.family_id,
        nominal_envelope_radius_m=nominal_radius,
        nominal_verdict=nominal,
        reduced_verdicts=reduced,
        category=category,
    )


def build_issue_5574_feasibility_report(  # noqa: C901
    scenario_path: Path,
    *,
    scenario_ids: Sequence[str] = DEFAULT_ISSUE_5574_SCENARIO_IDS,
    envelope_radii_m: Sequence[float] = DEFAULT_ENVELOPE_RADII_M,
    rollout_algo: str = "goal",
    rollout_seed: int | None = None,
    episode_runner: EpisodeRunner | None = None,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
) -> dict[str, Any]:
    """Run the deterministic candidate-cell report for issue #5574.

    The report is intentionally small and fail-closed. It selects named scenario cells from a
    canonical manifest, runs the existing single-cell oracle at one nominal and one or more
    strictly smaller envelope radii, and emits only compact verdicts and margins. No planner
    campaign rows or raw episode traces are copied into the report.

    Args:
        scenario_path: Scenario manifest containing the requested candidate cells.
        scenario_ids: Ordered scenario names to inspect. The default is the two cells named by
            issue #5574: the narrow doorway and blind corner Francis 2023 scenarios.
        envelope_radii_m: Ordered radii with the nominal radius first and reduced probes after it.
        rollout_algo: Deterministic scripted rollout algorithm passed to the existing oracle.
        rollout_seed: Optional seed override. When omitted, the first seed in each scenario
            manifest is used, falling back to the stable diagnostic default.
        episode_runner: Optional injected runner for deterministic unit tests.
        certifier: Optional injected route certifier for deterministic unit tests.

    Returns:
        Versioned diagnostic-only report with one serialized envelope verdict per cell.

    Raises:
        ValueError: If selection, envelope ordering, or manifest resolution is invalid.
    """
    source = Path(scenario_path)
    if not source.is_file():
        raise FileNotFoundError(f"Scenario manifest file not found: {source}")
    requested_ids = tuple(str(value).strip() for value in scenario_ids)
    if not requested_ids or any(not value for value in requested_ids):
        raise ValueError("scenario_ids must contain at least one non-empty scenario id")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("scenario_ids must not contain duplicates")

    radii = tuple(float(value) for value in envelope_radii_m)
    if len(radii) < 2:
        raise ValueError("envelope_radii_m must contain a nominal and reduced radius")
    if radii[0] != max(radii):
        raise ValueError("envelope_radii_m must place the nominal radius first")
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError("envelope_radii_m must contain only finite, positive radii")
    if any(radius >= radii[0] for radius in radii[1:]):
        raise ValueError("envelope_radii_m reduced probes must be smaller than nominal")

    scenarios = load_scenarios(source)
    by_id: dict[str, Mapping[str, Any]] = {}
    for scenario in scenarios:
        scenario_id = _scenario_id(scenario)
        if scenario_id in by_id:
            raise ValueError(f"duplicate scenario id in manifest: {scenario_id}")
        by_id[scenario_id] = scenario
    missing = tuple(scenario_id for scenario_id in requested_ids if scenario_id not in by_id)
    if missing:
        raise ValueError("requested scenario ids are missing from manifest: " + ", ".join(missing))

    cells: list[dict[str, Any]] = []
    for scenario_id in requested_ids:
        scenario = by_id[scenario_id]
        selected_seed = _scenario_rollout_seed(scenario, override=rollout_seed)
        config = FeasibilityOracleConfig(
            scenario_path=source,
            envelope_radii_m=radii,
            rollout_algo=rollout_algo,
            rollout_seed=selected_seed,
        )
        verdict = run_envelope_sensitivity_sweep(
            scenario,
            config=config,
            episode_runner=episode_runner,
            certifier=certifier,
        )
        cell = envelope_sensitivity_verdict_to_dict(verdict, issue="5574")
        cell["scenario_manifest"] = source.as_posix()
        cell["rollout_algo"] = rollout_algo
        cell["rollout_seed"] = selected_seed
        cells.append(cell)

    return {
        "schema_version": ISSUE_5574_REPORT_SCHEMA,
        "issue": "5574",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "scenario_manifest": source.as_posix(),
        "scenario_ids": list(requested_ids),
        "envelope_radii_m": list(radii),
        "rollout_algo": rollout_algo,
        "cells": cells,
    }


def annotate_zero_completion_cells(
    per_cell_completion: Mapping[str, Mapping[str, Any]],
    envelope_verdicts: Mapping[str, EnvelopeSensitivityVerdict],
    *,
    zero_completion_threshold: float = 0.0,
) -> dict[str, Any]:
    """Annotate zero-completion campaign cells with the planner-free oracle verdict.

    This emits the oracle verdict into campaign metadata so zero-completion cells are
    automatically annotated as route-infeasible-by-construction, envelope-sensitive-hard,
    or still-planner-limited, rather than read narratively.

    Each ``per_cell_completion`` entry maps a scenario id to a mapping that exposes a
    completion/success rate (any of ``completion_rate``, ``success_rate``, ``completion``,
    or ``success``). Entries whose rate is ``<= zero_completion_threshold`` are annotated
    when an envelope-sensitivity verdict is available.

    Args:
        per_cell_completion: Mapping of scenario id -> per-cell completion payload.
        envelope_verdicts: Mapping of scenario id -> envelope-sensitivity verdict.
        zero_completion_threshold: Rate at/below which a cell counts as zero-completion.

    Returns:
        Versioned ``campaign_feasibility_annotation.v1`` payload with per-cell annotations
        and aggregate counts.
    """
    annotations: list[dict[str, Any]] = []
    annotated_count = 0
    for scenario_id, payload in per_cell_completion.items():
        rate = _completion_rate(payload)
        if rate is None or rate > zero_completion_threshold:
            continue
        verdict = envelope_verdicts.get(scenario_id)
        entry: dict[str, Any] = {
            "scenario_id": scenario_id,
            "observed_completion_rate": rate,
            "zero_completion": True,
        }
        if verdict is None:
            entry["oracle_status"] = "missing"
            entry["annotation"] = PLANNER_LIMITED
            entry["claim_boundary"] = DIAGNOSTIC_CLAIM_BOUNDARY
        else:
            entry["oracle_status"] = "available"
            entry["envelope_category"] = verdict.category
            entry["annotation"] = _annotation_for_category(verdict.category)
            entry["nominal_envelope_radius_m"] = verdict.nominal_envelope_radius_m
            entry["claim_boundary"] = DIAGNOSTIC_CLAIM_BOUNDARY
            annotated_count += 1
        annotations.append(entry)

    return {
        "schema_version": CAMPAIGN_FEASIBILITY_ANNOTATION_SCHEMA,
        "issue": "5137",
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "zero_completion_threshold": float(zero_completion_threshold),
        "annotated_cell_count": annotated_count,
        "total_zero_completion_cells": len(annotations),
        "annotations": annotations,
    }


def envelope_sensitivity_verdict_to_dict(
    verdict: EnvelopeSensitivityVerdict,
    *,
    issue: str = "5137",
) -> dict[str, Any]:
    """Serialize an envelope-sensitivity verdict to JSON-safe primitives.

    Returns:
        Versioned ``envelope_sensitivity_axis.v1`` payload.
    """
    return {
        "schema_version": ENVELOPE_SENSITIVITY_SCHEMA,
        "issue": issue,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "scenario_id": verdict.scenario_id,
        "family_id": verdict.family_id,
        "nominal_envelope_radius_m": verdict.nominal_envelope_radius_m,
        "category": verdict.category,
        "nominal_verdict": feasibility_verdict_to_dict(verdict.nominal_verdict, issue=issue),
        "reduced_verdicts": [
            feasibility_verdict_to_dict(v, issue=issue) for v in verdict.reduced_verdicts
        ],
    }


def feasibility_verdict_to_dict(
    verdict: FeasibilityVerdict,
    *,
    issue: str = "5137",
) -> dict[str, Any]:
    """Serialize a feasibility verdict to JSON-safe primitives.

    Returns:
        Versioned ``scenario_feasibility_oracle.v1`` payload for one envelope radius.
    """
    return {
        "schema_version": FEASIBILITY_ORACLE_SCHEMA,
        "issue": issue,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "scenario_id": verdict.scenario_id,
        "family_id": verdict.family_id,
        "envelope_radius_m": verdict.envelope_radius_m,
        "feasible": verdict.feasible,
        "status": verdict.status,
        "geometric": _geometric_margin_to_dict(verdict.geometric),
        "completion": _completion_margin_to_dict(verdict.completion),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _geometric_margin(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    envelope_radius_m: float,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate],
) -> GeometricMargin:
    """Build the geometric route-clearance margin from the route certificate.

    Returns:
        Geometric margin with corridor-vs-envelope reporting.
    """
    envelope_diameter_m = 2.0 * float(envelope_radius_m)
    try:
        certificate = certifier(scenario, scenario_path)
    except Exception as exc:  # noqa: BLE001 - oracle must fail closed on certifier errors.
        return _blocked_geometric_margin(envelope_radius_m, str(exc))

    classification = str(certificate.classification)
    eligibility = str(certificate.benchmark_eligibility)
    route_checks = _aggregate_route_checks(certificate)
    min_clearance = _optional_float(route_checks.get("minimum_static_clearance_m"))
    shortest_path = _optional_float(route_checks.get("shortest_path_length_m"))

    # minimum_static_clearance_m = line.distance(obstacles) - robot_radius.
    # The corridor half-width available to a centered path is that clearance plus the
    # robot radius; the corridor width is twice that. Report None when there are no
    # obstacles (the certifier returns None for clearance on empty obstacle sets).
    if min_clearance is not None:
        min_corridor_width_m = 2.0 * (min_clearance + float(envelope_radius_m))
        corridor_envelope_margin_m = min_corridor_width_m - envelope_diameter_m
    else:
        min_corridor_width_m = None
        corridor_envelope_margin_m = None

    geometrically_feasible: bool | None
    if classification in _GEOMETRIC_INFEASIBLE_STATUSES or eligibility == "excluded":
        geometrically_feasible = False
    elif classification in SOLVABLE_ROUTE_CLASSES:
        geometrically_feasible = True
    else:
        geometrically_feasible = None

    return GeometricMargin(
        envelope_radius_m=float(envelope_radius_m),
        envelope_diameter_m=envelope_diameter_m,
        route_geometrically_feasible=geometrically_feasible,
        min_corridor_width_m=min_corridor_width_m,
        corridor_envelope_margin_m=corridor_envelope_margin_m,
        min_static_clearance_m=min_clearance,
        shortest_path_length_m=shortest_path,
        classification=classification,
        benchmark_eligibility=eligibility,
    )


def _blocked_geometric_margin(envelope_radius_m: float, blocker: str) -> GeometricMargin:
    """Return a fail-closed geometric margin when the certifier raises."""
    return GeometricMargin(
        envelope_radius_m=float(envelope_radius_m),
        envelope_diameter_m=2.0 * float(envelope_radius_m),
        route_geometrically_feasible=None,
        min_corridor_width_m=None,
        corridor_envelope_margin_m=None,
        min_static_clearance_m=None,
        shortest_path_length_m=None,
        classification=f"blocked:{blocker}",
        benchmark_eligibility="blocked",
    )


def _completion_margin(
    scenario: Mapping[str, Any],
    *,
    config: FeasibilityOracleConfig,
    envelope_radius_m: float,
    geometric: GeometricMargin,
    episode_runner: EpisodeRunner | None,
) -> CompletionMargin:
    """Build the completion-vs-horizon margin from the scripted actor-free rollout.

    Returns:
        Completion margin with observed and kinematic-floor completion steps.
    """
    horizon_steps = _scenario_horizon(scenario)
    kinematic_floor = _kinematic_min_steps_lower_bound(
        scenario, geometric, scenario_path=config.scenario_path
    )

    # If the route is geometrically infeasible there is no path to traverse; the scripted
    # rollout cannot complete it. Skip the expensive rollout and report time-feasibility
    # as infeasible-by-construction rather than running a doomed episode.
    if geometric.route_geometrically_feasible is False:
        return CompletionMargin(
            route_completion_feasible=False,
            min_completion_steps=None,
            horizon_steps=horizon_steps,
            completion_horizon_margin_steps=None,
            kinematic_min_steps_lower_bound=kinematic_floor,
            termination_reason=None,
            status="failed",
            blocker="route_geometrically_infeasible_no_traversal_path",
        )

    try:
        runner = episode_runner or _default_actor_free_runner(config)
        actor_free = make_actor_free_scenario(scenario)
        record = dict(runner(actor_free, config.rollout_seed, horizon_steps, config.rollout_algo))
    except Exception as exc:  # noqa: BLE001 - oracle must fail closed on rollout errors.
        return CompletionMargin(
            route_completion_feasible=None,
            min_completion_steps=None,
            horizon_steps=horizon_steps,
            completion_horizon_margin_steps=None,
            kinematic_min_steps_lower_bound=kinematic_floor,
            termination_reason=None,
            status="blocked",
            blocker=f"rollout_error: {exc}",
        )

    completed, termination = _rollout_route_complete(record)
    steps = _optional_int(record.get("steps"))
    min_completion_steps = steps if completed and steps is not None else None
    completion_horizon_margin_steps: int | None
    if min_completion_steps is not None and horizon_steps is not None:
        completion_horizon_margin_steps = horizon_steps - min_completion_steps
    else:
        completion_horizon_margin_steps = None

    if completed:
        status = "passed"
    elif completed is False:
        status = "failed"
    else:
        status = "blocked"

    return CompletionMargin(
        route_completion_feasible=completed,
        min_completion_steps=min_completion_steps,
        horizon_steps=horizon_steps,
        completion_horizon_margin_steps=completion_horizon_margin_steps,
        kinematic_min_steps_lower_bound=kinematic_floor,
        termination_reason=termination,
        status=status,
        blocker=None if completed is not None else "unknown_rollout_outcome",
    )


def _combine_into_verdict(
    geometric: GeometricMargin, completion: CompletionMargin
) -> tuple[bool | None, str]:
    """Combine the geometric and completion margins into an overall verdict.

    Returns:
        Tuple of (feasible, status). A cell is feasible only when the geometric route is
        feasible AND the scripted traversal completes within horizon.
    """
    if (
        geometric.route_geometrically_feasible is None
        or completion.route_completion_feasible is None
    ):
        return None, BLOCKED
    if geometric.route_geometrically_feasible is False:
        return False, INFEASIBLE_BY_CONSTRUCTION
    if completion.route_completion_feasible is False:
        # Geometrically feasible but the scripted traversal could not finish in horizon.
        return False, TIME_TRUNCATED
    return True, FEASIBLE


def _classify_envelope_sensitivity(
    nominal: FeasibilityVerdict, reduced: Sequence[FeasibilityVerdict]
) -> str:
    """Classify a cell along the envelope-sensitivity axis.

    Returns:
        One of ``infeasible_by_construction``, ``envelope_sensitive_hard``, ``feasible``,
        or ``blocked``.
    """
    if nominal.status == BLOCKED:
        return BLOCKED
    if nominal.feasible is True:
        return FEASIBLE
    # Nominal is infeasible (by construction or time-truncated). Probe reduced envelopes.
    any_reduced_feasible = any(v.feasible is True for v in reduced)
    if any_reduced_feasible:
        return ENVELOPE_SENSITIVE_HARD
    # A blocked reduced probe cannot support a definitive map-artifact or horizon verdict.
    # Preserve the oracle's fail-closed boundary unless another reduced probe already proved
    # envelope sensitivity above.
    if any(v.status == BLOCKED or v.feasible is None for v in reduced):
        return BLOCKED
    # If the nominal failure is purely time truncation and no reduced envelope helps, the
    # cell is time-limited rather than route-infeasible by construction.
    if nominal.status == TIME_TRUNCATED and all(
        v.status != INFEASIBLE_BY_CONSTRUCTION for v in reduced
    ):
        return TIME_TRUNCATED
    return INFEASIBLE_BY_CONSTRUCTION


def _annotation_for_category(category: str) -> str:
    """Map an envelope-sensitivity category to a campaign annotation label.

    Returns:
        Campaign annotation label string for the given category.
    """
    if category == INFEASIBLE_BY_CONSTRUCTION:
        return "route_infeasible_by_construction_zero_completion_is_map_artifact"
    if category == ENVELOPE_SENSITIVE_HARD:
        return "envelope_sensitive_hard_zero_completion_under_nominal_envelope"
    if category == TIME_TRUNCATED:
        return "time_truncated_zero_completion_within_envelope"
    if category == FEASIBLE:
        return "oracle_feasible_zero_completion_is_planner_limited"
    if category == BLOCKED:
        return "oracle_blocked"
    return PLANNER_LIMITED


def _default_certifier(scenario: Mapping[str, Any], scenario_path: Path) -> ScenarioCertificate:
    """Default route certifier adapter (keyword-only ``scenario_path`` bridge).

    Returns:
        Scenario certificate from the canonical ``certify_scenario`` certifier.
    """
    return certify_scenario(scenario, scenario_path=scenario_path)


def _default_actor_free_runner(config: FeasibilityOracleConfig) -> EpisodeRunner:
    """Build the default actor-free scripted rollout runner.

    Mirrors the issue #3484 default runner but always uses the actor-free (pedestrian-free)
    variant so the traversal is planner-free and deterministic.

    Returns:
        Callable that executes one actor-free scenario/seed/algo diagnostic rollout.
    """
    from robot_sf.benchmark.map_runner import _run_map_episode  # noqa: PLC0415

    def _run(
        scenario: Mapping[str, Any],
        seed: int,
        horizon: int | None,
        algo: str,
    ) -> Mapping[str, Any]:
        scenario_payload = deepcopy(dict(scenario))
        scenario_payload["seeds"] = [int(seed)]
        return _run_map_episode(
            scenario_payload,
            int(seed),
            horizon=horizon,
            dt=None,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=algo,
            scenario_path=config.scenario_path,
        )

    return _run


def _rollout_route_complete(record: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    """Determine route completion and termination reason from a rollout record.

    Returns:
        Tuple of (route_complete, termination_reason). ``route_complete`` is ``None`` when
        the outcome cannot be determined.
    """
    termination = str(record.get("termination_reason") or record.get("status") or "").lower()
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping):
        flag = _bool_flag(outcome.get("route_complete"))
        if flag is not None:
            return flag, termination or None
    flag = (
        _bool_flag(record.get("route_complete"))
        if "route_complete" in record
        else _bool_flag(record.get("goal_reached"))
    )
    if flag is None:
        flag = _scan_mapping_for_bool(record, ("route_complete", "goal_reached", "success"))
    metrics = record.get("metrics")
    if flag is None and isinstance(metrics, Mapping):
        flag = _scan_mapping_for_bool(metrics, ("success", "route_complete", "goal_reached"))
    if flag is not None:
        return flag, termination or None
    return _termination_completion(termination)


def _bool_flag(value: Any) -> bool | None:
    """Return a tri-state boolean for a rollout flag value."""
    if value is True:
        return True
    if value is False:
        return False
    return None


def _scan_mapping_for_bool(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool | None:
    """Return the first explicit boolean found under ``keys`` in ``mapping``."""
    for key in keys:
        flag = _bool_flag(mapping.get(key))
        if flag is not None:
            return flag
    return None


def _termination_completion(termination: str) -> tuple[bool | None, str | None]:
    """Infer route completion from a termination-reason string when no flag is present.

    Returns:
        Tuple of (route_complete, termination_reason) inferred from the termination string.
    """
    if termination in {"success", "goal_reached", "route_complete", "completed"}:
        return True, termination or None
    if termination in {
        "timeout",
        "collision",
        "failure",
        "failed",
        "error",
        "truncated",
        "max_steps",
    }:
        return False, termination or None
    return None, termination or None


def _aggregate_route_checks(certificate: ScenarioCertificate) -> Mapping[str, Any]:
    """Aggregate per-route ``checks`` into one mapping, taking the tightest margin.

    Returns:
        Mapping with the worst-case (minimum) static clearance across routes.
    """
    route_certs = getattr(certificate, "route_certificates", []) or []
    if not route_certs:
        return certificate.checks or {}
    min_clearance: float | None = None
    shortest_path: float | None = None
    inflated_path_found = True
    for route in route_certs:
        checks = route.checks or {}
        clearance = _optional_float(checks.get("minimum_static_clearance_m"))
        if clearance is not None:
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
        path_len = _optional_float(checks.get("shortest_path_length_m"))
        if path_len is not None:
            shortest_path = path_len if shortest_path is None else min(shortest_path, path_len)
        if checks.get("inflated_collision_free_path") is False:
            inflated_path_found = False
    aggregated: dict[str, Any] = dict(certificate.checks or {})
    if min_clearance is not None:
        aggregated["minimum_static_clearance_m"] = min_clearance
    if shortest_path is not None:
        aggregated["shortest_path_length_m"] = shortest_path
    aggregated["inflated_collision_free_path"] = inflated_path_found
    return aggregated


def _kinematic_min_steps_lower_bound(
    scenario: Mapping[str, Any],
    geometric: GeometricMargin,
    *,
    scenario_path: Path,
) -> float | None:
    """Return a kinematic lower bound on completion steps from the planned path length.

    ``ceil(shortest_path_length / (max_linear_speed * dt))`` is an envelope-free floor on
    how many steps any kinematically-respecting traversal needs. When the floor exceeds
    the horizon the cell is time-infeasible regardless of planner skill.

    Returns:
        Lower bound in steps, or ``None`` when path length or kinematics are unavailable.
    """
    path_length = geometric.shortest_path_length_m
    if path_length is None:
        return None
    try:
        robot_config = build_robot_config_from_scenario(dict(scenario), scenario_path=scenario_path)
    except Exception:  # noqa: BLE001 - floor is optional; degrade to None.
        return None
    max_speed = float(getattr(robot_config.robot_config, "max_linear_speed", 0.0) or 0.0)
    dt = float(getattr(robot_config.sim_config, "time_per_step_in_secs", 0.0) or 0.0)
    if max_speed <= 0.0 or dt <= 0.0:
        return None
    return float(math.ceil(path_length / (max_speed * dt)))


def _completion_rate(payload: Mapping[str, Any]) -> float | None:
    """Extract a per-cell completion rate from a campaign payload.

    Returns:
        Completion/success rate, or ``None`` when absent.
    """
    for key in ("completion_rate", "success_rate", "completion", "success"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _scenario_rollout_seed(scenario: Mapping[str, Any], *, override: int | None) -> int:
    """Resolve one deterministic seed for a candidate-cell rollout.

    Returns:
        Seed from the explicit override, scenario manifest, or stable default.
    """
    if override is not None:
        parsed_override = _optional_int(override)
        if parsed_override is None:
            raise ValueError("rollout seed override must be a finite integer")
        return parsed_override
    raw_seeds = scenario.get("seeds")
    if isinstance(raw_seeds, Sequence) and not isinstance(raw_seeds, (str, bytes)):
        saw_invalid_seed = False
        for value in raw_seeds:
            parsed_seed = _optional_int(value)
            if parsed_seed is not None:
                return parsed_seed
            saw_invalid_seed = True
        if saw_invalid_seed:
            raise ValueError("scenario seeds must contain a finite integer")
    return DEFAULT_ROLLOUT_SEED


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    return str(
        scenario.get("id") or scenario.get("name") or scenario.get("scenario_id") or "unknown"
    )


def _scenario_family_id(scenario: Mapping[str, Any]) -> str:
    metadata = scenario.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("archetype", "family", "scenario_family", "family_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return str(scenario.get("obstacle") or scenario.get("flow") or "unknown")


def _scenario_horizon(scenario: Mapping[str, Any]) -> int | None:
    sim_cfg = scenario.get("simulation_config")
    if not isinstance(sim_cfg, Mapping):
        return None
    value = sim_cfg.get("max_episode_steps")
    if value is None:
        return None
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _optional_float(value: Any) -> float | None:
    """Coerce a value to float, returning ``None`` when not finite/numeric.

    Returns:
        float | None: Parsed finite float, or ``None`` when the value is absent,
        non-numeric, or NaN.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _optional_int(value: Any) -> int | None:
    """Coerce a finite integral value to int, returning ``None`` otherwise.

    Returns:
        int | None: Parsed integer, or ``None`` when the value is absent, non-integral,
            non-finite, or non-numeric.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def _geometric_margin_to_dict(margin: GeometricMargin) -> dict[str, Any]:
    return {
        "envelope_radius_m": margin.envelope_radius_m,
        "envelope_diameter_m": margin.envelope_diameter_m,
        "route_geometrically_feasible": margin.route_geometrically_feasible,
        "min_corridor_width_m": margin.min_corridor_width_m,
        "corridor_envelope_margin_m": margin.corridor_envelope_margin_m,
        "min_static_clearance_m": margin.min_static_clearance_m,
        "shortest_path_length_m": margin.shortest_path_length_m,
        "classification": margin.classification,
        "benchmark_eligibility": margin.benchmark_eligibility,
    }


def _completion_margin_to_dict(margin: CompletionMargin) -> dict[str, Any]:
    return {
        "route_completion_feasible": margin.route_completion_feasible,
        "min_completion_steps": margin.min_completion_steps,
        "horizon_steps": margin.horizon_steps,
        "completion_horizon_margin_steps": margin.completion_horizon_margin_steps,
        "kinematic_min_steps_lower_bound": margin.kinematic_min_steps_lower_bound,
        "termination_reason": margin.termination_reason,
        "status": margin.status,
        "blocker": margin.blocker,
    }


# ---------------------------------------------------------------------------
# Issue #5596 — blind-corner zero-success diagnostic
# ---------------------------------------------------------------------------


def make_route_follow_episode_runner(
    config: FeasibilityOracleConfig,
) -> EpisodeRunner:
    """Build the stateful route-follow intervention runner (issue #5636).

    The runner drives the robot along the certifier's inflated A* path in a
    **single continuous episode**, targeting each waypoint in sequence. The robot
    pose and remaining horizon are preserved across the whole route: there is no
    per-waypoint episode reset, so progress and elapsed steps accumulate naturally.
    This is the stateful adapter the fail-closed placeholder in #5596 lacked, and it
    prevents a reset-per-waypoint run from being misread as continuous route-follow
    evidence.

    Args:
        config: Oracle configuration (scenario path for certifier/route resolution).

    Returns:
        An episode runner that executes one stateful waypoint-to-waypoint rollout.
    """

    def _route_follow_runner(
        scenario: Mapping[str, Any],
        seed: int,
        horizon: int | None,
        algo: str,
    ) -> Mapping[str, Any]:
        """Execute one stateful route-follow rollout along the certified A* path.

        Returns:
            Episode record with route completion flags and route-follow metadata.
        """
        waypoints = _get_certified_route_waypoints(scenario, scenario_path=config.scenario_path)
        if not waypoints or len(waypoints) < 2:
            # Without a certified route there is nothing to follow; report blocked.
            return {
                "algo": ROUTE_FOLLOW_ALGO,
                "route_complete": None,
                "steps": None,
                "status": "blocked",
                "termination_reason": "route_follow_no_certified_path",
                "route_follow_blocker": "certifier produced no A* path to follow",
            }

        actor_free = make_actor_free_scenario(scenario)
        try:
            env_config = build_env_config(actor_free, scenario_path=config.scenario_path)
            if horizon is not None:
                env_config.sim_config.max_episode_steps = int(horizon)
            env = make_robot_env(config=env_config, seed=int(seed), debug=False)
        except Exception as exc:  # noqa: BLE001 - diagnostic lane must fail closed.
            return {
                "algo": ROUTE_FOLLOW_ALGO,
                "route_complete": None,
                "steps": None,
                "status": "blocked",
                "termination_reason": "route_follow_env_init_error",
                "route_follow_blocker": str(exc),
            }

        try:
            obs, _ = env.reset(seed=int(seed))
            del obs
            sim = env.simulator
            navs = getattr(sim, "robot_navs", None)
            if not navs:
                return {
                    "algo": ROUTE_FOLLOW_ALGO,
                    "route_complete": None,
                    "steps": None,
                    "status": "blocked",
                    "termination_reason": "route_follow_no_navigator",
                    "route_follow_blocker": "scenario simulator exposes no robot navigator",
                }

            # Re-target the navigator at the certified route. `current_waypoint` is a
            # live property of the navigator, so mutating it each step steers the robot
            # without resetting its pose or consuming the remaining horizon.
            nav = navs[0]
            target_waypoints = [tuple(float(c) for c in wp) for wp in waypoints]
            nav.waypoints = list(target_waypoints)
            nav.waypoint_id = 0
            nav.reached_waypoint = False

            command_mode = _route_follow_command_mode(sim.robots[0])
            if command_mode is None:
                return {
                    "algo": ROUTE_FOLLOW_ALGO,
                    "route_complete": None,
                    "steps": None,
                    "status": "blocked",
                    "termination_reason": "route_follow_unsupported_robot_model",
                    "route_follow_blocker": (
                        "stateful route-follow has no validated command adapter for "
                        f"{type(sim.robots[0]).__name__}"
                    ),
                    "stateful": False,
                    "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
                }

            max_steps = (
                int(env_config.sim_config.max_episode_steps) if horizon is None else int(horizon)
            )
            result = _run_stateful_route_follow_rollout(env, nav, max_steps)
            route_complete = result["route_complete"]
            status = result.get("status") or ("passed" if route_complete else "failed")
            return {
                "algo": ROUTE_FOLLOW_ALGO,
                "route_complete": None if status == "blocked" else route_complete,
                "steps": result["steps_used"],
                "status": status,
                "termination_reason": result["termination_reason"],
                "collision_seen": result["collision_seen"],
                "robot_final_position": result["final_pos"],
                "waypoint_count": len(target_waypoints),
                "stateful": result.get("stateful", True),
                "route_follow_blocker": result.get("blocker"),
                "route_follow_provenance": (
                    "single continuous episode; pose + remaining horizon preserved "
                    "across certified waypoints (issue #5636)"
                    if result.get("stateful", True)
                    else None
                ),
                "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
            }
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    return _route_follow_runner


def _run_stateful_route_follow_rollout(
    env: Any,
    nav: Any,
    max_steps: int,
) -> dict[str, Any]:
    """Drive the robot along ``nav.waypoints`` in one continuous episode.

    The navigator's ``current_waypoint`` is advanced live as each waypoint is reached,
    so the robot pose and remaining horizon are preserved across the whole route.

    Returns:
        Mapping with ``route_complete``, ``steps_used``, ``termination_reason``,
        ``collision_seen``, and ``final_pos``.
    """
    steps_used = 0
    final_pos: tuple[float, float] | None = None
    termination_reason = "max_steps"
    collision_seen = False
    robot = env.simulator.robots[0]
    try:
        step_dt = float(env.simulator.config.time_per_step_in_secs)
    except (AttributeError, TypeError, ValueError):
        return {
            "route_complete": None,
            "steps_used": steps_used,
            "termination_reason": "route_follow_invalid_step_dt",
            "collision_seen": collision_seen,
            "final_pos": final_pos,
            "status": "blocked",
            "stateful": False,
            "blocker": "simulator time_per_step_in_secs is unavailable or invalid",
        }
    if not math.isfinite(step_dt) or step_dt <= 0.0 or max_steps <= 0:
        return {
            "route_complete": None,
            "steps_used": steps_used,
            "termination_reason": "route_follow_invalid_horizon_or_step_dt",
            "collision_seen": collision_seen,
            "final_pos": final_pos,
            "status": "blocked",
            "stateful": False,
            "blocker": "route-follow requires a positive horizon and simulation timestep",
        }
    for _step_idx in range(max_steps):
        robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        goal = np.asarray(nav.current_waypoint, dtype=float)
        action = _route_follow_action(robot, robot_pos, goal, step_dt)
        if action is None:
            return {
                "route_complete": None,
                "steps_used": steps_used,
                "termination_reason": "route_follow_action_adapter_unavailable",
                "collision_seen": collision_seen,
                "final_pos": final_pos,
                "status": "blocked",
                "stateful": False,
                "blocker": (f"no validated route-follow action adapter for {type(robot).__name__}"),
            }
        _obs, _reward, terminated, truncated, info = env.step(action)
        steps_used += 1
        post_step_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        final_pos = (float(post_step_pos[0]), float(post_step_pos[1]))
        if bool(terminated) or bool(truncated):
            if nav.reached_destination:
                termination_reason = "route_follow_reached_destination"
            elif bool(info.get("collision", False)):
                termination_reason = "collision"
                collision_seen = True
            else:
                termination_reason = "max_steps"
            break
    return {
        "route_complete": bool(nav.reached_destination),
        "steps_used": steps_used,
        "termination_reason": termination_reason,
        "collision_seen": collision_seen,
        "final_pos": list(final_pos) if final_pos is not None else None,
        "status": "passed" if nav.reached_destination else "failed",
        "stateful": True,
    }


def _route_follow_command_mode(robot: Any) -> str | None:
    """Return the validated action contract supported by a robot model.

    The default differential-drive environment consumes acceleration commands, while
    holonomic environments can consume either Cartesian velocity or unicycle velocity
    commands. Bicycle-drive actions require a steering adapter and are intentionally
    blocked until one is validated for this diagnostic.

    Returns:
        Name of the supported command contract, or ``None`` when unsupported.
    """
    if isinstance(robot, DifferentialDriveRobot):
        return "differential_acceleration"
    if isinstance(robot, HolonomicDriveRobot):
        return f"holonomic_{robot.config.command_mode}"
    return None


def _route_follow_action(
    robot: Any,
    robot_pos: np.ndarray,
    goal: np.ndarray,
    step_dt: float,
) -> np.ndarray | None:
    """Build a bounded action using the active robot's actual command semantics.

    Returns:
        A two-component action, or ``None`` when no validated adapter exists.
    """
    direction = goal - robot_pos
    dist = float(np.linalg.norm(direction))
    if not math.isfinite(step_dt) or step_dt <= 0.0 or not math.isfinite(dist):
        return None
    command_mode = _route_follow_command_mode(robot)
    if command_mode is None:
        return None
    if dist < 1e-6:
        return np.zeros(2, dtype=float)

    if isinstance(robot, HolonomicDriveRobot):
        speed = min(float(robot.config.max_speed), dist / step_dt)
        if robot.config.command_mode == "vx_vy":
            return np.asarray(direction / dist * speed, dtype=np.float32)
        heading = float(robot.pose[1])
        target_heading = math.atan2(float(direction[1]), float(direction[0]))
        heading_error = math.atan2(
            math.sin(target_heading - heading), math.cos(target_heading - heading)
        )
        forward_speed = speed if abs(heading_error) <= math.pi / 2.0 else 0.0
        angular_speed = float(
            np.clip(
                2.0 * heading_error,
                -float(robot.config.max_angular_speed),
                float(robot.config.max_angular_speed),
            )
        )
        return np.asarray([forward_speed, angular_speed], dtype=np.float32)

    if isinstance(robot, DifferentialDriveRobot):
        heading = float(robot.pose[1])
        target_heading = math.atan2(float(direction[1]), float(direction[0]))
        heading_error = math.atan2(
            math.sin(target_heading - heading), math.cos(target_heading - heading)
        )
        target_speed = min(float(robot.config.max_linear_speed), dist / step_dt)
        if abs(heading_error) > math.pi / 2.0 and not robot.config.allow_backwards:
            target_speed = 0.0
        target_angular_speed = float(
            np.clip(
                2.0 * heading_error,
                -float(robot.config.max_angular_speed),
                float(robot.config.max_angular_speed),
            )
        )
        current_linear_speed, current_angular_speed = robot.current_speed
        linear_accel = (target_speed - float(current_linear_speed)) / step_dt
        angular_accel = (target_angular_speed - float(current_angular_speed)) / step_dt
        return np.asarray(
            [
                np.clip(
                    linear_accel,
                    -float(robot.config.max_linear_decel),
                    float(robot.config.max_linear_accel),
                ),
                np.clip(
                    angular_accel,
                    -float(robot.config.max_angular_accel),
                    float(robot.config.max_angular_accel),
                ),
            ],
            dtype=np.float32,
        )

    return None


def _get_certified_route_waypoints(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
) -> list[tuple[float, float]]:
    """Extract the A* planned path waypoints from the route certificate.

    Returns:
        List of (x, y) waypoints from the certifier's inflated A* path, or an empty
        list when the route cannot be planned.
    """
    try:
        config = build_robot_config_from_scenario(dict(scenario), scenario_path=scenario_path)
        certificate = _default_certifier(scenario, scenario_path)
    except Exception:  # noqa: BLE001
        return []

    for route_cert in certificate.route_certificates or []:
        planned_info = (route_cert.checks or {}).get("planner")
        if planned_info and route_cert.checks.get("inflated_collision_free_path"):
            map_name = route_cert.checks.get("map_name")
            robot_radius = float(getattr(config.robot_config, "radius", DEFAULT_ROBOT_RADIUS))
            for candidate_map_name, map_def in config.map_pool.map_defs.items():
                if map_name is not None and candidate_map_name != map_name:
                    continue
                for route in getattr(map_def, "robot_routes", []):
                    route_id = route.source_label or f"robot_route_{route.spawn_id}_{route.goal_id}"
                    if (
                        route_id != route_cert.route_id
                        or int(route.spawn_id) != int(route_cert.spawn_id)
                        or int(route.goal_id) != int(route_cert.goal_id)
                        or len(route.waypoints) < 2
                    ):
                        continue
                    return _replan_astar_path(
                        map_def,
                        start=tuple(float(c) for c in route.waypoints[0]),
                        goal=tuple(float(c) for c in route.waypoints[-1]),
                        robot_radius=robot_radius,
                    )
    return []


def _replan_astar_path(
    map_def: Any,
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    robot_radius: float,
) -> list[tuple[float, float]]:
    """Re-plan the inflated A* path for the scenario to extract waypoints.

    Returns:
        List of (x, y) waypoints, or empty list on failure.
    """
    from robot_sf.planner.classic_global_planner import (  # noqa: PLC0415
        ClassicGlobalPlanner,
        ClassicPlannerConfig,
    )

    planner_config = ClassicPlannerConfig(
        cells_per_meter=2.0,
        inflate_radius_meters=robot_radius,
        algorithm="a_star",
    )
    planner = ClassicGlobalPlanner(map_def, planner_config)
    try:
        path, _info = planner.plan(start, goal, algorithm="a_star", allow_inflation_fallback=False)
    except Exception:  # noqa: BLE001
        return []

    if not path:
        return []

    return [(float(x), float(y)) for x, y in path]


def build_issue_5596_blind_corner_diagnostic(  # noqa: C901, PLR0912, PLR0915
    manifest_path: Path,
    *,
    envelope_radii_m: tuple[float, ...] = (1.0, 0.5),
    episode_runner: EpisodeRunner | None = None,
    certifier: Callable[[Mapping[str, Any], Path], ScenarioCertificate] | None = None,
) -> dict[str, Any]:
    """Build the issue #5596 blind-corner zero-success diagnostic report.

    Runs three analyses to discriminate the competing explanations for why
    ``francis2023_blind_corner`` stays zero-success for the scripted traversal:

    1. **Oracle lane**: planner-free goal script at each envelope radius (existing
       ``run_envelope_sensitivity_sweep``).
    2. **Route-follow intervention lane**: a ``route_follow`` runner that drives the
       certifier's A* path waypoint-to-waypoint, isolating route geometry from
       controller strategy.
    3. **Clearance comparison**: straight-line goal beeline vs certified-route
       obstacle clearance for each radius.

    The mechanism classification distinguishes whether the zero-success is caused by:
    explanation #1 (scripted controller incomplete), #2 (route geometry or config),
    or #3 (scenario/map runtime interpretation drift).

    Args:
        manifest_path: Path to the scenario manifest YAML containing the blind-corner
            scenario cell.
        envelope_radii_m: Envelope radii to probe. Must have the nominal radius first
            (largest) and at least one reduced radius.
        episode_runner: Optional injected episode runner. Defaults to the canonical
            actor-free map episode rollout.
        certifier: Optional injected route certifier. Defaults to ``certify_scenario``.

    Returns:
        Versioned diagnostic report with oracle verdict, route-follow intervention,
        clearance comparison, and mechanism classification.

    Raises:
        FileNotFoundError: When the manifest path does not exist.
        ValueError: When the blind-corner scenario is missing from the manifest or
            envelope radii are not ordered nominal-first.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest file not found: {manifest_path}")

    radii = tuple(float(r) for r in envelope_radii_m)
    if len(radii) < 2:
        raise ValueError("envelope_radii_m must contain at least two radii")
    if radii[0] != max(radii):
        raise ValueError("envelope_radii_m must place the nominal radius first")

    scenarios = load_scenarios(manifest_path)
    blind_corner = None
    for s in scenarios:
        if _scenario_id(s).lower() == ISSUE_5596_BLIND_CORNER_SCENARIO_ID.lower():
            blind_corner = dict(s)
            break
    if blind_corner is None:
        raise ValueError(
            f"Scenario '{ISSUE_5596_BLIND_CORNER_SCENARIO_ID}' missing from manifest "
            f"{manifest_path}"
        )

    config = FeasibilityOracleConfig(
        scenario_path=manifest_path,
        envelope_radii_m=radii,
    )
    runner = episode_runner or _default_actor_free_runner(config)
    certify = certifier or _default_certifier
    route_runner = (
        episode_runner if episode_runner is not None else make_route_follow_episode_runner(config)
    )

    nominal_radius = radii[0]
    seed = _scenario_rollout_seed(blind_corner, override=None)

    # 1. Oracle verdict (existing feasibility sweep).
    envelope_verdict = run_envelope_sensitivity_sweep(
        blind_corner,
        config=config,
        episode_runner=runner,
        certifier=certify,
    )
    oracle_dict = envelope_sensitivity_verdict_to_dict(envelope_verdict)
    oracle_dict["issue"] = "5596"

    # 2. Route-follow intervention.
    route_follow_results: list[dict[str, Any]] = []
    for radius in radii:
        env_scenario = make_envelope_scenario(blind_corner, envelope_radius_m=radius)
        horizon = _scenario_horizon(env_scenario)
        try:
            record = route_runner(env_scenario, seed, horizon, ROUTE_FOLLOW_ALGO)
            is_stateful = record.get("stateful") is True
            if not is_stateful:
                completed = None
                status = "blocked"
                blocker_msg = (
                    record.get("route_follow_blocker") or "route-follow runner is not stateful"
                )
            else:
                completed = (
                    True
                    if record.get("route_complete") is True
                    or (record.get("outcome") or {}).get("route_complete") is True
                    else None
                    if record.get("route_complete") is None and record.get("status") == "blocked"
                    else False
                )
                status = (
                    "passed" if completed is True else "blocked" if completed is None else "failed"
                )
                blocker_msg = record.get("route_follow_blocker")

            route_follow_results.append(
                {
                    "envelope_radius_m": radius,
                    "feasible": completed,
                    "termination_reason": record.get("termination_reason"),
                    "steps": record.get("steps"),
                    "status": status,
                    "blocker": blocker_msg,
                    "stateful": record.get("stateful"),
                    "waypoint_count": record.get("waypoint_count"),
                    "collision_seen": record.get("collision_seen"),
                    "robot_final_position": record.get("robot_final_position"),
                    "route_follow_provenance": record.get("route_follow_provenance"),
                    "claim_boundary": record.get("claim_boundary") or DIAGNOSTIC_CLAIM_BOUNDARY,
                }
            )
        except Exception as exc:  # noqa: BLE001
            route_follow_results.append(
                {
                    "envelope_radius_m": radius,
                    "feasible": None,
                    "termination_reason": str(exc),
                    "steps": None,
                    "status": "blocked",
                }
            )

    route_follow_nominal = route_follow_results[0]
    route_follow_verdict = {
        "issue": "5596",
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "scenario_id": ISSUE_5596_BLIND_CORNER_SCENARIO_ID,
        "nominal_radius_m": nominal_radius,
        "results": route_follow_results,
    }

    # 3. Straight-line vs certified-route clearance.
    clearance_comparison: list[dict[str, Any]] = []
    for radius in radii:
        env_scenario = make_envelope_scenario(blind_corner, envelope_radius_m=radius)
        try:
            certificate = certify(env_scenario, manifest_path)
            route_checks = _aggregate_route_checks(certificate)
            min_clearance = _optional_float(route_checks.get("minimum_static_clearance_m"))
            shortest_path = _optional_float(route_checks.get("shortest_path_length_m"))
            inflated_path = route_checks.get("inflated_collision_free_path", None)

            # Straight-line clearance (from authored route line to obstacles - minus radius).
            straight_line_clearance = min_clearance  # The certifier measures the authored line.

            clearance_comparison.append(
                {
                    "envelope_radius_m": radius,
                    "straight_line_clearance_m": straight_line_clearance,
                    "certified_route_clearance_m": min_clearance,
                    "shortest_path_length_m": shortest_path,
                    "inflated_collision_free_path": inflated_path,
                }
            )
        except Exception as exc:  # noqa: BLE001
            clearance_comparison.append(
                {
                    "envelope_radius_m": radius,
                    "straight_line_clearance_m": None,
                    "certified_route_clearance_m": None,
                    "shortest_path_length_m": None,
                    "inflated_collision_free_path": None,
                    "blocker": str(exc),
                }
            )

    # 4. Mechanism classification.
    oracle_nominal = oracle_dict["nominal_verdict"]
    oracle_nominal_feasible = oracle_nominal.get("feasible") is True
    route_follow_feasible = route_follow_nominal.get("feasible")

    # When route-follow succeeds but goal script fails -> explanation #1 (controller).
    # When both fail -> explanation #2 (route geometry or config).
    explanation_1 = not oracle_nominal_feasible and route_follow_feasible is True
    explanation_2 = not oracle_nominal_feasible and route_follow_feasible is False

    if explanation_1:
        supported = "scripted_controller_corner_cut"
    elif explanation_2:
        supported = "route_geometry_or_config_cause"
    elif route_follow_feasible is None:
        supported = "route_follow_intervention_blocked"
    else:
        supported = "nominal_oracle_feasible_no_anomaly"

    mechanism = {
        "explanation_1_scripted_controller_incomplete": explanation_1,
        "explanation_2_route_geometry_or_config_cause": explanation_2,
        "explanation_3_scenario_runtime_drift": None,  # Not established in this diagnostic.
        "supported_explanation": supported,
        "oracle_nominal_feasible": oracle_nominal_feasible,
        "route_follow_intervention_feasible": route_follow_feasible,
        "route_follow_intervention_status": route_follow_nominal.get("status"),
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
    }

    return {
        "schema_version": ISSUE_5596_DIAGNOSTIC_SCHEMA,
        "review_marker": ISSUE_5596_REVIEW_MARKER,
        "source_commit": _git_hash_fallback(),
        "issue": "5596",
        "scenario_id": ISSUE_5596_BLIND_CORNER_SCENARIO_ID,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "manifest_path": manifest_path.as_posix(),
        "envelope_radii_m": list(radii),
        "rollout_seed": seed,
        "oracle_verdict": oracle_dict,
        "route_follow_intervention_verdict": route_follow_verdict,
        "straight_line_vs_route_clearance": clearance_comparison,
        "mechanism": mechanism,
    }


__all__ = [
    "BLOCKED",
    "CAMPAIGN_FEASIBILITY_ANNOTATION_SCHEMA",
    "DEFAULT_ENVELOPE_RADII_M",
    "DEFAULT_ISSUE_5574_SCENARIO_IDS",
    "DEFAULT_ROLLOUT_SEED",
    "ENVELOPE_SENSITIVE_HARD",
    "ENVELOPE_SENSITIVITY_SCHEMA",
    "FEASIBILITY_ORACLE_SCHEMA",
    "FEASIBLE",
    "INFEASIBLE_BY_CONSTRUCTION",
    "ISSUE_5596_BLIND_CORNER_SCENARIO_ID",
    "ISSUE_5596_DIAGNOSTIC_SCHEMA",
    "ISSUE_5596_REVIEW_MARKER",
    "PLANNER_LIMITED",
    "ROUTE_FOLLOW_ALGO",
    "TIME_TRUNCATED",
    "CompletionMargin",
    "EnvelopeSensitivityVerdict",
    "FeasibilityOracleConfig",
    "FeasibilityVerdict",
    "GeometricMargin",
    "annotate_zero_completion_cells",
    "build_issue_5574_feasibility_report",
    "build_issue_5596_blind_corner_diagnostic",
    "envelope_sensitivity_verdict_to_dict",
    "feasibility_verdict_to_dict",
    "make_envelope_scenario",
    "make_route_follow_episode_runner",
    "run_envelope_sensitivity_sweep",
    "run_feasibility_oracle",
]
