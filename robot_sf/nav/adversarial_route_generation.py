"""Adversarial route generation utilities for scenario hardening workflows."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import optuna
import yaml
from loguru import logger
from shapely.geometry import LineString, Polygon

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.planner import ClassicGlobalPlanner, PlanningError

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition

ObjectiveMode = Literal["composite", "failure_only", "near_miss_only"]


@dataclass(slots=True)
class AdversarialRouteGenerationConfig:
    """Configuration for adversarial route generation and optimization."""

    scenario_id: str
    map_file: str
    objective_mode: ObjectiveMode = "composite"
    trial_count: int = 40
    seed: int = 123
    robot_route_count: int = 1
    ped_route_count: int = 2
    allow_inflation_fallback: bool = False
    feasibility_filter: bool = True
    top_k: int = 5
    min_valid_trial_ratio: float = 0.1
    near_miss_threshold_m: float = 1.5
    clearance_threshold_m: float = 0.75
    failure_weight: float = 0.45
    delay_weight: float = 0.25
    inefficiency_weight: float = 0.15
    near_miss_weight: float = 0.15

    def __post_init__(self) -> None:
        """Validate generation configuration values."""
        self._validate_identity_fields()
        self._validate_mode_and_counts()
        self._validate_thresholds()

    def _validate_identity_fields(self) -> None:
        """Ensure identifiers and map path fields are set."""
        scenario_id = self.scenario_id.strip()
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if "/" in scenario_id or "\\" in scenario_id or ".." in scenario_id:
            raise ValueError("scenario_id must not contain path separators or '..'")
        if not self.map_file.strip():
            raise ValueError("map_file must be non-empty")
        map_path = Path(self.map_file)
        if ".." in map_path.parts:
            raise ValueError("map_file must not contain parent-directory traversal ('..')")

    def _validate_mode_and_counts(self) -> None:
        """Validate objective mode and integer count fields."""
        if self.objective_mode not in {"composite", "failure_only", "near_miss_only"}:
            raise ValueError(
                "objective_mode must be one of: composite, failure_only, near_miss_only"
            )
        if self.trial_count <= 0:
            raise ValueError("trial_count must be > 0")
        if self.robot_route_count <= 0:
            raise ValueError("robot_route_count must be > 0")
        if self.ped_route_count < 0:
            raise ValueError("ped_route_count must be >= 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not 0.0 <= self.min_valid_trial_ratio <= 1.0:
            raise ValueError("min_valid_trial_ratio must be in [0, 1]")

    def _validate_thresholds(self) -> None:
        """Validate distance thresholds used in route-scoring heuristics."""
        if self.near_miss_threshold_m <= 0:
            raise ValueError("near_miss_threshold_m must be > 0")
        if self.clearance_threshold_m <= 0:
            raise ValueError("clearance_threshold_m must be > 0")


@dataclass(slots=True)
class CandidateRouteSet:
    """Candidate route set generated for one optimizer trial."""

    robot_routes: list[GlobalRoute]
    ped_routes: list[GlobalRoute]


@dataclass(slots=True)
class RouteEvaluation:
    """Scored hardness components for a candidate route set."""

    score: float
    failure_proxy: float
    delay_proxy: float
    path_inefficiency: float
    near_miss_stress: float


@dataclass(slots=True)
class OptimizationResult:
    """Optimization outputs required for artifact/report generation."""

    config: AdversarialRouteGenerationConfig
    best_candidate: CandidateRouteSet
    best_evaluation: RouteEvaluation
    failed_trials: int
    feasibility_rejection_counts: dict[str, int]
    top_k_scores: list[float]
    valid_trial_count: int


@dataclass(slots=True)
class _GenerationContext:
    """Shared context for one candidate generation attempt."""

    map_def: MapDefinition
    trial: optuna.Trial | None
    rng: random.Random
    allow_inflation_fallback: bool
    feasibility_filter: bool


def _route_length(route: GlobalRoute) -> float:
    """Return polyline length for a route."""
    if len(route.waypoints) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(route.waypoints)):
        x1, y1 = route.waypoints[idx - 1]
        x2, y2 = route.waypoints[idx]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _compute_path_inefficiency(routes: list[GlobalRoute]) -> float:
    """Compute mean route inefficiency against straight-line baselines.

    Returns:
        float: Mean inefficiency ratio above straight-line distance.
    """
    ineff_values: list[float] = []
    for route in routes:
        if len(route.waypoints) < 2:
            continue
        start = route.waypoints[0]
        goal = route.waypoints[-1]
        straight = max(math.hypot(goal[0] - start[0], goal[1] - start[1]), 1e-6)
        ineff_values.append(max(0.0, _route_length(route) / straight - 1.0))
    return sum(ineff_values) / len(ineff_values) if ineff_values else 0.0


def _compute_failure_proxy(
    routes: list[GlobalRoute],
    map_def: MapDefinition,
    clearance_threshold_m: float,
) -> float:
    """Compute low-clearance failure proxy against map obstacles.

    Returns:
        float: Mean normalized low-clearance risk across valid routes.
    """
    obstacle_polys = [Polygon(obstacle.vertices) for obstacle in map_def.obstacles]
    clearance_values: list[float] = []
    for route in routes:
        if len(route.waypoints) < 2:
            continue
        line = LineString(route.waypoints)
        if not obstacle_polys:
            clearance_values.append(clearance_threshold_m)
            continue
        min_clearance = min(line.distance(poly) for poly in obstacle_polys)
        clearance_values.append(float(min_clearance))

    if not clearance_values:
        return 0.0
    normalized = [
        max(0.0, 1.0 - (clearance / clearance_threshold_m)) for clearance in clearance_values
    ]
    return sum(normalized) / len(normalized)


def _compute_near_miss_stress(
    candidate: CandidateRouteSet,
    near_miss_threshold_m: float,
) -> float:
    """Compute near-miss stress via minimum robot-pedestrian route separation.

    Returns:
        float: Normalized near-miss stress in [0, 1].
    """
    if not candidate.robot_routes or not candidate.ped_routes:
        return 0.0
    robot_lines = [
        LineString(route.waypoints) for route in candidate.robot_routes if len(route.waypoints) >= 2
    ]
    ped_lines = [
        LineString(route.waypoints) for route in candidate.ped_routes if len(route.waypoints) >= 2
    ]
    if not robot_lines or not ped_lines:
        return 0.0
    min_dist = min(
        robot_line.distance(ped_line) for robot_line in robot_lines for ped_line in ped_lines
    )
    return max(0.0, 1.0 - (float(min_dist) / near_miss_threshold_m))


def _serialize_route(route: GlobalRoute) -> dict:
    """Serialize a route to YAML-compatible structure.

    Returns:
        dict: Serialized route payload.
    """
    return {
        "spawn_id": int(route.spawn_id),
        "goal_id": int(route.goal_id),
        "waypoints": [[float(x), float(y)] for x, y in route.waypoints],
    }


def _candidate_to_payload(candidate: CandidateRouteSet) -> dict:
    """Serialize candidate route set payload.

    Returns:
        dict: Serialized robot and pedestrian route payload.
    """
    return {
        "robot_routes": [_serialize_route(route) for route in candidate.robot_routes],
        "ped_routes": [_serialize_route(route) for route in candidate.ped_routes],
    }


def _sample_point(
    trial: optuna.Trial | None,
    rng: random.Random,
    *,
    prefix: str,
    map_def: MapDefinition,
) -> tuple[float, float]:
    """Sample one world-space point from map bounds.

    Returns:
        tuple[float, float]: Sampled map-space coordinate.
    """
    if trial is None:
        return (rng.uniform(0.0, map_def.width), rng.uniform(0.0, map_def.height))
    x = trial.suggest_float(f"{prefix}_x", 0.0, map_def.width)
    y = trial.suggest_float(f"{prefix}_y", 0.0, map_def.height)
    return (x, y)


def _plan_route(
    planner: ClassicGlobalPlanner,
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    template: GlobalRoute,
    allow_inflation_fallback: bool,
) -> GlobalRoute:
    """Plan route and clone template metadata.

    Returns:
        GlobalRoute: Planned route object with template metadata.
    """
    path_world, _path_info = planner.plan(
        start=start,
        goal=goal,
        allow_inflation_fallback=allow_inflation_fallback,
    )
    if len(path_world) < 2:
        raise PlanningError("Planner returned fewer than 2 waypoints for route.")
    return GlobalRoute(
        spawn_id=template.spawn_id,
        goal_id=template.goal_id,
        waypoints=path_world,
        spawn_zone=template.spawn_zone,
        goal_zone=template.goal_zone,
        source_label="generated_adversarial",
    )


def _generate_entity_routes(
    planner: ClassicGlobalPlanner,
    *,
    templates: list[GlobalRoute],
    count: int,
    entity_name: str,
    context: _GenerationContext,
) -> tuple[list[GlobalRoute] | None, str | None]:
    """Generate routes for one entity class (robot or pedestrian).

    Returns:
        tuple[list[GlobalRoute] | None, str | None]: Routes or rejection reason.
    """
    if count == 0:
        return [], None
    if not templates:
        return None, f"no_{entity_name}_templates"

    routes: list[GlobalRoute] = []
    for idx in range(count):
        template = templates[idx % len(templates)]
        start = _sample_point(
            context.trial,
            context.rng,
            prefix=f"{entity_name}_{idx}_start",
            map_def=context.map_def,
        )
        goal = _sample_point(
            context.trial,
            context.rng,
            prefix=f"{entity_name}_{idx}_goal",
            map_def=context.map_def,
        )
        if start == goal:
            return None, "degenerate_start_goal"
        try:
            planner.validate_point(start)
            planner.validate_point(goal)
        except PlanningError:
            if context.feasibility_filter:
                return None, "invalid_start_or_goal"
            raise
        try:
            route = _plan_route(
                planner,
                start=start,
                goal=goal,
                template=template,
                allow_inflation_fallback=context.allow_inflation_fallback,
            )
        except PlanningError:
            if context.feasibility_filter:
                return None, "planning_failed"
            raise
        routes.append(route)
    return routes, None


def generate_candidate_route_set(
    map_def: MapDefinition,
    planner: ClassicGlobalPlanner,
    config: AdversarialRouteGenerationConfig,
    *,
    trial: optuna.Trial | None = None,
    rng: random.Random | None = None,
) -> tuple[CandidateRouteSet | None, str | None]:
    """Generate one route-set candidate.

    Returns:
        CandidateRouteSet | None: Candidate routes when feasible.
        str | None: Rejection reason when candidate is infeasible.
    """
    generator = rng or random.Random(config.seed)
    context = _GenerationContext(
        map_def=map_def,
        trial=trial,
        rng=generator,
        allow_inflation_fallback=config.allow_inflation_fallback,
        feasibility_filter=config.feasibility_filter,
    )
    robot_routes, robot_reason = _generate_entity_routes(
        planner,
        templates=map_def.robot_routes,
        count=config.robot_route_count,
        entity_name="robot",
        context=context,
    )
    if robot_reason:
        return None, robot_reason
    assert robot_routes is not None

    ped_routes, ped_reason = _generate_entity_routes(
        planner,
        templates=map_def.ped_routes,
        count=config.ped_route_count,
        entity_name="ped",
        context=context,
    )
    if ped_reason:
        return None, ped_reason
    assert ped_routes is not None
    return CandidateRouteSet(robot_routes=robot_routes, ped_routes=ped_routes), None


def evaluate_route_set(
    candidate: CandidateRouteSet,
    map_def: MapDefinition,
    config: AdversarialRouteGenerationConfig,
) -> RouteEvaluation:
    """Evaluate route hardness for optimizer objective.

    Returns:
        RouteEvaluation: Scored objective components.
    """
    robot_lengths = [_route_length(route) for route in candidate.robot_routes]
    ped_lengths = [_route_length(route) for route in candidate.ped_routes]
    all_lengths = robot_lengths + ped_lengths
    all_routes = [*candidate.robot_routes, *candidate.ped_routes]
    diag = max(math.hypot(map_def.width, map_def.height), 1e-6)
    delay_proxy = (sum(all_lengths) / len(all_lengths) / diag) if all_lengths else 0.0

    path_inefficiency = _compute_path_inefficiency(all_routes)
    failure_proxy = _compute_failure_proxy(all_routes, map_def, config.clearance_threshold_m)
    near_miss_stress = _compute_near_miss_stress(candidate, config.near_miss_threshold_m)

    if config.objective_mode == "failure_only":
        score = failure_proxy
    elif config.objective_mode == "near_miss_only":
        score = near_miss_stress
    else:
        score = (
            config.failure_weight * failure_proxy
            + config.delay_weight * delay_proxy
            + config.inefficiency_weight * path_inefficiency
            + config.near_miss_weight * near_miss_stress
        )

    return RouteEvaluation(
        score=float(score),
        failure_proxy=float(failure_proxy),
        delay_proxy=float(delay_proxy),
        path_inefficiency=float(path_inefficiency),
        near_miss_stress=float(near_miss_stress),
    )


def optimize_route_set(
    map_def: MapDefinition,
    planner: ClassicGlobalPlanner,
    config: AdversarialRouteGenerationConfig,
) -> OptimizationResult:
    """Run Optuna TPE optimization for adversarial route generation.

    Returns:
        OptimizationResult: Best candidate and optimizer diagnostics.
    """
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    rng = random.Random(config.seed)
    candidate_by_trial: dict[int, CandidateRouteSet] = {}
    evaluation_by_trial: dict[int, RouteEvaluation] = {}
    rejection_counts: dict[str, int] = {}
    failed_trials = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal failed_trials
        candidate, reason = generate_candidate_route_set(
            map_def,
            planner,
            config,
            trial=trial,
            rng=rng,
        )
        if candidate is None:
            failed_trials += 1
            key = reason or "unknown_rejection"
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
            trial.set_user_attr("valid_candidate", False)
            trial.set_user_attr("rejection_reason", key)
            return -1e6

        evaluation = evaluate_route_set(candidate, map_def, config)
        trial.set_user_attr("valid_candidate", True)
        trial.set_user_attr("score", evaluation.score)
        trial.set_user_attr(
            "metrics",
            {
                "failure_proxy": evaluation.failure_proxy,
                "delay_proxy": evaluation.delay_proxy,
                "path_inefficiency": evaluation.path_inefficiency,
                "near_miss_stress": evaluation.near_miss_stress,
            },
        )
        candidate_by_trial[trial.number] = candidate
        evaluation_by_trial[trial.number] = evaluation
        return evaluation.score

    study.optimize(objective, n_trials=config.trial_count)

    if not candidate_by_trial:
        raise RuntimeError(
            "No feasible candidate route set found. "
            f"Rejections: {rejection_counts if rejection_counts else 'none'}"
        )

    valid_trials = [trial for trial in study.trials if trial.number in candidate_by_trial]
    valid_ratio = len(valid_trials) / config.trial_count
    if valid_ratio < config.min_valid_trial_ratio:
        raise RuntimeError(
            "Valid trial ratio too low for reliable optimization "
            f"({valid_ratio:.3f} < {config.min_valid_trial_ratio:.3f}). "
            f"Rejections: {rejection_counts if rejection_counts else 'none'}"
        )

    valid_trials_sorted = sorted(
        valid_trials,
        key=lambda trial: trial.value if trial.value is not None else -1e6,
        reverse=True,
    )
    best_trial = valid_trials_sorted[0]
    best_candidate = candidate_by_trial[best_trial.number]
    best_evaluation = evaluation_by_trial[best_trial.number]
    top_k_scores = [
        float(trial.value if trial.value is not None else -1e6)
        for trial in valid_trials_sorted[: config.top_k]
    ]

    logger.info(
        "Optimized adversarial routes: best score={score:.4f}, valid_trials={valid}/{total}, mode={mode}",
        score=best_evaluation.score,
        valid=len(valid_trials),
        total=config.trial_count,
        mode=config.objective_mode,
    )

    return OptimizationResult(
        config=config,
        best_candidate=best_candidate,
        best_evaluation=best_evaluation,
        failed_trials=failed_trials,
        feasibility_rejection_counts=rejection_counts,
        top_k_scores=top_k_scores,
        valid_trial_count=len(valid_trials),
    )


def write_route_override_artifact(
    result: OptimizationResult,
    *,
    output_root: Path | str = Path("output/adversarial_routes"),
) -> dict[str, Path]:
    """Write YAML override artifact plus report files.

    Returns:
        dict[str, Path]: Output artifact paths keyed by artifact type.
    """
    root = Path(output_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"{result.config.scenario_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    payload = _candidate_to_payload(result.best_candidate)
    artifact = {
        "scenario_id": result.config.scenario_id,
        "map_file": result.config.map_file,
        "optimizer": "optuna_tpe",
        "seed": result.config.seed,
        "trial_count": result.config.trial_count,
        "objective_mode": result.config.objective_mode,
        "best_score": result.best_evaluation.score,
        "route_payload": payload,
        "diagnostics": {
            "failed_trials": result.failed_trials,
            "valid_trial_count": result.valid_trial_count,
            "feasibility_rejection_counts": result.feasibility_rejection_counts,
            "top_k_scores": result.top_k_scores,
            "components": {
                "failure_proxy": result.best_evaluation.failure_proxy,
                "delay_proxy": result.best_evaluation.delay_proxy,
                "path_inefficiency": result.best_evaluation.path_inefficiency,
                "near_miss_stress": result.best_evaluation.near_miss_stress,
            },
        },
    }

    artifact_path = run_dir / "route_overrides.yaml"
    artifact_path.write_text(yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")

    json_summary_path = run_dir / "summary.json"
    json_summary_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    report_path = run_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Adversarial Route Generation Report ({result.config.scenario_id})",
                "",
                f"- objective_mode: `{result.config.objective_mode}`",
                "- optimizer: `optuna_tpe`",
                f"- seed: `{result.config.seed}`",
                f"- trial_count: `{result.config.trial_count}`",
                f"- valid_trial_count: `{result.valid_trial_count}`",
                f"- failed_trials: `{result.failed_trials}`",
                f"- best_score: `{result.best_evaluation.score:.6f}`",
                "",
                "## Objective Components",
                f"- failure_proxy: `{result.best_evaluation.failure_proxy:.6f}`",
                f"- delay_proxy: `{result.best_evaluation.delay_proxy:.6f}`",
                f"- path_inefficiency: `{result.best_evaluation.path_inefficiency:.6f}`",
                f"- near_miss_stress: `{result.best_evaluation.near_miss_stress:.6f}`",
                "",
                "## Rejections",
                f"- feasibility_rejection_counts: `{result.feasibility_rejection_counts}`",
                "",
                "## Replay",
                "Add this to your scenario entry:",
                f"- `route_overrides_file: {artifact_path.as_posix()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "run_dir": run_dir,
        "artifact_path": artifact_path,
        "json_summary_path": json_summary_path,
        "report_path": report_path,
    }


__all__ = [
    "AdversarialRouteGenerationConfig",
    "CandidateRouteSet",
    "ObjectiveMode",
    "OptimizationResult",
    "RouteEvaluation",
    "evaluate_route_set",
    "generate_candidate_route_set",
    "optimize_route_set",
    "write_route_override_artifact",
]
