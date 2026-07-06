"""CPU-only synthetic fixture diagnostics for pedestrian-model comparisons.

This module adds a *local shared-throat precursor harness* for issue #3481 without
claiming benchmark-strength evidence. It runs short, seed-controlled, no-robot
simulator traces for selected pedestrian-model variants and reports descriptive metrics
only:

- minimum pairwise pedestrian distance,
- mean maximum lateral displacement (a proxy for passive sliding),
- interaction-zone slow-run proxies, and
- finite-state checks.

The report is diagnostic-only. It does not apply realism thresholds, does not run a
benchmark campaign, and does not upgrade the evidence tier for the force-model family.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ANISOTROPIC_FOV_V1,
    HSFM_TOTAL_FORCE_V1,
    HSFM_TTC_PREDICTIVE_V1,
    SOCIAL_FORCE_DEFAULT,
    normalize_pedestrian_model,
)
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import Simulator

if TYPE_CHECKING:
    from robot_sf.common.types import Rect

Axis = Literal["x", "y"]
ThresholdDirection = Literal["min_required", "max_allowed"]

PED_MODEL_FIXTURE_REPORT_SCHEMA_VERSION = "pedestrian_model_fixture_diagnostics.v1"
CLAIM_BOUNDARY = (
    "diagnostic-only synthetic no-robot geometric fixture comparison for issue #3481; "
    "no benchmark-strength, calibration, planner-ranking, or paper-facing claim"
)
DEFAULT_PEDESTRIAN_MODELS = (
    SOCIAL_FORCE_DEFAULT,
    HSFM_TOTAL_FORCE_V1,
    HSFM_TTC_PREDICTIVE_V1,
    HSFM_ANISOTROPIC_FOV_V1,
)


@dataclass(frozen=True)
class DiagnosticThreshold:
    """Direction-aware local diagnostic threshold."""

    value: float
    direction: ThresholdDirection

    def __post_init__(self) -> None:
        """Validate threshold value and direction."""
        _require_finite_number(self.value, "diagnostic threshold value", non_negative=True)
        if self.direction not in ("min_required", "max_allowed"):
            raise ValueError(f"Unsupported diagnostic threshold direction: {self.direction}")


@dataclass(frozen=True)
class PedestrianModelFixtureSpec:
    """Synthetic scenario spec for a fixed set of single pedestrians."""

    scenario_id: str
    map_def: MapDefinition
    single_pedestrians: tuple[SinglePedestrianDefinition, ...]
    interaction_zone_center: tuple[float, float]
    interaction_zone_radius_m: float
    interaction_zone_min_pedestrians: int
    lane_axis: Axis = "x"
    lateral_axis: Axis = "y"
    metric_thresholds: dict[str, float | DiagnosticThreshold | dict[str, Any]] = field(
        default_factory=dict
    )
    diagnostic_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PedestrianModelFixtureRunConfig:
    """Runtime controls for a short local diagnostic fixture run."""

    duration_s: float = 4.0
    dt_s: float = 0.1
    seed: int = 3481
    freeze_speed_threshold_mps: float = 0.6
    freeze_window_steps: int = 3
    diagnostic_only_not_benchmark_gate: bool = True

    def __post_init__(self) -> None:
        """Validate finite positive runtime controls."""
        _require_finite_number(self.duration_s, "duration_s", positive=True)
        _require_finite_number(self.dt_s, "dt_s", positive=True)
        _require_finite_number(
            self.freeze_speed_threshold_mps,
            "freeze_speed_threshold_mps",
            non_negative=True,
        )
        if self.freeze_window_steps < 1:
            raise ValueError("freeze_window_steps must be >= 1")


@dataclass(frozen=True)
class PedestrianModelFixtureTrace:
    """Collected no-robot simulator trace for one scenario/model pair."""

    scenario_id: str
    pedestrian_model: str
    seed: int
    dt_s: float
    duration_s: float
    positions: np.ndarray
    velocities: np.ndarray


def _require_finite_number(
    value: float, field_name: str, *, positive: bool = False, non_negative: bool = False
) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    if positive and float(value) <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    if non_negative and float(value) < 0.0:
        raise ValueError(f"{field_name} must be non-negative")


def build_pedestrian_model_fixture_scenarios() -> dict[str, PedestrianModelFixtureSpec]:
    """Return the issue #3481 local diagnostic fixture scenarios."""

    scenarios = (
        _build_shared_throat_sliding_spec(),
        _build_shared_throat_congestion_spec(),
        _build_narrow_passage_lateral_sliding_spec(),
        _build_bottleneck_freeze_deadlock_spec(),
    )
    return {scenario.scenario_id: scenario for scenario in scenarios}


def run_pedestrian_model_fixture_trace(
    spec: PedestrianModelFixtureSpec,
    *,
    pedestrian_model: str,
    config: PedestrianModelFixtureRunConfig,
) -> PedestrianModelFixtureTrace:
    """Run one short no-robot fixture trace for the requested pedestrian model.

    Returns:
        Trace positions and velocities for one scenario/model pair.
    """

    normalized_model = normalize_pedestrian_model(pedestrian_model)
    sim_settings = SimulationSettings(
        sim_time_in_secs=config.duration_s,
        time_per_step_in_secs=config.dt_s,
        ped_density_by_difficulty=[0.0],
        difficulty=0,
        route_spawn_seed=config.seed,
        max_total_pedestrians=len(spec.single_pedestrians),
        pedestrian_model=normalized_model,
    )
    sim = Simulator(
        sim_settings,
        _copy_map_with_single_pedestrians(spec.map_def, list(spec.single_pedestrians)),
        robots=[],
        goal_proximity_threshold=0.0,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )

    positions = [np.asarray(sim.ped_pos, dtype=float).copy()]
    velocities = [np.asarray(sim.ped_vel, dtype=float).copy()]
    step_count = math.floor(config.duration_s / config.dt_s)
    for _ in range(step_count):
        sim.step_once([])
        positions.append(np.asarray(sim.ped_pos, dtype=float).copy())
        velocities.append(np.asarray(sim.ped_vel, dtype=float).copy())

    return PedestrianModelFixtureTrace(
        scenario_id=spec.scenario_id,
        pedestrian_model=normalized_model,
        seed=config.seed,
        dt_s=config.dt_s,
        duration_s=float(step_count * config.dt_s),
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
    )


def compute_fixture_metrics(
    trace: PedestrianModelFixtureTrace,
    spec: PedestrianModelFixtureSpec,
    *,
    freeze_speed_threshold_mps: float,
    freeze_window_steps: int,
) -> dict[str, Any]:
    """Compute descriptive sliding / interaction-zone slow proxies for one trace.

    Returns:
        JSON-safe descriptive metrics for the fixture run.
    """

    positions = np.asarray(trace.positions, dtype=float)
    velocities = np.asarray(trace.velocities, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("trace.positions must have shape (T, N, 2)")
    if velocities.shape != positions.shape:
        raise ValueError("trace.velocities must match trace.positions")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("trace positions and velocities must be finite")

    lateral_index = _axis_index(spec.lateral_axis)
    start_lateral = positions[0, :, lateral_index]
    lateral_offsets = np.abs(positions[:, :, lateral_index] - start_lateral[np.newaxis, :])
    mean_max_lateral_displacement = float(np.mean(np.max(lateral_offsets, axis=0)))

    min_pairwise_distance = float(_minimum_pairwise_distance(positions))
    speeds = np.linalg.norm(velocities, axis=-1)
    interaction_center = np.asarray(spec.interaction_zone_center, dtype=float)
    in_interaction_zone = np.linalg.norm(
        positions - interaction_center[np.newaxis, np.newaxis, :], axis=-1
    ) <= float(spec.interaction_zone_radius_m)
    origin_axis_index = _axis_index(spec.lane_axis)
    if len(spec.single_pedestrians) == positions.shape[1]:
        start_positions = np.asarray([ped.start for ped in spec.single_pedestrians], dtype=float)
    else:
        start_positions = positions[0]
    left_origin_mask = start_positions[:, origin_axis_index] < interaction_center[origin_axis_index]
    right_origin_mask = (
        start_positions[:, origin_axis_index] > interaction_center[origin_axis_index]
    )
    left_origin_in_zone = np.count_nonzero(in_interaction_zone[:, left_origin_mask], axis=1)
    right_origin_in_zone = np.count_nonzero(in_interaction_zone[:, right_origin_mask], axis=1)
    opposing_origins_in_zone = (left_origin_in_zone > 0) & (right_origin_in_zone > 0)
    slow_in_zone = in_interaction_zone & (speeds <= float(freeze_speed_threshold_mps))
    interaction_zone_slow = np.count_nonzero(slow_in_zone, axis=1) >= int(
        spec.interaction_zone_min_pedestrians
    )
    max_consecutive_interaction_zone_slow_steps = int(_max_consecutive_true(interaction_zone_slow))
    interaction_zone_slow_detected = max_consecutive_interaction_zone_slow_steps >= int(
        freeze_window_steps
    )
    threshold_checks = _evaluate_metric_thresholds(
        {
            "minimum_pairwise_distance_m": min_pairwise_distance,
            "mean_max_lateral_displacement_m": mean_max_lateral_displacement,
            "max_consecutive_interaction_zone_slow_steps": float(
                max_consecutive_interaction_zone_slow_steps
            ),
        },
        spec.metric_thresholds,
    )

    return {
        "minimum_pairwise_distance_m": min_pairwise_distance,
        "mean_max_lateral_displacement_m": mean_max_lateral_displacement,
        "mean_speed_mps": float(np.mean(speeds)),
        "entered_interaction_zone": bool(np.any(in_interaction_zone)),
        "max_pedestrians_in_interaction_zone": int(
            np.max(np.count_nonzero(in_interaction_zone, axis=1))
        ),
        "max_left_origin_pedestrians_in_interaction_zone": int(np.max(left_origin_in_zone)),
        "max_right_origin_pedestrians_in_interaction_zone": int(np.max(right_origin_in_zone)),
        "opposing_origins_cooccurred_in_interaction_zone": bool(np.any(opposing_origins_in_zone)),
        "interaction_zone_slow_steps": int(np.count_nonzero(interaction_zone_slow)),
        "max_consecutive_interaction_zone_slow_steps": max_consecutive_interaction_zone_slow_steps,
        "interaction_zone_slow_detected": interaction_zone_slow_detected,
        "finite_positions": bool(np.all(np.isfinite(positions))),
        "finite_velocities": bool(np.all(np.isfinite(velocities))),
        "pedestrian_count": int(positions.shape[1]),
        "diagnostic_thresholds": threshold_checks,
    }


def _evaluate_metric_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float | DiagnosticThreshold | dict[str, Any]],
) -> dict[str, dict[str, float | bool | str]]:
    """Evaluate local diagnostic thresholds without promoting benchmark claims.

    Returns:
        JSON-safe threshold verdicts keyed by metric name.
    """

    checks: dict[str, dict[str, float | bool | str]] = {}
    for metric_name, raw_threshold in thresholds.items():
        if metric_name not in metrics:
            raise ValueError(f"Unknown diagnostic threshold metric: {metric_name}")
        threshold = _normalize_diagnostic_threshold(raw_threshold, metric_name)
        observed = float(metrics[metric_name])
        criterion_met = (
            observed >= threshold.value
            if threshold.direction == "min_required"
            else observed <= threshold.value
        )
        check: dict[str, float | bool | str] = {
            "observed": observed,
            "threshold": threshold.value,
            "direction": threshold.direction,
            "criterion_met": criterion_met,
        }
        if threshold.direction == "min_required":
            check["meets_or_exceeds"] = criterion_met
        checks[metric_name] = check
    return checks


def _normalize_diagnostic_threshold(
    threshold: float | DiagnosticThreshold | dict[str, Any],
    metric_name: str,
) -> DiagnosticThreshold:
    """Normalize legacy and JSON-style diagnostic threshold specs.

    Returns:
        Direction-aware diagnostic threshold.
    """
    if isinstance(threshold, DiagnosticThreshold):
        normalized = threshold
    elif isinstance(threshold, dict):
        if set(threshold) != {"value", "direction"}:
            raise ValueError(f"threshold {metric_name} must contain exactly value and direction")
        normalized = DiagnosticThreshold(
            value=float(threshold["value"]),
            direction=threshold["direction"],
        )
    else:
        normalized = DiagnosticThreshold(value=float(threshold), direction="min_required")
    _require_finite_number(normalized.value, f"threshold {metric_name}", non_negative=True)
    return normalized


def run_pedestrian_model_fixture_diagnostics(
    *,
    config: PedestrianModelFixtureRunConfig | None = None,
    scenarios: tuple[str, ...] | None = None,
    pedestrian_models: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the configured synthetic fixtures and return a JSON-safe report.

    Returns:
        Compact diagnostic report spanning the selected scenarios and pedestrian models.
    """

    run_config = config or PedestrianModelFixtureRunConfig()
    scenario_specs = build_pedestrian_model_fixture_scenarios()
    selected_ids = tuple(scenarios or scenario_specs.keys())
    unknown = sorted(set(selected_ids) - set(scenario_specs))
    if unknown:
        raise ValueError(f"Unknown pedestrian-model fixture scenario(s): {unknown}")

    selected_models = tuple(
        normalize_pedestrian_model(model)
        for model in (pedestrian_models or DEFAULT_PEDESTRIAN_MODELS)
    )
    per_run: list[dict[str, Any]] = []
    for scenario_id in selected_ids:
        spec = scenario_specs[scenario_id]
        for pedestrian_model in selected_models:
            trace = run_pedestrian_model_fixture_trace(
                spec,
                pedestrian_model=pedestrian_model,
                config=run_config,
            )
            per_run.append(
                {
                    "scenario_id": scenario_id,
                    "pedestrian_model": pedestrian_model,
                    "seed": trace.seed,
                    "duration_s": trace.duration_s,
                    "dt_s": trace.dt_s,
                    "step_count": int(max(trace.positions.shape[0] - 1, 0)),
                    "metrics": compute_fixture_metrics(
                        trace,
                        spec,
                        freeze_speed_threshold_mps=run_config.freeze_speed_threshold_mps,
                        freeze_window_steps=run_config.freeze_window_steps,
                    ),
                    "diagnostic_metadata": dict(spec.diagnostic_metadata),
                }
            )

    report = {
        "schema_version": PED_MODEL_FIXTURE_REPORT_SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": "diagnostic-only",
        "diagnostic_only_not_benchmark_gate": run_config.diagnostic_only_not_benchmark_gate,
        "status": {
            "thresholds_applied": True,
            "build_gate": False,
            "slurm_or_gpu_used": False,
            "robot_inserted": False,
        },
        "run_config": {
            "duration_s": run_config.duration_s,
            "dt_s": run_config.dt_s,
            "seed": run_config.seed,
            "interaction_zone_slow_speed_threshold_mps": run_config.freeze_speed_threshold_mps,
            "interaction_zone_window_steps": run_config.freeze_window_steps,
            "pedestrian_models": list(selected_models),
        },
        "scenario_ids": list(selected_ids),
        "runs": per_run,
    }
    return _json_safe(report)


def render_pedestrian_model_fixture_markdown(report: dict[str, Any]) -> str:
    """Render a compact reviewer-readable Markdown summary.

    Returns:
        Markdown table summarizing the per-run descriptive metrics.
    """

    lines = [
        "# Pedestrian model fixture diagnostics",
        "",
        f"Claim boundary: {report['claim_boundary']}.",
        "Evidence status: diagnostic-only smoke evidence. No benchmark or realism threshold was applied.",
        "",
        "| scenario_id | pedestrian_model | min_pairwise_distance_m | "
        "mean_max_lateral_displacement_m | max_consecutive_interaction_zone_slow_steps | "
        "interaction_zone_slow_detected |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        lines.append(
            "| "
            f"{run['scenario_id']} | {run['pedestrian_model']} | "
            f"{metrics['minimum_pairwise_distance_m']:.3f} | "
            f"{metrics['mean_max_lateral_displacement_m']:.3f} | "
            f"{metrics['max_consecutive_interaction_zone_slow_steps']} | "
            f"{metrics['interaction_zone_slow_detected']} |"
        )
    lines.append("")
    lines.append(
        "Diagnostic thresholds are local fixture assertions only; they are not benchmark, "
        "calibration, planner-ranking, or paper-facing success criteria."
    )
    lines.append("")
    return "\n".join(lines)


def write_pedestrian_model_fixture_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write compact JSON and Markdown artifacts for a fixture diagnostics report.

    Returns:
        Paths to the written summary artifacts.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "README.md"
    summary_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(render_pedestrian_model_fixture_markdown(report), encoding="utf-8")
    return {"summary_json": summary_json, "summary_md": summary_md}


def _build_shared_throat_sliding_spec() -> PedestrianModelFixtureSpec:
    """Build a symmetric shared-throat crossing fixture for sliding diagnostics.

    Returns:
        Scenario spec for the local shared-throat sliding diagnostic.
    """

    map_def = _base_map("shared_throat_sliding", width=10.0, height=4.0, obstacles=[])
    single_pedestrians = (
        SinglePedestrianDefinition(
            id="narrow_left_low",
            start=(3.0, 1.7),
            trajectory=[(5.0, 1.7), (7.5, 1.7)],
            speed_m_s=1.0,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        ),
        SinglePedestrianDefinition(
            id="narrow_left_high",
            start=(3.0, 2.3),
            trajectory=[(5.0, 2.3), (7.5, 2.3)],
            speed_m_s=1.0,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        ),
        SinglePedestrianDefinition(
            id="narrow_right_low",
            start=(7.0, 1.7),
            trajectory=[(5.0, 1.7), (2.5, 1.7)],
            speed_m_s=1.0,
            start_delay_s=0.2,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        ),
        SinglePedestrianDefinition(
            id="narrow_right_high",
            start=(7.0, 2.3),
            trajectory=[(5.0, 2.3), (2.5, 2.3)],
            speed_m_s=1.0,
            start_delay_s=0.2,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        ),
    )
    return PedestrianModelFixtureSpec(
        scenario_id="shared_throat_sliding",
        map_def=map_def,
        single_pedestrians=single_pedestrians,
        interaction_zone_center=(5.0, 2.0),
        interaction_zone_radius_m=0.8,
        interaction_zone_min_pedestrians=2,
        diagnostic_metadata={"diagnostic_only_not_benchmark_gate": True},
    )


def _build_shared_throat_congestion_spec() -> PedestrianModelFixtureSpec:
    """Build a symmetric shared-throat congestion fixture for slowdown diagnostics.

    Returns:
        Scenario spec for the local shared-throat congestion diagnostic.
    """

    map_def = _base_map("shared_throat_congestion", width=10.0, height=4.0, obstacles=[])
    y_offsets = (1.4, 2.0, 2.6)
    left = tuple(
        SinglePedestrianDefinition(
            id=f"bottleneck_left_{idx}",
            start=(2.8, y),
            trajectory=[(5.0, 2.0), (7.4, y)],
            speed_m_s=1.0,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        )
        for idx, y in enumerate(y_offsets)
    )
    right = tuple(
        SinglePedestrianDefinition(
            id=f"bottleneck_right_{idx}",
            start=(7.2, y),
            trajectory=[(5.0, 2.0), (2.6, y)],
            speed_m_s=1.0,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        )
        for idx, y in enumerate(reversed(y_offsets))
    )
    return PedestrianModelFixtureSpec(
        scenario_id="shared_throat_congestion",
        map_def=map_def,
        single_pedestrians=left + right,
        interaction_zone_center=(5.0, 2.0),
        interaction_zone_radius_m=0.9,
        interaction_zone_min_pedestrians=3,
        diagnostic_metadata={
            "diagnostic_only_not_benchmark_gate": True,
            "synthetic_shared_throat": True,
        },
    )


def _build_narrow_passage_lateral_sliding_spec() -> PedestrianModelFixtureSpec:
    """Build narrow-passage geometric fixture for lateral-sliding diagnostics.

    Returns:
        Scenario spec for local narrow-passage diagnostic evidence.
    """

    obstacles = [
        Obstacle([(0.0, 0.0), (10.0, 0.0), (10.0, 1.55), (0.0, 1.55)]),
        Obstacle([(0.0, 2.45), (10.0, 2.45), (10.0, 4.0), (0.0, 4.0)]),
    ]
    map_def = _base_map(
        "narrow_passage_lateral_sliding", width=10.0, height=4.0, obstacles=obstacles
    )
    single_pedestrians = (
        SinglePedestrianDefinition(
            id="narrow_passage_left",
            start=(2.7, 1.82),
            trajectory=[(5.0, 1.82), (7.4, 1.82)],
            speed_m_s=0.95,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        ),
        SinglePedestrianDefinition(
            id="narrow_passage_right",
            start=(7.3, 2.18),
            trajectory=[(5.0, 2.18), (2.6, 2.18)],
            speed_m_s=0.95,
            start_delay_s=0.1,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        ),
    )
    return PedestrianModelFixtureSpec(
        scenario_id="narrow_passage_lateral_sliding",
        map_def=map_def,
        single_pedestrians=single_pedestrians,
        interaction_zone_center=(5.0, 2.0),
        interaction_zone_radius_m=0.85,
        interaction_zone_min_pedestrians=2,
        metric_thresholds={
            "mean_max_lateral_displacement_m": DiagnosticThreshold(
                value=0.04,
                direction="max_allowed",
            )
        },
        diagnostic_metadata={
            "diagnostic_only_not_benchmark_gate": True,
            "geometric_fixture": "narrow_passage_lateral_sliding",
            "wall_gap_m": 0.9,
        },
    )


def _build_bottleneck_freeze_deadlock_spec() -> PedestrianModelFixtureSpec:
    """Build pinched bottleneck geometric fixture for freeze/deadlock diagnostics.

    Returns:
        Scenario spec for local bottleneck freeze/deadlock diagnostic evidence.
    """

    obstacles = [
        Obstacle([(4.55, 0.0), (5.45, 0.0), (5.45, 1.35), (4.55, 1.35)]),
        Obstacle([(4.55, 2.65), (5.45, 2.65), (5.45, 4.0), (4.55, 4.0)]),
    ]
    map_def = _base_map("bottleneck_freeze_deadlock", width=10.0, height=4.0, obstacles=obstacles)
    y_offsets = (1.5, 2.0, 2.5)
    left = tuple(
        SinglePedestrianDefinition(
            id=f"geometric_bottleneck_left_{idx}",
            start=(4.2, y),
            trajectory=[(5.0, 2.0), (5.8, y)],
            speed_m_s=0.9,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        )
        for idx, y in enumerate(y_offsets)
    )
    right = tuple(
        SinglePedestrianDefinition(
            id=f"geometric_bottleneck_right_{idx}",
            start=(5.8, y),
            trajectory=[(5.0, 2.0), (4.2, y)],
            speed_m_s=0.9,
            start_delay_s=0.15,
            metadata={"diagnostic_only_not_benchmark_gate": True},
        )
        for idx, y in enumerate(reversed(y_offsets))
    )
    return PedestrianModelFixtureSpec(
        scenario_id="bottleneck_freeze_deadlock",
        map_def=map_def,
        single_pedestrians=left + right,
        interaction_zone_center=(5.0, 2.0),
        interaction_zone_radius_m=1.1,
        interaction_zone_min_pedestrians=3,
        metric_thresholds={
            "max_consecutive_interaction_zone_slow_steps": DiagnosticThreshold(
                value=2.0,
                direction="max_allowed",
            )
        },
        diagnostic_metadata={
            "diagnostic_only_not_benchmark_gate": True,
            "geometric_fixture": "bottleneck_freeze_deadlock",
            "neck_width_m": 1.3,
        },
    )


def _base_map(
    scenario_id: str,
    *,
    width: float,
    height: float,
    obstacles: list[Obstacle] | None = None,
) -> MapDefinition:
    robot_spawn = _rect(0.2, 0.2, 0.8, 0.8)
    robot_goal = _rect(width - 0.8, height - 0.8, width - 0.2, height - 0.2)
    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles or [],
        robot_spawn_zones=[robot_spawn],
        ped_spawn_zones=[],
        robot_goal_zones=[robot_goal],
        bounds=[
            ((0.0, 0.0), (width, 0.0)),
            ((width, 0.0), (width, height)),
            ((width, height), (0.0, height)),
            ((0.0, height), (0.0, 0.0)),
        ],
        robot_routes=[
            GlobalRoute(
                spawn_id=0,
                goal_id=0,
                waypoints=[(0.5, 0.5), (width - 0.5, height - 0.5)],
                spawn_zone=robot_spawn,
                goal_zone=robot_goal,
                source_path_id=f"{scenario_id}_diagnostic_robot_stub",
                source_label="diagnostic-only robot route stub; no robot inserted",
            )
        ],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
    )


def _copy_map_with_single_pedestrians(
    base_map: MapDefinition,
    single_pedestrians: list[SinglePedestrianDefinition],
) -> MapDefinition:
    return MapDefinition(
        width=base_map.width,
        height=base_map.height,
        obstacles=list(base_map.obstacles),
        robot_spawn_zones=list(base_map.robot_spawn_zones),
        ped_spawn_zones=list(base_map.ped_spawn_zones),
        robot_goal_zones=list(base_map.robot_goal_zones),
        bounds=list(base_map.bounds),
        robot_routes=list(base_map.robot_routes),
        ped_goal_zones=list(base_map.ped_goal_zones),
        ped_crowded_zones=list(base_map.ped_crowded_zones),
        ped_routes=list(base_map.ped_routes),
        single_pedestrians=single_pedestrians,
    )


def _minimum_pairwise_distance(positions: np.ndarray) -> float:
    if positions.shape[1] < 2:
        return 0.0
    deltas = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    mask = np.eye(distances.shape[1], dtype=bool)[np.newaxis, :, :]
    distances = np.where(mask, np.inf, distances)
    return float(np.min(distances))


def _max_consecutive_true(mask: np.ndarray) -> int:
    max_run = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _axis_index(axis: Axis) -> int:
    return 0 if axis == "x" else 1


def _rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    return ((x0, y1), (x1, y1), (x1, y0))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"non JSON-safe value: {value!r}")
