"""Diagnostic-only pedestrian flow validation harness for issue #3971.

The harness runs the pedestrian simulator with ``robots=[]`` on small synthetic
fixtures and emits descriptive metrics only. It intentionally does not apply
realism thresholds, build gates, or benchmark-strength claims.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from robot_sf.benchmark.ped_trajectory_quality import (
    TrajectoryQualityConfig,
    compute_trajectory_quality_distributions,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from robot_sf.common.types import Rect

PED_FLOW_REPORT_SCHEMA_VERSION = "pedestrian_flow_validation.report.v1"
CLAIM_BOUNDARY = (
    "diagnostic-only pedestrian-flow validation; no realism thresholds, build gates, "
    "robot-vs-crowd safety claims, or benchmark-strength claims"
)

Axis = Literal["x", "y"]
Direction = Literal["positive", "negative", "both"]


@dataclass(frozen=True)
class FlowGate:
    """Axis-aligned crossing gate used for descriptive flow-rate summaries."""

    gate_id: str
    axis: Axis
    coordinate: float
    direction: Direction = "both"


@dataclass(frozen=True)
class PedFlowScenarioSpec:
    """Synthetic no-robot pedestrian-flow fixture."""

    scenario_id: str
    map_def: MapDefinition
    measurement_area_m2: float
    flow_gates: tuple[FlowGate, ...]
    lane_axis: Axis = "x"
    lane_lateral_axis: Axis = "y"
    diagnostic_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PedFlowRunConfig:
    """Runtime controls for a short diagnostic flow run."""

    duration_s: float = 2.0
    dt_s: float = 0.1
    pedestrian_counts: tuple[int, ...] = (2, 6)
    seed: int = 3971
    speed_mps: float = 1.1
    jam_speed_threshold_mps: float = 0.08
    diagnostic_only_not_benchmark_gate: bool = True

    def __post_init__(self) -> None:
        """Validate finite positive run controls and non-negative pedestrian counts."""

        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if not self.pedestrian_counts:
            raise ValueError("pedestrian_counts must not be empty")
        if any(count < 0 for count in self.pedestrian_counts):
            raise ValueError("pedestrian_counts must be non-negative")


@dataclass(frozen=True)
class PedFlowTrace:
    """Collected no-robot pedestrian trace for one scenario/count/seed."""

    scenario_id: str
    pedestrian_count: int
    density_ped_per_m2: float
    seed: int
    dt_s: float
    duration_s: float
    measurement_area_m2: float
    pedestrian_model: str
    robot_count: int
    positions: np.ndarray
    velocities: np.ndarray


def build_ped_flow_scenarios() -> dict[str, PedFlowScenarioSpec]:
    """Return the three issue-requested diagnostic flow fixtures."""

    scenarios = (
        _build_bidirectional_corridor_spec(),
        _build_bottleneck_spec(),
        _build_forked_route_spec(),
    )
    return {scenario.scenario_id: scenario for scenario in scenarios}


def run_pedestrian_flow_validation(
    *,
    config: PedFlowRunConfig | None = None,
    scenarios: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run configured no-robot flow diagnostics and return a JSON-safe report.

    Returns
    -------
    dict[str, Any]
        Compact report with diagnostic-only status, flow metrics, and trajectory summaries.
    """

    run_config = config or PedFlowRunConfig()
    scenario_specs = build_ped_flow_scenarios()
    selected_ids = tuple(scenarios or scenario_specs.keys())
    unknown = sorted(set(selected_ids) - set(scenario_specs))
    if unknown:
        raise ValueError(f"Unknown pedestrian flow scenario(s): {unknown}")

    traces: list[PedFlowTrace] = []
    per_run: list[dict[str, Any]] = []
    for scenario_id in selected_ids:
        spec = scenario_specs[scenario_id]
        for count in run_config.pedestrian_counts:
            trace = run_ped_flow_trace(spec, pedestrian_count=count, config=run_config)
            traces.append(trace)
            flow = compute_flow_metrics(trace, spec, config=run_config)
            quality = compute_trajectory_quality_distributions(
                trace.positions,
                trace.velocities,
                dt_s=trace.dt_s,
                config=TrajectoryQualityConfig(),
            )
            per_run.append(
                {
                    "scenario_id": trace.scenario_id,
                    "pedestrian_count": trace.pedestrian_count,
                    "density_ped_per_m2": trace.density_ped_per_m2,
                    "seed": trace.seed,
                    "step_count": int(max(trace.positions.shape[0] - 1, 0)),
                    "duration_s": trace.duration_s,
                    "dt_s": trace.dt_s,
                    "measurement_area_m2": trace.measurement_area_m2,
                    "pedestrian_model": trace.pedestrian_model,
                    "robot_count": trace.robot_count,
                    "flow_metrics": flow,
                    "trajectory_quality": quality,
                }
            )

    report = {
        "schema_version": PED_FLOW_REPORT_SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": "diagnostic-only",
        "diagnostic_only_not_benchmark_gate": run_config.diagnostic_only_not_benchmark_gate,
        "status": {
            "thresholds_applied": False,
            "build_gate": False,
            "pedestrian_dynamics_changed": False,
            "slurm_or_gpu_used": False,
            "robot_inserted": False,
        },
        "run_config": {
            "duration_s": run_config.duration_s,
            "dt_s": run_config.dt_s,
            "pedestrian_counts": list(run_config.pedestrian_counts),
            "seed": run_config.seed,
            "speed_mps": run_config.speed_mps,
        },
        "scenario_ids": list(selected_ids),
        "flow_metrics": _aggregate_flow_metrics(per_run),
        "trajectory_quality": _aggregate_trajectory_quality(per_run),
        "runs": per_run,
    }
    return _json_safe(report)


def run_ped_flow_trace(
    spec: PedFlowScenarioSpec,
    *,
    pedestrian_count: int,
    config: PedFlowRunConfig,
) -> PedFlowTrace:
    """Run one synthetic fixture with no robots and collect pedestrian states.

    Returns
    -------
    PedFlowTrace
        Positions, velocities, and run metadata for one fixture.
    """

    map_def = _copy_map_with_single_pedestrians(
        spec.map_def,
        _build_single_pedestrians(spec.scenario_id, pedestrian_count, speed_mps=config.speed_mps),
    )
    sim_settings = SimulationSettings(
        sim_time_in_secs=config.duration_s,
        time_per_step_in_secs=config.dt_s,
        ped_density_by_difficulty=[0.0],
        difficulty=0,
        route_spawn_seed=config.seed,
        max_total_pedestrians=pedestrian_count,
    )
    sim = Simulator(
        sim_settings,
        map_def,
        robots=[],
        goal_proximity_threshold=0.0,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )

    positions = [np.asarray(sim.ped_pos, dtype=float).copy()]
    velocities = [np.asarray(sim.ped_vel, dtype=float).copy()]
    step_count = round(config.duration_s / config.dt_s)
    for _ in range(step_count):
        sim.step_once([])
        positions.append(np.asarray(sim.ped_pos, dtype=float).copy())
        velocities.append(np.asarray(sim.ped_vel, dtype=float).copy())

    return PedFlowTrace(
        scenario_id=spec.scenario_id,
        pedestrian_count=pedestrian_count,
        density_ped_per_m2=float(pedestrian_count / spec.measurement_area_m2),
        seed=config.seed,
        dt_s=config.dt_s,
        duration_s=float(step_count * config.dt_s),
        measurement_area_m2=spec.measurement_area_m2,
        pedestrian_model=str(sim.pedestrian_model),
        robot_count=len(sim.robots),
        positions=np.stack(positions, axis=0),
        velocities=np.stack(velocities, axis=0),
    )


def compute_flow_metrics(
    trace: PedFlowTrace,
    spec: PedFlowScenarioSpec,
    *,
    config: PedFlowRunConfig | None = None,
) -> dict[str, Any]:
    """Compute collective-flow metrics for one trace.

    Returns
    -------
    dict[str, Any]
        Flow metric families requested by issue #3971.
    """

    cfg = config or PedFlowRunConfig()
    return {
        "average_speed_vs_density": average_speed_vs_density(trace),
        "flow_rate": [compute_flow_rate(trace, gate) for gate in spec.flow_gates],
        "lane_formation_score": lane_formation_score(
            trace,
            movement_axis=spec.lane_axis,
            lateral_axis=spec.lane_lateral_axis,
        ),
        "jam_duration": jam_duration(trace, speed_threshold_mps=cfg.jam_speed_threshold_mps),
    }


def average_speed_vs_density(trace: PedFlowTrace) -> dict[str, float | int | str]:
    """Return one fundamental-diagram point for a trace.

    Returns
    -------
    dict[str, float | int | str]
        Density, average speed, sample count, and status.
    """

    speed = np.linalg.norm(trace.velocities, axis=2)
    summary = _finite_summary(speed)
    return {
        "scenario_id": trace.scenario_id,
        "pedestrian_count": trace.pedestrian_count,
        "density_ped_per_m2": trace.density_ped_per_m2,
        "average_speed_mps": summary["mean"],
        "speed_sample_count": summary["count"],
        "status": summary["status"],
    }


def compute_flow_rate(trace: PedFlowTrace, gate: FlowGate) -> dict[str, float | int | str]:
    """Count unique pedestrian crossings through a gate divided by duration.

    Returns
    -------
    dict[str, float | int | str]
        Crossing count and flow rate for the gate.
    """

    if trace.positions.shape[1] == 0:
        return _unavailable_metric(gate.gate_id, "no_pedestrians")
    axis_idx = _axis_index(gate.axis)
    samples = trace.positions[:, :, axis_idx]
    finite = np.isfinite(samples[:-1]) & np.isfinite(samples[1:])
    previous = samples[:-1] - gate.coordinate
    current = samples[1:] - gate.coordinate
    crossed = finite & (previous * current <= 0.0) & (previous != current)
    if gate.direction == "positive":
        crossed &= current > previous
    elif gate.direction == "negative":
        crossed &= current < previous
    crossing_pedestrians = set(np.where(crossed)[1].tolist())
    crossings = len(crossing_pedestrians)
    duration = max(trace.duration_s, trace.dt_s)
    return {
        "gate_id": gate.gate_id,
        "axis": gate.axis,
        "coordinate": gate.coordinate,
        "direction": gate.direction,
        "crossing_pedestrian_count": crossings,
        "flow_rate_ped_per_s": float(crossings / duration),
        "status": "ok",
    }


def lane_formation_score(
    trace: PedFlowTrace,
    *,
    movement_axis: Axis = "x",
    lateral_axis: Axis = "y",
) -> dict[str, float | int | str]:
    """Proxy lane formation by lateral separation of opposite-moving pedestrians.

    Returns
    -------
    dict[str, float | int | str]
        Normalized lateral separation summary or an unavailable status.
    """

    if trace.positions.shape[1] < 2:
        return _unavailable_metric("lane_formation_score", "fewer_than_two_pedestrians")
    move_idx = _axis_index(movement_axis)
    lateral_idx = _axis_index(lateral_axis)
    movement = trace.velocities[:, :, move_idx]
    lateral = trace.positions[:, :, lateral_idx]
    positive = movement > 0.05
    negative = movement < -0.05
    scores: list[float] = []
    for lateral_t, pos_mask, neg_mask in zip(lateral, positive, negative, strict=True):
        if np.count_nonzero(pos_mask) == 0 or np.count_nonzero(neg_mask) == 0:
            continue
        pos_mean = float(np.nanmean(lateral_t[pos_mask]))
        neg_mean = float(np.nanmean(lateral_t[neg_mask]))
        spread = float(np.nanmax(lateral_t) - np.nanmin(lateral_t))
        if np.isfinite(spread) and spread > 0.0:
            scores.append(abs(pos_mean - neg_mean) / spread)
    summary = _finite_summary(np.asarray(scores, dtype=float))
    return {
        "metric_id": "lane_formation_score",
        "movement_axis": movement_axis,
        "lateral_axis": lateral_axis,
        "score": summary["mean"],
        "sample_count": summary["count"],
        "status": summary["status"],
    }


def jam_duration(
    trace: PedFlowTrace,
    *,
    speed_threshold_mps: float = 0.08,
) -> dict[str, float | int | str]:
    """Return seconds where mean pedestrian speed stays below the jam threshold.

    Returns
    -------
    dict[str, float | int | str]
        Jam duration, sample count, and diagnostic threshold used for measurement.
    """

    if trace.positions.shape[1] == 0:
        return _unavailable_metric("jam_duration", "no_pedestrians")
    speeds = np.linalg.norm(trace.velocities, axis=2)
    mean_speed = np.nanmean(speeds, axis=1)
    finite = np.isfinite(mean_speed)
    jam_steps = finite & (mean_speed < speed_threshold_mps)
    return {
        "metric_id": "jam_duration",
        "speed_threshold_mps": speed_threshold_mps,
        "jam_duration_s": float(np.count_nonzero(jam_steps) * trace.dt_s),
        "sample_count": int(np.count_nonzero(finite)),
        "status": "ok" if np.any(finite) else "empty",
    }


def write_pedestrian_flow_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write compact JSON, Markdown, and trajectory-quality CSV evidence.

    Returns
    -------
    dict[str, Path]
        Paths to the written summary JSON, Markdown summary, and quality CSV.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "README.md"
    quality_csv = output_dir / "trajectory_quality.csv"

    compact_report = _compact_evidence_report(report)
    summary_json.write_text(json.dumps(_json_safe(compact_report), indent=2, sort_keys=True) + "\n")
    summary_md.write_text(render_markdown_summary(compact_report))
    _write_trajectory_quality_csv(report, quality_csv)
    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "trajectory_quality_csv": quality_csv,
    }


def render_markdown_summary(report: dict[str, Any]) -> str:
    """Render a compact reviewer-readable diagnostic summary.

    Returns
    -------
    str
        Markdown summary with claim boundary and caveats first.
    """

    lines = [
        "# Issue #3971 Pedestrian Flow Validation Evidence",
        "",
        f"Claim boundary: {report['claim_boundary']}.",
        "",
        "Evidence status: diagnostic-only smoke evidence. No pass/fail realism threshold was applied.",
        "",
        "Major caveats: short CPU fixtures, synthetic maps, no robot inserted, no human-subject "
        "realism validation, and no benchmark-strength claim.",
        "",
        "## Run Summary",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Scenarios: {', '.join(report['scenario_ids'])}",
        f"- Pedestrian counts: {report['run_config']['pedestrian_counts']}",
        f"- Duration: {report['run_config']['duration_s']} s",
        f"- Timestep: {report['run_config']['dt_s']} s",
        f"- Robot inserted: {report['status']['robot_inserted']}",
        f"- Thresholds applied: {report['status']['thresholds_applied']}",
        "",
        "## Flow Metrics",
        "",
        "| scenario | peds | density | avg speed | jam s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        avg = run["flow_metrics"]["average_speed_vs_density"]
        jam = run["flow_metrics"]["jam_duration"]
        lines.append(
            "| {scenario} | {peds} | {density:.4f} | {speed:.4f} | {jam:.4f} |".format(
                scenario=run["scenario_id"],
                peds=run["pedestrian_count"],
                density=run["density_ped_per_m2"],
                speed=float(avg.get("average_speed_mps", 0.0)),
                jam=float(jam.get("jam_duration_s", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _compact_evidence_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return the small durable JSON evidence shape written under docs/context/evidence."""

    compact_runs = []
    for run in report["runs"]:
        compact_runs.append(
            {
                "scenario_id": run["scenario_id"],
                "pedestrian_count": run["pedestrian_count"],
                "density_ped_per_m2": run["density_ped_per_m2"],
                "seed": run["seed"],
                "step_count": run["step_count"],
                "duration_s": run["duration_s"],
                "dt_s": run["dt_s"],
                "measurement_area_m2": run["measurement_area_m2"],
                "pedestrian_model": run["pedestrian_model"],
                "robot_count": run["robot_count"],
                "flow_metrics": run["flow_metrics"],
            }
        )
    return {
        "schema_version": report["schema_version"],
        "claim_boundary": report["claim_boundary"],
        "evidence_tier": report["evidence_tier"],
        "diagnostic_only_not_benchmark_gate": report["diagnostic_only_not_benchmark_gate"],
        "status": report["status"],
        "run_config": report["run_config"],
        "scenario_ids": report["scenario_ids"],
        "flow_metrics": report["flow_metrics"],
        "trajectory_quality": report["trajectory_quality"],
        "runs": compact_runs,
    }


def _aggregate_flow_metrics(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "average_speed_vs_density": [
            run["flow_metrics"]["average_speed_vs_density"] for run in per_run
        ],
        "flow_rate": [item for run in per_run for item in run["flow_metrics"]["flow_rate"]],
        "lane_formation_score": [run["flow_metrics"]["lane_formation_score"] for run in per_run],
        "jam_duration": [run["flow_metrics"]["jam_duration"] for run in per_run],
    }


def _aggregate_trajectory_quality(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_run:
        return {}
    first = per_run[0]["trajectory_quality"]
    return {
        key: first[key]
        for key in (
            "speed_mps",
            "acceleration_mps2",
            "curvature_1pm",
            "turning_angle_rad",
            "pairwise_distance_m",
            "stop_frequency_hz",
            "stop_fraction",
        )
        if key in first
    }


def _write_trajectory_quality_csv(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in report["runs"]:
        quality = run["trajectory_quality"]
        for metric, summary in quality.items():
            if not isinstance(summary, dict) or "count" not in summary:
                continue
            rows.append(
                {
                    "scenario_id": run["scenario_id"],
                    "pedestrian_count": run["pedestrian_count"],
                    "density_ped_per_m2": run["density_ped_per_m2"],
                    "metric": metric,
                    "status": summary.get("status", ""),
                    "count": summary.get("count", 0),
                    "mean": summary.get("mean", ""),
                    "std": summary.get("std", ""),
                    "min": summary.get("min", ""),
                    "max": summary.get("max", ""),
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scenario_id",
                "pedestrian_count",
                "density_ped_per_m2",
                "metric",
                "status",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_bidirectional_corridor_spec() -> PedFlowScenarioSpec:
    map_def = _base_map("bidirectional_corridor", width=12.0, height=4.0)
    return PedFlowScenarioSpec(
        scenario_id="bidirectional_corridor",
        map_def=map_def,
        measurement_area_m2=48.0,
        flow_gates=(FlowGate("center_x", "x", 6.0, "both"),),
        diagnostic_metadata={"diagnostic_only_not_benchmark_gate": True},
    )


def _build_bottleneck_spec() -> PedFlowScenarioSpec:
    obstacles = [
        Obstacle([(5.4, 0.0), (6.6, 0.0), (6.6, 1.45), (5.4, 1.45)]),
        Obstacle([(5.4, 2.55), (6.6, 2.55), (6.6, 4.0), (5.4, 4.0)]),
    ]
    map_def = _base_map("bottleneck", width=12.0, height=4.0, obstacles=obstacles)
    return PedFlowScenarioSpec(
        scenario_id="bottleneck",
        map_def=map_def,
        measurement_area_m2=48.0,
        flow_gates=(FlowGate("neck_x", "x", 6.0, "positive"),),
        diagnostic_metadata={"diagnostic_only_not_benchmark_gate": True, "bottleneck_width_m": 1.1},
    )


def _build_forked_route_spec() -> PedFlowScenarioSpec:
    map_def = _base_map("forked_route", width=12.0, height=6.0)
    return PedFlowScenarioSpec(
        scenario_id="forked_route",
        map_def=map_def,
        measurement_area_m2=72.0,
        flow_gates=(
            FlowGate("upper_branch_x", "x", 8.0, "positive"),
            FlowGate("lower_branch_x", "x", 8.0, "positive"),
        ),
        diagnostic_metadata={"diagnostic_only_not_benchmark_gate": True},
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


def _build_single_pedestrians(
    scenario_id: str,
    count: int,
    *,
    speed_mps: float,
) -> list[SinglePedestrianDefinition]:
    builders = {
        "bidirectional_corridor": _bidirectional_pedestrian,
        "bottleneck": _bottleneck_pedestrian,
        "forked_route": _forked_route_pedestrian,
    }
    builder = builders[scenario_id]
    return [builder(idx, count, speed_mps) for idx in range(count)]


def _bidirectional_pedestrian(
    idx: int,
    count: int,
    speed_mps: float,
) -> SinglePedestrianDefinition:
    lane_count = max(count, 1)
    y = 0.7 + (idx + 0.5) * (2.6 / lane_count)
    if idx % 2 == 0:
        start, goal = (1.0, y), (11.0, y)
    else:
        start, goal = (11.0, y), (1.0, y)
    return SinglePedestrianDefinition(
        id=f"bidirectional_corridor_p{idx}",
        start=start,
        goal=goal,
        speed_m_s=speed_mps,
        metadata={"diagnostic_only_not_benchmark_gate": True},
    )


def _bottleneck_pedestrian(idx: int, count: int, speed_mps: float) -> SinglePedestrianDefinition:
    y_offsets = np.linspace(-1.35, 1.35, max(count, 1))
    y = float(2.0 + y_offsets[idx])
    return SinglePedestrianDefinition(
        id=f"bottleneck_p{idx}",
        start=(1.0, y),
        trajectory=[(5.2, 2.0), (6.8, 2.0), (11.0, y)],
        speed_m_s=speed_mps,
        metadata={"diagnostic_only_not_benchmark_gate": True},
    )


def _forked_route_pedestrian(idx: int, count: int, speed_mps: float) -> SinglePedestrianDefinition:
    y = 2.1 + (idx % max(count, 1)) * 0.55
    upper = idx % 2 == 0
    goal = (11.0, 4.8) if upper else (11.0, 1.2)
    branch = (7.0, 4.2) if upper else (7.0, 1.8)
    return SinglePedestrianDefinition(
        id=f"forked_route_p{idx}",
        start=(1.0, y),
        trajectory=[(5.5, 3.0), branch, goal],
        speed_m_s=speed_mps,
        metadata={"diagnostic_only_not_benchmark_gate": True},
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


def _finite_summary(values: np.ndarray) -> dict[str, float | int | str]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"status": "empty", "count": 0, "mean": 0.0}
    return {"status": "ok", "count": int(finite.size), "mean": float(np.mean(finite))}


def _unavailable_metric(metric_id: str, status: str) -> dict[str, float | int | str]:
    return {"metric_id": metric_id, "status": status, "count": 0}


def _axis_index(axis: Axis) -> int:
    return 0 if axis == "x" else 1


def _rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    return ((x0, y1), (x1, y1), (x1, y0))


def _json_safe(value: Any) -> Any:
    """Return JSON-safe nested dict/list/scalar values."""

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
