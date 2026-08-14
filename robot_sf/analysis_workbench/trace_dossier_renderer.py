"""Static multi-panel dossier rendering for ``simulation_trace_export.v1`` traces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
from jsonschema import Draft202012Validator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceFrame,
    load_simulation_trace_export,
)
from robot_sf.errors import RobotSfError

TRACE_DOSSIER_MANIFEST_SCHEMA_VERSION = "trace_dossier_manifest.v1"
TRACE_DOSSIER_RENDERER_VERSION = "issue_7086.v1"
TRACE_DOSSIER_MANIFEST_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "trace_dossier_manifest.v1.json"
)
DEFAULT_DOSSIER_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)


class TraceDossierRenderError(RobotSfError, ValueError):
    """Raised when a trace cannot be rendered without inventing dossier fields."""


@dataclass(frozen=True, slots=True)
class ClearancePoint:
    """One frame's closest robot-to-pedestrian clearance observation."""

    step: int
    time_s: float
    pedestrian_id: str
    value_m: float
    mode: str


@dataclass(frozen=True, slots=True)
class TraceDossierRenderResult:
    """Output paths and manifest for one rendered trace dossier."""

    png_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def render_trace_dossier(
    trace_path: Path,
    *,
    output_png: Path,
    manifest_path: Path,
    command: str,
) -> TraceDossierRenderResult:
    """Render one ``simulation_trace_export.v1`` trace as a four-panel PNG dossier.

    Returns:
        Paths and manifest payload for the generated diagnostic artifact.
    """

    try:
        trace = load_simulation_trace_export(trace_path)
    except OverflowError as error:
        raise TraceDossierRenderError(
            "trace contains a numeric value outside the supported float range"
        ) from error
    _validate_renderable_trace(trace)
    clearance_points = _clearance_points(trace)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _render_png(trace, clearance_points, output_png)
    manifest = _manifest_payload(
        trace,
        trace_path=trace_path,
        output_png=output_png,
        command=command,
        clearance_points=clearance_points,
    )
    validate_trace_dossier_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TraceDossierRenderResult(
        png_path=output_png,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_trace_dossier_manifest_schema() -> dict[str, Any]:
    """Load the manifest schema used by the static trace dossier renderer.

    Returns:
        Parsed ``trace_dossier_manifest.v1`` JSON Schema.
    """

    return _load_trace_dossier_manifest_schema()


@lru_cache(maxsize=1)
def _load_trace_dossier_manifest_schema() -> dict[str, Any]:
    return json.loads(TRACE_DOSSIER_MANIFEST_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_trace_dossier_manifest(payload: dict[str, Any]) -> None:
    """Validate a trace dossier manifest payload."""

    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in sorted(
            Draft202012Validator(_load_trace_dossier_manifest_schema()).iter_errors(payload),
            key=lambda item: list(item.absolute_path),
        )
    ]
    if errors:
        raise TraceDossierRenderError("; ".join(errors))


def _validate_renderable_trace(trace: SimulationTraceExport) -> None:
    """Fail closed when required panel inputs are absent from the trace contract."""

    if trace.evidence_boundary != "analysis_workbench_only":
        raise TraceDossierRenderError(
            "trace dossier renderer accepts only analysis_workbench_only traces"
        )
    if trace.coordinate_frame != "world":
        raise TraceDossierRenderError("trajectory dossier requires world coordinate_frame")
    if not trace.frames:
        raise TraceDossierRenderError("trace dossier requires at least one frame")
    if not any(frame.pedestrians for frame in trace.frames):
        raise TraceDossierRenderError(
            "clearance-over-time panel requires at least one pedestrian position"
        )
    missing_event_steps = [frame.step for frame in trace.frames if not _event_name(frame)]
    if missing_event_steps:
        raise TraceDossierRenderError(
            "event timeline requires planner.event on every frame; missing steps: "
            + ", ".join(str(step) for step in missing_event_steps)
        )
    _radius_mode(trace)


def _render_png(
    trace: SimulationTraceExport,
    clearance_points: list[ClearancePoint],
    output_png: Path,
) -> None:
    """Write the deterministic four-panel Matplotlib PNG."""

    with plt.rc_context(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "figure.constrained_layout.use": True,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
        _plot_trajectory(trace, axes[0][0])
        _plot_speed_profile(trace, axes[0][1])
        _plot_clearance(clearance_points, axes[1][0])
        _plot_event_timeline(trace, axes[1][1])
        fig.suptitle(
            f"{trace.source.planner_id} / {trace.source.scenario_id} / {trace.source.episode_id}",
            fontsize=11,
        )
        fig.text(
            0.01,
            0.01,
            "diagnostic-only simulation_trace_export.v1 dossier; not benchmark evidence",
            fontsize=8,
        )
        fig.savefig(
            output_png,
            format="png",
            metadata={"Software": "robot_sf trace_dossier_renderer issue_7086.v1"},
        )
        plt.close(fig)


def _plot_trajectory(trace: SimulationTraceExport, ax: Any) -> None:
    robot_xy = [
        _xy(frame.robot["position"], context=f"/frames/{index}/robot/position")
        for index, frame in enumerate(trace.frames)
    ]
    robot_x, robot_y = zip(*robot_xy, strict=True)
    ax.plot(robot_x, robot_y, color="#1f77b4", linewidth=2.0, marker="o", label="robot")
    ax.scatter([robot_x[0]], [robot_y[0]], color="#2ca02c", s=42, zorder=4, label="start")
    ax.scatter([robot_x[-1]], [robot_y[-1]], color="#d62728", s=42, zorder=4, label="end")

    tracks: dict[str, list[tuple[float, float]]] = {}
    for frame_index, frame in enumerate(trace.frames):
        for pedestrian in frame.pedestrians:
            ped_id = str(pedestrian["id"])
            tracks.setdefault(ped_id, []).append(
                _xy(
                    pedestrian["position"],
                    context=f"/frames/{frame_index}/pedestrians/{ped_id}/position",
                )
            )
    for ped_id, points in sorted(tracks.items()):
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color="#7f7f7f", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.scatter(xs, ys, s=18, alpha=0.75, label=ped_id)

    ax.set_title("Trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    _dedup_legend(ax)


def _plot_speed_profile(trace: SimulationTraceExport, ax: Any) -> None:
    speeds = [
        _speed(frame.robot["velocity"], context=f"/frames/{index}/robot/velocity")
        for index, frame in enumerate(trace.frames)
    ]
    selected = [
        _finite_number(
            frame.planner["selected_action"]["linear_velocity"],
            context=f"/frames/{index}/planner/selected_action/linear_velocity",
        )
        for index, frame in enumerate(trace.frames)
    ]
    times = [
        _finite_number(frame.time_s, context=f"/frames/{index}/time_s")
        for index, frame in enumerate(trace.frames)
    ]
    ax.plot(times, speeds, color="#1f77b4", marker="o", label="robot speed")
    ax.plot(times, selected, color="#ff7f0e", marker=".", linestyle="--", label="selected linear")
    ax.set_title("Speed Profile")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def _plot_clearance(clearance_points: list[ClearancePoint], ax: Any) -> None:
    times = [point.time_s for point in clearance_points]
    values = [point.value_m for point in clearance_points]
    labels = [point.pedestrian_id for point in clearance_points]
    mode = clearance_points[0].mode
    title = "Clearance Over Time"
    ylabel = "edge clearance (m)" if mode == "edge_distance_m" else "center distance (m)"
    ax.plot(times, values, color="#9467bd", marker="o")
    minimum = min(clearance_points, key=lambda point: (point.value_m, point.time_s, point.step))
    ax.scatter([minimum.time_s], [minimum.value_m], color="#d62728", s=42, zorder=4)
    ax.annotate(
        f"min {minimum.value_m:.3f} m\n{minimum.pedestrian_id}",
        xy=(minimum.time_s, minimum.value_m),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
    )
    for time_s, value, label in zip(times, values, labels, strict=True):
        ax.annotate(
            label, xy=(time_s, value), xytext=(3, -10), textcoords="offset points", fontsize=7
        )
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _plot_event_timeline(trace: SimulationTraceExport, ax: Any) -> None:
    y_by_event = {
        event: index
        for index, event in enumerate(sorted({_event_name(frame) for frame in trace.frames}))
    }
    times = [frame.time_s for frame in trace.frames]
    ys = [y_by_event[_event_name(frame)] for frame in trace.frames]
    ax.scatter(times, ys, color="#2ca02c", s=38)
    for frame, y_value in zip(trace.frames, ys, strict=True):
        event_id = str(frame.planner.get("event_id") or f"step-{frame.step}")
        ax.annotate(
            f"{frame.step}: {_event_name(frame)}\n{event_id}",
            xy=(frame.time_s, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_title("Annotated Event Timeline")
    ax.set_xlabel("time (s)")
    ax.set_yticks(list(y_by_event.values()), list(y_by_event.keys()))
    ax.grid(True, axis="x", alpha=0.25)


def _clearance_points(trace: SimulationTraceExport) -> list[ClearancePoint]:
    mode = _radius_mode(trace)
    points: list[ClearancePoint] = []
    for frame_index, frame in enumerate(trace.frames):
        if not frame.pedestrians:
            continue
        time_s = _finite_number(frame.time_s, context=f"/frames/{frame_index}/time_s")
        robot_position = _xy(
            frame.robot["position"], context=f"/frames/{frame_index}/robot/position"
        )
        robot_radius = (
            _radius(frame.robot, context=f"/frames/{frame_index}/robot/radius")
            if mode == "edge_distance_m"
            else 0.0
        )
        candidates: list[ClearancePoint] = []
        for ped_index, pedestrian in enumerate(frame.pedestrians):
            ped_position = _xy(
                pedestrian["position"],
                context=f"/frames/{frame_index}/pedestrians/{ped_index}/position",
            )
            distance = math.dist(robot_position, ped_position)
            value = distance
            if mode == "edge_distance_m":
                value = (
                    distance
                    - robot_radius
                    - _radius(
                        pedestrian,
                        context=f"/frames/{frame_index}/pedestrians/{ped_index}/radius",
                    )
                )
            if not math.isfinite(value):
                raise TraceDossierRenderError(
                    f"/frames/{frame_index}: clearance arithmetic produced a non-finite value"
                )
            candidates.append(
                ClearancePoint(
                    step=frame.step,
                    time_s=time_s,
                    pedestrian_id=str(pedestrian["id"]),
                    value_m=value,
                    mode=mode,
                )
            )
        points.append(min(candidates, key=lambda point: (point.value_m, point.pedestrian_id)))
    if not points:
        raise TraceDossierRenderError(
            "clearance-over-time panel requires frames containing pedestrians"
        )
    return points


def _radius_mode(trace: SimulationTraceExport) -> str:
    present = 0
    missing = 0
    for frame in trace.frames:
        actors = [frame.robot, *frame.pedestrians]
        for actor in actors:
            if "radius" in actor:
                _radius(actor, context="/radius")
                present += 1
            else:
                missing += 1
    if present and missing:
        raise TraceDossierRenderError(
            "clearance-over-time cannot mix actor radius metadata with missing radii"
        )
    return "edge_distance_m" if present else "center_distance_m"


def _manifest_payload(
    trace: SimulationTraceExport,
    *,
    trace_path: Path,
    output_png: Path,
    command: str,
    clearance_points: list[ClearancePoint],
) -> dict[str, Any]:
    minimum = min(clearance_points, key=lambda point: (point.value_m, point.time_s, point.step))
    return {
        "schema_version": TRACE_DOSSIER_MANIFEST_SCHEMA_VERSION,
        "trace_schema_version": trace.schema_version,
        "trace_id": trace.trace_id,
        "source_trace": {
            "path": str(trace_path),
            "sha256": _sha256_file(trace_path),
        },
        "evidence_boundary": "diagnostic_only",
        "renderer": {
            "name": "trace_dossier_renderer",
            "version": TRACE_DOSSIER_RENDERER_VERSION,
            "command": command,
        },
        "outputs": {
            "png": {
                "path": str(output_png),
                "sha256": _sha256_file(output_png),
            }
        },
        "panels": [
            "trajectory",
            "speed_profile",
            "clearance_over_time",
            "event_timeline",
        ],
        "clearance_semantics": {
            "mode": clearance_points[0].mode,
            "units": "m",
            "minimum_clearance": {
                "time_s": minimum.time_s,
                "step": minimum.step,
                "pedestrian_id": minimum.pedestrian_id,
                "value_m": minimum.value_m,
            },
        },
        "limitations": _limitations(clearance_points[0].mode),
    }


def _limitations(clearance_mode: str) -> list[str]:
    limitations = [
        "diagnostic-only renderer; does not run simulation or admit benchmark evidence",
        "events are rendered only from planner.event and optional planner.event_id fields present in each frame",
    ]
    if clearance_mode == "center_distance_m":
        limitations.append(
            "actor radii are absent, so clearance panel reports center-to-center distance, not body-edge clearance"
        )
    else:
        limitations.append(
            "all actor radii are present, so clearance panel reports body-edge distance"
        )
    return limitations


def _event_name(frame: SimulationTraceFrame) -> str:
    event = frame.planner.get("event")
    return str(event).strip() if event is not None else ""


def _xy(value: Any, *, context: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise TraceDossierRenderError(f"{context}: expected [x, y]")
    x = _finite_number(value[0], context=f"{context}/0")
    y = _finite_number(value[1], context=f"{context}/1")
    return x, y


def _speed(value: Any, *, context: str) -> float:
    x, y = _xy(value, context=context)
    return math.hypot(x, y)


def _radius(actor: dict[str, Any], *, context: str) -> float:
    value = actor.get("radius")
    radius = _finite_number(value, context=context)
    if radius < 0:
        raise TraceDossierRenderError(f"{context}: radius must be non-negative")
    return radius


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TraceDossierRenderError(f"{context}: expected finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise TraceDossierRenderError(f"{context}: number is outside the float range") from error
    if not math.isfinite(number):
        raise TraceDossierRenderError(f"{context}: expected finite number")
    return number


def _dedup_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles, strict=False))
    ax.legend(dedup.values(), dedup.keys(), loc="best")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
