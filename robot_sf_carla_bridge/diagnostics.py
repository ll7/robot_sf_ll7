"""Conservative diagnostics for Robot-SF traces and CARLA replay summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from robot_sf_carla_bridge.parity import DEFAULT_PARITY_METRICS, DEGRADED_MODES

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

CARLA_REPLAY_DIAGNOSTICS_SCHEMA_VERSION = "carla-replay-diagnostics.v1"
_DIAGNOSTICS_SCHEMA_RESOURCE = "schemas/carla_replay_diagnostics.v1.json"
_VALID_STATUSES = {"available", "degraded", "not_available", "unsupported"}


@dataclass(frozen=True)
class DiagnosticsRow:
    """One capability or semantic diagnostics row."""

    axis: str
    status: str
    robot_sf_value: Any = None
    carla_value: Any = None
    reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostics row."""
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"unsupported diagnostics status: {self.status}")
        return {
            "axis": self.axis,
            "status": self.status,
            "robot_sf_value": self.robot_sf_value,
            "carla_value": self.carla_value,
            "reason": self.reason,
        }


@cache
def load_carla_replay_diagnostics_schema() -> dict[str, Any]:
    """Load the versioned CARLA replay diagnostics JSON schema.

    Returns:
        Parsed JSON schema dictionary.
    """
    schema_path = files("robot_sf_carla_bridge").joinpath(_DIAGNOSTICS_SCHEMA_RESOURCE)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def build_carla_replay_diagnostics(
    robot_sf_record: dict[str, Any],
    carla_summary: dict[str, Any],
    *,
    metric_names: Iterable[str] = DEFAULT_PARITY_METRICS,
) -> dict[str, Any]:
    """Build diagnostics without treating CARLA replay as simulator-equivalence proof.

    Returns:
        JSON-safe diagnostics report with capability, metric, and unsupported-semantic rows.
    """
    capability_rows = [
        *_required_carla_summary_rows(carla_summary),
        _replay_status_row(carla_summary),
        _static_geometry_row(carla_summary),
        _map_coordinate_row(carla_summary),
        _timing_row(carla_summary),
        _robot_terminal_row(robot_sf_record, carla_summary),
        _pedestrian_row(carla_summary),
    ]
    metric_rows = [
        _metric_row(name, _metrics(robot_sf_record), _metrics(carla_summary), carla_summary)
        for name in metric_names
    ]
    unsupported_rows = _unsupported_semantic_rows(carla_summary)
    all_rows = [*capability_rows, *metric_rows, *unsupported_rows]
    return {
        "schema_version": CARLA_REPLAY_DIAGNOSTICS_SCHEMA_VERSION,
        "status": _overall_status(all_rows),
        "interpretation_boundary": (
            "Diagnostics classify comparability surfaces only; they are not simulator-equivalence "
            "or benchmark-transfer evidence by themselves."
        ),
        "capability_matrix": [row.to_json_dict() for row in capability_rows],
        "metric_fields": [row.to_json_dict() for row in metric_rows],
        "unsupported_semantics": [row.to_json_dict() for row in unsupported_rows],
    }


def write_carla_replay_diagnostics_outputs(
    report: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    """Write JSON, Markdown, and CSV diagnostics artifacts to ``output_dir``.

    Returns:
        Mapping from artifact role to written path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "carla_replay_diagnostics.json"
    markdown_path = output_dir / "carla_replay_diagnostics.md"
    capability_path = output_dir / "carla_capability_matrix.csv"
    unsupported_path = output_dir / "unsupported_semantics.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_rows_csv(capability_path, report["capability_matrix"], field_name="axis")
    _write_rows_csv(unsupported_path, report["unsupported_semantics"], field_name="axis")
    return {
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        "capability_matrix": capability_path.as_posix(),
        "unsupported_semantics": unsupported_path.as_posix(),
    }


def _required_carla_summary_rows(carla_summary: dict[str, Any]) -> list[DiagnosticsRow]:
    required = {
        "summary_schema_version": carla_summary.get("schema_version"),
        "replay_status": carla_summary.get("status"),
        "carla_map": _nested_get(carla_summary, ("carla", "map")),
        "actor_summary": carla_summary.get("actors"),
    }
    return [
        DiagnosticsRow(
            axis=name,
            status="available" if value not in (None, "", {}, []) else "not_available",
            carla_value=value,
            reason=None
            if value not in (None, "", {}, [])
            else "required CARLA summary field missing",
        )
        for name, value in required.items()
    ]


def _replay_status_row(carla_summary: dict[str, Any]) -> DiagnosticsRow:
    mode = str(carla_summary.get("mode") or "").lower()
    status = str(carla_summary.get("status") or "").lower()
    degraded = next((value for value in (mode, status) if value in DEGRADED_MODES), None)
    if degraded:
        return DiagnosticsRow(
            axis="replay_status",
            status="degraded",
            carla_value={"mode": mode, "status": status},
            reason=f"CARLA replay mode/status is not native/comparable: {degraded}",
        )
    if status in {"oracle-replay", "native", "available"} or mode in {"oracle-replay", "native"}:
        return DiagnosticsRow(
            axis="replay_status", status="available", carla_value=carla_summary.get("status")
        )
    return DiagnosticsRow(
        axis="replay_status",
        status="not_available",
        carla_value=carla_summary.get("status"),
        reason="CARLA replay status is missing or not recognized as replay evidence",
    )


def _static_geometry_row(carla_summary: dict[str, Any]) -> DiagnosticsRow:
    unsupported = carla_summary.get("unsupported")
    if isinstance(unsupported, dict) and unsupported.get("unsupported_static_obstacle_count"):
        return DiagnosticsRow(
            axis="static_geometry_support",
            status="unsupported",
            carla_value=unsupported,
            reason="CARLA replay summary reports unsupported static obstacle geometry",
        )
    boundary = carla_summary.get("boundary")
    if isinstance(boundary, dict) and boundary.get("static_geometry_replay") is True:
        return DiagnosticsRow(axis="static_geometry_support", status="available", carla_value=True)
    return DiagnosticsRow(
        axis="static_geometry_support",
        status="not_available",
        reason="static-geometry replay support metadata is absent",
    )


def _map_coordinate_row(carla_summary: dict[str, Any]) -> DiagnosticsRow:
    carla_map = _nested_get(carla_summary, ("carla", "map"))
    alignment = carla_summary.get("coordinate_alignment")
    if carla_map and isinstance(alignment, dict):
        replay_mode = str(alignment.get("replay_mode") or "").lower()
        if replay_mode in DEGRADED_MODES:
            return DiagnosticsRow(
                axis="map_coordinate_frame",
                status="degraded",
                carla_value={"map": carla_map, "coordinate_alignment": alignment},
                reason=f"coordinate replay mode is not native/comparable: {replay_mode}",
            )
        return DiagnosticsRow(
            axis="map_coordinate_frame",
            status="available",
            carla_value={"map": carla_map, "coordinate_alignment": alignment},
        )
    return DiagnosticsRow(
        axis="map_coordinate_frame",
        status="not_available",
        carla_value={"map": carla_map, "coordinate_alignment": alignment},
        reason="map or coordinate-frame metadata is missing",
    )


def _timing_row(carla_summary: dict[str, Any]) -> DiagnosticsRow:
    trajectory = carla_summary.get("trajectory")
    if isinstance(trajectory, dict) and "steps_replayed" in trajectory:
        return DiagnosticsRow(axis="timing_step_sync", status="available", carla_value=trajectory)
    return DiagnosticsRow(
        axis="timing_step_sync",
        status="not_available",
        reason="trajectory step synchronization metadata is missing",
    )


def _robot_terminal_row(
    robot_sf_record: dict[str, Any], carla_summary: dict[str, Any]
) -> DiagnosticsRow:
    robot_metrics = _metrics(robot_sf_record)
    carla_metrics = _metrics(carla_summary)
    fields = ("success", "collision")
    if all(field in robot_metrics and field in carla_metrics for field in fields):
        return DiagnosticsRow(
            axis="robot_pose_terminal_event",
            status="available",
            robot_sf_value={field: robot_metrics[field] for field in fields},
            carla_value={field: carla_metrics[field] for field in fields},
        )
    return DiagnosticsRow(
        axis="robot_pose_terminal_event",
        status="not_available",
        robot_sf_value={field: robot_metrics.get(field) for field in fields},
        carla_value={field: carla_metrics.get(field) for field in fields},
        reason="success/collision terminal fields are not present in both inputs",
    )


def _pedestrian_row(carla_summary: dict[str, Any]) -> DiagnosticsRow:
    actors = carla_summary.get("actors")
    if isinstance(actors, dict) and isinstance(actors.get("pedestrians"), int):
        if actors["pedestrians"] > 0:
            return DiagnosticsRow(axis="pedestrian_replay", status="available", carla_value=actors)
        return DiagnosticsRow(
            axis="pedestrian_replay",
            status="not_available",
            carla_value=actors,
            reason="summary reports no pedestrian actors",
        )
    return DiagnosticsRow(
        axis="pedestrian_replay",
        status="not_available",
        reason="actor summary does not expose pedestrian replay count",
    )


def _metric_row(
    metric_name: str,
    robot_metrics: dict[str, Any],
    carla_metrics: dict[str, Any],
    carla_summary: dict[str, Any],
) -> DiagnosticsRow:
    degraded = _degraded_reason(carla_summary)
    if degraded is not None:
        return DiagnosticsRow(axis=metric_name, status="degraded", reason=degraded)
    if metric_name not in robot_metrics:
        return DiagnosticsRow(
            axis=metric_name,
            status="not_available",
            reason="missing Robot-SF metric",
            carla_value=carla_metrics.get(metric_name),
        )
    if metric_name not in carla_metrics:
        return DiagnosticsRow(
            axis=metric_name,
            status="not_available",
            robot_sf_value=robot_metrics.get(metric_name),
            reason="missing CARLA metric",
        )
    return DiagnosticsRow(
        axis=metric_name,
        status="available",
        robot_sf_value=robot_metrics[metric_name],
        carla_value=carla_metrics[metric_name],
    )


def _unsupported_semantic_rows(carla_summary: dict[str, Any]) -> list[DiagnosticsRow]:
    rows = [
        DiagnosticsRow(
            axis="sensor_perception_replay",
            status="unsupported",
            reason="CARLA replay diagnostics do not compare sensor or perception pipelines",
        ),
        DiagnosticsRow(
            axis="broad_simulator_equivalence",
            status="unsupported",
            reason="live replay diagnostics are not simulator-equivalence evidence",
        ),
    ]
    unsupported = carla_summary.get("unsupported")
    if isinstance(unsupported, dict):
        for key, value in sorted(unsupported.items()):
            if key in ("boundary", "reason"):
                continue
            rows.append(
                DiagnosticsRow(
                    axis=f"carla_summary.{key}",
                    status="unsupported",
                    carla_value=value,
                    reason=str(
                        unsupported.get("reason") or "CARLA summary marks this semantic unsupported"
                    ),
                )
            )
    return rows


def _overall_status(rows: Sequence[DiagnosticsRow]) -> str:
    statuses = {row.status for row in rows}
    if "degraded" in statuses:
        return "degraded"
    if "available" in statuses:
        return "available"
    return "not_available"


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", record)
    nested_replay = record.get("replay")
    if "metrics" not in record and isinstance(nested_replay, dict):
        nested_metrics = nested_replay.get("metrics")
        if isinstance(nested_metrics, dict):
            return nested_metrics
    return metrics if isinstance(metrics, dict) else {}


def _degraded_reason(carla_summary: dict[str, Any]) -> str | None:
    mode = str(carla_summary.get("mode") or "").lower()
    status = str(carla_summary.get("status") or "").lower()
    degraded = next((value for value in (mode, status) if value in DEGRADED_MODES), None)
    if degraded is not None:
        return f"CARLA replay mode/status is not native/comparable: {degraded}"
    alignment = carla_summary.get("coordinate_alignment")
    if isinstance(alignment, dict):
        replay_mode = str(alignment.get("replay_mode") or "").lower()
        if replay_mode in DEGRADED_MODES:
            return f"coordinate replay mode is not native/comparable: {replay_mode}"
    return None


def _nested_get(record: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CARLA Replay Diagnostics",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Status: `{report['status']}`",
        f"- Boundary: {report['interpretation_boundary']}",
        "",
        "## Capability Matrix",
        "",
        "| Axis | Status | Reason |",
        "| --- | --- | --- |",
    ]
    lines.extend(_markdown_rows(report["capability_matrix"]))
    lines.extend(
        ["", "## Metric Fields", "", "| Metric | Status | Reason |", "| --- | --- | --- |"]
    )
    lines.extend(_markdown_rows(report["metric_fields"]))
    lines.extend(
        [
            "",
            "## Unsupported Semantics",
            "",
            "| Semantic | Status | Reason |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(_markdown_rows(report["unsupported_semantics"]))
    return "\n".join(lines) + "\n"


def _markdown_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [f"| `{row['axis']}` | `{row['status']}` | {row.get('reason') or ''} |" for row in rows]


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]], *, field_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field_name, "status", "robot_sf_value", "carla_value", "reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field_name: row["axis"],
                    "status": row["status"],
                    "robot_sf_value": _csv_json_cell(row.get("robot_sf_value")),
                    "carla_value": _csv_json_cell(row.get("carla_value")),
                    "reason": row.get("reason") or "",
                }
            )


def _csv_json_cell(value: Any) -> str:
    """Return a CSV cell for an optional JSON value, leaving missing values empty."""
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)
