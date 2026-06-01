"""Render ``simulation_trace_export.v1`` traces as static Markdown reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# This tool only renders JSON traces. Keep optional ML backend import chatter out of CLI output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    SimulationTraceFrame,
    load_simulation_trace_export,
)
from robot_sf.analysis_workbench.trace_annotation import (
    TraceAnnotation,
    TraceAnnotationSet,
    TraceAnnotationSetValidationError,
    load_trace_annotation_set,
)

ANNOTATION_KEYS = {
    "annotation",
    "annotations",
    "clearance",
    "collision",
    "metrics",
    "min_ttc",
    "near_miss",
    "note",
    "notes",
    "pet",
    "risk",
    "risk_score",
}


def render_trace_report(
    trace: SimulationTraceExport,
    *,
    trace_path: Path | None = None,
    annotation_set: TraceAnnotationSet | None = None,
    annotation_path: Path | None = None,
) -> str:
    """Render a loaded trace export as Markdown.

    Returns:
        Markdown text suitable for writing to ``report.md``.
    """

    lines: list[str] = [
        "# Simulation Trace Report",
        "",
        (
            "Trace reports are analysis/debug artifacts for local inspection. They are "
            "not benchmark evidence by themselves and do not replace benchmark summaries, "
            "episode JSONL, manifests, or paper-facing evidence contracts."
        ),
        "",
    ]
    lines.extend(_trace_metadata(trace))
    lines.extend(_summary(trace))
    if trace_path is not None or annotation_set is not None:
        lines.extend(_source_fixture_summary(trace_path, annotation_set, annotation_path))
    lines.extend(_event_summary(trace.frames))
    lines.extend(_planner_key_summary(trace.frames))
    lines.extend(_annotation_summary(trace.frames))
    if annotation_set is not None:
        lines.extend(_annotation_set_summary(annotation_set))
    lines.extend(_frame_table(trace.frames))
    return "\n".join(lines).rstrip() + "\n"


def write_trace_report(
    *,
    trace_path: Path,
    output: Path,
    annotation_path: Path | None = None,
) -> Path:
    """Load a trace export and write a Markdown report.

    Raises:
        SimulationTraceExportValidationError: if ``trace_path`` is not a valid trace export.
        TraceAnnotationSetValidationError: if ``annotation_path`` is invalid or mismatched.
        OSError: if input or output files cannot be read or written.
    """

    _validate_tracked_fixture_path(trace_path, label="trace")
    trace = load_simulation_trace_export(trace_path)
    annotation_set = None
    if annotation_path is not None:
        _validate_tracked_fixture_path(annotation_path, label="annotation")
        annotation_set = load_trace_annotation_set(annotation_path)
        _validate_annotation_matches_trace(
            trace=trace,
            trace_path=trace_path,
            annotation_set=annotation_set,
        )
    markdown = render_trace_report(
        trace,
        trace_path=trace_path,
        annotation_set=annotation_set,
        annotation_path=annotation_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return output


def _trace_metadata(trace: SimulationTraceExport) -> list[str]:
    """Render trace and source metadata tables."""

    rows = [
        ("schema_version", trace.schema_version),
        ("trace_id", trace.trace_id),
        ("evidence_boundary", trace.evidence_boundary),
        ("coordinate_frame", trace.coordinate_frame),
        ("units", ", ".join(f"{key}={value}" for key, value in sorted(trace.units.items()))),
    ]
    source_rows = [
        ("scenario_id", trace.source.scenario_id),
        ("seed", trace.source.seed),
        ("planner_id", trace.source.planner_id),
        ("episode_id", trace.source.episode_id),
        ("generated_by", trace.source.generated_by),
    ]
    return [
        "## Trace Metadata",
        "",
        *_markdown_table(("field", "value"), rows),
        "",
        "## Source",
        "",
        *_markdown_table(("field", "value"), source_rows),
        "",
    ]


def _summary(trace: SimulationTraceExport) -> list[str]:
    """Render high-level frame, time, and pedestrian counts."""

    steps = [frame.step for frame in trace.frames]
    times = [frame.time_s for frame in trace.frames]
    pedestrian_counts = [len(frame.pedestrians) for frame in trace.frames]
    event_counts = Counter(_event_name(frame) for frame in trace.frames)
    rows = [
        ("frames", len(trace.frames)),
        ("step_range", f"{min(steps)} to {max(steps)}"),
        ("time_range_s", f"{min(times):.3f} to {max(times):.3f}"),
        ("duration_s", f"{max(times) - min(times):.3f}"),
        ("unique_events", len(event_counts)),
        ("max_pedestrians_per_frame", max(pedestrian_counts)),
    ]
    return [
        "## Summary",
        "",
        *_markdown_table(("metric", "value"), rows),
        "",
    ]


def _source_fixture_summary(
    trace_path: Path | None,
    annotation_set: TraceAnnotationSet | None,
    annotation_path: Path | None,
) -> list[str]:
    """Render source fixture paths and annotation metadata."""

    rows: list[tuple[str, Any]] = []
    if trace_path is not None:
        rows.append(("trace_path", _display_path(trace_path)))
    if annotation_path is not None:
        rows.append(("annotation_path", _display_path(annotation_path)))
    if annotation_set is not None:
        rows.extend(
            [
                ("annotation_set_id", annotation_set.annotation_set_id),
                ("annotation_source_issue", annotation_set.provenance.source_issue),
                ("annotation_evidence_boundary", annotation_set.provenance.evidence_boundary),
                ("annotation_count", len(annotation_set.annotations)),
            ]
        )
    return [
        "## Source Fixtures",
        "",
        *_markdown_table(("field", "value"), rows),
        "",
    ]


def _event_summary(frames: Iterable[SimulationTraceFrame]) -> list[str]:
    """Render event and selected-action counts."""

    event_counts = Counter(_event_name(frame) for frame in frames)
    rows = [(event, count) for event, count in sorted(event_counts.items())]
    return [
        "## Events",
        "",
        *_markdown_table(("event", "frames"), rows),
        "",
    ]


def _planner_key_summary(frames: Iterable[SimulationTraceFrame]) -> list[str]:
    """Render planner fields observed across frames."""

    key_counts: Counter[str] = Counter()
    for frame in frames:
        key_counts.update(str(key) for key in frame.planner)

    rows = [(key, count) for key, count in sorted(key_counts.items())]
    return [
        "## Planner State",
        "",
        *_markdown_table(("planner_field", "frames"), rows),
        "",
    ]


def _annotation_summary(frames: Iterable[SimulationTraceFrame]) -> list[str]:
    """Render annotation-like planner fields without inventing new trace data."""

    rows: list[tuple[str, str, str]] = []
    for frame in frames:
        for key, value in frame.planner.items():
            if str(key) not in ANNOTATION_KEYS:
                continue
            for name, formatted in _flatten_annotation_value(str(key), value):
                rows.append((str(frame.step), name, formatted))

    if not rows:
        rows = [("-", "none", "No annotation-like planner fields were present.")]

    return [
        "## Notable Annotations",
        "",
        *_markdown_table(("step", "annotation", "value"), rows),
        "",
    ]


def _annotation_set_summary(annotation_set: TraceAnnotationSet) -> list[str]:
    """Render external qualitative annotation anchors and evidence boundaries."""

    rows = [_annotation_row(annotation) for annotation in annotation_set.annotations]
    return [
        "## Annotation Review",
        "",
        (
            "Annotation review rows preserve their evidence type. `observed` rows cite trace "
            "contents, `hypothesis` rows are candidate interpretations, and `commentary` rows "
            "are reviewer notes. These qualitative rows are diagnostic-only and not benchmark "
            "evidence."
        ),
        "",
        *_markdown_table(
            (
                "annotation_id",
                "evidence_type",
                "category",
                "frame_range",
                "event_ids",
                "entities",
                "summary",
                "details",
            ),
            rows,
        ),
        "",
    ]


def _annotation_row(annotation: TraceAnnotation) -> tuple[str, str, str, str, str, str, str, str]:
    """Format one external annotation as a Markdown-table row."""

    anchor = annotation.anchor
    return (
        annotation.annotation_id,
        annotation.evidence_type,
        annotation.category,
        f"{anchor.frame_start}-{anchor.frame_end}",
        ", ".join(anchor.event_ids) if anchor.event_ids else "none",
        ", ".join(f"{entity.type}:{entity.id}" for entity in anchor.entities)
        if anchor.entities
        else "none",
        annotation.summary,
        annotation.details or "",
    )


def _frame_table(frames: Iterable[SimulationTraceFrame]) -> list[str]:
    """Render a compact per-frame inspection table."""

    rows = [
        (
            frame.step,
            f"{frame.time_s:.3f}",
            _event_name(frame),
            _selected_action(frame),
            _robot_state(frame),
            _pedestrian_state(frame),
            _planner_state(frame),
        )
        for frame in frames
    ]
    return [
        "## Frames",
        "",
        *_markdown_table(
            (
                "step",
                "time_s",
                "event",
                "selected_action",
                "robot",
                "pedestrians",
                "planner_state",
            ),
            rows,
        ),
        "",
    ]


def _event_name(frame: SimulationTraceFrame) -> str:
    """Return the planner event label for a frame."""

    event = frame.planner.get("event", "unknown")
    return str(event) if event else "unknown"


def _selected_action(frame: SimulationTraceFrame) -> str:
    """Format selected linear/angular velocity for one frame."""

    action = frame.planner.get("selected_action")
    if not isinstance(action, Mapping):
        return "missing"
    linear = action.get("linear_velocity")
    angular = action.get("angular_velocity")
    if isinstance(linear, int | float) and isinstance(angular, int | float):
        return f"{float(linear):.3f}, {float(angular):.3f}"
    return _format_value(action)


def _robot_state(frame: SimulationTraceFrame) -> str:
    """Format robot pose and velocity for one frame."""

    position = _format_vector2(frame.robot.get("position"))
    velocity = _format_vector2(frame.robot.get("velocity"))
    heading = frame.robot.get("heading")
    heading_text = f"{float(heading):.3f}" if isinstance(heading, int | float) else "missing"
    return f"pos=({position}); heading={heading_text}; vel=({velocity})"


def _pedestrian_state(frame: SimulationTraceFrame) -> str:
    """Format pedestrian positions and velocities for one frame."""

    if not frame.pedestrians:
        return "none"

    items: list[str] = []
    for pedestrian in frame.pedestrians:
        pedestrian_id = str(pedestrian.get("id", "unknown"))
        position = _format_vector2(pedestrian.get("position"))
        velocity = _format_vector2(pedestrian.get("velocity"))
        items.append(f"{pedestrian_id} @ ({position}) vel=({velocity})")
    return "; ".join(items)


def _planner_state(frame: SimulationTraceFrame) -> str:
    """Format non-action planner fields for one frame."""

    skipped = {"selected_action"}
    parts = [
        f"{key}={_format_value(value)}"
        for key, value in sorted(frame.planner.items())
        if key not in skipped
    ]
    return "; ".join(parts) if parts else "none"


def _flatten_annotation_value(key: str, value: Any) -> list[tuple[str, str]]:
    """Flatten annotation-like values into display rows."""

    if isinstance(value, Mapping):
        return [
            (f"{key}: {subkey}", _format_value(subvalue))
            for subkey, subvalue in sorted(value.items())
        ]
    if isinstance(value, list):
        return [(key, _format_value(item)) for item in value]
    return [(key, _format_value(value))]


def _format_vector2(value: Any) -> str:
    """Format a two-element vector for compact Markdown tables."""

    if (
        isinstance(value, list | tuple)
        and len(value) >= 2
        and isinstance(value[0], int | float)
        and isinstance(value[1], int | float)
    ):
        return f"{float(value[0]):.3f}, {float(value[1]):.3f}"
    return "missing"


def _format_value(value: Any) -> str:
    """Format JSON-like values for compact Markdown cells."""

    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {_format_value(item)}" for key, item in sorted(value.items()))
    if isinstance(value, list | tuple):
        return ", ".join(_format_value(item) for item in value)
    return json.dumps(value, sort_keys=True)


def _validate_tracked_fixture_path(path: Path, *, label: str) -> None:
    """Reject output-only paths before report rendering."""

    relative_path = _repo_relative_path(path)
    if any(part in {"output", "results"} for part in relative_path.parts):
        raise ValueError(f"{label} path must be a tracked fixture, not generated output: {path}")


def _validate_annotation_matches_trace(
    *,
    trace: SimulationTraceExport,
    trace_path: Path,
    annotation_set: TraceAnnotationSet,
) -> None:
    """Fail closed when the external annotation set points at another trace."""

    if annotation_set.timeline.trace_id != trace.trace_id:
        raise TraceAnnotationSetValidationError(
            [
                "/timeline/trace_id: "
                f"expected supplied trace_id {trace.trace_id!r}, "
                f"got {annotation_set.timeline.trace_id!r}"
            ],
            source=annotation_set.annotation_set_id,
        )
    expected_name = Path(annotation_set.timeline.path).name
    if expected_name != trace_path.name:
        raise TraceAnnotationSetValidationError(
            [
                "/timeline/path: "
                f"annotation references {annotation_set.timeline.path!r}, "
                f"but supplied trace path is {str(trace_path)!r}"
            ],
            source=annotation_set.annotation_set_id,
        )


def _display_path(path: Path) -> str:
    """Return a stable repo-friendly display path when possible."""

    return str(_repo_relative_path(path))


def _repo_relative_path(path: Path) -> Path:
    """Return a path relative to the repository root when possible."""

    repo_root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repo_root)
    except ValueError:
        return path


def _markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> list[str]:
    """Render a simple GitHub-flavored Markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_escape_markdown_cell(_format_value(cell)) for cell in row) + " |"
        )
    return lines


def _escape_markdown_cell(value: str) -> str:
    """Escape table separators and line breaks in a Markdown table cell."""

    return value.replace("|", "\\|").replace("\n", " ")


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for trace-report rendering."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path, required=True, help="simulation_trace_export.v1 JSON."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional trace_annotation_set.v1 JSON to cite in the report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report.md"),
        help="Markdown output path. Defaults to ./report.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render a trace report from the command line."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output = write_trace_report(
            trace_path=args.trace,
            annotation_path=args.annotations,
            output=args.output,
        )
    except (
        OSError,
        ValueError,
        SimulationTraceExportValidationError,
        TraceAnnotationSetValidationError,
    ) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"wrote trace report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
