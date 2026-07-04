"""Agent discussion workflow for simulation trace sequences (issue #1646 child).

This script takes a simulation trace export and an optional annotation set,
selects a frame/event sequence, and generates a structured discussion/debate transcript
between an ObserverAgent and a TheoristAgent, strictly separating empirical observations
(facts) from theoretical hypotheses (interpretations).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

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


def _repo_relative_path(path: Path) -> Path:
    """Return a path relative to the repository root when possible."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(repo_root)
    except ValueError:
        return path


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
                f"expected referenced trace_id {trace.trace_id!r}, "
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


def _render_annotation_roster(overlapping_annotations: list[TraceAnnotation]) -> list[str]:
    """Render annotation roster table in Markdown."""
    lines = [
        "",
        "## Annotation Roster",
        "",
    ]
    if overlapping_annotations:
        lines.append("| Annotation ID | Type | Category | Frame Range | Summary |")
        lines.append("| --- | --- | --- | --- | --- |")
        for ann in overlapping_annotations:
            lines.append(
                f"| `{ann.annotation_id}` | `{ann.evidence_type}` | `{ann.category}` | "
                f"{ann.anchor.frame_start}-{ann.anchor.frame_end} | {ann.summary} |"
            )
    else:
        lines.append("*No overlapping annotations found in this frame range.*")
    return lines


def _format_frame_discussion_block(
    frame: SimulationTraceFrame,
    overlapping_annotations: list[TraceAnnotation],
) -> list[str]:
    """Format ObserverAgent and TheoristAgent dialogue for a single frame."""
    lines = [
        f"### Frame/Step {frame.step} (time_s={frame.time_s:.3f})",
        "",
    ]
    # 1. ObserverAgent (Strict Facts)
    robot_pos = frame.robot.get("position", "missing")
    robot_vel = frame.robot.get("velocity", "missing")
    action = frame.planner.get("selected_action")
    if isinstance(action, Mapping):
        lin = action.get("linear_velocity")
        ang = action.get("angular_velocity")
        action_str = (
            f"linear={lin:.3f}, angular={ang:.3f}"
            if lin is not None and ang is not None
            else str(action)
        )
    else:
        action_str = str(action)

    ped_strings = []
    for ped in frame.pedestrians:
        ped_id = ped.get("id", "unknown")
        ped_pos = ped.get("position", "missing")
        ped_strings.append(f"Pedestrian {ped_id} @ {ped_pos}")
    peds_str = "; ".join(ped_strings) if ped_strings else "None"

    event = frame.planner.get("event")
    event_str = f", Event: `{event}`" if event and event != "unknown" else ""

    lines.append(
        "**ObserverAgent:** I record the following empirical facts from the trace: "
        f"Robot is at `{robot_pos}` with velocity `{robot_vel}`. "
        f"Planner selected action: `{action_str}`{event_str}. "
        f"Pedestrian states: {peds_str}."
    )

    # Include observed annotations
    for ann in overlapping_annotations:
        if (
            ann.evidence_type == "observed"
            and ann.anchor.frame_start <= frame.step <= ann.anchor.frame_end
        ):
            lines.append(
                f"**ObserverAgent:** Ground-truth annotation `{ann.annotation_id}` "
                f'({ann.category}) notes: "{ann.summary}".'
            )

    lines.append("")

    # 2. TheoristAgent (Hypothesis & Analysis)
    collision = frame.planner.get("collision")
    collision_str = f"collision={collision}" if collision is not None else "collision=none"
    reward = frame.planner.get("reward")
    reward_str = f"reward={reward}" if reward is not None else "reward=none"

    lines.append(
        "**TheoristAgent:** I hypothesize the following interpretation: "
        f"Under the current planner state ({collision_str}; {reward_str}), "
        f"the action choice indicates intent to steer relative to the observed pedestrian trajectories."
    )

    # Include hypothesis/commentary annotations
    for ann in overlapping_annotations:
        if (
            ann.evidence_type in ("hypothesis", "commentary")
            and ann.anchor.frame_start <= frame.step <= ann.anchor.frame_end
        ):
            role = "TheoristAgent" if ann.evidence_type == "hypothesis" else "ReviewerNote"
            lines.append(
                f"**{role}:** Qualitative interpretation `{ann.annotation_id}` "
                f'({ann.category}) proposes: "{ann.summary}".'
            )

    lines.append("")
    return lines


def generate_agent_discussion(
    trace: SimulationTraceExport,
    *,
    trace_path: Path,
    annotation_set: TraceAnnotationSet | None = None,
    annotation_path: Path | None = None,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> str:
    """Generate a structured Markdown discussion report separating observation from hypothesis.

    Args:
        trace: The loaded trace export.
        trace_path: Path to the trace export.
        annotation_set: The optional loaded annotation set.
        annotation_path: Path to the annotation set.
        frame_start: The start frame step to filter.
        frame_end: The end frame step to filter.

    Returns:
        Markdown string representing the debate/discussion report.
    """
    frame_steps = {f.step for f in trace.frames}
    if not frame_steps:
        raise ValueError("Trace contains no frames.")

    min_step = min(frame_steps)
    max_step = max(frame_steps)

    # Set default frame_end if not provided
    actual_end = frame_end if frame_end is not None else max_step

    if frame_start < min_step or actual_end > max_step:
        raise ValueError(
            f"Requested frame range {frame_start}-{actual_end} is outside "
            f"trace step range {min_step}-{max_step}."
        )
    if frame_start > actual_end:
        raise ValueError(f"frame_start ({frame_start}) must be <= frame_end ({actual_end}).")

    selected_frames = [f for f in trace.frames if frame_start <= f.step <= actual_end]

    # Find overlapping annotations
    overlapping_annotations: list[TraceAnnotation] = []
    if annotation_set is not None:
        for annotation in annotation_set.annotations:
            anchor = annotation.anchor
            if not (anchor.frame_end < frame_start or anchor.frame_start > actual_end):
                overlapping_annotations.append(annotation)

    # Begin Markdown generation
    lines = [
        "# Agent Discussion Report",
        "",
        "> [!IMPORTANT]",
        "> **Qualitative Analysis Boundary:** This agent discussion workflow is diagnostic-only ",
        "> and does not establish benchmark evidence. Observations and hypotheses are kept strictly separate.",
        "",
        "## Trace Context",
        "",
        f"- **Trace ID:** `{trace.trace_id}`",
        f"- **Scenario ID:** `{trace.source.scenario_id}`",
        f"- **Planner ID:** `{trace.source.planner_id}`",
        f"- **Frame Step Range:** {frame_start} to {actual_end}",
        f"- **Source Trace Path:** `{_repo_relative_path(trace_path)}`",
    ]

    if annotation_path is not None:
        lines.append(f"- **Annotation Path:** `{_repo_relative_path(annotation_path)}`")

    lines.extend(_render_annotation_roster(overlapping_annotations))

    lines.extend(
        [
            "",
            "## Sequence Debate Transcript",
            "",
        ]
    )

    for frame in selected_frames:
        lines.extend(_format_frame_discussion_block(frame, overlapping_annotations))

    # Summary section separating observation and hypothesis
    lines.extend(
        [
            "## Discussion Summary",
            "",
            "### Empirical Takeaways (ObserverAgent)",
            "",
            f"- Total frames analyzed: {len(selected_frames)}",
            f"- Steps covered: {frame_start} to {actual_end}",
        ]
    )

    events_observed = sorted(
        {
            f.planner.get("event")
            for f in selected_frames
            if f.planner.get("event") and f.planner.get("event") != "unknown"
        }
    )
    if events_observed:
        lines.append(f"- Events registered: {', '.join(f'`{e}`' for e in events_observed)}")

    lines.extend(
        [
            "",
            "### Interpretive Takeaways (TheoristAgent)",
            "",
            "- The planner velocity commands are consistent with local avoidance logic.",
            "- These qualitative annotations suggest candidate explanations of failure modes, ",
            "  which must be validated quantitatively via statistical counters.",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def write_agent_discussion(
    *,
    trace_path: Path,
    output: Path,
    annotation_path: Path | None = None,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> Path:
    """Load trace/annotations and write the Markdown agent discussion report."""
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

    markdown = generate_agent_discussion(
        trace,
        trace_path=trace_path,
        annotation_set=annotation_set,
        annotation_path=annotation_path,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for trace-report rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path, required=True, help="simulation_trace_export.v1 JSON."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional trace_annotation_set.v1 JSON to include in debate.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Start frame step for sequence window.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="End frame step for sequence window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("discussion_report.md"),
        help="Markdown output path. Defaults to ./discussion_report.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render a trace report from the command line."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output = write_agent_discussion(
            trace_path=args.trace,
            annotation_path=args.annotations,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
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

    print(f"wrote agent discussion report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
