"""Build compact signalized-crossing failure-case pack from trace/metric inputs or fixtures.

This script implements the issue #2754 requirement to extract failure cases from signalized
crossing episodes, capturing trace ranges, signal phases, stop lines, states, and claim wording.
If no failures are present, it outputs a negative-control pack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    load_simulation_trace_export,
)
from robot_sf.benchmark.failure_extractor import is_failure

# Canonical allowed claim wording for signalized behavior (Issue #2760 / Dissertation Ledger)
ALLOWED_CLAIM_WORDING = (
    "The repository can now produce simulator-backed signalized-crossing rows that separate "
    "planner-observable denominator evidence from unavailable/proxy exclusions; this proves "
    "denominator plumbing, not traffic-light realism or crossing-legality compliance."
)


def _sanitize(val: Any) -> Any:
    """Recursively sanitize a value to make it JSON-serializable.

    Converts numpy arrays to lists, numpy generics to native types, and NaN/inf to None.
    """
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, list | tuple | set):
        return [_sanitize(v) for v in val]
    if isinstance(val, np.ndarray):
        return _sanitize(val.tolist())
    if isinstance(val, float) and not np.isfinite(val):
        return None
    if isinstance(val, np.generic):
        return _sanitize(val.item())
    return val


def _expand_timeline(timeline: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    """Expand phase durations into per-step phase records."""
    if dt <= 0.0 or not np.isfinite(dt):
        return []
    expanded: list[dict[str, Any]] = []
    for phase_info in timeline:
        duration = float(phase_info.get("duration", 0.0))
        # Use ceil to ensure at least 1 step if duration > 0
        steps = max(1, int(np.ceil(duration / dt))) if duration > 0.0 else 0
        expanded.extend([phase_info] * steps)
    return expanded


def find_failure_step(trace: SimulationTraceExport) -> int:
    """Find the step at which a collision or near-miss occurs in a trace.

    Defaults to the step of closest approach if no event is explicitly flagged.
    """
    for frame in trace.frames:
        event = str(frame.planner.get("event", "")).lower()
        if "collision" in event or "fail" in event:
            return frame.step

    # Fallback: step with minimum clearance between robot and any pedestrian
    min_dist = float("inf")
    closest_step = 0
    for frame in trace.frames:
        r_pos = np.array(frame.robot.get("position") or [0.0, 0.0], dtype=float)
        for ped in frame.pedestrians:
            p_pos = np.array(ped.get("position") or [0.0, 0.0], dtype=float)
            dist = float(np.linalg.norm(r_pos - p_pos))
            if dist < min_dist:
                min_dist = dist
                closest_step = frame.step
    return closest_step


def get_signal_phase_at_step(
    trace: SimulationTraceExport,
    step: int,
    timeline: list[dict[str, Any]],
    dt: float,
) -> str:
    """Get the active signal phase (red/green) at a specific step in the trace."""
    # 1. Check if pedestrian signal_state is stored in the frame
    for frame in trace.frames:
        if frame.step == step:
            for ped in frame.pedestrians:
                sig = ped.get("signal_state")
                if sig and sig.get("available"):
                    return str(sig.get("label", "unknown"))

    # 2. Fallback to timeline
    expanded = _expand_timeline(timeline, dt)
    if expanded and 0 <= step < len(expanded):
        return str(expanded[step].get("state", "unknown"))
    return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read lines of JSONL into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_failure_pack(
    traces: list[SimulationTraceExport],
    records: list[dict[str, Any]],
    artifact_status: str,
    allowed_claim_wording: str,
    collision_threshold: float,
    comfort_threshold: float,
    near_miss_threshold: float,
) -> dict[str, Any]:
    """Assemble failure cases from paired trace exports and metrics records."""
    cases = []

    for trace in traces:
        # Match trace with episodes records by episode_id
        matched_rec = None
        for record in records:
            rec_id = str(record.get("episode_id", ""))
            trace_id = trace.source.episode_id
            if rec_id == trace_id or rec_id.startswith(trace_id) or trace_id.startswith(rec_id):
                matched_rec = record
                break

        if not matched_rec:
            continue

        metrics = matched_rec.get("metrics") or {}
        if not is_failure(
            matched_rec,
            collision_threshold=collision_threshold,
            comfort_threshold=comfort_threshold,
            near_miss_threshold=near_miss_threshold,
        ):
            continue

        # Extract signal metadata
        scenario_params = matched_rec.get("scenario_params") or {}
        scenario_metadata = scenario_params.get("metadata") or {}
        signal_state = scenario_metadata.get("signal_state") or {}
        if not signal_state:
            # Fallback to episode_metadata if present
            signal_state = (matched_rec.get("episode_metadata") or {}).get("signal_state") or {}

        timeline = signal_state.get("timeline") or []
        stop_line = signal_state.get("stop_line")

        dt = 0.1
        if len(trace.frames) > 1:
            dt = float(trace.frames[1].time_s - trace.frames[0].time_s)

        failure_step = find_failure_step(trace)
        phase = get_signal_phase_at_step(trace, failure_step, timeline, dt)

        # Get robot and pedestrian state at failure step
        failure_frame = next(
            (f for f in trace.frames if f.step == failure_step),
            trace.frames[-1] if trace.frames else None,
        )

        robot_state = failure_frame.robot if failure_frame else {}
        ped_state = failure_frame.pedestrians if failure_frame else []

        # Determine denominator and eligibility
        denominator = int(metrics.get("signal_metrics_denominator", 0) or 0)
        evidence = metrics.get("signal_metrics_evidence") or {}
        state = evidence.get("state", "unavailable")
        exclusion = evidence.get("exclusion_reason", "")

        planner_observable = state == "planner_observable" and not exclusion
        benchmark_evidence = planner_observable and denominator > 0
        eligible = planner_observable and benchmark_evidence

        denominator_status = "eligible" if eligible else "excluded"
        diagnostic_only = not eligible
        figure_eligible = eligible

        # Proxy/unavailable signal rows remain diagnostic-only and figure-ineligible
        if diagnostic_only:
            claim_wording = (
                "Unavailable or proxy signal rows are diagnostic-only and ineligible "
                "for compliance claims."
            )
        else:
            claim_wording = allowed_claim_wording

        cases.append(
            {
                "episode_id": trace.source.episode_id,
                "scenario_id": trace.source.scenario_id,
                "trace_row_range": (
                    [trace.frames[0].step, trace.frames[-1].step] if trace.frames else None
                ),
                "signal_phase": phase,
                "stop_line_geometry": stop_line,
                "robot_state": robot_state,
                "pedestrian_state": ped_state,
                "metric_row": metrics,
                "denominator_status": denominator_status,
                "stale_current_status": artifact_status,
                "allowed_claim_wording": claim_wording,
                "diagnostic_only": diagnostic_only,
                "figure_eligible": figure_eligible,
            }
        )

    if not cases:
        return {
            "schema_version": "signalized_crossing_failure_pack.v1",
            "negative_control": True,
            "status": "insufficiently_adversarial",
            "message": "The fixture is insufficiently adversarial; no real failures were detected.",
            "cases": [],
        }

    return {
        "schema_version": "signalized_crossing_failure_pack.v1",
        "negative_control": False,
        "status": "failures_present",
        "cases": cases,
    }


def load_trace_with_fallback(path: Path) -> SimulationTraceExport:
    """Load a simulation trace export, falling back to a raw JSON load if validation fails."""
    try:
        return load_simulation_trace_export(path)
    except Exception:
        # Fall back to raw JSON load and bypass strict schema checks
        from robot_sf.analysis_workbench.simulation_trace_export import (
            SimulationTraceFrame,
            SimulationTraceSource,
        )

        raw = json.loads(path.read_text(encoding="utf-8"))
        src = raw["source"]
        source = SimulationTraceSource(
            scenario_id=src["scenario_id"],
            seed=int(src["seed"]),
            planner_id=src["planner_id"],
            episode_id=src["episode_id"],
            generated_by=src["generated_by"],
        )
        frames = []
        for f in raw["frames"]:
            frames.append(
                SimulationTraceFrame(
                    step=int(f["step"]),
                    time_s=float(f["time_s"]),
                    robot=f["robot"],
                    pedestrians=f["pedestrians"],
                    planner=f["planner"],
                )
            )
        return SimulationTraceExport(
            schema_version=raw.get("schema_version", "simulation_trace_export.v1"),
            trace_id=raw["trace_id"],
            source=source,
            evidence_boundary=raw["evidence_boundary"],
            coordinate_frame=raw["coordinate_frame"],
            units=raw["units"],
            frames=frames,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for signalized-crossing failure case pack building."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        help="One or more simulation trace export files (JSON).",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        nargs="+",
        help="One or more episodes metrics files (JSONL).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("result.json"),
        help="Output path for the failure pack JSON.",
    )
    parser.add_argument(
        "--artifact-status",
        choices=["current", "stale", "unknown"],
        default="current",
        help="Status label for the generated/inspected artifacts.",
    )
    parser.add_argument(
        "--allowed-claim-wording",
        default=ALLOWED_CLAIM_WORDING,
        help="Allowed claim wording for eligible rows.",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=1.0,
        help="Minimum number of collisions to flag a failure.",
    )
    parser.add_argument(
        "--comfort-threshold",
        type=float,
        default=0.2,
        help="Minimum comfort exposure to flag a failure.",
    )
    parser.add_argument(
        "--near-miss-threshold",
        type=float,
        default=0.0,
        help="Minimum near-misses to flag a failure.",
    )

    args = parser.parse_args(argv)

    loaded_traces = []
    if args.traces:
        for p in args.traces:
            if p.is_file():
                loaded_traces.append(load_trace_with_fallback(p))

    loaded_records = []
    if args.episodes_jsonl:
        for p in args.episodes_jsonl:
            if p.is_file():
                loaded_records.extend(_read_jsonl(p))

    pack = build_failure_pack(
        traces=loaded_traces,
        records=loaded_records,
        artifact_status=args.artifact_status,
        allowed_claim_wording=args.allowed_claim_wording,
        collision_threshold=args.collision_threshold,
        comfort_threshold=args.comfort_threshold,
        near_miss_threshold=args.near_miss_threshold,
    )

    sanitized_pack = _sanitize(pack)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(sanitized_pack, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote signalized crossing failure pack to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
