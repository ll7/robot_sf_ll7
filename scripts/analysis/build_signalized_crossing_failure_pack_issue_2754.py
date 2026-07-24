"""Build compact signalized-crossing failure-case pack from trace/metric inputs or fixtures.

This script implements the issue #2754 requirement to extract failure cases from signalized
crossing episodes, capturing trace ranges, signal phases, stop lines, states, and claim wording.
If no failures are present, it outputs a negative-control pack.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
)
from robot_sf.benchmark.failure_extractor import is_failure

# Canonical allowed claim wording for signalized behavior (Issue #2760 / Dissertation Ledger)
ALLOWED_CLAIM_WORDING = (
    "The repository can now produce simulator-backed signalized-crossing rows that separate "
    "planner-observable denominator evidence from unavailable/proxy exclusions; this proves "
    "denominator plumbing, not traffic-light realism or crossing-legality compliance."
)

SOURCE_KIND_CHOICES = ("live_execution", "durable_replay", "fixture", "synthetic", "unknown")
EVIDENCE_TIER_CHOICES = (
    "benchmark",
    "nominal_benchmark",
    "paper_grade",
    "smoke",
    "diagnostic",
    "analysis_only",
    "proposal",
    "unknown",
)
EXECUTION_MODE_CHOICES = (
    "native",
    "durable_replay",
    "fixture",
    "synthetic",
    "fallback",
    "degraded",
    "proxy",
    "unavailable",
    "not_available",
    "unknown",
)
CLAIM_MATRIX_STATUS_CHOICES = (
    "allowed",
    "claimable",
    "diagnostic_only",
    "not_claimable",
    "blocked",
    "unknown",
)

_DURABLE_SOURCE_KINDS = {"live_execution", "durable_replay"}
_FIGURE_EVIDENCE_TIERS = {"benchmark", "nominal_benchmark", "paper_grade"}
_FIGURE_EXECUTION_MODES = {"native", "durable_replay"}
_ALLOWED_CLAIM_STATUSES = {"allowed", "claimable"}
_METRIC_INPUT_PATH_KEY = "_failure_pack_metric_input_path"
_METRIC_INPUT_LINE_KEY = "_failure_pack_metric_input_line"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FailurePackProvenance:
    """Provenance fields that control claim and figure eligibility."""

    trace_source_kind: str
    metric_source_kind: str
    execution_performed: bool
    evidence_tier: str
    artifact_status: str
    execution_mode: str
    fallback_or_degraded: bool
    claim_matrix_status: str


@dataclass(frozen=True)
class FailurePredicateThresholds:
    """Thresholds for generic and signal-specific failure-pack predicates."""

    collision: float = 1.0
    comfort: float = 0.2
    near_miss: float = 0.0
    signal_red_phase_violation: float = 1.0
    signal_stop_line_crossing: float = 1.0


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


def _portable_input_path(path: Path, *, repo_root: Path | None = None) -> str:
    """Return a non-absolute path suitable for tracked provenance output."""
    root = (repo_root or _REPO_ROOT).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        if path.is_absolute():
            return path.name
        return path.as_posix()


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


def signal_specific_failure_predicates(
    rec: dict[str, Any],
    *,
    red_phase_violation_threshold: float = 1.0,
    stop_line_crossing_threshold: float = 1.0,
) -> list[str]:
    """Return signal-specific metric predicates that intentionally define a failure-pack case."""
    metrics = rec.get("metrics") or {}

    def _metric(name: str) -> float:
        try:
            return float(metrics.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    predicates: list[str] = []
    if _metric("signal_red_phase_violations") >= float(red_phase_violation_threshold):
        predicates.append("signal_red_phase_violations")
    if _metric("signal_stop_line_crossings_under_red") >= float(stop_line_crossing_threshold):
        predicates.append("signal_stop_line_crossings_under_red")
    return predicates


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
    input_path = _portable_input_path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"Invalid JSONL in {input_path} at line {line_number}: {exc.msg}"
                    raise ValueError(msg) from exc
                if not isinstance(record, dict):
                    msg = f"Invalid JSONL in {input_path} at line {line_number}: expected object"
                    raise ValueError(msg)
                record[_METRIC_INPUT_PATH_KEY] = input_path
                record[_METRIC_INPUT_LINE_KEY] = line_number
                records.append(record)
    return records


def _run_provenance_ineligibility_reasons(provenance: FailurePackProvenance) -> list[str]:
    """Return ineligibility reasons based strictly on run-level provenance."""
    checks = [
        (
            provenance.trace_source_kind not in _DURABLE_SOURCE_KINDS,
            f"trace_source_kind={provenance.trace_source_kind}",
        ),
        (
            provenance.metric_source_kind not in _DURABLE_SOURCE_KINDS,
            f"metric_source_kind={provenance.metric_source_kind}",
        ),
        (not provenance.execution_performed, "execution_performed=false"),
        (
            provenance.evidence_tier not in _FIGURE_EVIDENCE_TIERS,
            f"evidence_tier={provenance.evidence_tier}",
        ),
        (provenance.artifact_status != "current", f"artifact_status={provenance.artifact_status}"),
        (
            provenance.execution_mode not in _FIGURE_EXECUTION_MODES,
            f"execution_mode={provenance.execution_mode}",
        ),
        (provenance.fallback_or_degraded, "fallback_or_degraded=true"),
        (
            provenance.claim_matrix_status not in _ALLOWED_CLAIM_STATUSES,
            f"claim_matrix_status={provenance.claim_matrix_status}",
        ),
    ]
    return [reason for failed, reason in checks if failed]


def _case_ineligibility_reasons(
    *,
    denominator: int,
    evidence_state: str,
    evidence_exclusion: str,
) -> list[str]:
    """Return ineligibility reasons based strictly on case-level metrics."""
    checks = [
        (denominator <= 0, "signal_metrics_denominator<=0"),
        (evidence_state != "planner_observable", f"signal_metrics_evidence.state={evidence_state}"),
        (
            bool(evidence_exclusion),
            f"signal_metrics_evidence.exclusion_reason={evidence_exclusion}",
        ),
    ]
    return [reason for failed, reason in checks if failed]


def _figure_ineligibility_reasons(
    *,
    denominator: int,
    evidence_state: str,
    evidence_exclusion: str,
    provenance: FailurePackProvenance,
) -> list[str]:
    """Return fail-closed reasons that make a failure-pack row diagnostic-only."""
    case_reasons = _case_ineligibility_reasons(
        denominator=denominator,
        evidence_state=evidence_state,
        evidence_exclusion=evidence_exclusion,
    )
    run_reasons = _run_provenance_ineligibility_reasons(provenance)
    return case_reasons + run_reasons


def _diagnostic_claim_wording(reasons: list[str]) -> str:
    """Build conservative claim wording for rows that fail figure eligibility."""
    reason_text = ", ".join(reasons) if reasons else "figure eligibility was not proven"
    return (
        "Figure-ineligible diagnostic-only row; do not use for benchmark, dissertation figure, "
        f"or paper-facing claims. Reasons: {reason_text}."
    )


def build_failure_pack(
    traces: list[SimulationTraceExport],
    records: list[dict[str, Any]],
    allowed_claim_wording: str,
    thresholds: FailurePredicateThresholds,
    provenance: FailurePackProvenance,
    trace_input_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble failure cases from paired trace exports and metrics records."""
    cases = []
    trace_input_paths = trace_input_paths or {}

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
        signal_failure_predicates = signal_specific_failure_predicates(
            matched_rec,
            red_phase_violation_threshold=thresholds.signal_red_phase_violation,
            stop_line_crossing_threshold=thresholds.signal_stop_line_crossing,
        )
        if (
            not is_failure(
                matched_rec,
                collision_threshold=thresholds.collision,
                comfort_threshold=thresholds.comfort,
                near_miss_threshold=thresholds.near_miss,
            )
            and not signal_failure_predicates
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
        denominator_eligible = planner_observable and denominator > 0
        ineligibility_reasons = _figure_ineligibility_reasons(
            denominator=denominator,
            evidence_state=str(state),
            evidence_exclusion=str(exclusion),
            provenance=provenance,
        )
        figure_eligible = not ineligibility_reasons

        denominator_status = "eligible" if denominator_eligible else "excluded"
        diagnostic_only = not figure_eligible

        # Provenance, fallback, stale, proxy, and unavailable rows fail closed.
        if diagnostic_only:
            claim_wording = _diagnostic_claim_wording(ineligibility_reasons)
        else:
            claim_wording = allowed_claim_wording

        cases.append(
            {
                "episode_id": trace.source.episode_id,
                "scenario_id": trace.source.scenario_id,
                "trace_path": trace_input_paths.get(trace.source.episode_id),
                "episodes_jsonl_path": matched_rec.get(_METRIC_INPUT_PATH_KEY),
                "metric_row_line_number": matched_rec.get(_METRIC_INPUT_LINE_KEY),
                "metric_row_claim_boundary": matched_rec.get("claim_boundary"),
                "trace_row_range": (
                    [trace.frames[0].step, trace.frames[-1].step] if trace.frames else None
                ),
                "signal_phase": phase,
                "stop_line_geometry": stop_line,
                "robot_state": robot_state,
                "pedestrian_state": ped_state,
                "metric_row": metrics,
                "signal_failure_predicates": signal_failure_predicates,
                "denominator_status": denominator_status,
                "stale_current_status": provenance.artifact_status,
                "artifact_status": provenance.artifact_status,
                "trace_source_kind": provenance.trace_source_kind,
                "metric_source_kind": provenance.metric_source_kind,
                "execution_performed": provenance.execution_performed,
                "evidence_tier": provenance.evidence_tier,
                "execution_mode": provenance.execution_mode,
                "fallback_or_degraded": provenance.fallback_or_degraded,
                "claim_matrix_status": provenance.claim_matrix_status,
                "figure_ineligibility_reasons": ineligibility_reasons,
                "allowed_claim_wording": claim_wording,
                "diagnostic_only": diagnostic_only,
                "figure_eligible": figure_eligible,
            }
        )

    run_reasons = _run_provenance_ineligibility_reasons(provenance)
    run_level_eligible = len(run_reasons) == 0

    if not cases:
        top_figure_eligible = run_level_eligible
        top_diagnostic_only = not top_figure_eligible
        return {
            "schema_version": "signalized_crossing_failure_pack.v1",
            "negative_control": True,
            "status": "insufficiently_adversarial",
            "message": "The fixture is insufficiently adversarial; no real failures were detected.",
            "diagnostic_only": top_diagnostic_only,
            "figure_eligible": top_figure_eligible,
            "cases": [],
        }

    top_figure_eligible = run_level_eligible and all(case["figure_eligible"] for case in cases)
    top_diagnostic_only = not top_figure_eligible

    return {
        "schema_version": "signalized_crossing_failure_pack.v1",
        "negative_control": False,
        "status": "failures_present",
        "diagnostic_only": top_diagnostic_only,
        "figure_eligible": top_figure_eligible,
        "cases": cases,
    }


def load_trace_with_fallback(path: Path) -> SimulationTraceExport:
    """Load a simulation trace export, falling back to a raw JSON load if validation fails."""
    try:
        return load_simulation_trace_export(path)
    except SimulationTraceExportValidationError:
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


def _find_eligibility_flags(data: Any) -> list[tuple[str, Any]]:
    """Recursively find all occurrences of figure_eligible or diagnostic_only in data."""
    found = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("figure_eligible", "diagnostic_only"):
                found.append((k, v))
            else:
                found.extend(_find_eligibility_flags(v))
    elif isinstance(data, list):
        for item in data:
            found.extend(_find_eligibility_flags(item))
    return found


def _scan_json_file(
    json_file: Path,
    readme_path: Path,
    matched_keyword: str,
    path_name: str,
    disagreements: list[dict[str, Any]],
) -> None:
    """Scan a single JSON file for eligibility disagreements."""
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception:
        # Malformed JSON or read error; skip
        return

    flags = _find_eligibility_flags(data)
    conflicts = []
    for k, v in flags:
        if k == "figure_eligible" and v is True:
            conflicts.append((k, v))
        elif k == "diagnostic_only" and v is False:
            conflicts.append((k, v))

    if conflicts:
        try:
            rel_readme = readme_path.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            rel_readme = readme_path.as_posix()
        try:
            rel_json = json_file.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            rel_json = json_file.as_posix()

        disagreements.append(
            {
                "dir": path_name,
                "readme": rel_readme,
                "json": rel_json,
                "matched_keyword": matched_keyword,
                "conflicts": conflicts,
            }
        )


def _scan_single_directory(
    path: Path,
    keywords_pattern: re.Pattern,
    disagreements: list[dict[str, Any]],
) -> None:
    """Scan a single directory for eligibility disagreements between README and JSON."""
    readme_path = path / "README.md"
    if not readme_path.is_file():
        return

    # Find all JSON files directly under the directory
    json_files = sorted(path.glob("*.json"))
    if not json_files:
        return

    try:
        readme_text = readme_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {readme_path}: {e}")
        return

    keyword_match = keywords_pattern.search(readme_text)
    if not keyword_match:
        return

    matched_keyword = keyword_match.group(0)

    for json_file in json_files:
        _scan_json_file(
            json_file=json_file,
            readme_path=readme_path,
            matched_keyword=matched_keyword,
            path_name=path.name,
            disagreements=disagreements,
        )


def run_evidence_scan(evidence_dir: Path | None = None) -> int:
    """Scan docs/context/evidence/ for prose vs machine-readable eligibility disagreements.

    Returns:
        0 if no disagreements are found, 1 otherwise.
    """
    if evidence_dir is None:
        evidence_dir = _REPO_ROOT / "docs/context/evidence"
    if not evidence_dir.is_dir():
        print(f"Evidence directory {evidence_dir} does not exist.")
        return 1

    ineligibility_keywords = [
        "smoke",
        "synthetic",
        "fixture",
        "stale",
        "unknown",
        "fallback",
        "degraded",
        "proxy",
        "unavailable",
    ]
    # Match whole words case-insensitively
    keywords_pattern = re.compile(r"\b(" + "|".join(ineligibility_keywords) + r")\b", re.IGNORECASE)

    disagreements = []

    for readme_path in sorted(evidence_dir.rglob("README.md")):
        _scan_single_directory(readme_path.parent, keywords_pattern, disagreements)

    if disagreements:
        print("PROSE VS MACHINE-READABLE ELIGIBILITY DISAGREEMENTS FOUND:")
        for diag in disagreements:
            print(f"\n- Directory: {diag['dir']}")
            print(f"  Prose: {diag['readme']} (matched '{diag['matched_keyword']}')")
            print(f"  JSON: {diag['json']}")
            for k, v in diag["conflicts"]:
                print(f"    Disagreement: {k} is set to {v}")
        return 1

    print(
        "No prose vs machine-readable eligibility disagreements found under docs/context/evidence/."
    )
    return 0


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
        "--trace-source-kind",
        choices=SOURCE_KIND_CHOICES,
        default="unknown",
        help="Provenance of the trace input.",
    )
    parser.add_argument(
        "--metric-source-kind",
        choices=SOURCE_KIND_CHOICES,
        default="unknown",
        help="Provenance of the metric row input.",
    )
    parser.add_argument(
        "--execution-performed",
        action="store_true",
        help="Set when this pack is backed by an actual planner/simulator execution.",
    )
    parser.add_argument(
        "--evidence-tier",
        choices=EVIDENCE_TIER_CHOICES,
        default="unknown",
        help="Evidence tier for this pack.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODE_CHOICES,
        default="unknown",
        help="Execution mode backing the pack.",
    )
    parser.add_argument(
        "--fallback-or-degraded",
        action="store_true",
        help="Mark the row as fallback or degraded; figure eligibility then fails closed.",
    )
    parser.add_argument(
        "--claim-matrix-status",
        choices=CLAIM_MATRIX_STATUS_CHOICES,
        default="unknown",
        help="Claim-matrix status for the requested claim boundary.",
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
    parser.add_argument(
        "--signal-red-phase-violation-threshold",
        type=float,
        default=1.0,
        help="Minimum signal red-phase violations flag signal-specific failure.",
    )
    parser.add_argument(
        "--signal-stop-line-crossing-threshold",
        type=float,
        default=1.0,
        help="Minimum stop-line crossings under red flag signal-specific failure.",
    )

    parser.add_argument(
        "--scan-evidence",
        action="store_true",
        help="Scan docs/context/evidence/ for prose vs machine-readable eligibility disagreements.",
    )

    args = parser.parse_args(argv)

    if args.scan_evidence:
        return run_evidence_scan()

    loaded_traces = []
    trace_input_paths = {}
    if args.traces:
        for p in args.traces:
            if p.is_file():
                trace = load_trace_with_fallback(p)
                loaded_traces.append(trace)
                trace_input_paths[trace.source.episode_id] = _portable_input_path(p)

    loaded_records = []
    if args.episodes_jsonl:
        for p in args.episodes_jsonl:
            if p.is_file():
                loaded_records.extend(_read_jsonl(p))

    pack = build_failure_pack(
        traces=loaded_traces,
        records=loaded_records,
        allowed_claim_wording=args.allowed_claim_wording,
        thresholds=FailurePredicateThresholds(
            collision=args.collision_threshold,
            comfort=args.comfort_threshold,
            near_miss=args.near_miss_threshold,
            signal_red_phase_violation=args.signal_red_phase_violation_threshold,
            signal_stop_line_crossing=args.signal_stop_line_crossing_threshold,
        ),
        provenance=FailurePackProvenance(
            trace_source_kind=args.trace_source_kind,
            metric_source_kind=args.metric_source_kind,
            execution_performed=args.execution_performed,
            evidence_tier=args.evidence_tier,
            artifact_status=args.artifact_status,
            execution_mode=args.execution_mode,
            fallback_or_degraded=args.fallback_or_degraded,
            claim_matrix_status=args.claim_matrix_status,
        ),
        trace_input_paths=trace_input_paths,
    )

    sanitized_pack = _sanitize(pack)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(sanitized_pack, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote signalized crossing failure pack to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
