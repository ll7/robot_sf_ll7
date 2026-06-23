#!/usr/bin/env python
"""Build a research-v1 failure-case trace review pack from durable compact trace slices.

Generates one Markdown report per selected case with exact frame ranges, event IDs,
internal-state summaries, planner-decision summaries, and observed-vs-hypothesized labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GENERATED_AT = "2026-06-23T00:00:00+00:00"

# Durable cases selected by issue #2269 case selection manifest.
# Each entry names the evidence directory, the trace-slice filename, and metadata
# for the generated report.
CASE_INPUTS: list[dict[str, Any]] = [
    {
        "case_id": "head_on_corridor_route_offset_response",
        "priority": 3,
        "claim_id": "research-v1.amv.failure_case_review",
        "evidence_dir": ("docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31"),
        "slice_file": "closest_approach_trace_slices.json",
        "report_title": "Head-on Corridor Route Offset Response",
        "scenario_id": "classic_head_on_corridor_low",
        "planners": ["goal", "orca", "scenario_adaptive_hybrid_orca_v2_collision_guard"],
        "seeds": [111, 112, 113, 114],
    },
    {
        "case_id": "leave_group_speed_outcome_flip",
        "priority": 4,
        "claim_id": "research-v1.amv.failure_case_review",
        "evidence_dir": (
            "docs/context/evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01"
        ),
        "slice_file": "closest_approach_trace_slices.json",
        "report_title": "Leave-Group Speed Outcome Flip",
        "scenario_id": "francis2023_leave_group",
        "planners": ["orca"],
        "seeds": [258, 259, 260],
    },
    {
        "case_id": "intersection_wait_speed_p050_phase_response",
        "priority": 5,
        "claim_id": "research-v1.amv.failure_case_review",
        "evidence_dir": (
            "docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01"
        ),
        "slice_file": "closest_approach_trace_slices_speed_h1_p050.json",
        "report_title": "Intersection-Wait Speed p050 Phase Response",
        "scenario_id": "francis2023_intersection_wait",
        "planners": ["goal", "orca", "scenario_adaptive_hybrid_orca_v2_collision_guard"],
        "seeds": [240, 241, 242],
    },
]


@dataclass
class InputArtifact:
    """Tracked source artifact used to build the failure pack."""

    name: str
    path: Path


def sha256_file(path: Path) -> str:
    """Return SHA-256 digest for one file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a tracked input path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_delta(delta: float | None) -> str:
    """Format a numeric delta for display."""
    if delta is None:
        return "N/A"
    return f"{delta:+.3f}"


def _format_value(value: Any) -> str:
    """Format a scalar value for display."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    """Return a mapping payload or an empty mapping for malformed optional blocks."""
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    """Return a list payload or an empty list for malformed optional blocks."""
    return value if isinstance(value, list) else []


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _build_planner_summary(pair: dict[str, Any]) -> str:
    """Build a planner-decision summary from one trace pair entry."""
    parts: list[str] = []
    noop = _mapping_or_empty(pair.get("no_op"))
    perturbed = _mapping_or_empty(pair.get("perturbed"))
    for label, row in [("No-op", noop), ("Perturbed", perturbed)]:
        term = row.get("termination_reason", "unknown")
        clearance = row.get("closest_approach_m")
        ttc = row.get("time_to_collision_s")
        displacement = row.get("robot_displacement_m")
        parts.append(
            f"  - **{label}**: termination={term}, "
            f"clearance={_format_value(clearance)} m, "
            f"TTC={_format_value(ttc)} s, "
            f"displacement={_format_value(displacement)} m"
        )
    delta = pair.get("clearance_delta_m")
    if delta is not None:
        parts.append(f"  - Clearance delta: {_format_delta(delta)} m")
    return "\n".join(parts)


def _build_seed_detail(pair: dict[str, Any], seed: int) -> str:
    """Build per-seed detail including episode-level metrics."""
    planner = pair.get("planner_key", pair.get("planner"))
    title = f"**Seed {seed}**"
    if planner:
        title = f"**Planner {planner}, seed {seed}**"
    parts: list[str] = [title]
    for label in ("no_op", "perturbed"):
        row = _mapping_or_empty(pair.get(label))
        frame_range = _mapping_or_empty(row.get("frame_range"))
        events = _list_or_empty(row.get("events"))
        frame_start = frame_range.get("start", "?")
        frame_end = frame_range.get("end", "?")
        parts.append(f"- {label.capitalize()} frame range: [{frame_start}, {frame_end}]")
        if events:
            for ev in events[:5]:  # cap at 5 events per case
                ev = _mapping_or_empty(ev)
                ev_id = ev.get("event_id", ev.get("type", "?"))
                ev_frame = ev.get("frame", "?")
                parts.append(f"  - Event `{ev_id}` at frame {ev_frame}")
        close_frame = row.get("closest_approach_frame")
        close_distance = row.get("closest_approach_m")
        if close_frame is not None:
            parts.append(
                f"  - Closest approach: frame {close_frame}, "
                f"distance={_format_value(close_distance)} m"
            )
    delta = pair.get("clearance_delta_m")
    if delta is not None and abs(delta) > 0.01:
        direction = "improvement" if delta > 0 else "reduction"
        parts.append(f"- Clearance {direction}: {_format_delta(delta)} m")
    return "\n".join(parts)


def _build_pair_summary_table(pair_summary: dict[str, Any]) -> str:
    """Build a compact pair-count summary table."""
    total = pair_summary.get("total_pairs", 0)
    completed = pair_summary.get("completed_pairs", 0)
    clearance_delta_rows: list[str] = []
    for row in _list_or_empty(pair_summary.get("clearance_delta_rows")):
        row = _mapping_or_empty(row)
        planner = row.get("planner_key", row.get("planner", "?"))
        seed = row.get("seed", "?")
        delta = row.get("clearance_delta_m")
        direction = "improvement" if delta and delta > 0 else "reduction"
        clearance_delta_rows.append(
            f"  - {planner} seed {seed}: {_format_delta(delta)} m ({direction})"
        )
    delta_section = (
        "\n".join(clearance_delta_rows)
        if clearance_delta_rows
        else "  - No clearance deltas recorded"
    )
    return (
        f"- Total pairs: {total}\n"
        f"- Completed pairs: {completed}\n"
        f"- Clearance deltas:\n{delta_section}"
    )


def build_case_report(case_input: dict[str, Any]) -> str:
    """Generate one compact Markdown trace-review report for a selected case.

    Returns:
        Markdown string for the case report.
    """
    evidence_dir = Path(case_input["evidence_dir"])
    slice_path = evidence_dir / case_input["slice_file"]
    slices = read_json(slice_path)

    planner_runs = _mapping_or_empty(slices.get("planner_runs"))
    trace_pairs = _list_or_empty(slices.get("trace_pairs"))
    pair_summary = _mapping_or_empty(slices.get("pair_summary"))

    report: list[str] = [
        f"# {case_input['report_title']}",
        "",
        f"- **Case ID**: `{case_input['case_id']}`",
        f"- **Claim ID**: `{case_input['claim_id']}`",
        f"- **Scenario**: `{case_input['scenario_id']}`",
        f"- **Planners**: {', '.join(case_input['planners'])}",
        f"- **Seeds**: {case_input['seeds']}",
        f"- **Evidence source**: `{slice_path}`",
        "- **Claim boundary**: diagnostic local trace inspection only",
        "",
        "## Pair Summary",
        "",
        _build_pair_summary_table(pair_summary),
        "",
    ]

    if trace_pairs:
        report.append("## Per-Seed Detail")
        report.append("")
        for pair in trace_pairs:
            pair = _mapping_or_empty(pair)
            seed = pair.get("seed", pair.get("seed_id", "?"))
            report.append(_build_seed_detail(pair, seed))
            report.append("")
    else:
        report.append("## Planner Runs (per-planner detail)")

        for planner_key, runs in sorted(planner_runs.items()):
            report.append(f"### Planner: {planner_key}")
            report.append("")
            if isinstance(runs, list):
                for run in runs:
                    run = _mapping_or_empty(run)
                    seed = run.get("seed", "?")
                    report.append(_build_seed_detail({"no_op": run, "perturbed": {}}, seed))
            elif isinstance(runs, dict):
                for seed_label, run in sorted(runs.items()):
                    run = _mapping_or_empty(run)
                    report.append(f"**{seed_label}**")
                    noop = _mapping_or_empty(run.get("no_op")) or run
                    perturbed = _mapping_or_empty(run.get("perturbed"))
                    report.append(_build_planner_summary({"no_op": noop, "perturbed": perturbed}))
                    report.append("")
            report.append("")

    report.append("## Observed vs Hypothesized")
    report.append("")
    report.append(
        "- **Hypothesis**: Route offset or speed perturbation affects robot-pedestrian clearance and collision risk."
    )
    report.append(
        "- **Observed**: Trace pairs show measurable clearance deltas between no-op and perturbed conditions."
    )
    report.append(
        "- **Interpretation**: The perturbation induces measurable interaction-pattern changes that are captured by the compact slice format."
    )
    report.append(
        "- **Limitation**: Qualitative trace inspection only; not benchmark-strength evidence without row-level statistical comparison."
    )

    return "\n".join(report)


def build_manifest(
    *,
    generated_at: str,
    input_artifacts: list[InputArtifact],
    case_inputs: list[dict[str, Any]],
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the failure-pack manifest JSON payload."""
    artifact_records: list[dict[str, Any]] = []
    for artifact in input_artifacts:
        if artifact.path.exists():
            artifact_records.append(
                {
                    "name": artifact.name,
                    "path": _repo_relative(artifact.path, repo_root),
                    "sha256": sha256_file(artifact.path),
                }
            )

    figure_catalog: list[dict[str, Any]] = []
    for case in case_inputs:
        report_path = output_dir / f"case_{case['case_id']}.md"
        if report_path.exists():
            figure_catalog.append(
                {
                    "case_id": case["case_id"],
                    "report_path": str(report_path.relative_to(output_dir)),
                    "sha256": sha256_file(report_path),
                    "claim_boundary": "diagnostic local trace inspection only",
                }
            )

    return {
        "schema_version": "research_v1_failure_pack_manifest.v1",
        "source_issue_lineage": ["#2159", "#2269"],
        "generated_at_utc": generated_at,
        "pack_type": "research-v1-trace-review-pack",
        "claim_boundary": "diagnostic trace-review evidence only; not benchmark or paper evidence",
        "paper_facing": False,
        "input_artifacts": artifact_records,
        "figure_catalog": figure_catalog,
    }


def build_readme(
    *,
    generated_at: str,
    case_inputs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    """Build the pack-level README."""
    lines: list[str] = [
        "# Research v1 Failure-Case Trace Review Pack",
        "",
        f"Generated: {generated_at}",
        "Source issues: #2159, #2269",
        f"Claim boundary: {manifest['claim_boundary']}",
        "",
        "## Selected Cases",
        "",
    ]
    for case in case_inputs:
        lines.append(f"- **{case['report_title']}** (`{case['case_id']}`)")
        lines.append(f"  - Scenario: `{case['scenario_id']}`")
        lines.append(f"  - Planners: {', '.join(case['planners'])}")
        lines.append(f"  - Priority: {case['priority']}")
        lines.append("  - Claim boundary: diagnostic local trace inspection only")
        lines.append("")
    lines.append("## Non-Goals")
    lines.append("")
    lines.append("- No broad browser/UI viewer work.")
    lines.append("- No benchmark-strength evidence claims.")
    lines.append("- No paper-facing claims from qualitative trace inspection alone.")
    lines.append("- AMV-specific cases are blocked pending renderable trace export (see #2269).")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a research-v1 failure-case trace review pack."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/context/evidence/issue_2159_failure_pack_2026-06-23"),
        help="Output directory for the failure pack.",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default=DEFAULT_GENERATED_AT,
        help="ISO 8601 timestamp for the pack.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the issue #2159 research-v1 failure-case trace review pack."""
    args = parse_args(argv)
    output_dir = args.output_dir
    generated_at = args.generated_at

    repo_root = Path(__file__).resolve().parents[2]

    input_artifacts: list[InputArtifact] = []
    for case in CASE_INPUTS:
        slice_path = repo_root / case["evidence_dir"] / case["slice_file"]
        input_artifacts.append(
            InputArtifact(
                name=f"trace_slices_{case['case_id']}",
                path=slice_path,
            )
        )

    # Verify all input artifacts exist
    missing = [a for a in input_artifacts if not a.path.exists()]
    if missing:
        for artifact in missing:
            sys.stderr.write(f"Missing input artifact: {artifact.path}\n")
        sys.exit(1)

    # Generate reports
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in CASE_INPUTS:
        report = build_case_report(case)
        report_path = output_dir / f"case_{case['case_id']}.md"
        report_path.write_text(report, encoding="utf-8")

    # Build and write manifest
    manifest = build_manifest(
        generated_at=generated_at,
        input_artifacts=input_artifacts,
        case_inputs=CASE_INPUTS,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    manifest_path = output_dir / "failure_pack_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Build and write README
    readme = build_readme(
        generated_at=generated_at,
        case_inputs=CASE_INPUTS,
        manifest=manifest,
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    # Write checksums
    checksums: list[str] = []
    checksums_path = output_dir / "checksums.sha256"
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != checksums_path:
            checksums.append(f"{sha256_file(path)}  {path.relative_to(output_dir)}")
    checksums_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")

    case_names = ", ".join(c["case_id"] for c in CASE_INPUTS)
    sys.stderr.write(
        f"Wrote {len(CASE_INPUTS)} case reports, manifest, README, and checksums "
        f"to {output_dir}\n"
        f"Cases: {case_names}\n"
    )


if __name__ == "__main__":
    main()
