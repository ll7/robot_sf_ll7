#!/usr/bin/env python3
"""Resolve and render the two native worked-example trace figures for #5756.

The command consumes the byte-pinned #5446 request list and a versioned,
artifact-bound mapping receipt from the pinned rerun.  It writes the resolution
manifest before rendering and fails closed unless all 90 requests reproduce
their release outcomes with matching source identity, pinned provenance, exact
trace digests, and valid ``simulation_trace_export.v1`` payloads.  The real
#6412 package is intentionally an explicit 88-admitted/2-excluded surface: the
two outcome mismatches remain visible and are never presented as resolved
release evidence.  It does not launch a campaign or infer missing outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from robot_sf.benchmark.candidate_trace_resolution import (
    ISSUE_5756_NOT_ADMITTED_TUPLES,
    ISSUE_5756_REQUEST_COUNT,
    WORKED_EXAMPLE_OUTCOMES,
    CandidateTraceResolutionError,
    EpisodeMappingReceipt,
    load_episode_mapping,
    load_episode_requests,
    resolve_episode_requests,
    validate_candidate_trace_resolution,
)
from robot_sf.benchmark.figure_qa import lint_figure
from robot_sf.benchmark.trace_scene_figure import (
    TraceSchemaError,
    load_episode_from_trace_export,
    render_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-requests", type=Path, required=True)
    parser.add_argument(
        "--episode-mapping",
        "--episode-map",
        dest="episode_mapping",
        type=Path,
        required=True,
        help="Versioned issue_5756_trace_mapping_receipt.v1 JSON with pinned provenance.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution-json", type=Path, default=None)
    parser.add_argument("--qa", action="store_true", help="run the figure linter and emit PNGs")
    parser.add_argument(
        "--qa-report",
        type=Path,
        default=None,
        help="JSON report path for the figure-QA results; implies --qa when supplied.",
    )
    return parser


def _mapping_row(mapping: EpisodeMappingReceipt, row: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(row["episode_id"])
    mapped = mapping.get(episode_id)
    if mapped is None:
        raise CandidateTraceResolutionError(f"resolved row has no mapping for {episode_id}")
    return mapped


def _outcome(row: dict[str, Any]) -> str | None:
    value = row.get("rerun_outcome")
    return str(value) if value in WORKED_EXAMPLE_OUTCOMES else None


def _require_complete_resolution(resolution: dict[str, Any]) -> None:
    """Require exactly 88 admitted rows and two named explicit exclusions."""
    summary = resolution.get("summary")
    rows = resolution.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise CandidateTraceResolutionError("resolution has no summary or rows")
    expected = {
        "n_candidates": ISSUE_5756_REQUEST_COUNT,
        "n_resolved": 88,
        "n_trace_missing": 0,
        "n_schema_mismatch": 0,
        "n_provenance_incomplete": 2,
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected or len(rows) != ISSUE_5756_REQUEST_COUNT:
        raise CandidateTraceResolutionError(
            f"rendering requires 88 admitted plus 2 explicit exclusions; "
            f"summary={observed}, rows={len(rows)}"
        )
    identities = {(row.get("scenario_id"), row.get("planner_id"), row.get("seed")) for row in rows}
    if len(identities) != ISSUE_5756_REQUEST_COUNT:
        raise CandidateTraceResolutionError(
            "rendering requires one resolution row for every requested tuple"
        )
    admitted = [row for row in rows if row.get("admission_status") == "admitted"]
    excluded = [row for row in rows if row.get("admission_status") == "not_admitted"]
    if len(admitted) != 88 or len(excluded) != 2:
        raise CandidateTraceResolutionError(
            f"rendering requires 88 admitted and 2 not_admitted rows; "
            f"admitted={len(admitted)}, excluded={len(excluded)}"
        )
    if any(row.get("resolution_status") != "resolved" for row in admitted):
        raise CandidateTraceResolutionError("every admitted row must resolve successfully")
    if any(
        row.get("resolution_status") != "provenance-incomplete"
        or row.get("exclusion_reason") != "outcome_mismatch"
        or row.get("release_outcome") == row.get("rerun_outcome")
        for row in excluded
    ):
        raise CandidateTraceResolutionError(
            "each excluded row must be an explicit outcome mismatch with unequal outcomes"
        )
    excluded_identities = {
        (row.get("scenario_id"), row.get("planner_id"), row.get("seed")) for row in excluded
    }
    expected_excluded = {
        (scenario_id, planner, seed)
        for scenario_id, planner, seed in ISSUE_5756_NOT_ADMITTED_TUPLES
    }
    if excluded_identities != expected_excluded:
        raise CandidateTraceResolutionError(
            f"unexpected #5756 exclusions: observed={sorted(excluded_identities)}, "
            f"expected={sorted(expected_excluded)}"
        )


def _find_exemplar(
    resolution: dict[str, Any],
    mapping: EpisodeMappingReceipt,
    *,
    scenario_id: str,
    planner: str,
    seed: int,
    allowed_outcomes: set[str],
) -> tuple[dict[str, Any], str]:
    matches = [
        row
        for row in resolution["rows"]
        if row["scenario_id"] == scenario_id
        and row["planner_id"] == planner
        and row["seed"] == seed
    ]
    if len(matches) != 1:
        raise CandidateTraceResolutionError(
            f"expected one request for {scenario_id}/{planner}/{seed}, found {len(matches)}"
        )
    row = matches[0]
    if row["resolution_status"] != "resolved":
        raise CandidateTraceResolutionError(
            f"required exemplar {scenario_id}/{planner}/{seed} is {row['resolution_status']}: "
            f"{row['reason_code']}"
        )
    if row.get("admission_status") != "admitted":
        raise CandidateTraceResolutionError(
            f"required exemplar {scenario_id}/{planner}/{seed} is not admitted"
        )
    observed = _outcome(_mapping_row(mapping, row))
    if observed not in allowed_outcomes:
        raise CandidateTraceResolutionError(
            f"required exemplar {scenario_id}/{planner}/{seed} has outcome {observed!r}; "
            f"expected one of {sorted(allowed_outcomes)}"
        )
    return row, observed


def _render_pair(
    rows: tuple[dict[str, Any], dict[str, Any]],
    outcomes: tuple[str, str],
    *,
    output: Path,
    emit_png: bool,
) -> dict[str, Any]:
    episodes = [
        load_episode_from_trace_export(Path(row["trace_artifact_uri"]), outcome=outcome)
        for row, outcome in zip(rows, outcomes, strict=True)
    ]
    render_result = render_comparison(episodes, output, return_figure=emit_png)
    if not emit_png:
        return {
            "figure": output.name,
            "status": "rendered_without_qa",
            "n_defects": None,
            "n_error_defects": None,
            "defects": [],
        }
    if not isinstance(render_result, tuple):
        raise TraceSchemaError("comparison renderer did not return a Figure for QA")
    _, figure = render_result
    try:
        defects = lint_figure(figure)
        defect_records = [
            {
                "defect_type": defect.defect_type,
                "severity": defect.severity,
                "message": defect.message,
                "location": list(defect.location) if defect.location is not None else None,
            }
            for defect in defects
        ]
        hard_defects = [defect for defect in defects if defect.severity == "error"]
        if hard_defects:
            raise TraceSchemaError(
                "figure QA failed: "
                + "; ".join(f"{d.defect_type}: {d.message}" for d in hard_defects)
            )
        figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        return {
            "figure": output.name,
            "status": "passed",
            "n_defects": len(defects),
            "n_error_defects": len(hard_defects),
            "defects": defect_records,
        }
    finally:
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Resolve the 90 requests and render both required pair figures."""
    args = _parser().parse_args(argv)
    if args.qa_report is not None:
        args.qa = True
    resolution_json = args.resolution_json or args.out_dir / "candidate_trace_resolution.v1.json"
    qa_report = args.qa_report or (args.out_dir / "figure_qa.json" if args.qa else None)
    try:
        request_manifest = load_episode_requests(args.episode_requests)
        mapping = load_episode_mapping(args.episode_mapping)
        resolution = resolve_episode_requests(request_manifest, mapping)
        resolution_json.parent.mkdir(parents=True, exist_ok=True)
        resolution_json.write_text(json.dumps(resolution, indent=2) + "\n", encoding="utf-8")
        validation = validate_candidate_trace_resolution(resolution)
        if not validation["ok"]:
            raise CandidateTraceResolutionError(
                "resolution manifest failed schema validation: " + "; ".join(validation["errors"])
            )
        _require_complete_resolution(resolution)
        doorway_success, doorway_collision = (
            _find_exemplar(
                resolution,
                mapping,
                scenario_id="classic_doorway_medium",
                planner="ppo",
                seed=113,
                allowed_outcomes={"success", "route_complete"},
            ),
            _find_exemplar(
                resolution,
                mapping,
                scenario_id="classic_doorway_medium",
                planner="ppo",
                seed=114,
                allowed_outcomes={"collision_event"},
            ),
        )
        bottleneck_goal, bottleneck_ppo = (
            _find_exemplar(
                resolution,
                mapping,
                scenario_id="classic_realworld_double_bottleneck_high",
                planner="goal",
                seed=118,
                allowed_outcomes={"success", "route_complete"},
            ),
            _find_exemplar(
                resolution,
                mapping,
                scenario_id="classic_realworld_double_bottleneck_high",
                planner="ppo",
                seed=118,
                allowed_outcomes={"collision_event"},
            ),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        figure_reports = [
            _render_pair(
                (doorway_success[0], doorway_collision[0]),
                (doorway_success[1], doorway_collision[1]),
                output=args.out_dir / "doorway_ppo_seed113_vs_114.pdf",
                emit_png=args.qa,
            ),
            _render_pair(
                (bottleneck_goal[0], bottleneck_ppo[0]),
                (bottleneck_goal[1], bottleneck_ppo[1]),
                output=args.out_dir / "double_bottleneck_goal_vs_ppo_seed118.pdf",
                emit_png=args.qa,
            ),
        ]
        if qa_report is not None:
            payload = {
                "schema_version": "issue_6412_figure_qa.v1",
                "status": "passed",
                "visualization_only": True,
                "claim_boundary": (
                    "Figure rendering only; no release statistics or manuscript evidence admission."
                ),
                "n_figures": len(figure_reports),
                "n_error_defects": sum(
                    int(report["n_error_defects"] or 0) for report in figure_reports
                ),
                "figures": figure_reports,
            }
            qa_report.parent.mkdir(parents=True, exist_ok=True)
            qa_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except (CandidateTraceResolutionError, TraceSchemaError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"resolved {resolution['summary']['n_resolved']} requests; rendered 2 pair figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
