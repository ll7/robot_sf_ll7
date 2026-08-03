#!/usr/bin/env python3
"""Gate 3 radius rank-stability analysis and durable evidence bundle (issue #6643).

Analyze the Gate 2 production radius sweep (#6642) for the #6600 campaign and register
a durable evidence bundle: planner-ranking tables (success, typed collisions, SNQI),
Kendall rank correlation and rank-flip counts versus the 1.0 m baseline, per-planner
paired changes with uncertainty, scenario-family and feasibility transitions including
the narrow-doorway family, a fail-closed missingness/degradation ledger, and immutable
config/command/commit/seed-roster/checksum/reproduction provenance.

The command fails closed per the #6600 stop rules:

- no Gate 2 sweep summary -> ``blocked_pending_gate2`` (exit 2); no scientific verdict;
- incomplete row-identity accounting or any fallback/degraded/failed/missing/duplicate/
  provenance-invalid row -> ``invalid_missing_or_inconsistent_evidence`` (exit 0).

A scientific verdict (one of ``stable_within_tested_radii``, ``radius_dependent``,
``non_identifiable``, ``invalid_missing_or_inconsistent_evidence``) exits 0. The verdict
comment for #6600 and the propagation comment for #3207 are printed so an authorized
operator can post them; this command does not post to GitHub itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.radius_rank_stability import (
    ANALYSIS_BLOCKED_PENDING_GATE2,
    RadiusSensitivityReport,
    analyze_radius_sensitivity,
    build_evidence_provenance,
    current_git_sha,
    evidence_tier_for_verdict,
    load_sweep_summary,
    render_propagation_comment,
    render_verdict_comment,
    sweep_summary_available,
    write_evidence_bundle,
)

EXIT_VERDICT_PRODUCED = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_BLOCKED_PENDING_GATE2 = 2


def _default_command() -> str:
    """Return the canonical reproduction command for this invocation."""
    return " ".join(["uv run python", *sys.argv])


def _build_report(args: argparse.Namespace) -> RadiusSensitivityReport:
    """Build the radius-sensitivity report, failing closed when the sweep is absent."""
    if not sweep_summary_available(args.sweep_summary):
        return analyze_radius_sensitivity(None, baseline_radius=args.baseline_radius)
    summary = load_sweep_summary(args.sweep_summary)
    return analyze_radius_sensitivity(summary, baseline_radius=args.baseline_radius)


def _write_bundle(
    report: RadiusSensitivityReport,
    args: argparse.Namespace,
) -> dict[str, Path]:
    """Register the durable evidence bundle and return the written paths."""
    config_sha256 = None
    config_path = Path(args.config) if args.config else None
    if config_path is not None and config_path.is_file():
        config_sha256 = sha256_file(config_path)
    input_paths = {}
    if args.sweep_summary is not None and Path(args.sweep_summary).is_file():
        input_paths["sweep_summary.json"] = Path(args.sweep_summary)
    if args.gate1_canary_receipt is not None and Path(args.gate1_canary_receipt).is_file():
        input_paths["gate1_canary_receipt.json"] = Path(args.gate1_canary_receipt)
    summary = (
        load_sweep_summary(args.sweep_summary)
        if args.sweep_summary is not None and Path(args.sweep_summary).is_file()
        else None
    )
    declared_campaign_commit = None
    if isinstance(summary, dict):
        provenance_by_radius = summary.get("campaign_provenance")
        if isinstance(provenance_by_radius, dict):
            baseline_provenance = provenance_by_radius.get(str(args.baseline_radius))
            if baseline_provenance is None:
                baseline_provenance = provenance_by_radius.get(f"{args.baseline_radius:g}")
            if isinstance(baseline_provenance, dict):
                candidate = baseline_provenance.get("campaign_commit")
                if isinstance(candidate, str):
                    declared_campaign_commit = candidate
    provenance = build_evidence_provenance(
        report,
        config_path=args.config or "not supplied (blocked before Gate 2)",
        command=args.command or _default_command(),
        campaign_commit=args.campaign_commit or declared_campaign_commit or current_git_sha(),
        analysis_commit=current_git_sha(),
        config_sha256=config_sha256,
        input_paths=input_paths,
        sweep_summary=summary,
    )
    return write_evidence_bundle(report, provenance, args.output_dir)


def main(argv: list[str] | None = None) -> int:
    """Run the Gate 3 analysis, register the bundle, and return a process status.

    Returns:
        Exit status: 0 when a scientific verdict is produced, 2 when blocked pending the
        Gate 2 sweep, and 1 on an unexpected input error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=None,
        help="Path to the Gate 2 radius sweep summary JSON. Omit when Gate 2 has not run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the durable evidence bundle.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Immutable campaign config path; required for promoted evidence.",
    )
    parser.add_argument(
        "--baseline-radius",
        type=float,
        default=1.0,
        help="Baseline collision-envelope radius in metres (default 1.0).",
    )
    parser.add_argument(
        "--campaign-commit",
        default=None,
        help="Expected immutable Gate 2 campaign commit SHA; must match every summary arm.",
    )
    parser.add_argument(
        "--gate1-canary-receipt",
        type=Path,
        default=None,
        help="Gate 1 canary receipt whose SHA-256 must match every summary arm for promotion.",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Explicit reproduction command recorded in provenance.",
    )
    parser.add_argument(
        "--print-comments",
        action="store_true",
        help="Print the #6600 verdict comment and the #3207 propagation comment.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a compact machine-readable summary instead of prose.",
    )
    args = parser.parse_args(argv)

    try:
        report = _build_report(args)
        written = _write_bundle(report, args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED_ERROR

    verdict = report.verdict.verdict
    if args.json:
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "analysis_status": report.verdict.analysis_status,
                    "evidence_tier": evidence_tier_for_verdict(verdict),
                    "interpretation_promoted": report.verdict.interpretation_promoted,
                    "reasons": list(report.verdict.reasons),
                    "bundle": {name: str(path) for name, path in written.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"verdict: {verdict}")
        print(f"analysis_status: {report.verdict.analysis_status}")
        print(f"evidence_tier: {evidence_tier_for_verdict(verdict)}")
        print(f"interpretation_promoted: {report.verdict.interpretation_promoted}")
        print(f"reasons: {', '.join(report.verdict.reasons) or 'none'}")
        print("bundle:")
        for name, path in written.items():
            print(f"  {name}: {path}")

    if args.print_comments:
        print("\n----- verdict comment for #6600 -----\n")
        print(render_verdict_comment(report))
        print("\n----- propagation comment for #3207 -----\n")
        print(render_propagation_comment(report))

    if verdict == ANALYSIS_BLOCKED_PENDING_GATE2:
        return EXIT_BLOCKED_PENDING_GATE2
    return EXIT_VERDICT_PRODUCED


if __name__ == "__main__":
    sys.exit(main())
