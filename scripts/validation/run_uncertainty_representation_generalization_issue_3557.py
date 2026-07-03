#!/usr/bin/env python3
"""Run issue #3557 uncertainty-representation generalization report.

This script composes the merged #4187 ScenarioBelief episode harness across the
three supported uncertainty representations. It writes a small diagnostic report
artifact only: controlled #3471 episode evidence, not a full benchmark campaign
and not paper-grade claim promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from robot_sf.representation.uncertainty_source_generalization import (
    SourceContrast,
    assess_source_generalization,
)
from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    UNCERTAINTY_REPRESENTATIONS,
    EpisodeParams,
    load_config,
    run_matrix,
)

ISSUE = 3557
SCHEMA_VERSION = "uncertainty_representation_generalization_report.v1"
DEFAULT_CONFIG = Path("configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml")
DEFAULT_OUTPUT_DIR = Path(
    "docs/context/evidence/issue_3557_uncertainty_representation_generalization"
)
DEFAULT_REPRESENTATIONS = ("belief_drop", "conformal_radius", "envelope_inflation")
CLAIM_BOUNDARY = (
    "Controlled #3471 crossing scenario using the real stream_gap planner and "
    "ScenarioBelief uncertainty gate. This is diagnostic-tier representation "
    "generalization evidence only: no full benchmark campaign, no Slurm/GPU "
    "submission, and no paper/dissertation claim edit."
)


def _unsafe_commit_rate(report: dict[str, Any], mode: str) -> float:
    rows = report["episodes"][mode]
    total_steps = sum(int(row["runtime_steps"]) for row in rows)
    if total_steps <= 0:
        return 0.0
    unsafe_steps = sum(int(row["unsafe_commit_steps"]) for row in rows)
    return unsafe_steps / total_steps


def contrast_from_representation_report(report: dict[str, Any]) -> SourceContrast:
    """Build the existing issue #3557 contrast primitive from one representation run."""
    retained = report["by_mode"]["uncertain_retained"]
    dropped = report["by_mode"]["uncertain_dropped"]
    min_separation_delta_m = float(dropped["worst_min_separation"]) - float(
        retained["worst_min_separation"]
    )
    return SourceContrast(
        source=str(report["uncertainty_representation"]),
        retained_unsafe_commit_rate=_unsafe_commit_rate(report, "uncertain_retained"),
        dropped_unsafe_commit_rate=_unsafe_commit_rate(report, "uncertain_dropped"),
        min_separation_delta_m=round(min_separation_delta_m, 6),
        n_episodes=int(dropped["episodes"]),
    )


def summarize_representation_report(report: dict[str, Any], verdict: str) -> dict[str, Any]:
    """Return one compact row for summary JSON and CSV output."""
    retained = report["by_mode"]["uncertain_retained"]
    dropped = report["by_mode"]["uncertain_dropped"]
    retained_rate = _unsafe_commit_rate(report, "uncertain_retained")
    dropped_rate = _unsafe_commit_rate(report, "uncertain_dropped")
    return {
        "representation": report["uncertainty_representation"],
        "harness_decision": report["decision"]["decision"],
        "harness_reason": report["decision"]["reason"],
        "generalization_verdict": verdict,
        "episodes": dropped["episodes"],
        "retained_unsafe_commit_rate": round(retained_rate, 6),
        "dropped_unsafe_commit_rate": round(dropped_rate, 6),
        "unsafe_commit_rate_delta_dropped_minus_retained": round(dropped_rate - retained_rate, 6),
        "retained_worst_min_separation_m": retained["worst_min_separation"],
        "dropped_worst_min_separation_m": dropped["worst_min_separation"],
        "min_separation_delta_dropped_minus_retained_m": round(
            float(dropped["worst_min_separation"]) - float(retained["worst_min_separation"]),
            6,
        ),
        "uncertainty_consumed_any": dropped["uncertainty_consumed_any"],
        "fail_closed_any": dropped["fail_closed_any"],
    }


def build_report(
    *,
    seeds: list[int],
    params: EpisodeParams,
    representations: list[str],
) -> dict[str, Any]:
    """Run all requested representations and assemble the issue #3557 report."""
    unknown = sorted(set(representations) - set(UNCERTAINTY_REPRESENTATIONS))
    if unknown:
        raise ValueError(f"unknown uncertainty representation(s): {unknown}")

    representation_reports = [
        run_matrix(seeds, params, uncertainty_representation=representation)
        for representation in representations
    ]
    contrasts = [contrast_from_representation_report(report) for report in representation_reports]
    assessment = assess_source_generalization(contrasts)
    verdict_by_representation = {row["source"]: row["verdict"] for row in assessment["per_source"]}
    per_representation = [
        summarize_representation_report(
            report,
            verdict_by_representation[str(report["uncertainty_representation"])],
        )
        for report in representation_reports
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "evidence_tier": "diagnostic",
        "claim_boundary": CLAIM_BOUNDARY,
        "representations": representations,
        "seed_count": len(seeds),
        "seeds": seeds,
        "params": vars(params),
        "generalization": assessment["generalization"],
        "decision_layer": assessment,
        "per_representation": per_representation,
        "representation_reports": representation_reports,
    }


def write_report_artifacts(report: dict[str, Any], output_dir: Path, command: str) -> None:
    """Write reviewable issue #3557 evidence artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_representation_decisions.csv"
    readme_path = output_dir / "README.md"

    persisted_report = {
        key: value for key, value in report.items() if key != "representation_reports"
    }
    summary_path.write_text(
        json.dumps(persisted_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(report["per_representation"][0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["per_representation"])

    rows = "\n".join(
        "| {representation} | {harness_decision} | {generalization_verdict} | "
        "{unsafe_commit_rate_delta_dropped_minus_retained} | "
        "{min_separation_delta_dropped_minus_retained_m} |".format(**row)
        for row in report["per_representation"]
    )
    readme_path.write_text(
        "\n".join(
            [
                "# Issue #3557 Uncertainty-Representation Generalization",
                "",
                "Plain-language summary: this artifact runs the merged #4187 controlled "
                "episode harness across `belief_drop`, `conformal_radius`, and "
                "`envelope_inflation` representations. It asks whether dropping uncertain "
                "agents remains worse in the same diagnostic crossing scenario.",
                "",
                f"- Issue: #{ISSUE}",
                f"- Schema: `{SCHEMA_VERSION}`",
                f"- Evidence tier: `{report['evidence_tier']}`",
                f"- Generalization verdict: `{report['generalization']}`",
                f"- Seeds: `{report['seeds']}`",
                f"- Command: `{command}`",
                "",
                "## Claim Boundary",
                "",
                CLAIM_BOUNDARY,
                "",
                "This is not a full benchmark campaign result. It does not use Slurm or "
                "GPU resources and does not edit paper or dissertation claims.",
                "",
                "## Per-Representation Decisions",
                "",
                "| Representation | Harness decision | Generalization verdict | "
                "Unsafe-rate delta | Min-separation delta (m) |",
                "| --- | --- | --- | ---: | ---: |",
                rows,
                "",
                "Detailed machine-readable outputs:",
                "",
                "- `summary.json`",
                "- `per_representation_decisions.csv`",
                "",
            ]
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=sorted(UNCERTAINTY_REPRESENTATIONS),
        default=list(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the report and write artifacts."""
    args = parse_args(argv)
    seeds, params = load_config(args.config)
    if args.seeds is not None:
        seeds = args.seeds
    if args.max_steps is not None:
        params = replace(params, max_steps=args.max_steps)

    report = build_report(
        seeds=seeds,
        params=params,
        representations=list(args.representations),
    )
    try:
        script = Path(__file__).relative_to(Path.cwd())
    except ValueError:
        # Invoked from outside the repo root; fall back to the script name so the
        # recorded command string never crashes on cwd-relative resolution.
        script = Path(__file__).name
    command_args = sys.argv[1:] if argv is None else argv
    command = " ".join(["uv run python", str(script), *command_args])
    write_report_artifacts(report, args.output_dir, command)
    compact = {
        "schema_version": report["schema_version"],
        "issue": report["issue"],
        "evidence_tier": report["evidence_tier"],
        "generalization": report["generalization"],
        "seed_count": report["seed_count"],
        "per_representation": report["per_representation"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
