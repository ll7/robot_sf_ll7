#!/usr/bin/env python3
"""Run issue #3557 uncertainty-source generalization report.

This composes the controlled #3471 ScenarioBelief episode harness across the
registered uncertainty sources. The output is diagnostic-tier evidence only:
it is not a full benchmark campaign and does not promote paper-facing claims.
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
    INCONCLUSIVE,
    NO_EFFECT,
    REPRODUCES,
    SourceContrast,
    assess_source_generalization,
)
from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    UNCERTAINTY_SOURCES,
    EpisodeParams,
    load_config,
    run_matrix,
)

ISSUE = 3557
SCHEMA_VERSION = "uncertainty_source_generalization_report.v1"
DEFAULT_CONFIG = Path("configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml")
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_3557_uncertainty_source_generalization")
DEFAULT_SOURCES = tuple(UNCERTAINTY_SOURCES)
CLAIM_BOUNDARY = (
    "Controlled #3471 crossing scenario using real stream_gap planner and ScenarioBelief "
    "uncertainty gate. This is diagnostic-tier uncertainty-source generalization evidence "
    "only: no full benchmark campaign, no Slurm/GPU submission, no paper/dissertation claim."
)

_ISSUE_DECISIONS = {
    REPRODUCES: "reproduces_unsafe_drop",
    NO_EFFECT: "no_measurable_difference",
    INCONCLUSIVE: "inconclusive",
}


def _unsafe_commit_rate(report: dict[str, Any], mode: str) -> float:
    """Return unsafe-commit rate for one mode aggregate."""
    aggregate = report["by_mode"][mode]
    episodes = int(aggregate["episodes"])
    if episodes <= 0:
        raise ValueError(f"mode {mode!r} has no episodes")
    return float(aggregate["total_unsafe_commit_steps"]) / episodes


def contrast_from_source_report(report: dict[str, Any]) -> SourceContrast:
    """Convert one source harness report into the merged issue #3557 classifier input."""
    retained = report["by_mode"]["uncertain_retained"]
    dropped = report["by_mode"]["uncertain_dropped"]
    return SourceContrast(
        source=str(report["uncertainty_source"]),
        retained_unsafe_commit_rate=_unsafe_commit_rate(report, "uncertain_retained"),
        dropped_unsafe_commit_rate=_unsafe_commit_rate(report, "uncertain_dropped"),
        min_separation_delta_m=round(
            float(dropped["worst_min_separation"]) - float(retained["worst_min_separation"]),
            6,
        ),
        n_episodes=int(dropped["episodes"]),
    )


def _row_from_source_report(
    report: dict[str, Any],
    verdict_by_source: dict[str, str],
) -> dict[str, Any]:
    """Return one CSV/Markdown row for a source report."""
    retained = report["by_mode"]["uncertain_retained"]
    dropped = report["by_mode"]["uncertain_dropped"]
    source = str(report["uncertainty_source"])
    retained_rate = _unsafe_commit_rate(report, "uncertain_retained")
    dropped_rate = _unsafe_commit_rate(report, "uncertain_dropped")
    verdict = verdict_by_source[source]
    return {
        "source": source,
        "condition_builder": report["uncertainty_source_contract"]["selected"]["condition_builder"],
        "harness_decision": report["decision"]["decision"],
        "source_decision": _ISSUE_DECISIONS.get(verdict, "blocked"),
        "classifier_verdict": verdict,
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
    sources: list[str],
) -> dict[str, Any]:
    """Run all requested sources and assemble issue #3557 diagnostic report."""
    unknown = sorted(set(sources) - set(UNCERTAINTY_SOURCES))
    if unknown:
        raise ValueError(f"unknown uncertainty source(s): {unknown}")

    source_reports = [run_matrix(seeds, params, uncertainty_source=source) for source in sources]
    decision_layer = assess_source_generalization(
        [contrast_from_source_report(report) for report in source_reports]
    )
    verdict_by_source = {
        str(row["source"]): str(row["verdict"]) for row in decision_layer["per_source"]
    }
    rows = [_row_from_source_report(report, verdict_by_source) for report in source_reports]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "evidence_tier": "diagnostic",
        "claim_boundary": CLAIM_BOUNDARY,
        "sources": list(sources),
        "seed_count": len(seeds),
        "seeds": seeds,
        "params": vars(params),
        "generalization": decision_layer["generalization"],
        "decision_layer": decision_layer,
        "per_source": rows,
        "source_reports": source_reports,
    }


def write_report_artifacts(report: dict[str, Any], output_dir: Path, command: str) -> None:
    """Write compact tracked source-generalization artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_source_decisions.csv"
    readme_path = output_dir / "README.md"

    compact = dict(report)
    compact.pop("source_reports", None)
    summary_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = list(report["per_source"][0].keys()) if report["per_source"] else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["per_source"])

    rows = "\n".join(
        "| {source} | {condition_builder} | {source_decision} | "
        "{unsafe_commit_rate_delta_dropped_minus_retained} | "
        "{min_separation_delta_dropped_minus_retained_m} |".format(**row)
        for row in report["per_source"]
    )
    readme_path.write_text(
        "\n".join(
            [
                "# Issue #3557 Uncertainty-Source Generalization",
                "",
                "Plain-language summary: this artifact runs the controlled #3471 "
                "episode harness across registered ScenarioBelief uncertainty sources. "
                "It records per-source oracle/retained/dropped aggregate decisions "
                "without promoting the diagnostic result into benchmark evidence.",
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
                "This is not a full benchmark campaign result. It does not use Slurm "
                "or GPU resources and does not edit paper or dissertation claims.",
                "",
                "## Per-Source Decisions",
                "",
                "| Source | Condition builder | Decision | Unsafe-rate delta | "
                "Min-separation delta (m) |",
                "| --- | --- | --- | ---: | ---: |",
                rows,
                "",
                "Detailed machine-readable outputs:",
                "",
                "- `summary.json`",
                "- `per_source_decisions.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sources", choices=sorted(UNCERTAINTY_SOURCES), nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    seeds, params = (
        load_config(args.config)
        if args.config is not None
        else (list(range(101, 113)), EpisodeParams())
    )
    if args.seeds is not None:
        seeds = args.seeds
    if args.max_steps is not None:
        params = replace(params, max_steps=args.max_steps)
    sources = list(args.sources or DEFAULT_SOURCES)
    report = build_report(seeds=seeds, params=params, sources=sources)
    try:
        script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
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
        "source_count": len(report["sources"]),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
