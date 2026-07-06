#!/usr/bin/env python3
"""Fail-closed real-trajectory readiness report for issue #4013.

The checker reads only a real-trajectory ingestion manifest. It never downloads,
copies, or converts external data. The intended use is to keep #4013 Phase 3
honest: diagnostic synthetic evidence remains useful, but real-trajectory
training and representative evaluation stay blocked until a local dataset copy is
checksum-validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_sf.data_ingestion.real_trajectory_contract import (
    ContractError,
    load_manifest,
    run_preflight,
)

SCHEMA_VERSION = "issue_4013.real_trajectory_readiness.v1"
DEFAULT_MANIFEST = Path("configs/data/issue_4013_eth_biwi_real_trajectory_manifest.yaml")
DEFAULT_OUTPUT_JSON = Path(
    "docs/context/evidence/issue_4013_learned_model_based_planning/"
    "real_trajectory_readiness.v1.json"
)
DEFAULT_OUTPUT_MARKDOWN = Path(
    "docs/context/evidence/issue_4013_learned_model_based_planning/real_trajectory_readiness.v1.md"
)


def build_report(manifest_path: Path) -> dict[str, Any]:
    """Build #4013 Phase 3 readiness report from one manifest."""
    try:
        manifest = load_manifest(manifest_path)
        preflight = run_preflight(manifest)
        input_error: str | None = None
    except (ContractError, jsonschema.ValidationError) as exc:
        manifest = {}
        preflight = None
        input_error = str(exc)

    if input_error:
        status = "blocked_manifest_contract"
        blockers = [{"code": "manifest.input_error", "message": input_error}]
        dataset_id = None
        availability = "unknown"
        benchmark_eligibility = "unknown"
        issues: list[dict[str, str]] = []
    else:
        assert preflight is not None
        issues = [
            {"code": issue.code, "severity": issue.severity, "message": issue.message}
            for issue in preflight.issues
        ]
        dataset_id = preflight.dataset_id
        availability = preflight.availability
        benchmark_eligibility = preflight.benchmark_eligibility
        blockers = [issue for issue in issues if issue["severity"] == "error"]
        if blockers:
            status = "blocked_manifest_contract"
        elif availability != "validated":
            status = "blocked_real_trajectory_data_unavailable"
            blockers = [
                {
                    "code": "real_trajectory.availability_not_validated",
                    "message": (
                        "Phase 3 requires a checksum-validated real trajectory dataset; "
                        f"manifest availability is {availability!r}."
                    ),
                }
            ]
        elif benchmark_eligibility not in {"research_only", "benchmark_candidate"}:
            status = "blocked_real_trajectory_not_evaluation_ready"
            blockers = [
                {
                    "code": "real_trajectory.eligibility_too_low",
                    "message": (
                        "Representative evaluation requires research_only or "
                        "benchmark_candidate eligibility; manifest currently records "
                        f"{benchmark_eligibility!r}."
                    ),
                }
            ]
        else:
            status = "ready_for_real_trajectory_training"

    acceptance_evidence = [
        {
            "criterion": "short-horizon predictor trained",
            "evidence": (
                "Met at diagnostic tier by PR #4629 using seeded synthetic data; "
                "Phase 3 real-trajectory retraining remains blocked until this report "
                "reaches ready_for_real_trajectory_training."
            ),
        },
        {
            "criterion": "model-based action selection runs on a smoke scenario",
            "evidence": "Met at diagnostic tier by PR #4644 and PR #4655.",
        },
        {
            "criterion": "comparison against cv_prediction_mpc and one model-free baseline",
            "evidence": (
                "Met at diagnostic tier by PR #4655 with paired seed count 1; "
                "representative real-trajectory evaluation remains blocked by dataset readiness."
            ),
        },
        {
            "criterion": "fallback/degraded rows are non-evidence",
            "evidence": "Met by PR #4587 report contract and PR #4655 diagnostic report.",
        },
        {
            "criterion": "claim boundary excludes world-model and paper-grade claims",
            "evidence": "Met in existing #4013 design/evidence docs; this report preserves that boundary.",
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 4013,
        "status": status,
        "manifest_path": str(manifest_path),
        "dataset_id": dataset_id,
        "availability": availability,
        "benchmark_eligibility": benchmark_eligibility,
        "preflight_issues": issues,
        "blockers": blockers,
        "acceptance_evidence": acceptance_evidence,
        "claim_boundary": (
            "Real-trajectory readiness only. No raw data is staged by this checker; "
            "no benchmark, navigation-quality, paper, or dissertation claim is made."
        ),
        "next_action": (
            "Stage ETH/BIWI or another approved real pedestrian trajectory dataset under "
            "$ROBOT_SF_EXTERNAL_DATA_ROOT, validate its checksum, update the manifest, "
            "then run real-trajectory training and representative evaluation."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable report."""
    lines = [
        "# Issue #4013 Real-Trajectory Readiness",
        "",
        f"Status: `{report['status']}`",
        "",
        "Claim boundary: real-trajectory readiness only. No raw data is staged, no full "
        "benchmark campaign is run, and no paper/dissertation claim is made.",
        "",
        "## Dataset Gate",
        "",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Dataset: `{report['dataset_id']}`",
        f"- Availability: `{report['availability']}`",
        f"- Benchmark eligibility: `{report['benchmark_eligibility']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report["blockers"]
    if blockers:
        lines.extend(f"- `{item['code']}`: {item['message']}" for item in blockers)
    else:
        lines.append("- None.")
    lines.extend(["", "## Acceptance Evidence", "", "| Criterion | Evidence |", "| --- | --- |"])
    for item in report["acceptance_evidence"]:
        lines.append(f"| {item['criterion']} | {item['evidence']} |")
    lines.extend(["", f"Next action: {report['next_action']}", ""])
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--check", action="store_true", help="Do not write outputs; print JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness checker and optionally write report artifacts."""
    args = _build_parser().parse_args(argv)
    report = build_report(args.manifest)
    if args.check:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
