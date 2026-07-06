#!/usr/bin/env python3
"""Build the issue #2910 publication-suite certification integration report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "issue_2910_publication_suite_certification_report.v1"
DEFAULT_SCENARIO_CERTIFICATION_SUMMARY = Path(
    "docs/context/evidence/issue_2910_release_scenario_certification/summary.json"
)
DEFAULT_RELEASE_CLAIM_MATRIX = Path(
    "docs/context/evidence/issue_3294_release_claim_matrix/release_claim_matrix.json"
)
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2910_publication_suite_certification_report")
ACCEPTED_CERTIFICATION_VALUES = {
    "scenario_cert.v1:accepted",
    "scenario_cert.v1:accepted_reviewed",
}


@dataclass(frozen=True)
class InputPaths:
    """Repository-relative inputs used by the report builder."""

    scenario_certification_summary: Path
    release_claim_matrix: Path


def _repo_relative(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    """Load ``path`` as a JSON object."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_list(payload: dict[str, Any], key: str) -> list[Any]:
    """Return ``payload[key]`` as a list or fail with a useful error."""

    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _created_at_utc(summary: dict[str, Any]) -> str:
    """Return deterministic report timestamp from source summary when available."""

    raw_created_at = summary.get("created_at_utc")
    if isinstance(raw_created_at, str) and raw_created_at.strip():
        created_at = raw_created_at.strip()
        date_part, separator, time_part = created_at.partition("T")
        if ":" in date_part and separator:
            return f"{date_part.replace(':', '-')}T{time_part}"
        return created_at
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _scenario_id(item: object) -> str:
    """Return a scenario id for report sorting."""

    if isinstance(item, dict):
        return str(item.get("scenario_id", ""))
    return str(item)


def _release_artifact_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Return release-artifact rows from the release claim matrix."""

    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ValueError("release claim matrix must contain a rows list")
    return [
        row for row in rows if isinstance(row, dict) and row.get("section") == "release_artifact"
    ]


def _certification_blocked_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Return release rows that still lack accepted scenario certification."""

    blocked: list[dict[str, Any]] = []
    for row in _release_artifact_rows(matrix):
        status = str(row.get("scenario_certification", "")).strip()
        if status not in ACCEPTED_CERTIFICATION_VALUES:
            blocked.append(
                {
                    "row_id": row.get("row_id"),
                    "classification": row.get("classification"),
                    "scenario_certification": status or "missing",
                    "missing_prerequisites": row.get("missing_prerequisites", []),
                }
            )
    return blocked


def build_report(
    scenario_summary: dict[str, Any],
    release_matrix: dict[str, Any],
    *,
    inputs: InputPaths,
) -> dict[str, Any]:
    """Build a deterministic publication-suite certification report."""

    excluded = sorted(
        _as_list(scenario_summary, "excluded_scenarios"),
        key=_scenario_id,
    )
    stress_only = sorted(
        _as_list(scenario_summary, "stress_only_scenarios"),
        key=_scenario_id,
    )
    blocked_rows = _certification_blocked_rows(release_matrix)
    eligibility_counts = scenario_summary.get("benchmark_eligibility_counts", {})
    if not isinstance(eligibility_counts, dict):
        raise ValueError("benchmark_eligibility_counts must be an object")

    status = "pass"
    blockers: list[dict[str, Any]] = []
    if excluded:
        blockers.append(
            {
                "check": "excluded_scenarios",
                "severity": "blocker",
                "count": len(excluded),
                "next_action": (
                    "Remove these geometrically infeasible scenarios from the v0.1 "
                    "publication suite or repair the geometry and regenerate "
                    "scenario_cert.v1 evidence."
                ),
            }
        )
    if stress_only:
        blockers.append(
            {
                "check": "stress_only_scenarios",
                "severity": "blocker",
                "count": len(stress_only),
                "next_action": (
                    "Route these scenarios through an explicit stress-suite claim "
                    "boundary or keep them out of the nominal v0.1 publication set."
                ),
            }
        )
    if blocked_rows:
        blockers.append(
            {
                "check": "release_claim_matrix_rows",
                "severity": "blocker",
                "count": len(blocked_rows),
                "next_action": (
                    "Regenerate the release claim matrix after the publication suite "
                    "exclusion/stress-only policy is applied."
                ),
            }
        )
    if blockers:
        status = "blocked"

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2910,
        "created_at_utc": _created_at_utc(scenario_summary),
        "status": status,
        "claim_boundary": (
            "CPU-only integration report over tracked scenario_cert.v1 summary and "
            "release claim matrix. It does not run a benchmark campaign, publish a "
            "release, submit compute, or promote blocked/stress-only rows as "
            "benchmark evidence."
        ),
        "inputs": {
            "scenario_certification_summary": _repo_relative(inputs.scenario_certification_summary),
            "release_claim_matrix": _repo_relative(inputs.release_claim_matrix),
        },
        "summary": {
            "scenario_count": scenario_summary.get("scenario_count"),
            "benchmark_eligibility_counts": eligibility_counts,
            "release_artifact_rows": len(_release_artifact_rows(release_matrix)),
            "release_artifact_rows_blocked_on_certification": len(blocked_rows),
            "blocker_count": len(blockers),
        },
        "blockers": blockers,
        "excluded_scenarios": excluded,
        "stress_only_scenarios": stress_only,
        "release_claim_matrix_blocked_rows": blocked_rows,
        "acceptance_evidence": [
            {
                "criterion": "All included scenarios have contracts and certification status.",
                "evidence": (
                    "scenario_cert.v1 summary covers "
                    f"{scenario_summary.get('scenario_count')} scenarios with "
                    f"{eligibility_counts.get('eligible')} eligible, "
                    f"{eligibility_counts.get('excluded')} excluded, and "
                    f"{eligibility_counts.get('stress_only')} stress-only."
                ),
                "result": "blocked_until_excluded_and_stress_only_scenarios_are_handled",
            },
            {
                "criterion": "All planner rows classify fallback/degraded/not-available fail-closed.",
                "evidence": (
                    "Release claim matrix remains the source for planner-row caveats; "
                    "this report does not alter planner row classification."
                ),
                "result": "unchanged_from_merged_release_claim_matrix_and_gate_prs",
            },
            {
                "criterion": "All claims link durable artifacts.",
                "evidence": (
                    "Report inputs are tracked docs/context/evidence JSON files and "
                    "generated outputs are tracked compact JSON/Markdown evidence."
                ),
                "result": "met_for_this_integration_report",
            },
            {
                "criterion": "Release summary states exactly what benchmark supports.",
                "evidence": (
                    "Report states claim boundary and lists blockers that prevent "
                    "publication readiness."
                ),
                "result": "improved_but_release_publication_still_blocked",
            },
            {
                "criterion": "If intended evidence cannot be produced, fail closed.",
                "evidence": (
                    "status remains blocked while excluded/stress-only scenarios and "
                    "matrix certification blockers remain."
                ),
                "result": "met",
            },
        ],
        "next_empirical_action": (
            "Apply a versioned v0.1 suite policy that excludes or repairs the 2 "
            "geometrically infeasible scenarios and explicitly routes or removes "
            "the 9 stress-only scenarios, then regenerate scenario_cert.v1 summary, "
            "release claim matrix, and publication gate output."
        ),
    }


def _scenario_table_rows(items: list[Any], *, stress_only: bool) -> list[str]:
    """Render compact scenario rows for Markdown output."""

    rows: list[str] = []
    for item in items:
        if isinstance(item, dict):
            scenario_id = str(item.get("scenario_id", ""))
            classification = str(item.get("classification", ""))
            reason = str(item.get("reason", ""))
        else:
            scenario_id = str(item)
            classification = "stress_only" if stress_only else ""
            reason = ""
        rows.append(f"| `{scenario_id}` | {classification} | {reason} |")
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as compact reviewable Markdown."""

    summary = report["summary"]
    lines = [
        "# Issue #2910 Publication Suite Certification Report",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Summary",
        "",
        f"- Scenario count: {summary['scenario_count']}",
        f"- Benchmark eligibility: {summary['benchmark_eligibility_counts']}",
        "- Release artifact rows blocked on certification: "
        f"{summary['release_artifact_rows_blocked_on_certification']}",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report["blockers"]:
        lines.append(f"- `{blocker['check']}` ({blocker['count']}): {blocker['next_action']}")
    if not report["blockers"]:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Excluded Scenarios",
            "",
            "| Scenario | Classification | Reason |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(_scenario_table_rows(report["excluded_scenarios"], stress_only=False))

    lines.extend(
        [
            "",
            "## Stress-Only Scenarios",
            "",
            "| Scenario | Classification | Reason |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(_scenario_table_rows(report["stress_only_scenarios"], stress_only=True))

    lines.extend(
        [
            "",
            "## Release Claim Matrix Rows Still Blocked",
            "",
            "| Row | Classification | Scenario certification |",
            "| --- | --- | --- |",
        ]
    )
    for row in report["release_claim_matrix_blocked_rows"]:
        lines.append(
            f"| `{row['row_id']}` | {row['classification']} | {row['scenario_certification']} |"
        )

    lines.extend(
        [
            "",
            "## Next Empirical Action",
            "",
            report["next_empirical_action"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-certification-summary",
        type=Path,
        default=DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
    )
    parser.add_argument(
        "--release-claim-matrix",
        type=Path,
        default=DEFAULT_RELEASE_CLAIM_MATRIX,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write JSON and Markdown report files."""

    args = parse_args(argv)
    inputs = InputPaths(
        scenario_certification_summary=args.scenario_certification_summary,
        release_claim_matrix=args.release_claim_matrix,
    )
    report = build_report(
        _load_json(inputs.scenario_certification_summary),
        _load_json(inputs.release_claim_matrix),
        inputs=inputs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
