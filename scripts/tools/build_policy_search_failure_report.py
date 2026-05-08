#!/usr/bin/env python3
"""Build a failure-taxonomy report from a policy-search run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.validation.policy_search_common import (
    classify_failure_mode,
    normalize_scenario_exclusion,
    summarize_policy_search_records,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/policy_search/failure_report"))
    return parser.parse_args()


def _read_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read JSON object records from a JSONL file.

    Returns:
        list[dict[str, Any]]: Parsed records, skipping blank lines.
    """
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def main() -> int:
    """Build JSON and Markdown failure reports."""
    args = parse_args()
    if args.jsonl is None and args.summary_json is None:
        raise SystemExit("Provide --jsonl or --summary-json")
    jsonl_path = args.jsonl
    if jsonl_path is None and args.summary_json is not None:
        payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
        jsonl_raw = payload.get("jsonl_path") if isinstance(payload, dict) else None
        if not isinstance(jsonl_raw, str) or not jsonl_raw.strip():
            raise SystemExit("summary JSON does not contain jsonl_path")
        jsonl_path = Path(jsonl_raw)
    assert jsonl_path is not None

    records = _read_records(jsonl_path)
    summary = summarize_policy_search_records(records)
    scenario_counts: Counter[tuple[str, str]] = Counter()
    exclusions = []
    for record in records:
        exclusion = normalize_scenario_exclusion(record)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue
        failure_mode = classify_failure_mode(record)
        if failure_mode is None:
            continue
        scenario_counts[(failure_mode, str(record.get("scenario_id", "unknown")))] += 1

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_report.json"
    json_path.write_text(
        json.dumps(
            {
                "jsonl_path": str(jsonl_path),
                "summary": summary,
                "scenario_failure_counts": [
                    {"failure_mode": key[0], "scenario_id": key[1], "count": value}
                    for key, value in scenario_counts.most_common()
                ],
                "scenario_exclusions": exclusions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Policy Search Failure Report",
        "",
        f"- Source JSONL: `{jsonl_path}`",
        "",
        "## Failure Taxonomy",
        "",
        "| Failure Mode | Count |",
        "|---|---:|",
    ]
    lines.extend(
        [
            f"| raw_success_rate | {float(summary.get('success_rate', 0.0)):.4f} |",
            f"| raw_collision_rate | {float(summary.get('collision_rate', 0.0)):.4f} |",
        ]
    )
    adjusted_raw = summary.get("evidence_adjusted")
    adjusted = adjusted_raw if isinstance(adjusted_raw, dict) else {}
    lines.extend(
        [
            (
                "| evidence_adjusted_success_rate | "
                f"{float(adjusted.get('success_rate', 0.0)):.4f} |"
            ),
            (
                "| evidence_adjusted_collision_rate | "
                f"{float(adjusted.get('collision_rate', 0.0)):.4f} |"
            ),
            f"| excluded_episodes | {int(adjusted.get('excluded_episodes', 0))} |",
        ]
    )
    failure_counts_raw = summary.get("failure_mode_counts")
    failure_counts = failure_counts_raw if isinstance(failure_counts_raw, dict) else {}
    for key, value in sorted(failure_counts.items()):
        lines.append(f"| {key} | {int(value)} |")

    lines.extend(
        [
            "",
            "## Scenario Exclusions",
            "",
            "These rows are not classified as policy failures because they carry explicit exclusion "
            "metadata with evidence.",
            "",
            "| Scenario | Seed | Status | Reason | Evidence |",
            "|---|---:|---|---|---|",
        ]
    )
    for exclusion in exclusions:
        evidence = "; ".join(str(item) for item in exclusion.get("evidence", []))
        lines.append(
            "| "
            f"{exclusion.get('scenario_id', 'unknown')} | "
            f"{exclusion.get('seed', '')} | "
            f"{exclusion.get('status', '')} | "
            f"{exclusion.get('reason', '')} | "
            f"{evidence} |"
        )

    lines.extend(
        [
            "",
            "## Top Failure Scenarios",
            "",
            "| Failure Mode | Scenario | Count |",
            "|---|---|---:|",
        ]
    )
    for (failure_mode, scenario_id), count in scenario_counts.most_common(10):
        lines.append(f"| {failure_mode} | {scenario_id} | {count} |")
    md_path = output_dir / "failure_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
