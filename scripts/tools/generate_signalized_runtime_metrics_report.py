"""Generate runtime signalized crossing metrics evidence for issue #2799."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2799_signalized_runtime")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def _row_type(metrics: dict[str, Any]) -> str:
    evidence = metrics.get("signal_metrics_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    state = str(evidence.get("state", "unavailable"))
    exclusion = str(evidence.get("exclusion_reason", ""))
    denominator = int(metrics.get("signal_metrics_denominator", 0) or 0)
    if state == "planner_observable" and not exclusion and denominator > 0:
        red_violations = int(metrics.get("signal_red_phase_violations", 0) or 0)
        red_crossings = int(metrics.get("signal_stop_line_crossings_under_red", 0) or 0)
        return "red_required_stop" if red_violations or red_crossings else "green_proceed"
    if state == "unavailable":
        return "unavailable_no_claim"
    return "proxy_only_denominator_excluded"


def _row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    evidence = metrics.get("signal_metrics_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    denominator = int(metrics.get("signal_metrics_denominator", 0) or 0)
    planner_observable = (
        evidence.get("state") == "planner_observable"
        and not evidence.get("exclusion_reason")
        and denominator > 0
    )
    row_type = _row_type(metrics)
    min_dist = metrics.get("signal_min_distance_to_stop_line_before_crossing_m")
    delay = metrics.get("signal_delay_after_green_onset_s")
    return {
        "episode_id": record.get("episode_id", "unknown"),
        "scenario_id": record.get("scenario_id", "unknown"),
        "status": record.get("status", "unknown"),
        "row_type": row_type,
        "planner_observable": planner_observable,
        "benchmark_evidence": planner_observable,
        "signal_compliance_eligible": planner_observable,
        "signal_metrics_denominator": denominator,
        "signal_unavailable_exclusion_count": int(
            metrics.get("signal_unavailable_exclusion_count", 0) or 0
        ),
        "exclusion_reason": str(evidence.get("exclusion_reason", "")),
        "stop_line_behaviour": {
            "crossed_under_red": bool(
                int(metrics.get("signal_stop_line_crossings_under_red", 0) or 0)
            ),
            "red_violation_count": int(metrics.get("signal_red_phase_violations", 0) or 0),
            "min_distance_m": min_dist,
        },
        "pedestrian_conflict": {
            "count": int(
                metrics.get("signal_pedestrian_conflict_during_legal_crossing_count", 0) or 0
            ),
            "label": (
                "conflict_detected"
                if int(
                    metrics.get("signal_pedestrian_conflict_during_legal_crossing_count", 0) or 0
                )
                else "no_conflict"
                if planner_observable
                else "not_applicable"
            ),
            "eligible": planner_observable,
        },
        "delay_after_green_onset_s": delay,
        "signal_metrics_evidence": evidence,
    }


def _finite_or_na(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.2f}" if np.isfinite(number) else "N/A"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| episode_id | scenario_id | row_type | eligible | denominator | exclusion_reason | crossed_red | min_dist_m | ped_conflicts | delay_green_s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        stop = row["stop_line_behaviour"]
        ped = row["pedestrian_conflict"]
        lines.append(
            f"| {row['episode_id']} "
            f"| {row['scenario_id']} "
            f"| {row['row_type']} "
            f"| {str(row['signal_compliance_eligible']).lower()} "
            f"| {row['signal_metrics_denominator']} "
            f"| {row['exclusion_reason'] or '-'} "
            f"| {str(stop['crossed_under_red']).lower()} "
            f"| {_finite_or_na(stop['min_distance_m'])} "
            f"| {ped['count']} "
            f"| {_finite_or_na(row['delay_after_green_onset_s'])} |"
        )
    return "\n".join(lines)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observable = [row for row in rows if row["signal_compliance_eligible"]]
    excluded = [row for row in rows if not row["signal_compliance_eligible"]]
    row_types = sorted({str(row["row_type"]) for row in rows})
    return {
        "issue": 2799,
        "title": "Signalized crossing simulator-backed runtime metrics evidence",
        "claim_boundary": (
            "Runtime evidence for denominator and exclusion semantics. Compliance claims require "
            "at least one planner_observable benchmark-evidence row with denominator > 0."
        ),
        "episodes_source": "worktree-local episodes.jsonl generated by the README reproduction command",
        "source_command": "see README.md fenced reproduction command",
        "total_rows": len(rows),
        "row_types_present": row_types,
        "observable_count": len(observable),
        "excluded_count": len(excluded),
        "eligible_rows": [
            {
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "row_type": row["row_type"],
                "planner_observable": row["planner_observable"],
                "benchmark_evidence": row["benchmark_evidence"],
                "signal_metrics_denominator": row["signal_metrics_denominator"],
            }
            for row in observable
        ],
        "excluded_rows": [
            {
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "row_type": row["row_type"],
                "planner_observable": row["planner_observable"],
                "benchmark_evidence": row["benchmark_evidence"],
                "signal_metrics_denominator": row["signal_metrics_denominator"],
                "exclusion_reason": row["exclusion_reason"],
            }
            for row in excluded
        ],
        "has_observable_runtime_evidence": bool(observable),
        "excluded_denominator_zero": all(
            row["signal_metrics_denominator"] == 0 for row in excluded
        ),
        "all_required_runtime_row_classes_present": {
            "red_required_stop": "red_required_stop" in row_types,
            "green_proceed": "green_proceed" in row_types,
            "unavailable_no_claim": "unavailable_no_claim" in row_types,
            "proxy_only_denominator_excluded": "proxy_only_denominator_excluded" in row_types,
        },
    }


def _report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return f"""# Issue #2799 Signalized Crossing Runtime Metrics Evidence (2026-06-13)

Issue: #2799

Claim boundary: {summary["claim_boundary"]}

## Result

- Total runtime rows: {summary["total_rows"]}
- Observable denominator rows: {summary["observable_count"]}
- Excluded rows: {summary["excluded_count"]}
- Row types present: {", ".join(summary["row_types_present"])}
- Source episodes: {summary["episodes_source"]}.

## Rows

{_markdown_table(rows)}

## Interpretation

Rows with `planner_observable=true`, `benchmark_evidence=true`, and
`signal_metrics_denominator > 0` are runtime benchmark-evidence candidates for denominator
semantics. Excluded rows remain non-claim rows.
"""


def _readme(source_command: str) -> str:
    return f"""# Issue #2799 Signalized Crossing Runtime Evidence (2026-06-13)

This bundle preserves simulator-backed runtime evidence for signalized-crossing metric
denominator and exclusion semantics.

## Reproduction

```bash
{source_command}
```

Then regenerate the bundle with:

```bash
uv run python scripts/tools/generate_signalized_runtime_metrics_report.py \\
  --episodes-jsonl <campaign-root>/runs/goal__differential_drive/episodes.jsonl
```

## Files

- [summary.json](summary.json): machine-readable runtime row summary.
- [report.md](report.md): human-readable runtime row table and interpretation.
"""


def generate(
    *,
    episodes_jsonl: Path,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    source_command: str,
) -> dict[str, Path]:
    """Generate the durable signalized runtime evidence bundle."""
    records = _read_jsonl(episodes_jsonl)
    rows = [_row(record) for record in records]
    summary = _summary(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_report(summary, rows), encoding="utf-8")
    (output_dir / "README.md").write_text(_readme(source_command), encoding="utf-8")
    return {
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "readme": output_dir / "README.md",
    }


def main() -> int:
    """CLI entry point for signalized runtime evidence generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-command",
        default=(
            "uv run python scripts/tools/run_camera_ready_benchmark.py "
            "--config configs/benchmarks/signalized_runtime_smoke_issue_2799.yaml "
            "--output-root output/benchmarks/issue_2799_signalized_runtime "
            "--label issue_2799_signalized_runtime --skip-publication-bundle"
        ),
    )
    args = parser.parse_args()
    paths = generate(
        episodes_jsonl=args.episodes_jsonl,
        output_dir=args.output_dir,
        source_command=args.source_command,
    )
    print(json.dumps({key: path.as_posix() for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
