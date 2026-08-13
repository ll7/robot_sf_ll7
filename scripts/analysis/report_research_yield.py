#!/usr/bin/env python3
"""Build a source-backed research-yield report from a frozen JSON snapshot.

The report separates empirical answers from infrastructure and preflight
throughput. It never infers scientific value from issue closure or pull-request
merge state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA = "research_yield_snapshot.v1"
REPORT_SCHEMA = "research_yield_report.v1"
_KINDS = {"empirical_result", "infrastructure", "preflight", "coordination"}
_STATUSES = {
    "authorized",
    "launched",
    "completed",
    "admitted",
    "inconclusive",
    "invalid",
    "blocked",
    "not_started",
}


class ResearchYieldError(ValueError):
    """Raised when a research-yield snapshot is incomplete or malformed."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchYieldError(f"{field} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchYieldError(f"{field} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load and structurally validate a frozen research-yield snapshot."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchYieldError(f"cannot load snapshot {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ResearchYieldError(f"snapshot must declare {SNAPSHOT_SCHEMA}")
    window = _mapping(payload.get("window"), "window")
    _text(window.get("start"), "window.start")
    _text(window.get("end"), "window.end")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ResearchYieldError("records must be a list")
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"records[{index}]")
        _text(record.get("id"), f"records[{index}].id")
        kind = _text(record.get("kind"), f"records[{index}].kind")
        status = _text(record.get("status"), f"records[{index}].status")
        if kind not in _KINDS:
            raise ResearchYieldError(f"records[{index}].kind is unsupported: {kind}")
        if status not in _STATUSES:
            raise ResearchYieldError(f"records[{index}].status is unsupported: {status}")
    return payload


def _lag_summary(records: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ResearchYieldError(f"{field} must contain non-negative numeric values")
        values.append(float(value))
    return {
        "n": len(values),
        "median_days": statistics.median(values) if values else None,
        "values_available": bool(values),
    }


def build_research_yield_report(
    snapshot: Mapping[str, Any], *, source_path: Path | None = None
) -> dict[str, Any]:
    """Aggregate a validated snapshot without collapsing distinct yield dimensions."""
    payload = dict(snapshot)
    if payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ResearchYieldError(f"snapshot must declare {SNAPSHOT_SCHEMA}")
    records = [_mapping(record, "record") for record in payload.get("records", [])]
    window = _mapping(payload.get("window"), "window")
    by_kind_status = Counter(f"{record['kind']}:{record['status']}" for record in records)
    by_kind = Counter(str(record["kind"]) for record in records)
    by_status = Counter(str(record["status"]) for record in records)
    failure_reasons = Counter(
        str(record["failure_reason"]) for record in records if record.get("failure_reason")
    )
    empirical = [record for record in records if record["kind"] == "empirical_result"]
    infrastructure = [
        record
        for record in records
        if record["kind"] in {"infrastructure", "preflight", "coordination"}
    ]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "window": {"start": window["start"], "end": window["end"]},
        "records_total": len(records),
        "by_kind": dict(sorted(by_kind.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_kind_status": dict(sorted(by_kind_status.items())),
        "empirical_answers": {
            "records": len(empirical),
            "statuses": dict(
                sorted(Counter(str(record["status"]) for record in empirical).items())
            ),
        },
        "infrastructure_throughput": {
            "records": len(infrastructure),
            "statuses": dict(
                sorted(Counter(str(record["status"]) for record in infrastructure).items())
            ),
        },
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "lag_days": {
            "approval_to_first_result": _lag_summary(records, "approval_to_first_result_days"),
            "result_to_package": _lag_summary(records, "result_to_package_days"),
        },
        "definitions": {
            "empirical_answers": "records where kind == empirical_result; status is reported without inferring admission from closure or merge",
            "infrastructure_throughput": "records where kind is infrastructure, preflight, or coordination",
            "failure_reasons": "explicit failure_reason values only; missing reasons are not reconstructed",
            "lag_days": "explicit non-negative day fields on the source records; unavailable values remain null",
        },
    }
    if source_path is not None:
        report["source_snapshot"] = {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        }
    else:
        report["source_snapshot"] = {"path": None, "sha256": None}
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable report while preserving source definitions."""
    lines = [
        "# Research Yield Report",
        "",
        f"- Window: `{report['window']['start']}` to `{report['window']['end']}`",
        f"- Records: `{report['records_total']}`",
        f"- Source snapshot: `{report['source_snapshot']['path']}`",
        "",
        "## Empirical Answers",
        "",
        f"- Records: `{report['empirical_answers']['records']}`",
    ]
    for status, count in report["empirical_answers"]["statuses"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Infrastructure Throughput",
            "",
            f"- Records: `{report['infrastructure_throughput']['records']}`",
        ]
    )
    for status, count in report["infrastructure_throughput"]["statuses"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Definitions", ""])
    for name, definition in report["definitions"].items():
        lines.append(f"- **{name}:** {definition}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-backed research-yield report from a frozen JSON snapshot."
    )
    parser.add_argument("snapshot", type=Path, help="Versioned research_yield_snapshot.v1 JSON.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    parser.add_argument(
        "--markdown-output", type=Path, help="Write a Markdown report to this path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the research-yield report CLI."""
    args = _parse_args(argv)
    try:
        snapshot = load_snapshot(args.snapshot)
        report = build_research_yield_report(snapshot, source_path=args.snapshot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if args.markdown_output:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, ResearchYieldError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
