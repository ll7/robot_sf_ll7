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
_DIMENSION_DEFINITIONS = {
    "duplicate_competing_prs": {
        "definition": "query-defined count of work items classified as duplicate, competing, both, or neither; the report does not infer this from PR state",
        "buckets": {
            "no_duplicate_or_competing",
            "duplicate_pr",
            "competing_pr",
            "duplicate_and_competing",
        },
    },
    "post_merge_repairs": {
        "definition": "query-defined count of items that required a repair after merge versus items with no post-merge repair",
        "buckets": {"no_post_merge_repair", "post_merge_repair"},
    },
    "admitted_result_packets": {
        "definition": "query-defined count of items with an admitted result packet versus items without admitted packet evidence",
        "buckets": {"no_admitted_packet", "admitted_packet"},
    },
    "blocked_age_categories": {
        "definition": "query-defined age buckets for blocked items; unblocked items remain separate from blocked-age buckets",
        "buckets": {
            "not_blocked",
            "blocked_0_7_days",
            "blocked_8_30_days",
            "blocked_over_30_days",
        },
    },
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


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchYieldError(f"{field} must be a non-negative integer")
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
    _validate_dimensions(payload.get("dimensions"))
    return payload


def _validate_dimensions(value: Any) -> None:
    dimensions = _mapping(value, "dimensions")
    known_names = set(_DIMENSION_DEFINITIONS)
    names = set(dimensions)
    unknown_names = sorted(names - known_names)
    if unknown_names:
        raise ResearchYieldError(f"dimensions contain unsupported names: {unknown_names}")
    missing_names = sorted(known_names - names)
    if missing_names:
        raise ResearchYieldError(f"dimensions are missing required names: {missing_names}")
    for name, raw_dimension in dimensions.items():
        dimension = _mapping(raw_dimension, f"dimensions.{name}")
        unknown_fields = sorted(set(dimension) - {"query", "denominator", "buckets"})
        if unknown_fields:
            raise ResearchYieldError(
                f"dimensions.{name} contains unsupported fields: {unknown_fields}"
            )
        _text(dimension.get("query"), f"dimensions.{name}.query")
        denominator = _non_negative_int(
            dimension.get("denominator"), f"dimensions.{name}.denominator"
        )
        buckets = _mapping(dimension.get("buckets"), f"dimensions.{name}.buckets")
        allowed_buckets = _DIMENSION_DEFINITIONS[name]["buckets"]
        bucket_names = set(buckets)
        unknown_buckets = sorted(bucket_names - allowed_buckets)
        if unknown_buckets:
            raise ResearchYieldError(
                f"dimensions.{name}.buckets contain unsupported names: {unknown_buckets}"
            )
        missing_buckets = sorted(allowed_buckets - bucket_names)
        if missing_buckets:
            raise ResearchYieldError(
                f"dimensions.{name}.buckets are missing required names: {missing_buckets}"
            )
        bucket_total = sum(
            _non_negative_int(count, f"dimensions.{name}.buckets.{bucket}")
            for bucket, count in buckets.items()
        )
        if bucket_total != denominator:
            raise ResearchYieldError(
                f"dimensions.{name}.buckets sum to {bucket_total}, expected denominator {denominator}"
            )


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
    _validate_dimensions(payload.get("dimensions"))
    raw_dimensions = _mapping(payload.get("dimensions"), "dimensions")
    dimensions = {
        name: {
            "definition": _DIMENSION_DEFINITIONS[name]["definition"],
            "query": raw_dimension["query"],
            "denominator": raw_dimension["denominator"],
            "buckets": dict(sorted(raw_dimension["buckets"].items())),
        }
        for name, raw_dimension in sorted(raw_dimensions.items())
    }
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
        "dimensions": dimensions,
        "definitions": {
            "empirical_answers": "records where kind == empirical_result; status is reported without inferring admission from closure or merge",
            "infrastructure_throughput": "records where kind is infrastructure, preflight, or coordination",
            "failure_reasons": "explicit failure_reason values only; missing reasons are not reconstructed",
            "lag_days": "explicit non-negative day fields on the source records; unavailable values remain null",
            "dimensions": "explicit top-level query-defined dimensions from the frozen snapshot; unknown dimensions and bucket names are rejected",
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
    lines.extend(["", "## Query-Defined Dimensions", ""])
    for name, dimension in report["dimensions"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Definition: {dimension['definition']}")
        lines.append(f"- Query: `{dimension['query']}`")
        lines.append(f"- Denominator: `{dimension['denominator']}`")
        for bucket, count in dimension["buckets"].items():
            lines.append(f"- `{bucket}`: {count}")
        lines.append("")
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
