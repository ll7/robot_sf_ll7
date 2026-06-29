#!/usr/bin/env python3
"""Compare frozen before/after EpisodeEventLedger.v1 JSONL exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.frozen_trace_reconciliation import (
    build_frozen_trace_reconciliation_report,
    build_missing_frozen_trace_export_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-ledgers",
        required=True,
        type=Path,
        help="Frozen before JSONL rows containing EpisodeEventLedger.v1 payloads.",
    )
    parser.add_argument(
        "--new-ledgers",
        required=True,
        type=Path,
        help="Frozen after JSONL rows containing EpisodeEventLedger.v1 payloads.",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help=(
            "Optional JSON list, or object with an artifacts list, declaring claim/table/figure "
            "consumes_event_fields entries."
        ),
    )
    parser.add_argument("--old-label", default="old", help="Label for the old frozen input.")
    parser.add_argument("--new-label", default="new", help="Label for the new frozen input.")
    parser.add_argument(
        "--out", required=True, type=Path, help="Output reconciliation report JSON."
    )
    parser.add_argument(
        "--diagnose-missing-exports",
        action="store_true",
        help=(
            "Write a diagnostic blocker report instead of reading inputs when one or both "
            "expected durable event-ledger exports are absent."
        ),
    )
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON object lines from ``path``."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {err}") from err
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    return rows


def _read_artifact_manifest(path: Path | None) -> list[dict[str, Any]]:
    """Read an optional affected artifact manifest."""

    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON in artifact manifest {path}: {err}") from err
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else payload
    if not isinstance(artifacts, list):
        raise ValueError("artifact manifest must be a JSON list or object with an artifacts list")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"artifact manifest entry {index} must be a JSON object")
    return artifacts


def _missing_export_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    """Return missing old/new durable export inputs for diagnostic reporting."""

    missing: list[dict[str, str]] = []
    for label, path in (
        (args.old_label, args.old_ledgers),
        (args.new_label, args.new_ledgers),
    ):
        if not path.exists():
            missing.append(
                {
                    "label": str(label),
                    "path": str(path),
                    "required_contract": (
                        "JSONL rows carrying EpisodeEventLedger.v1 exact_events and "
                        "surrogate_events"
                    ),
                }
            )
    return missing


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    missing_exports = _missing_export_inputs(args)
    if args.diagnose_missing_exports and missing_exports:
        report = build_missing_frozen_trace_export_report(
            missing_exports=missing_exports,
            artifact_manifest=_read_artifact_manifest(args.artifact_manifest),
            old_label=args.old_label,
            new_label=args.new_label,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "wrote frozen event-ledger missing-export diagnostic "
            f"{args.out} ({len(missing_exports)} missing exports)"
        )
        return 0

    report = build_frozen_trace_reconciliation_report(
        _read_jsonl(args.old_ledgers),
        _read_jsonl(args.new_ledgers),
        artifact_manifest=_read_artifact_manifest(args.artifact_manifest),
        old_label=args.old_label,
        new_label=args.new_label,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        "wrote frozen event-ledger reconciliation report "
        f"{args.out} ({summary['changed_row_count']} changed rows, "
        f"{summary['unchanged_row_count']} unchanged rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
