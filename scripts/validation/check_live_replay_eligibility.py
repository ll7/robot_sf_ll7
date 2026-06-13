#!/usr/bin/env python3
"""Classify diagnostic evidence records for live-replay promotion eligibility.

This checker is a gatekeeping helper only. It does not run live replay and does
not promote diagnostic evidence to benchmark, dissertation, or paper evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "live_replay_eligibility.v1"
ELIGIBLE = "eligible"
MISSING_FIELDS = "missing-fields"
BLOCKED = "blocked"
DIAGNOSTIC_ONLY = "diagnostic-only"

REQUIRED_TEXT_PREREQUISITES = {
    "scenario": ("scenario_id", "scenario_path", "executable_scenario"),
    "policy_candidate": ("policy_candidate", "planner_id"),
    "perturbation_config": ("perturbation_config",),
    "output_report_path": ("output_report_path",),
}
BOUNDARY_MARKERS = (
    "diagnostic",
    "not benchmark",
    "not paper",
    "fail-closed",
    "promotion",
    "live replay",
)


def _text_value(record: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty string value among keys."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _truthy_bool(value: Any) -> bool:
    """Return True only for an explicit boolean true value."""
    return value is True


def _evidence_class(record: dict[str, Any]) -> str:
    """Return a normalized evidence class label."""
    raw = (
        record.get("evidence_class")
        or record.get("classification")
        or record.get("claim_scope")
        or ""
    )
    return str(raw).strip().lower().replace("_", "-")


def _has_live_replay_candidate(record: dict[str, Any]) -> bool:
    """Return whether the record asks to be considered for live replay."""
    if _truthy_bool(record.get("live_replay_candidate")):
        return True
    promotion_path = str(record.get("promotion_path") or "").lower()
    normalized = promotion_path.replace("_", " ").replace("-", " ")
    return normalized.strip() in {"live replay", "live replay promotion"}


def _safe_claim_boundary(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether claim boundary text is present and conservative."""
    boundary = _text_value(record, "claim_boundary")
    if boundary is None:
        return False, "missing_claim_boundary"
    lowered = boundary.lower()
    if any(marker in lowered for marker in BOUNDARY_MARKERS):
        return True, None
    return False, "unsafe_claim_boundary"


def _fixture_exists(record: dict[str, Any], *, base_dir: Path) -> tuple[bool, str | None]:
    """Return whether the fixture path is declared and exists."""
    fixture = _text_value(record, "fixture_path")
    if fixture is None:
        return False, "fixture_path"
    path = Path(fixture)
    if not path.is_absolute():
        path = base_dir / path
    if path.exists():
        return True, None
    return False, "fixture_path"


def _missing_prerequisites(record: dict[str, Any], *, base_dir: Path) -> list[str]:
    """Return missing live-replay promotion prerequisites."""
    missing: list[str] = []
    fixture_ok, fixture_missing = _fixture_exists(record, base_dir=base_dir)
    if not fixture_ok and fixture_missing is not None:
        missing.append(fixture_missing)

    for requirement, keys in REQUIRED_TEXT_PREREQUISITES.items():
        if _text_value(record, *keys) is None:
            missing.append(requirement)

    if not _truthy_bool(record.get("live_metrics_supported")):
        missing.append("live_metrics_supported")
    return missing


def classify_record(record: Any, *, base_dir: Path | None = None, index: int = 0) -> dict[str, Any]:
    """Classify one diagnostic evidence record."""
    base = (base_dir or Path.cwd()).resolve()
    if not isinstance(record, dict):
        return {
            "index": index,
            "status": BLOCKED,
            "missing_prerequisites": ["record_object"],
            "blocked_reasons": ["malformed_record"],
            "diagnostic_reasons": [],
        }

    blocked: list[str] = []
    diagnostic: list[str] = []
    missing = _missing_prerequisites(record, base_dir=base)

    boundary_ok, boundary_reason = _safe_claim_boundary(record)
    if not boundary_ok and boundary_reason is not None:
        blocked.append(boundary_reason)

    blocked_reason = _text_value(record, "blocked_reason")
    if blocked_reason is not None:
        blocked.append(blocked_reason)

    evidence_class = _evidence_class(record)
    if evidence_class in {"diagnostic-only", "diagnostic"} and not _has_live_replay_candidate(
        record
    ):
        diagnostic.append("record_declares_diagnostic_only_without_live_replay_candidate")

    if blocked:
        status = BLOCKED
    elif missing:
        status = MISSING_FIELDS
    elif diagnostic:
        status = DIAGNOSTIC_ONLY
    else:
        status = ELIGIBLE

    return {
        "index": index,
        "status": status,
        "missing_prerequisites": sorted(set(missing)),
        "blocked_reasons": blocked,
        "diagnostic_reasons": diagnostic,
        "fixture_path": record.get("fixture_path"),
        "policy_candidate": record.get("policy_candidate") or record.get("planner_id"),
        "claim_boundary": record.get("claim_boundary"),
    }


def load_records(path: Path) -> list[Any]:
    """Load a JSON object, list, or object with an evidence_records list."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("evidence_records"), list):
        return raw["evidence_records"]
    if isinstance(raw, list):
        return raw
    return [raw]


def evaluate_records(records: list[Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate records and return a machine-readable report."""
    classifications = [
        classify_record(record, base_dir=base_dir, index=index)
        for index, record in enumerate(records)
    ]
    counts: dict[str, int] = {}
    for row in classifications:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "live_replay_eligibility": {
            "schema_version": SCHEMA_VERSION,
            "classifications": classifications,
            "counts": counts,
            "claim_boundary": (
                "Eligibility means prerequisites for a future live-replay promotion check are "
                "present. It is not benchmark evidence, paper evidence, or live-replay proof."
            ),
        }
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON record, list, or evidence_records object.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the live-replay eligibility checker."""
    args = build_parser().parse_args(argv)
    report = evaluate_records(load_records(args.records), base_dir=args.base_dir)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    counts = report["live_replay_eligibility"]["counts"]
    total = sum(counts.values())
    return 0 if total > 0 and counts == {ELIGIBLE: total} else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
