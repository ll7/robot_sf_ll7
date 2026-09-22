#!/usr/bin/env python3
"""Validate and reconcile protected PR-readiness artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pr_ready_artifact_consistency.v1"
TERMINATION_SCHEMA = "pr_ready_termination.v1"

_STATUS_ALIASES = {
    "complete": "passed",
    "completed": "passed",
    "fail": "failed",
    "failure": "failed",
    "in_progress": "interim",
    "interrupted": "terminated",
    "nonzero": "failed",
    "pass": "passed",
    "success": "passed",
    "timeout": "terminated",
}
_KNOWN_STATUSES = frozenset({"failed", "interim", "passed", "pending", "terminated"})
_STATUS_VALUE = r"(?:failed|failure|interim|in[_ -]?progress|passed?|pending|success|terminated|interrupted|timeout|complete[d]?)"
_EXPLICIT_STATUS_RE = re.compile(
    rf"(?im)^\s*(?:pr\s+)?readiness\s+(?:result|status|outcome)\s*[:=]\s*`?({_STATUS_VALUE})\b"
)
_GENERIC_STATUS_RE = re.compile(
    rf"(?im)^\s*(?:result|status|outcome)\s*[:=]\s*`?({_STATUS_VALUE})\b"
)
_TERMINATION_MARKERS = (
    re.compile(r"(?im)\bpr\s+readiness\s+received\s+sig(?:term|int)\b"),
    re.compile(r"(?im)\bpr_ready_termination\.v\d+\b"),
    re.compile(r"(?im)\btermination\s+receipt\b"),
    re.compile(r"(?im)\bexit\s+(?:code|status)\s+14[3-9]\b"),
)
_PASS_MARKERS = (
    re.compile(r"(?im)^\s*===\s*pr\s+readiness[^\n]*\bpassed\b"),
    re.compile(r"(?im)\b(?:pr\s+)?readiness\s+(?:run\s+)?passed\b"),
    re.compile(r"(?im)\ball\s+checks\s+passed\b"),
)
_FAILURE_MARKERS = (
    re.compile(r"(?im)^\s*===\s*pr\s+readiness[^\n]*\b(?:failed|failure)\b"),
    re.compile(r"(?im)\b(?:pr\s+)?readiness\s+(?:run\s+)?failed\b"),
)


def _normalize_status(value: Any) -> str | None:
    """Map one explicit readiness status to the versioned vocabulary."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace(" ", "_")
    normalized = _STATUS_ALIASES.get(normalized, normalized)
    return normalized if normalized in _KNOWN_STATUSES else None


def _read_text(path: Path) -> tuple[str | None, str | None]:
    """Read a text artifact, preserving strict decoding failures."""
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeError) as exc:
        return None, f"artifact_unreadable:{path.name}:{exc}"


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load one object from byte zero; human prefixes and non-objects fail closed."""
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"json_malformed:{path.name}:{exc}"
    if not isinstance(value, dict):
        return None, f"json_root_not_object:{path.name}"
    return value, None


def _machine_observation(path: Path) -> tuple[str | None, list[str]]:
    """Extract a readiness state from one machine-readable artifact."""
    payload, error = load_json_object(path)
    if error is not None or payload is None:
        return None, [error or f"json_malformed:{path.name}"]
    if payload.get("schema") == TERMINATION_SCHEMA:
        return "terminated", []
    status = _normalize_status(payload.get("status"))
    if status is None:
        return None, [f"machine_status_missing:{path.name}"]
    return status, []


def _text_statuses(text: str) -> set[str]:
    """Collect explicit and legacy readiness status markers from text."""
    statuses = {
        status
        for match in _EXPLICIT_STATUS_RE.finditer(text)
        if (status := _normalize_status(match.group(1))) is not None
    }
    statuses.update(
        status
        for match in _GENERIC_STATUS_RE.finditer(text)
        if (status := _normalize_status(match.group(1))) is not None
    )
    if any(pattern.search(text) for pattern in _TERMINATION_MARKERS):
        statuses.add("terminated")
    if any(pattern.search(text) for pattern in _PASS_MARKERS):
        statuses.add("passed")
    if any(pattern.search(text) for pattern in _FAILURE_MARKERS):
        statuses.add("failed")
    return statuses


def _text_observation(path: Path, *, source: str) -> tuple[str | None, list[str]]:
    """Extract one unambiguous readiness state from a human/log artifact."""
    text, error = _read_text(path)
    if error is not None or text is None:
        return None, [error or f"artifact_unreadable:{path.name}"]
    statuses = _text_statuses(text)
    if not statuses:
        return None, [f"{source}_status_missing:{path.name}"]
    if len(statuses) != 1:
        return None, [f"{source}_status_conflict:{path.name}"]
    return next(iter(statuses)), []


def reconcile_readiness_artifacts(
    machine_json: list[Path],
    *,
    human_summary: Path | None = None,
    command_log: Path | None = None,
) -> dict[str, Any]:
    """Reconcile machine, human, and command-log readiness evidence fail-closed."""
    observations: list[dict[str, str]] = []
    reasons: list[str] = []
    if not machine_json:
        reasons.append("machine_artifact_missing")
    for path in machine_json:
        status, errors = _machine_observation(path)
        reasons.extend(errors)
        if status is not None:
            observations.append({"source": str(path), "status": status})
    for path, source in ((human_summary, "human_summary"), (command_log, "command_log")):
        if path is None:
            continue
        status, errors = _text_observation(path, source=source)
        reasons.extend(errors)
        if status is not None:
            observations.append({"source": str(path), "status": status})

    statuses = {item["status"] for item in observations}
    if len(statuses) > 1:
        reasons.append("readiness_status_disagreement")
        status = "conflict"
    elif not observations:
        status = "unavailable"
    else:
        status = next(iter(statuses))
    if reasons and status == "passed":
        status = "conflict"
    return {
        "schema": SCHEMA_VERSION,
        "status": status,
        "passed": status == "passed" and not reasons,
        "reason_codes": sorted(set(reasons)),
        "observations": observations,
    }


def _parser() -> argparse.ArgumentParser:
    """Build the strict artifact-reconciliation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-json", action="append", type=Path, required=True)
    parser.add_argument("--human-summary", type=Path)
    parser.add_argument("--command-log", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate artifacts and emit JSON without a human prefix."""
    args = _parser().parse_args(argv)
    result = reconcile_readiness_artifacts(
        args.machine_json,
        human_summary=args.human_summary,
        command_log=args.command_log,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
