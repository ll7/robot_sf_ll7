#!/usr/bin/env python3
"""Route stale open-state findings into a fail-closed exact-fix review queue.

``open_state_label_hygiene.py`` deliberately stops at a verified merged-PR
reference.  This command consumes that report and records the evidence still
required before an issue may be closed or relabeled: a named symbol, the
failure signature, the failing file/line, a regression proof, and the exact
covering PR.  It never mutates GitHub state; a later maintainer-facing
issue-audit decision owns any disposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "open_state_label_hygiene.v1"
QUEUE_SCHEMA = "exact_fix_review_queue.v1"
EVIDENCE_SCHEMA = "exact_fix_evidence.v1"
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
REQUIRED_EVIDENCE_FIELDS = (
    "named_symbol",
    "failure_signature",
    "failing_file_line",
    "regression_proof",
    "current_main_sha",
)


class InputContractError(ValueError):
    """Raised when a report or evidence file cannot authorize routing."""


def _canonical_json(payload: object) -> bytes:
    """Serialize JSON deterministically for an auditable source digest."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(payload: object) -> str:
    """Return the SHA-256 digest of a normalized JSON payload."""
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _nonempty_text(value: object) -> str | None:
    """Return a trimmed non-empty string, otherwise ``None``."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _positive_int(value: object, *, field: str) -> int:
    """Validate a positive issue or pull-request number."""
    if isinstance(value, bool):
        raise InputContractError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise InputContractError(f"{field} must be a positive integer") from exc
    if number <= 0:
        raise InputContractError(f"{field} must be a positive integer")
    return number


def _validate_report(report: object) -> Mapping[str, Any]:
    """Validate the report-level fail-closed contract."""
    if not isinstance(report, Mapping):
        raise InputContractError("source report must be a JSON object")
    if report.get("schema") != REPORT_SCHEMA:
        raise InputContractError(f"source report schema must be {REPORT_SCHEMA}")
    if report.get("read_only") is not True:
        raise InputContractError("source report must declare read_only: true")
    if report.get("issue_writes") is not False or report.get("project_writes") is not False:
        raise InputContractError("source report must declare issue_writes/project_writes false")
    if report.get("complete_for_open_issues") is not True or report.get("truncated_any") is True:
        raise InputContractError(
            "source report is incomplete; exact-fix routing requires complete issue/timeline coverage"
        )
    issues = report.get("issues")
    if not isinstance(issues, list):
        raise InputContractError("source report issues must be a list")
    raw_count = report.get("candidate_count")
    if raw_count is not None and raw_count != len(issues):
        raise InputContractError("source report candidate_count does not match issues")
    return report


def _validate_covering_pr(raw_pr: object, *, issue_number: int) -> dict[str, Any]:
    """Validate one merged-PR reference before carrying it into a packet."""
    if not isinstance(raw_pr, Mapping):
        raise InputContractError(f"issue #{issue_number} contains a non-object merged PR")
    number = _positive_int(raw_pr.get("number"), field=f"issue #{issue_number} PR number")
    url = _nonempty_text(raw_pr.get("url"))
    merge_commit_sha = _nonempty_text(raw_pr.get("merge_commit_sha"))
    merged_at = _nonempty_text(raw_pr.get("merged_at"))
    if not url or not merge_commit_sha or not SHA_RE.fullmatch(merge_commit_sha) or not merged_at:
        raise InputContractError(
            f"issue #{issue_number} PR #{number} lacks a verified URL, merge time, or 40-hex merge SHA"
        )
    return {
        "number": number,
        "title": _nonempty_text(raw_pr.get("title")) or "",
        "url": url,
        "merged_at": merged_at,
        "merge_commit_sha": merge_commit_sha.lower(),
        "coverage_source": _nonempty_text(raw_pr.get("coverage_source"))
        or "open_state_label_hygiene",
    }


def _evidence_record(raw: object, *, issue_number: int) -> dict[str, str]:
    """Validate optional explicit exact-fix evidence for one issue."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise InputContractError(f"evidence for issue #{issue_number} must be an object")
    evidence: dict[str, str] = {}
    for field in REQUIRED_EVIDENCE_FIELDS:
        value = _nonempty_text(raw.get(field))
        if value:
            evidence[field] = value
    if "current_main_sha" in evidence and not SHA_RE.fullmatch(evidence["current_main_sha"]):
        raise InputContractError(
            f"evidence for issue #{issue_number} has an invalid current_main_sha"
        )
    covering_pr = raw.get("covering_pr")
    if covering_pr is not None:
        evidence["covering_pr"] = str(
            _positive_int(covering_pr, field=f"issue #{issue_number} covering_pr")
        )
    return evidence


def _guard_row(
    *,
    covering_prs: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, str],
) -> dict[str, Any]:
    """Build the explicit guard checklist carried by every review packet."""
    checks: dict[str, dict[str, Any]] = {
        "covering_pr": {
            "status": "verified" if covering_prs else "missing",
            "observed": [int(pr["number"]) for pr in covering_prs],
            "required": "issue-timeline merged PR plus merge commit",
        }
    }
    for field in REQUIRED_EVIDENCE_FIELDS:
        checks[field] = {
            "status": "provided" if evidence.get(field) else "missing",
            "required": "explicit current-main exact-fix evidence",
        }
    return checks


def _evidence_index(evidence: object) -> dict[int, Mapping[str, Any]]:
    """Normalize an optional exact-fix evidence manifest by issue number."""
    if evidence is None:
        return {}
    if not isinstance(evidence, Mapping) or evidence.get("schema") != EVIDENCE_SCHEMA:
        raise InputContractError(f"evidence manifest schema must be {EVIDENCE_SCHEMA}")
    rows = evidence.get("issues")
    if not isinstance(rows, list):
        raise InputContractError("evidence manifest issues must be a list")
    index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise InputContractError("evidence manifest issue rows must be objects")
        number = _positive_int(row.get("number"), field="evidence issue number")
        if number in index:
            raise InputContractError(f"evidence manifest repeats issue #{number}")
        index[number] = row
    return index


def build_review_queue(
    report: Mapping[str, Any],
    *,
    evidence: object | None = None,
) -> dict[str, Any]:
    """Build a no-write exact-fix queue from a complete hygiene report."""
    report = _validate_report(report)
    evidence_by_issue = _evidence_index(evidence)
    source_issue_numbers = {
        _positive_int(raw_issue.get("number"), field="source issue number")
        for raw_issue in report["issues"]
        if isinstance(raw_issue, Mapping)
    }
    unknown_evidence = sorted(set(evidence_by_issue) - source_issue_numbers)
    if unknown_evidence:
        numbers = ", ".join(f"#{number}" for number in unknown_evidence)
        raise InputContractError(
            f"evidence manifest names issues absent from the source report: {numbers}"
        )
    candidates: list[dict[str, Any]] = []
    for raw_issue in report["issues"]:
        if not isinstance(raw_issue, Mapping):
            raise InputContractError("source report issue rows must be objects")
        issue_number = _positive_int(raw_issue.get("number"), field="source issue number")
        if raw_issue.get("classification") != "merged_reference_needs_exact_fix_review":
            raise InputContractError(f"issue #{issue_number} is not an exact-fix hygiene candidate")
        raw_prs = raw_issue.get("merged_prs")
        if not isinstance(raw_prs, list) or not raw_prs:
            raise InputContractError(f"issue #{issue_number} has no verified merged PR reference")
        covering_prs = [
            _validate_covering_pr(raw_pr, issue_number=issue_number) for raw_pr in raw_prs
        ]
        record = _evidence_record(
            evidence_by_issue.get(issue_number),
            issue_number=issue_number,
        )
        if "covering_pr" in record and int(record["covering_pr"]) not in {
            int(pr["number"]) for pr in covering_prs
        }:
            raise InputContractError(
                f"evidence for issue #{issue_number} names a PR absent from the source report"
            )
        guard = _guard_row(covering_prs=covering_prs, evidence=record)
        missing = [field for field, check in guard.items() if check["status"] == "missing"]
        classification = (
            "ready_for_manual_exact_fix_review" if not missing else "needs_exact_fix_evidence"
        )
        candidates.append(
            {
                "issue": f"#{issue_number}",
                "number": issue_number,
                "title": _nonempty_text(raw_issue.get("title")) or "",
                "url": _nonempty_text(raw_issue.get("url")) or "",
                "state": "open",
                "active_labels": sorted(
                    str(label)
                    for label in raw_issue.get("active_labels", [])
                    if isinstance(label, str)
                ),
                "classification": classification,
                "question_source": "open_state_label_hygiene report plus exact-fix evidence manifest",
                "covering_prs": covering_prs,
                "exact_fix_guard": guard,
                "missing_evidence": missing,
                "evidence": record,
                "recommended_action": "present_to_issue_audit_for_manual_disposition",
                "safe_mutations": [],
            }
        )
    candidates.sort(key=lambda row: int(row["number"]))
    ready_count = sum(
        row["classification"] == "ready_for_manual_exact_fix_review" for row in candidates
    )
    missing_count = len(candidates) - ready_count
    source_digest = _digest(report)
    return {
        "schema": QUEUE_SCHEMA,
        "repo": str(report.get("repo") or ""),
        "source": {
            "schema": REPORT_SCHEMA,
            "report_digest": source_digest,
            "candidate_count": len(candidates),
            "complete_for_open_issues": True,
        },
        "route_complete": True,
        "disposition_authorized": False,
        "read_only": True,
        "issue_writes": False,
        "project_writes": False,
        "claim_boundary": (
            "A merged PR reference is not proof of an exact fix. No issue is closed or relabeled "
            "until a maintainer-facing issue-audit review verifies the guard."
        ),
        "required_exact_fix_evidence": list(REQUIRED_EVIDENCE_FIELDS),
        "candidates": candidates,
        "pending_decisions": [
            {
                "issue": row["issue"],
                "number": row["number"],
                "title": row["title"],
                "url": row["url"],
                "state": row["state"],
                "labels": row["active_labels"],
                "classification": row["classification"],
                "decision_required": True,
                "question_source": row["question_source"],
                "blocking_evidence": "; ".join(row["missing_evidence"])
                or "exact-fix evidence supplied; manual review still required",
                "evidence_sources": row["covering_prs"],
                "documented_options": [],
                "safe_mutations_applied": [],
            }
            for row in candidates
        ],
        "counts": {
            "candidates": len(candidates),
            "ready_for_manual_review": ready_count,
            "needs_exact_fix_evidence": missing_count,
        },
    }


def _load_json(path: Path) -> object:
    """Load one JSON file with a contextual error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputContractError(f"could not read JSON input {path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    """Build the route CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--evidence-file",
        type=Path,
        help=f"optional JSON manifest with schema {EVIDENCE_SCHEMA}",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Route a complete report and fail closed on malformed or partial input."""
    args = _parser().parse_args(argv)
    try:
        report = _load_json(args.report)
        evidence = _load_json(args.evidence_file) if args.evidence_file else None
        queue = build_review_queue(report, evidence=evidence)
    except InputContractError as exc:
        queue = {
            "schema": QUEUE_SCHEMA,
            "route_complete": False,
            "disposition_authorized": False,
            "read_only": True,
            "issue_writes": False,
            "project_writes": False,
            "error": str(exc),
        }
        rendered = json.dumps(queue, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 2
    rendered = json.dumps(queue, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
