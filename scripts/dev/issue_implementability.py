#!/usr/bin/env python3
"""Check whether one GitHub issue is safe for autonomous implementation claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from scripts.dev import gh_issue_rest, issue_claim

SCHEMA = "issue_implementability.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
READY_LABEL = "state:ready"

PARENT_LABELS = frozenset({"epic", "parent", "type:epic"})
HUMAN_DECISION_LABELS = frozenset(
    {
        "author-decision",
        "decision-required",
        "state:author-decision",
        "state:blocked-human-decision",
    }
)
COMPUTE_LABELS = frozenset(
    {
        "campaign",
        "needs-campaign",
        "resource:slurm",
        "routing:needs-compute",
        "slurm",
    }
)
EXTERNAL_LABELS = frozenset(
    {
        "resource:external-data",
        "state:blocked-external-input",
    }
)
WORKING_LABELS = frozenset({"state:running", "state:working"})
REVIEW_LABELS = frozenset({"needs-review", "state:review"})
BLOCKING_LABELS = frozenset(
    {
        "blocked",
        "duplicate",
        "evidence:blocked",
        "invalid",
        "needs-triage",
        "state:blocked",
        "state:hold",
        "state:parked",
        "wontfix",
    }
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "objective": (
        "ask",
        "current failure mode",
        "goal",
        "objective",
        "plain language summary",
        "problem",
        "purpose",
        "question",
        "research question",
        "summary",
    ),
    "scope": (
        "allowed paths",
        "forbidden work",
        "in scope",
        "non goals",
        "out of scope",
        "scope",
        "scope boundary",
    ),
    "inputs": (
        "affected files",
        "candidate paths",
        "canonical entry points",
        "current evidence",
        "exact source",
        "exact surface",
        "inputs",
        "prerequisites",
        "proposed implementation surface",
        "required changes",
        "required contract fields",
    ),
    "acceptance": (
        "acceptance criteria",
        "completion",
        "definition of done",
        "required outputs",
        "success metrics",
    ),
    "verification": (
        "proof",
        "testing",
        "validation",
        "validation proof",
        "validation testing",
        "verification",
    ),
}

HEADING_RE = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$")
LEADING_NUMBER_RE = re.compile(r"^\d+(?:[.)]|\s+-)\s*")
SPACE_RE = re.compile(r"\s+")
PARENT_TITLE_RE = re.compile(r"^\s*\[(?:epic|parent)\]", re.IGNORECASE)


def _normalize_heading(value: str) -> str:
    """Return a stable comparison form for one Markdown heading."""
    text = value.strip().lower()
    text = re.sub(r"[`*_]", "", text)
    text = LEADING_NUMBER_RE.sub("", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[/|:—–-]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def _heading_records(body: str) -> list[tuple[str, str]]:
    """Return non-empty Markdown sections while ignoring headings inside fences."""
    lines = body.splitlines(keepends=True)
    spans: list[tuple[int, int, str]] = []
    offset = 0
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            offset += len(line)
            continue
        if fence is None:
            match = HEADING_RE.match(line.rstrip("\r\n"))
            if match is not None:
                spans.append((offset, offset + len(line), _normalize_heading(match.group("title"))))
        offset += len(line)

    records: list[tuple[str, str]] = []
    for index, (_start, content_start, heading) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(body)
        content = body[content_start:next_start].strip()
        if content:
            records.append((heading, content))
    return records


def _heading_matches(heading: str, alias: str) -> bool:
    """Return whether a normalized heading identifies one contract section."""
    return heading == alias or heading.startswith(f"{alias} ")


def inspect_contract(body: str) -> dict[str, Any]:
    """Inspect required implementation-contract sections without inferring intent."""
    records = _heading_records(body)
    headings = sorted({heading for heading, _ in records})
    fields: dict[str, dict[str, Any]] = {}
    missing_fields: list[str] = []
    for field, aliases in FIELD_ALIASES.items():
        matched = sorted(
            {
                heading
                for heading, _ in records
                if any(_heading_matches(heading, alias) for alias in aliases)
            }
        )
        fields[field] = {"present": bool(matched), "matched_headings": matched}
        if not matched:
            missing_fields.append(field)
    return {
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "headings": headings,
        "fields": fields,
        "missing_fields": missing_fields,
        "complete": not missing_fields,
    }


def _normalize_labels(raw: Any) -> list[str]:
    """Normalize REST or offline label values to sorted names."""
    if not isinstance(raw, list):
        raise ValueError("labels must be a list")
    labels: list[str] = []
    for value in raw:
        if isinstance(value, str):
            name = value
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            name = value["name"]
        else:
            raise ValueError("each label must be a string or an object with a string name")
        name = name.strip()
        if name:
            labels.append(name)
    return sorted(set(labels))


def _normalize_assignees(raw: Any) -> list[str]:
    """Normalize REST or offline assignee values to sorted logins."""
    if not isinstance(raw, list):
        raise ValueError("assignees must be a list")
    assignees: list[str] = []
    for value in raw:
        if isinstance(value, str):
            login = value
        elif isinstance(value, dict) and isinstance(value.get("login"), str):
            login = value["login"]
        else:
            raise ValueError("each assignee must be a string or an object with a string login")
        login = login.strip()
        if login:
            assignees.append(login)
    return sorted(set(assignees))


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one issue payload."""
    number = issue.get("number")
    if type(number) is not int or number <= 0:
        raise ValueError("issue number must be a positive integer")
    title = issue.get("title")
    body = issue.get("body")
    state = issue.get("state")
    url = issue.get("url", "")
    if not isinstance(title, str):
        raise ValueError("issue title must be a string")
    if not isinstance(body, str):
        raise ValueError("issue body must be a string")
    if not isinstance(state, str):
        raise ValueError("issue state must be a string")
    if not isinstance(url, str):
        raise ValueError("issue url must be a string")
    return {
        "number": number,
        "title": title.strip(),
        "body": body,
        "state": state.strip().upper(),
        "url": url,
        "labels": _normalize_labels(issue.get("labels", [])),
        "assignees": _normalize_assignees(issue.get("assignees", [])),
    }


def _has_blocked_prefix(labels: set[str]) -> bool:
    """Return whether any label uses the explicit blocker namespace."""
    return any(label.startswith("blocked:") for label in labels)


def _pending_decision_heading(contract: dict[str, Any], labels: set[str]) -> bool:
    """Detect an unresolved decision heading unless a ruling label is present."""
    if "ruled" in labels or "domain-approved" in labels:
        return False
    decision_headings = {
        "decision required",
        "maintainer decision required",
        "required maintainer decision",
    }
    return bool(decision_headings & set(contract["headings"]))


def _classify_issue(
    normalized: dict[str, Any],
    claim: dict[str, Any],
    contract: dict[str, Any],
    labels: set[str],
) -> tuple[str, list[str]]:
    """Classify one normalized issue using the documented precedence order."""
    rules = [
        (
            normalized["state"] != "OPEN",
            "closed",
            f"issue state is {normalized['state'] or 'unknown'}",
        ),
        (claim.get("ok") is not True, "error", "claim state is unavailable"),
        (
            claim.get("claimed") is True,
            "already_claimed",
            "an atomic issue claim already exists",
        ),
        (
            bool(normalized["assignees"]),
            "assigned",
            "the issue already has an assignee",
        ),
        (
            bool(labels & PARENT_LABELS) or PARENT_TITLE_RE.match(normalized["title"]) is not None,
            "parent",
            "parent or epic issues are not implementation leaves",
        ),
        (
            bool(labels & HUMAN_DECISION_LABELS) or _pending_decision_heading(contract, labels),
            "human_decision",
            "a maintainer or author decision is required",
        ),
        (
            bool(labels & COMPUTE_LABELS),
            "needs_compute",
            "the issue is routed to compute or campaign execution",
        ),
        (
            bool(labels & EXTERNAL_LABELS),
            "blocked",
            "the issue requires external input",
        ),
        (
            bool(labels & WORKING_LABELS),
            "working",
            "the issue is already in an active work state",
        ),
        (
            bool(labels & REVIEW_LABELS),
            "review",
            "the issue is already in review",
        ),
        (
            bool(labels & BLOCKING_LABELS) or _has_blocked_prefix(labels),
            "blocked",
            "a blocking workflow label is present",
        ),
        (
            READY_LABEL not in labels,
            "needs_ready_label",
            f"required label {READY_LABEL!r} is absent",
        ),
        (
            not contract["complete"],
            "needs_spec",
            "missing implementation-contract fields: " + ", ".join(contract["missing_fields"]),
        ),
    ]
    for condition, classification, reason in rules:
        if condition:
            return classification, [reason]
    return "ready", ["issue state and execution contract permit claim admission"]


def evaluate_issue(issue: dict[str, Any], claim: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic, fail-closed issue implementability report."""
    normalized = normalize_issue(issue)
    contract = inspect_contract(normalized["body"])
    labels = set(normalized["labels"])
    classification, reasons = _classify_issue(normalized, claim, contract, labels)

    return {
        "schema": SCHEMA,
        "issue": {
            "number": normalized["number"],
            "title": normalized["title"],
            "state": normalized["state"],
            "url": normalized["url"],
            "labels": normalized["labels"],
            "assignees": normalized["assignees"],
        },
        "claim": {
            "ok": claim.get("ok"),
            "claimed": claim.get("claimed"),
            "claim_ref": claim.get("claim_ref"),
            "sha": claim.get("sha"),
        },
        "contract": contract,
        "classification": classification,
        "reasons": reasons,
        "ready": classification == "ready",
        "write_allowed": classification == "ready",
    }


def fetch_live_issue(number: int, *, repo: str) -> dict[str, Any]:
    """Fetch one issue through the canonical REST-backed normalized reader."""
    payload = gh_issue_rest.fetch_issue(number, repo=repo)
    if not isinstance(payload, dict):
        raise ValueError("issue reader returned a non-object payload")
    if payload.get("status") != "ok":
        raise RuntimeError(str(payload.get("error", "issue read failed")))
    return payload


def live_issue_report(number: int, *, repo: str, remote: str) -> dict[str, Any]:
    """Read one live issue and its atomic claim state, then evaluate it."""
    issue = fetch_live_issue(number, repo=repo)
    claim = issue_claim.status_issue(number, remote=remote)
    return evaluate_issue(issue, claim)


def _parse_claimed(value: str) -> dict[str, Any]:
    """Return a normalized offline claim-state fixture."""
    if value == "unknown":
        return {"ok": False, "claimed": None, "claim_ref": None, "sha": None}
    claimed = value == "true"
    return {
        "ok": True,
        "claimed": claimed,
        "claim_ref": "offline/claim" if claimed else None,
        "sha": "offline" if claimed else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int, help="Positive GitHub issue number.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument(
        "--remote", default=DEFAULT_REMOTE, help="Git remote used for claim status."
    )
    parser.add_argument("--body-file", help="Offline mode: read the issue body from this file.")
    parser.add_argument("--title", default="offline issue", help="Offline issue title.")
    parser.add_argument("--state", default="OPEN", help="Offline issue state.")
    parser.add_argument("--label", action="append", default=[], help="Offline label; repeatable.")
    parser.add_argument(
        "--assignee", action="append", default=[], help="Offline assignee; repeatable."
    )
    parser.add_argument(
        "--claimed",
        choices=("false", "true", "unknown"),
        default="false",
        help="Offline atomic-claim state.",
    )
    return parser


def _error_report(number: int, message: str) -> dict[str, Any]:
    """Return a stable fail-closed error payload."""
    return {
        "schema": SCHEMA,
        "issue": {"number": number},
        "classification": "error",
        "reasons": [message],
        "ready": False,
        "write_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    try:
        if args.issue <= 0:
            raise ValueError("issue number must be positive")
        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
            issue = {
                "number": args.issue,
                "title": args.title,
                "body": body,
                "state": args.state,
                "url": "",
                "labels": args.label,
                "assignees": args.assignee,
            }
            report = evaluate_issue(issue, _parse_claimed(args.claimed))
        else:
            report = live_issue_report(args.issue, repo=args.repo, remote=args.remote)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = _error_report(args.issue, str(exc))

    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("ready") is True:
        return 0
    if report.get("classification") == "error":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
