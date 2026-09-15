#!/usr/bin/env python3
"""Audit (and, with explicit confirmation, migrate) native GitHub issue relationships.

Why this exists
---------------
Issue bodies carry a canonical ``## Relationships`` mirror block, but the
native GitHub parent/sub-issue and blocked-by links are what the UI and API
surface. The relationship guide (``docs/context/issue_relationships.md``)
names this helper; it parses only that block, rejects ambiguous, cross-repo,
or conflicting links, writes native parent/dependency relationships through
current GitHub REST endpoints only after explicit ``RELATIONSHIP_MIGRATION``
confirmation, and verifies read-back.

Read model (all calls bounded to 30 seconds, fail closed on transport errors)
----
- mirror: REST issue body ``## Relationships`` block.
- native parent/children: GraphQL ``issue.parent`` / ``issue.subIssues``
  (the REST sub-issue list endpoint is unavailable on this repository).
- native blocked-by/blocking: REST ``issues/{n}/dependencies/blocked_by``
  and ``.../blocking`` list endpoints.

Write model (``--apply --confirm RELATIONSHIP_MIGRATION`` only)
--------------------------------------------------------------
Additive migration only: set a missing native parent, add missing native
blocked-by links. ``Blocking:`` entries are owned by the other issue's
``Blocked by`` row and are reported, never written. ``Relates to:`` is
informational and never written. Anything requiring deletion, disambiguation,
or a cross-repository link is refused with a stable reason.

CLI
---
::

    python scripts/dev/audit_issue_relationships.py [--repo <owner/repo>] [--json] <issue>
    python scripts/dev/audit_issue_relationships.py [--repo <owner/repo>] [--json] \\
        --apply --confirm RELATIONSHIP_MIGRATION <issue>

The default mode is a read-only audit. ``--apply`` without the exact
confirmation token is an error (exit 2). Any failed write or read-back
mismatch fails closed (exit 1) with a machine-readable reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Direct execution must resolve this checkout's transport helpers ahead of
    # any competing checkout or editable installation on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev._gh_rest import parse_json as _parse_json
from scripts.dev._gh_rest import run_gh_api as _gh_api
from scripts.dev._gh_rest import run_gh_command as _gh

DEFAULT_REPO = "ll7/robot_sf_ll7"
APPLY_CONFIRM_TOKEN = "RELATIONSHIP_MIGRATION"
GH_TIMEOUT = 30

_MIRROR_HEADING_RE = re.compile(r"^##\s+Relationships\s*$", re.IGNORECASE | re.MULTILINE)
_ROW_RE = re.compile(r"^\s*-\s*(Parent issue|Blocked by|Blocking|Relates to)\s*:\s*(.*?)\s*$")
_REF_RE = re.compile(r"#(\d+)")
_URL_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/(issues|pull)/(\d+)"
)
_NONE_RE = re.compile(r"^none\b", re.IGNORECASE)
_AMBIGUOUS_TOKEN_RE = re.compile(r"\b(tbd|todo|unknown|n/a|na|tba|later|pending)\b", re.IGNORECASE)

_PARENT_QUERY = """query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      parent { number }
      subIssues(first: 100) { nodes { number } }
    }
  }
}"""


@dataclass(frozen=True)
class RelationshipMirror:
    """Parsed canonical mirror block."""

    parent: int | None
    blocked_by: tuple[int, ...]
    blocking: tuple[int, ...]
    relates_to: tuple[int, ...]
    ambiguous: tuple[str, ...] = ()
    cross_repo: tuple[str, ...] = ()


@dataclass
class AuditResult:
    """JSON-ready audit outcome for one issue."""

    issue: int
    repo: str
    mirror_present: bool = False
    mirror: dict[str, Any] = field(default_factory=dict)
    native: dict[str, Any] = field(default_factory=dict)
    drift: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    verdict: str = "unknown"
    applied: list[str] = field(default_factory=list)
    error: str | None = None


def _split_owner_repo(repo: str) -> tuple[str, str]:
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name or "/" in name:
        raise ValueError(f"Invalid repo {repo!r}; expected OWNER/REPO")
    return owner, name


def _row_numbers(rows: dict[str, str], key: str) -> tuple[int, ...]:
    """Return same-repository ``#N`` references from one mirror row."""
    text = rows.get(key, "")
    if not text or _NONE_RE.match(text):
        return ()
    return tuple(int(n) for n in _REF_RE.findall(text))


def _mirror_section(body: str) -> str | None:
    """Return the ``## Relationships`` section text, or ``None`` when absent."""
    match = _MIRROR_HEADING_RE.search(body or "")
    if match is None:
        return None
    section = (body or "")[match.end() :]
    next_heading = re.search(r"^##\s+", section, re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]
    return section


def _mirror_rows(section: str) -> dict[str, str]:
    """Return canonical row label to raw value for one mirror section."""
    rows: dict[str, str] = {}
    for line in section.splitlines():
        row = _ROW_RE.match(line)
        if row is not None:
            rows[row.group(1).lower()] = row.group(2)
    return rows


def parse_relationship_block(body: str) -> RelationshipMirror | None:
    """Parse the canonical ``## Relationships`` mirror block.

    Returns:
        ``None`` when the block is absent; otherwise the parsed mirror with
        ambiguous tokens and cross-repository references preserved for
        fail-closed refusal instead of silent guessing.
    """
    section = _mirror_section(body)
    if section is None:
        return None
    rows = _mirror_rows(section)
    if not rows:
        return RelationshipMirror(
            parent=None,
            blocked_by=(),
            blocking=(),
            relates_to=(),
            ambiguous=("empty_relationships_block",),
        )
    ambiguous: list[str] = []
    cross_repo: list[str] = []
    for url in _URL_RE.finditer(section):
        ref = f"{url.group(1)}/{url.group(2)}#{url.group(4)}"
        if ref not in cross_repo:
            cross_repo.append(ref)
    if _AMBIGUOUS_TOKEN_RE.search(section):
        ambiguous.append("ambiguous_placeholder_token")

    def _numbers(key: str) -> tuple[int, ...]:
        return _row_numbers(rows, key)

    parents = _numbers("parent issue")
    parent: int | None = None
    if len(parents) > 1:
        ambiguous.append("multiple_parent_refs")
    elif parents:
        parent = parents[0]
    return RelationshipMirror(
        parent=parent,
        blocked_by=_numbers("blocked by"),
        blocking=_numbers("blocking"),
        relates_to=_numbers("relates to"),
        ambiguous=tuple(ambiguous),
        cross_repo=tuple(cross_repo),
    )


def _rest_issue_body(number: int, repo: str) -> tuple[str | None, str | None]:
    """Return ``(body, error)`` for one issue body read."""
    result = _gh_api(f"repos/{repo}/issues/{number}", timeout=GH_TIMEOUT)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "issue read failed").strip()
    payload, error = _parse_json(result, what="issue read")
    if error or not isinstance(payload, dict):
        return None, error or "issue response was not a JSON object"
    body = payload.get("body")
    if not isinstance(body, str):
        return None, "issue response has no string body"
    return body, None


def _graphql_parent_children(number: int, repo: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``({"parent": int|None, "children": [...]}, error)`` via GraphQL."""
    try:
        owner, name = _split_owner_repo(repo)
    except ValueError as exc:
        return None, str(exc)
    result = _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={_PARENT_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={name}",
            "-F",
            f"number={number}",
        ],
        timeout=GH_TIMEOUT,
    )
    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "graphql read failed").strip()
    payload, error = _parse_json(result, what="graphql read")
    if error or not isinstance(payload, dict):
        return None, error or "GraphQL response was not a JSON object"
    if isinstance(payload.get("errors"), list) and payload["errors"]:
        message = payload["errors"][0].get("message", "graphql error")
        return None, f"graphql error: {message}"
    try:
        issue = payload["data"]["repository"]["issue"]
    except (KeyError, TypeError):
        return None, "graphql response missing repository.issue"
    if issue is None:
        return None, f"issue #{number} not found"
    parent = (issue.get("parent") or {}).get("number")
    children = [
        node.get("number")
        for node in ((issue.get("subIssues") or {}).get("nodes") or [])
        if isinstance(node.get("number"), int)
    ]
    return {"parent": parent, "children": sorted(children)}, None


def _rest_dependency_numbers(
    number: int, repo: str, kind: str
) -> tuple[list[int] | None, str | None]:
    """Return native ``blocked_by``/``blocking`` issue numbers, or (None, error)."""
    result = _gh_api(f"repos/{repo}/issues/{number}/dependencies/{kind}", timeout=GH_TIMEOUT)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "dependency read failed").strip()
        return None, f"dependencies/{kind} unavailable: {detail}"
    payload, error = _parse_json(result, what=f"dependencies/{kind} read")
    if error or not isinstance(payload, list):
        return None, error or f"dependencies/{kind} response was not a list"
    numbers: list[int] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        inner = entry.get("issue") if isinstance(entry.get("issue"), dict) else entry
        value = inner.get("number") if isinstance(inner, dict) else None
        if isinstance(value, int) and value > 0:
            numbers.append(value)
    return sorted(set(numbers)), None


def _compute_drift(
    mirror: RelationshipMirror, native_parent: int | None, blocked_by: list[int]
) -> list[str]:
    """Return stable drift codes comparing mirror links against native links."""
    drift: list[str] = []
    if mirror.parent is not None and native_parent != mirror.parent:
        drift.append(f"parent_missing_native:{mirror.parent}")
    if mirror.parent is None and native_parent is not None:
        drift.append(f"parent_mirror_missing:{native_parent}")
    for ref in mirror.blocked_by:
        if ref not in blocked_by:
            drift.append(f"blocked_by_missing_native:{ref}")
    for ref in blocked_by:
        if ref not in mirror.blocked_by:
            drift.append(f"blocked_by_extra_native:{ref}")
    return drift


def audit_issue(number: int, repo: str = DEFAULT_REPO) -> AuditResult:
    """Run a read-only mirror-vs-native audit for one issue."""
    result = AuditResult(issue=number, repo=repo)
    body, error = _rest_issue_body(number, repo)
    if error is not None:
        result.error = error
        result.verdict = "error"
        return result
    mirror = parse_relationship_block(body)
    if mirror is None:
        result.verdict = "no_mirror_block"
        result.drift.append("no_mirror_block")
        return result
    result.mirror_present = True
    result.mirror = {
        "parent": mirror.parent,
        "blocked_by": list(mirror.blocked_by),
        "blocking": list(mirror.blocking),
        "relates_to": list(mirror.relates_to),
    }
    if mirror.ambiguous:
        result.refused.extend(f"ambiguous_mirror:{token}" for token in mirror.ambiguous)
    if mirror.cross_repo:
        result.refused.extend(f"cross_repo_ref:{ref}" for ref in mirror.cross_repo)
    native_rel, error = _graphql_parent_children(number, repo)
    if error is not None:
        result.unavailable.append(f"native_parent_children:{error}")
        native_parent: int | None = None
        native_children: list[int] = []
    else:
        native_parent = native_rel["parent"]
        native_children = native_rel["children"]
    blocked_by, error = _rest_dependency_numbers(number, repo, "blocked_by")
    if error is not None:
        result.unavailable.append(f"native_blocked_by:{error}")
        blocked_by = []
    blocking, error = _rest_dependency_numbers(number, repo, "blocking")
    if error is not None:
        result.unavailable.append(f"native_blocking:{error}")
        blocking = []
    result.native = {
        "parent": native_parent,
        "children": native_children,
        "blocked_by": blocked_by,
        "blocking": blocking,
    }
    if not result.refused and not result.unavailable:
        result.drift.extend(_compute_drift(mirror, native_parent, blocked_by))
    result.verdict = (
        "in_sync"
        if not result.drift
        and not result.refused
        and not result.unavailable
        and result.error is None
        else "drift"
        if result.drift
        else "refused"
        if result.refused
        else "unavailable"
        if result.unavailable
        else "error"
    )
    return result


def _rest_issue_id(number: int, repo: str) -> tuple[int | None, str | None]:
    """Return the numeric REST issue id used by relationship write endpoints."""
    result = _gh_api(f"repos/{repo}/issues/{number}", timeout=GH_TIMEOUT)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "issue read failed").strip()
    payload, error = _parse_json(result, what="issue id read")
    if error or not isinstance(payload, dict):
        return None, error or "issue response was not a JSON object"
    value = payload.get("id")
    if not isinstance(value, int) or value <= 0:
        return None, "issue response has no numeric id"
    return value, None


def _post_native_link(path: str, payload: dict[str, Any]) -> str | None:
    """POST one native relationship write; return an error string or None."""
    result = _gh_api(path, payload=payload, method="POST", timeout=GH_TIMEOUT)
    if result.returncode != 0:
        return (result.stderr or result.stdout or "relationship write failed").strip()
    return None


def _apply_missing_parent(audit: AuditResult) -> str | None:
    """Set the missing native parent; return an error string or None."""
    repo = audit.repo
    number = audit.issue
    mirror_parent = audit.mirror.get("parent")
    child_id, error = _rest_issue_id(number, repo)
    if error is not None:
        return f"child id read failed: {error}"
    assert isinstance(mirror_parent, int)
    error = _post_native_link(
        f"repos/{repo}/issues/{mirror_parent}/sub-issues",
        {"sub_issue_id": child_id},
    )
    if error is not None:
        return f"parent write failed: {error}"
    native, error = _graphql_parent_children(number, repo)
    if error is not None or (native or {}).get("parent") != mirror_parent:
        return "parent read-back mismatch after write"
    audit.applied.append(f"parent:{mirror_parent}")
    return None


def _apply_missing_blocked_by(audit: AuditResult, ref: int) -> str | None:
    """Add one missing native blocked-by link; return an error string or None."""
    repo = audit.repo
    number = audit.issue
    target_id, error = _rest_issue_id(ref, repo)
    if error is not None:
        return f"blocked-by target read failed: {error}"
    error = _post_native_link(
        f"repos/{repo}/issues/{number}/dependencies/blocked_by",
        {"issue_id": target_id},
    )
    if error is not None:
        return f"blocked-by write failed: {error}"
    current, error = _rest_dependency_numbers(number, repo, "blocked_by")
    if error is not None or ref not in (current or []):
        return "blocked-by read-back mismatch after write"
    audit.applied.append(f"blocked_by:{ref}")
    return None


def apply_migration(audit: AuditResult) -> AuditResult:
    """Apply the additive native migration described by a clean drift audit."""
    if audit.verdict != "drift" or not audit.drift:
        audit.error = f"nothing applicable: verdict is {audit.verdict}"
        return audit
    if audit.refused or audit.unavailable:
        audit.error = "refused or unavailable evidence blocks apply"
        return audit
    if any(item.startswith("parent_missing_native:") for item in audit.drift):
        error = _apply_missing_parent(audit)
        if error is not None:
            audit.error = error
            return audit
    for item in [d for d in audit.drift if d.startswith("blocked_by_missing_native:")]:
        error = _apply_missing_blocked_by(audit, int(item.rsplit(":", 1)[1]))
        if error is not None:
            audit.error = error
            return audit
    audit.verdict = "applied" if audit.applied else "drift"
    return audit


def _result_payload(audit: AuditResult) -> dict[str, Any]:
    return {
        "schema": "issue_relationship_audit.v1",
        "issue": audit.issue,
        "repo": audit.repo,
        "verdict": audit.verdict,
        "mirror_present": audit.mirror_present,
        "mirror": audit.mirror,
        "native": audit.native,
        "drift": audit.drift,
        "refused": audit.refused,
        "unavailable": audit.unavailable,
        "applied": audit.applied,
        "error": audit.error,
    }


def _render_human(audit: AuditResult) -> str:
    lines = [
        f"issue #{audit.issue} ({audit.repo}): {audit.verdict}",
        f"  mirror: {json.dumps(audit.mirror) if audit.mirror else 'absent'}",
        f"  native: {json.dumps(audit.native) if audit.native else 'unread'}",
    ]
    for item in audit.drift:
        lines.append(f"  drift: {item}")
    for item in audit.refused:
        lines.append(f"  refused: {item}")
    for item in audit.unavailable:
        lines.append(f"  unavailable: {item}")
    for item in audit.applied:
        lines.append(f"  applied: {item}")
    if audit.error:
        lines.append(f"  error: {audit.error}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int, help="Issue number to audit or migrate.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Migrate additive native links; requires --confirm.",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        help="Explicit migration confirmation; must be RELATIONSHIP_MIGRATION.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the audit (default) or the guarded apply migration."""
    args = _parse_args(argv or sys.argv[1:])
    if args.issue < 1:
        print("error: issue must be a positive integer", file=sys.stderr)
        return 2
    if args.apply and args.confirm != APPLY_CONFIRM_TOKEN:
        print(
            "error: --apply requires --confirm RELATIONSHIP_MIGRATION",
            file=sys.stderr,
        )
        return 2
    try:
        _split_owner_repo(args.repo)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    audit = audit_issue(args.issue, repo=args.repo)
    if args.apply:
        audit = apply_migration(audit)
    if args.json:
        print(json.dumps(_result_payload(audit), indent=2, sort_keys=True))
    else:
        print(_render_human(audit))
    if audit.error is not None:
        return 1
    if args.apply and not audit.applied:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
