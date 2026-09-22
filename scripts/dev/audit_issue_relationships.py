#!/usr/bin/env python3
"""Audit and explicitly link GitHub issue relationships.

The issue body is parsed only when it contains the canonical ``## Relationships`` block from
``docs/context/issue_relationships.md``.  Legacy headings and incidental issue mentions are
reported for review, but never become write proposals.  The default command is read-only.  An
explicit confirmation token is required before adding native parent or dependency links; no body,
parent replacement, or ``Relates to`` mutation is performed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from scripts.dev._gh_rest import parse_json, run_gh_api

SCHEMA = "issue_relationship_audit.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
CONFIRMATION_TOKEN = "RELATIONSHIP_MIGRATION"
RELATION_KINDS = ("parent", "blocked_by", "blocking", "relates_to")
RELATION_LABELS = {
    "parent": "Parent issue",
    "blocked_by": "Blocked by",
    "blocking": "Blocking",
    "relates_to": "Relates to",
}
CANONICAL_LABELS = {
    "parent issue": "parent",
    "parent": "parent",
    "blocked by": "blocked_by",
    "blocking": "blocking",
    "relates to": "relates_to",
}
LEGACY_LABELS = {
    "parent": "parent",
    "parent issue": "parent",
    "original parent": "parent",
    "child": "parent",
    "children": "parent",
    "child issue": "parent",
    "child issues": "parent",
    "blocked by": "blocked_by",
    "blocked-by": "blocked_by",
    "blocking": "blocking",
    "related": "relates_to",
    "related issue": "relates_to",
    "related issues": "relates_to",
    "related work": "relates_to",
    "dependencies": "blocked_by",
}
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>[^#].*?)\s*#*\s*$")
FIELD_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?P<label>parent(?:\s+issue)?|blocked\s+by|blocking|relates?\s+to)"
    r"\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s#]+)/issues/(?P<number>\d+)"
)
QUALIFIED_RE = re.compile(
    r"(?<![\w./-])(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#(?P<number>\d+)\b"
)
BARE_RE = re.compile(r"(?<![\w/])#(?P<number>\d+)\b")

ApiRunner = Callable[[str, object | None, str | None], Any]


@dataclass(frozen=True, slots=True)
class Declaration:
    """One explicitly declared relationship in an issue body."""

    issue: int
    kind: str
    target: int
    origin: str
    line: int
    raw: str


@dataclass(frozen=True, slots=True)
class Mention:
    """A legacy relationship-shaped heading or line that is not write eligible."""

    issue: int
    kind: str
    line: int
    raw: str
    targets: tuple[int, ...]


def _api_default(path: str, payload: object | None = None, method: str | None = None) -> Any:
    """Run a bounded GitHub REST request through the shared transport helper."""

    return run_gh_api(path, payload, method=method, timeout=60)


def _repo_slug(repo: str) -> str:
    """Normalize a repository identifier used in references."""

    return repo.strip().strip("/").lower()


def _extract_refs(value: str, *, repo: str) -> tuple[tuple[int, ...], list[str]]:
    """Extract same-repository issue references and reject external URLs/repositories."""

    targets: list[int] = []
    errors: list[str] = []
    covered_spans: list[tuple[int, int]] = []
    expected = _repo_slug(repo)

    for match in URL_RE.finditer(value):
        covered_spans.append(match.span())
        referenced_repo = f"{match.group('owner')}/{match.group('repo')}".lower()
        number = int(match.group("number"))
        if referenced_repo != expected:
            errors.append(f"cross-repository reference {match.group(0)!r}")
        else:
            targets.append(number)

    remainder = list(value)
    for start, end in covered_spans:
        remainder[start:end] = [" "] * (end - start)
    remainder_text = "".join(remainder)

    for match in QUALIFIED_RE.finditer(remainder_text):
        referenced_repo = match.group("repo").lower()
        number = int(match.group("number"))
        if referenced_repo != expected:
            errors.append(f"cross-repository reference {match.group(0)!r}")
        else:
            targets.append(number)

    for match in BARE_RE.finditer(remainder_text):
        targets.append(int(match.group("number")))

    unique_targets = tuple(sorted(set(targets)))
    return unique_targets, sorted(set(errors))


def _section_lines(body: str) -> tuple[list[tuple[int, str]], bool]:
    """Return the canonical relationship section's one-based lines and presence."""

    lines = body.splitlines()
    start: int | None = None
    end = len(lines)
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match:
            continue
        title = match.group("title").strip().lower()
        if start is None:
            if title == "relationships":
                start = index + 1
            continue
        end = index
        break
    if start is None:
        return [], False
    return [(index + 1, lines[index]) for index in range(start, end)], True


def _legacy_mentions(  # noqa: C901 - explicit heading/line parsing remains fail-closed
    body: str, *, issue: int, repo: str
) -> tuple[Mention, ...]:
    """Collect relationship-shaped legacy headings without making write proposals."""

    lines = body.splitlines()
    canonical_line_numbers = {
        line_number
        for line_number, line in _section_lines(body)[0]
        if FIELD_RE.match(line) is not None
    }
    mentions: list[Mention] = []
    active_kind: str | None = None
    active_line = 0
    active_raw = ""
    active_targets: list[int] = []

    def flush() -> None:
        nonlocal active_kind, active_line, active_raw, active_targets
        if active_kind is not None:
            mentions.append(
                Mention(
                    issue=issue,
                    kind=active_kind,
                    line=active_line,
                    raw=active_raw,
                    targets=tuple(sorted(set(active_targets))),
                )
            )
        active_kind = None
        active_line = 0
        active_raw = ""
        active_targets = []

    for index, line in enumerate(lines, start=1):
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            raw_title = re.sub(r"\s+", " ", heading.group("title").strip().lower())
            title, separator, inline_value = raw_title.partition(":")
            title = title.strip()
            if title in LEGACY_LABELS and title != "relationships":
                active_kind = LEGACY_LABELS[title]
                active_line = index
                active_raw = line.strip()
                targets, _ = _extract_refs(inline_value if separator else "", repo=repo)
                active_targets.extend(targets)
            continue
        if active_kind is not None:
            targets, _ = _extract_refs(line, repo=repo)
            active_targets.extend(targets)
    flush()

    # A small set of explicit legacy key/value lines occurs outside a heading (for example,
    # ``Original parent: #123``).  Keep these review-only as well.
    line_pattern = re.compile(
        r"^\s*(?:[-*+]\s+)?(?P<label>original\s+parent|parent(?:\s+issue)?|blocked\s+by|blocking|"
        r"related(?:\s+(?:issues?|work))?)\s*:\s*(?P<value>.*?)\s*$",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines, start=1):
        match = line_pattern.match(line)
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group("label").strip().lower())
        if index in canonical_line_numbers:
            # Canonical sections are parsed separately; do not duplicate their fields as legacy.
            continue
        kind = LEGACY_LABELS.get(label)
        if kind is None:
            continue
        targets, _ = _extract_refs(match.group("value"), repo=repo)
        mentions.append(
            Mention(
                issue=issue,
                kind=kind,
                line=index,
                raw=line.strip(),
                targets=targets,
            )
        )
    return tuple(mentions)


def parse_relationships(  # noqa: C901 - canonical and legacy states are intentionally explicit
    body: str, *, issue: int, repo: str = DEFAULT_REPO
) -> dict[str, Any]:
    """Parse one issue body into canonical declarations and review-only legacy mentions."""

    if not isinstance(body, str):
        return {
            "section_present": False,
            "declarations": [],
            "errors": ["issue body must be a string"],
            "legacy_mentions": [],
        }

    section, present = _section_lines(body)
    declarations: list[Declaration] = []
    errors: list[str] = []
    saw_kind: dict[str, list[int]] = {kind: [] for kind in RELATION_KINDS}
    saw_none: dict[str, bool] = dict.fromkeys(RELATION_KINDS, False)
    saw_field: set[str] = set()
    seen_declarations: set[tuple[str, int]] = set()

    for line_number, line in section:
        match = FIELD_RE.match(line)
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group("label").strip().lower())
        kind = CANONICAL_LABELS[label]
        saw_field.add(kind)
        value = match.group("value").strip()
        if value.lower() in {"none", "n/a", "not applicable"}:
            saw_none[kind] = True
            continue
        if not value:
            errors.append(
                f"line {line_number}: {RELATION_LABELS[kind]} must use explicit `none` when empty"
            )
            continue
        targets, reference_errors = _extract_refs(value, repo=repo)
        errors.extend(f"line {line_number}: {error}" for error in reference_errors)
        if not targets:
            errors.append(
                f"line {line_number}: {RELATION_LABELS[kind]} must name an issue or `none`"
            )
            continue
        for target in targets:
            if (kind, target) in seen_declarations:
                errors.append(
                    f"line {line_number}: duplicate {RELATION_LABELS[kind]} reference #{target}"
                )
                continue
            seen_declarations.add((kind, target))
            saw_kind[kind].append(target)
            declarations.append(
                Declaration(
                    issue=issue,
                    kind=kind,
                    target=target,
                    origin="canonical",
                    line=line_number,
                    raw=line.strip(),
                )
            )

    for kind in RELATION_KINDS:
        values = sorted(set(saw_kind[kind]))
        if saw_none[kind] and values:
            errors.append(f"{RELATION_LABELS[kind]} mixes `none` with issue references")
        if kind == "parent" and len(values) > 1:
            errors.append("Parent issue must name at most one issue")
        if issue in values:
            errors.append(f"{RELATION_LABELS[kind]} cannot reference the issue itself (#{issue})")

    if present:
        missing_fields = [RELATION_LABELS[kind] for kind in RELATION_KINDS if kind not in saw_field]
        if missing_fields:
            errors.append(
                "canonical ## Relationships section is missing field(s): "
                + ", ".join(missing_fields)
            )

    legacy = _legacy_mentions(body, issue=issue, repo=repo)
    if not present:
        errors.append("missing canonical ## Relationships section")
    return {
        "section_present": present,
        "declarations": [asdict(item) for item in declarations],
        "errors": sorted(set(errors)),
        "legacy_mentions": [asdict(item) for item in legacy],
    }


def _normalise_issue_row(raw: Mapping[str, Any], *, repo: str) -> dict[str, Any] | None:
    """Keep only canonical issue rows from the GitHub issues endpoint."""

    number = raw.get("number")
    if type(number) is not int or number < 1 or raw.get("pull_request") is not None:
        return None
    body = raw.get("body")
    if body is not None and not isinstance(body, str):
        return None
    issue_id = raw.get("id")
    if type(issue_id) is not int or issue_id < 1:
        return None
    return {
        "number": number,
        "id": issue_id,
        "title": str(raw.get("title") or ""),
        "body": body or "",
        "url": str(raw.get("html_url") or f"https://github.com/{repo}/issues/{number}"),
    }


def _read_json(api: ApiRunner, path: str, *, what: str) -> tuple[Any, str]:
    """Read and parse one API response using the shared diagnostic contract."""

    result = api(path, None, None)
    if isinstance(result, tuple):
        # A tuple is convenient for tiny offline runners: (payload, error).
        if len(result) == 2:
            return result
    return parse_json(result, what=what)


def _list_issues(  # noqa: C901 - inventory validation keeps partial reads visible
    api: ApiRunner,
    *,
    repo: str,
    state: str,
    per_page: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a bounded issue inventory and report truncation explicitly."""

    issues: list[dict[str, Any]] = []
    errors: list[str] = []
    pages_read = 0
    truncated = False
    malformed = 0
    for page in range(1, max_pages + 1):
        path = f"repos/{repo}/issues?state={state}&per_page={per_page}&page={page}"
        payload, error = _read_json(api, path, what=f"issues page {page}")
        if error:
            errors.append(error)
            break
        if not isinstance(payload, list):
            errors.append(f"issues page {page} returned a non-list payload")
            break
        pages_read += 1
        for raw in payload:
            if not isinstance(raw, Mapping):
                malformed += 1
                continue
            row = _normalise_issue_row(raw, repo=repo)
            if row is None:
                # Pull requests are expected in the issues endpoint; only malformed non-PR rows
                # affect completeness.
                if raw.get("pull_request") is None:
                    malformed += 1
                continue
            issues.append(row)
        if len(payload) < per_page:
            break
    else:
        truncated = True
    if malformed:
        errors.append(f"{malformed} malformed issue row(s) were excluded")
    if pages_read == 0 and not errors:
        errors.append("issue inventory returned no readable pages")
    return issues, {
        "status": "complete" if pages_read and not truncated and not errors else "unavailable",
        "pages_read": pages_read,
        "per_page": per_page,
        "page_budget": max_pages,
        "truncated": truncated,
        "errors": errors,
    }


def _exact_issue(
    api: ApiRunner, *, repo: str, number: int, cache: dict[int, dict[str, Any] | None]
) -> tuple[dict[str, Any] | None, str]:
    """Read one exact issue for target-id resolution, never accepting a pull request."""

    if number in cache:
        row = cache[number]
        return row, "" if row is not None else f"issue #{number} is unavailable"
    payload, error = _read_json(api, f"repos/{repo}/issues/{number}", what=f"issue #{number}")
    if error:
        cache[number] = None
        return None, error
    if not isinstance(payload, Mapping):
        cache[number] = None
        return None, f"issue #{number} returned a non-object payload"
    row = _normalise_issue_row(payload, repo=repo)
    if row is None:
        cache[number] = None
        return None, f"issue #{number} is not a canonical issue"
    cache[number] = row
    return row, ""


def _native_numbers(payload: Any) -> tuple[int, ...]:
    """Normalize issue numbers from parent/dependency endpoint payloads."""

    rows: Iterable[Any]
    if isinstance(payload, Mapping):
        rows = (payload,)
    elif isinstance(payload, list):
        rows = payload
    else:
        return ()
    numbers: list[int] = []
    for row in rows:
        if isinstance(row, Mapping) and type(row.get("number")) is int and row["number"] > 0:
            numbers.append(row["number"])
    return tuple(sorted(set(numbers)))


def _native_state(
    api: ApiRunner,
    *,
    repo: str,
    issue: int,
    max_reads: int,
    reads: list[str],
) -> tuple[dict[str, tuple[int, ...]], list[str]]:
    """Read native relationships for one issue, bounded by the caller's read budget."""

    state: dict[str, tuple[int, ...]] = {}
    errors: list[str] = []
    endpoints = {
        "parent": f"repos/{repo}/issues/{issue}/parent",
        "blocked_by": f"repos/{repo}/issues/{issue}/dependencies/blocked_by",
        "blocking": f"repos/{repo}/issues/{issue}/dependencies/blocking",
    }
    for kind, path in endpoints.items():
        if len(reads) >= max_reads:
            errors.append(f"native read budget exhausted before {path}")
            break
        reads.append(path)
        payload, error = _read_json(api, path, what=f"native {kind} for issue #{issue}")
        if error:
            # A missing parent is a valid empty relation; dependency endpoints should be readable.
            if kind == "parent" and "404" in error:
                state[kind] = ()
                continue
            errors.append(error)
            continue
        state[kind] = _native_numbers(payload)
    return state, errors


def _operation_for(
    *,
    issue: int,
    declaration: Mapping[str, Any],
    native: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Turn one canonical declaration into a conservative add-only operation."""

    kind = str(declaration["kind"])
    target = int(declaration["target"])
    operation = {
        "issue": issue,
        "kind": kind,
        "target": target,
        "status": "proposed",
        "reason": "native relationship is not present",
    }
    if kind == "relates_to":
        operation.update(status="manual", reason="Relates to has no supported REST write path")
    elif kind == "parent" and native.get("parent") and target not in native["parent"]:
        operation.update(
            status="conflict",
            reason=(
                "native parent differs; the audit never replaces an existing parent relationship"
            ),
        )
    elif target in native.get(kind, ()):
        operation.update(status="already_present", reason="native relationship already exists")
    elif kind not in native:
        operation.update(status="blocked", reason="native relationship read was unavailable")
    return operation


def _readback_path(repo: str, *, kind: str, issue: int) -> str:
    """Return the exact read-back endpoint path for a given relationship kind."""
    if kind == "parent":
        return f"repos/{repo}/issues/{issue}/parent"
    if kind in {"blocked_by", "blocking"}:
        return f"repos/{repo}/issues/{issue}/dependencies/{kind}"
    return ""


def _is_idempotent_collision_error(error: str) -> bool:
    """Return whether a write error indicates the native relationship may already exist."""
    lower = error.lower()
    collision_phrases = (
        "already been taken",
        "already taken",
        "already exists",
        "already a sub-issue",
        "already a sub issue",
        "already linked",
        "already added",
        "has already been added",
    )
    if any(phrase in lower for phrase in collision_phrases):
        return True
    return "422" in lower and "already" in lower


def _canonical_edge(kind: str, issue: int, target: int) -> tuple[str, int, int] | None:
    """Return a canonical directed edge representation for dependency and parent relations."""
    if kind == "blocked_by":
        return ("dependency", target, issue)
    if kind == "blocking":
        return ("dependency", issue, target)
    if kind == "parent":
        return ("parent", target, issue)
    return None


def _verify_readback(
    api: ApiRunner,
    *,
    repo: str,
    kind: str,
    issue: int,
    target: int,
) -> tuple[bool, str]:
    """Verify with an exact read-back that the target issue is in the native relationship."""
    readback_path = _readback_path(repo, kind=kind, issue=issue)
    if not readback_path:
        return False, f"unsupported read-back for kind {kind}"
    payload_read, error = _read_json(
        api, readback_path, what=f"read back {kind} for issue #{issue}"
    )
    if error:
        return False, error
    numbers = _native_numbers(payload_read)
    if target not in numbers:
        return False, f"read-back did not contain #{target} at {readback_path}"
    return True, ""


def _write_operation(  # noqa: C901 - each relation direction has distinct REST/read-back paths
    api: ApiRunner,
    *,
    repo: str,
    operation: Mapping[str, Any],
    issue_rows: Mapping[int, Mapping[str, Any]],
    target_cache: dict[int, dict[str, Any] | None],
) -> tuple[str, str]:
    """Apply one add-only native operation and verify it with an exact read-back."""

    issue = int(operation["issue"])
    target = int(operation["target"])
    kind = str(operation["kind"])
    current = issue_rows.get(issue)
    if current is None:
        return "blocked", f"issue #{issue} id is unavailable"
    target_row = issue_rows.get(target)
    if target_row is None:
        target_row, error = _exact_issue(api, repo=repo, number=target, cache=target_cache)
        if target_row is None:
            return "blocked", error
    current_id = current["id"]
    target_id = target_row["id"]
    if kind == "parent":
        path = f"repos/{repo}/issues/{target}/sub_issues"
        payload = {"sub_issue_id": current_id, "replace_parent": False}
    elif kind == "blocked_by":
        path = f"repos/{repo}/issues/{issue}/dependencies/blocked_by"
        payload = {"issue_id": target_id}
    elif kind == "blocking":
        path = f"repos/{repo}/issues/{target}/dependencies/blocked_by"
        payload = {"issue_id": current_id}
    else:
        return "manual", "Relates to has no supported REST write path"

    result = api(path, payload, "POST")
    if getattr(result, "returncode", 1) != 0:
        _, error = parse_json(result, what=f"write {kind} #{issue} -> #{target}")
        if _is_idempotent_collision_error(error):
            verified, _ = _verify_readback(api, repo=repo, kind=kind, issue=issue, target=target)
            if verified:
                return (
                    "already_applied",
                    "native relationship already present on write; verified with read-back",
                )
        return "failed", error
    verified, error = _verify_readback(api, repo=repo, kind=kind, issue=issue, target=target)
    if not verified:
        return "failed", error
    return "applied", "native relationship added and read back"


def audit_relationships(  # noqa: C901, PLR0912, PLR0913, PLR0915 - bounded audit/apply orchestration
    *,
    repo: str = DEFAULT_REPO,
    state: str = "open",
    issue_numbers: Sequence[int] | None = None,
    per_page: int = 100,
    max_pages: int = 20,
    max_native_reads: int = 1200,
    apply: bool = False,
    confirmation: str | None = None,
    api: ApiRunner = _api_default,
) -> dict[str, Any]:
    """Run a bounded audit and optionally apply reviewed canonical add-only links."""

    errors: list[str] = []
    if issue_numbers:
        issues: list[dict[str, Any]] = []
        cache: dict[int, dict[str, Any] | None] = {}
        for number in sorted(set(issue_numbers)):
            row, error = _exact_issue(api, repo=repo, number=number, cache=cache)
            if row is None:
                errors.append(error)
            else:
                issues.append(row)
        source = {
            "status": "complete" if not errors else "unavailable",
            "kind": "exact_issue_reads",
            "pages_read": 0,
            "truncated": False,
            "errors": list(errors),
        }
    else:
        issues, source = _list_issues(
            api, repo=repo, state=state, per_page=per_page, max_pages=max_pages
        )
        errors.extend(source["errors"])

    issue_rows = {row["number"]: row for row in issues}
    native_reads: list[str] = []
    target_cache: dict[int, dict[str, Any] | None] = dict(issue_rows)
    reports: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    legacy_count = 0
    canonical_count = 0
    contract_finding_count = 0
    missing_section_count = 0

    for row in issues:
        parsed = parse_relationships(row["body"], issue=row["number"], repo=repo)
        declarations = parsed["declarations"]
        canonical_count += len(declarations)
        contract_finding_count += len(parsed["errors"])
        missing_section_count += int(
            "missing canonical ## Relationships section" in parsed["errors"]
        )
        errors.extend(f"issue #{row['number']}: {error}" for error in parsed["errors"])
        native: dict[str, tuple[int, ...]] = {}
        native_errors: list[str] = []
        # Only concrete canonical declarations need native reads.  This keeps a whole-repository
        # audit bounded while still verifying every proposed mutation.
        if declarations:
            native, native_errors = _native_state(
                api,
                repo=repo,
                issue=row["number"],
                max_reads=max_native_reads,
                reads=native_reads,
            )
            errors.extend(native_errors)
        issue_operations = [
            _operation_for(issue=row["number"], declaration=item, native=native)
            for item in declarations
        ]
        errors.extend(
            f"issue #{row['number']}: {operation['reason']}"
            for operation in issue_operations
            if operation["status"] == "conflict"
        )
        operations.extend(issue_operations)
        legacy = parsed["legacy_mentions"]
        legacy_count += len(legacy)
        reports.append(
            {
                "number": row["number"],
                "url": row["url"],
                "section_present": parsed["section_present"],
                "declarations": declarations,
                "native": {kind: list(values) for kind, values in native.items()},
                "native_errors": native_errors,
                "operations": issue_operations,
                "errors": parsed["errors"],
                "legacy_mentions": legacy,
            }
        )

    apply_result: dict[str, Any] = {
        "requested": apply,
        "status": "not_requested",
        "confirmation": confirmation == CONFIRMATION_TOKEN,
        "operations": [],
    }
    if apply:
        if confirmation != CONFIRMATION_TOKEN:
            apply_result.update(status="blocked", reason=f"pass --confirm {CONFIRMATION_TOKEN}")
        elif (
            source.get("status") != "complete"
            or source.get("truncated")
            or errors
            or contract_finding_count
        ):
            apply_result.update(
                status="blocked",
                reason="source, native reads, or parsing is incomplete; no writes were attempted",
            )
        else:
            # Resolve every target before the first write.  This avoids a partial migration when
            # one referenced issue is missing, closed behind permissions, or malformed.
            target_errors: list[str] = []
            for operation in operations:
                if operation["status"] != "proposed":
                    continue
                target = int(operation["target"])
                if target not in issue_rows:
                    _, target_error = _exact_issue(
                        api, repo=repo, number=target, cache=target_cache
                    )
                    if target_error:
                        target_errors.append(target_error)
            if target_errors:
                apply_result.update(
                    status="blocked",
                    reason="target resolution failed; no writes were attempted",
                    errors=sorted(set(target_errors)),
                )
            else:
                statuses: list[str] = []
                applied_canonical_edges: set[tuple[str, int, int]] = set()
                for operation in operations:
                    edge = _canonical_edge(
                        str(operation["kind"]),
                        int(operation["issue"]),
                        int(operation["target"]),
                    )
                    if operation["status"] != "proposed":
                        apply_result["operations"].append(dict(operation))
                        statuses.append(operation["status"])
                        if edge is not None and operation["status"] in {
                            "already_present",
                            "already_applied",
                        }:
                            applied_canonical_edges.add(edge)
                        continue

                    # If this exact canonical edge was already applied or present in this run,
                    # verify with exact read-back instead of calling the redundant POST endpoint.
                    if edge is not None and edge in applied_canonical_edges:
                        issue = int(operation["issue"])
                        target = int(operation["target"])
                        kind = str(operation["kind"])
                        verified, _ = _verify_readback(
                            api, repo=repo, kind=kind, issue=issue, target=target
                        )
                        if verified:
                            applied = dict(operation)
                            applied.update(
                                status="already_applied",
                                reason="reciprocal native relationship already added and read back",
                            )
                            apply_result["operations"].append(applied)
                            statuses.append("already_applied")
                            continue

                    status, reason = _write_operation(
                        api,
                        repo=repo,
                        operation=operation,
                        issue_rows=issue_rows,
                        target_cache=target_cache,
                    )
                    applied = dict(operation)
                    applied.update(status=status, reason=reason)
                    apply_result["operations"].append(applied)
                    statuses.append(status)
                    if edge is not None and status in {"applied", "already_applied"}:
                        applied_canonical_edges.add(edge)

                if any(status in {"failed", "blocked"} for status in statuses):
                    apply_result["status"] = (
                        "partial"
                        if any(s in {"applied", "already_applied"} for s in statuses)
                        else "failed"
                    )
                else:
                    apply_result["status"] = "complete"

    return {
        "schema": SCHEMA,
        "repository": repo,
        "state": state,
        "dry_run": not apply,
        "source": source,
        "issue_count": len(issues),
        "canonical_declaration_count": canonical_count,
        "legacy_mention_count": legacy_count,
        "contract_finding_count": contract_finding_count,
        "missing_section_count": missing_section_count,
        "native_reads": {"count": len(native_reads), "paths": native_reads},
        "issues": reports,
        "apply": apply_result,
        "errors": sorted(set(errors)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--state", choices=("open", "closed", "all"), default="open")
    parser.add_argument("--issue", type=int, action="append", dest="issue_numbers")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-native-reads", type=int, default=1200)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    parser.add_argument("--apply", action="store_true", help="add reviewed native links")
    parser.add_argument("--confirm", help=f"required with --apply: {CONFIRMATION_TOKEN}")
    return parser


def _text_summary(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable report without hiding partial evidence."""

    lines = [
        f"{report['schema']} repository={report['repository']} state={report['state']}",
        f"issues={report['issue_count']} canonical_declarations={report['canonical_declaration_count']} "
        f"legacy_mentions={report['legacy_mention_count']} contract_findings={report['contract_finding_count']} "
        f"dry_run={report['dry_run']}",
        f"source={report['source']['status']} truncated={report['source'].get('truncated', False)} "
        f"native_reads={report['native_reads']['count']}",
    ]
    for issue in report["issues"]:
        for operation in issue["operations"]:
            lines.append(
                f"#{issue['number']} {operation['kind']} -> #{operation['target']}: "
                f"{operation['status']} ({operation['reason']})"
            )
        for error in issue["errors"]:
            lines.append(f"#{issue['number']} error: {error}")
        for mention in issue["legacy_mentions"]:
            lines.append(f"#{issue['number']} legacy line {mention['line']}: {mention['raw']}")
    for error in report["errors"]:
        lines.append(f"error: {error}")
    if report["apply"]["requested"]:
        lines.append(f"apply={report['apply']['status']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _parser().parse_args(argv)
    if (
        args.per_page <= 0
        or args.max_pages <= 0
        or args.max_native_reads < 0
        or any(number <= 0 for number in (args.issue_numbers or ()))
    ):
        print(
            "--per-page, --max-pages, and --issue values must be positive; "
            "--max-native-reads cannot be negative",
            file=sys.stderr,
        )
        return 2
    report = audit_relationships(
        repo=args.repo,
        state=args.state,
        issue_numbers=args.issue_numbers,
        per_page=args.per_page,
        max_pages=args.max_pages,
        max_native_reads=args.max_native_reads,
        apply=args.apply,
        confirmation=args.confirm,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_text_summary(report))
    if (
        report["source"].get("status") != "complete"
        or report["errors"]
        or report["contract_finding_count"]
    ):
        return 1
    if args.apply and report["apply"]["status"] not in {"complete", "not_requested"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
