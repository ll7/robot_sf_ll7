#!/usr/bin/env python3
# ruff: noqa: C901, PLR0912, PLR0913, PLR0915
"""Shared, fail-closed issue-audit inventory, classification, and mutation core.

The two issue-audit skills deliberately have different authority boundaries, but
they consume the same plan produced here.  This module keeps the policy-bearing
parts deterministic and testable:

* bounded REST inventory of issues, pull requests, labels, and repository state;
* local correlation of issue references with PRs, claims, worktrees, and jobs;
* composable versus mutually-exclusive label classification;
* conservative mutation planning and URI-safe label deletion;
* bounded mutation execution followed by REST readback; and
* explicit closure and decision-gate evidence.

The module never creates labels and never writes Project #5 fields.  Callers may
use the plan as a dry-run artifact or apply only the operations it contains.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from scripts.tools.issue_archetype_sync import SAFE_ARCHETYPE_LABEL_MAP
from scripts.tools.issue_template_audit import audit_archetype_metadata

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
PER_PAGE = 100
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_COMMENT_PAGES = 3
DEFAULT_MAX_MUTATIONS = 250
PLAN_SCHEMA = "issue_audit_plan.v1"

STATE_PREFIX = "state:"
RESOURCE_PREFIX = "resource:"
TYPE_PREFIX = "type:"
EVIDENCE_PREFIX = "evidence:"
DECISION_LABEL = "decision-required"

# Canonical execution states are mutually exclusive.  The repository also has
# composable ``state:*`` qualifiers such as ``state:review`` and
# ``state:needs-artifact-promotion``; those qualify an issue without replacing
# its execution state and must be preserved during cleanup.  Resource and
# evidence labels are composable; type labels are expected to be singular, but
# ambiguity is preserved rather than guessed.
EXECUTION_STATE_LABELS = frozenset(
    {
        "state:blocked-external-input",
        "state:blocked",
        "state:hold",
        "state:running",
        "state:ready",
    }
)
STATE_PRIORITY = (
    "state:blocked-external-input",
    "state:blocked",
    "state:hold",
    "state:running",
    "state:ready",
)

BLOCKER_TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "provenance": (
        ("provenance", "checksum", "digest", "lineage", "seed", "compatibility"),
        (
            "blocked",
            "missing",
            "unavailable",
            "mismatch",
            "incompatible",
            "unsupported",
            "cannot verify",
            "not available",
        ),
    ),
    "rights": (
        ("license", "licensing", "rights", "permission", "consent", "redistribution"),
        (
            "blocked",
            "missing",
            "unavailable",
            "pending",
            "not granted",
            "cannot publish",
            "cannot release",
            "not available",
        ),
    ),
    "compute": (
        ("slurm", "sbatch", "gpu", "compute", "quota", "allocation"),
        (
            "blocked",
            "missing",
            "unavailable",
            "pending",
            "not authorized",
            "authorization unavailable",
            "authorization missing",
            "quota exhausted",
            "quota unavailable",
            "quota exceeded",
            "no allocation",
            "cannot submit",
        ),
    ),
    "external-input": (
        ("external data", "external asset", "checkpoint", "dataset", "model weights"),
        (
            "missing",
            "unavailable",
            "pending",
            "not available",
            "waiting for",
            "requires access",
        ),
    ),
}

DECISION_PATTERNS = (
    re.compile(
        r"\b(?:decision|maintainer\s+(?:decision|choice|approval)|"
        r"(?:owner|maintainer)\s+decisions?)"
        r"\s+(?:is\s+)?(?:required|needed|pending|requested)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:maintainer|owner)\b.{0,40}\b(?:must|needs?\s+to|should)\b"
        r".{0,40}\b(?:choose|select|approve|confirm|decide)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:choose|select)\s+(?:one|an?\s+option|the\s+(?:option|approach|policy|scope))\b"
        r"|\bapprove\s+or\s+waive\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
READY_HEADING_PATTERN = re.compile(
    r"^#{2,4}\s+(?:acceptance(?: criteria)?|definition of done|success criteria|"
    r"validation(?: / testing| command| commands)?|implementation plan)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CHECKBOX_PATTERN = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+", re.MULTILINE)
UNCHECKED_CHECKBOX_PATTERN = re.compile(r"^\s*[-*]\s+\[ \]\s+", re.MULTILINE)
ISSUE_REF_PATTERN = re.compile(r"(?<![\w-])#(\d+)\b|(?:issue|issues)[ -](\d+)\b", re.IGNORECASE)
PARENT_TITLE_PATTERN = re.compile(
    r"\b(parent|roadmap|epic|tracking|multi[- ]slice|umbrella)\b", re.IGNORECASE
)
EXPLICIT_MERGED_CLOSE_PATTERN = re.compile(
    r"(?im)^\s*(?:close|completion|done)\s+"
    r"(?:condition|when|criteria)\s*:\s*.*\b(?:merged?|merge)\b.*\b(?:pr|pull request)\b"
)
PARENT_CLOSE_PATTERN = re.compile(
    r"(?im)^\s*parent\s+close\s+condition\s*:\s*all\s+(?:linked\s+)?children\s+closed\s*$"
)

Runner = Callable[[list[str], str | None], subprocess.CompletedProcess[str]]


def _run_gh(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run gh with captured text output, returning failures to the caller."""
    try:
        return subprocess.run(
            ["gh", *args],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            ["gh", *args],
            127,
            "",
            "gh CLI not found on PATH; install GitHub CLI and authenticate it",
        )


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a local discovery command without turning missing optional tools into writes."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, "", f"{args[0]} not found on PATH")


def _parse_json(result: subprocess.CompletedProcess[str], *, what: str) -> tuple[Any | None, str]:
    """Decode a command result into JSON or a useful fail-closed error."""
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return None, f"{what} failed: {detail or f'exit code {result.returncode}'}"
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"{what} returned invalid JSON: {exc}"


def _runner_or_default(runner: Runner | None) -> Runner:
    return runner or _run_gh


def _issue_ref_numbers(*values: object) -> set[int]:
    """Extract explicit issue references from titles, bodies, branches, and job names."""
    numbers: set[int] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for match in ISSUE_REF_PATTERN.finditer(value):
            raw = match.group(1) or match.group(2)
            if raw:
                numbers.add(int(raw))
    return numbers


def _label_names(raw: object) -> list[str]:
    """Normalize GitHub issue or repository label rows."""
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str) and item:
            names.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
            names.append(str(item["name"]))
    return sorted(set(names))


def _normalize_issue(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub issue row onto the plan's stable issue shape."""
    url = str(raw.get("html_url") or raw.get("url") or "")
    raw_assignees = raw.get("assignees")
    assignees = raw_assignees if isinstance(raw_assignees, list) else []
    raw_comments = raw.get("comments")
    comments = raw_comments if isinstance(raw_comments, list) else []
    return {
        "number": int(raw.get("number", 0)),
        "title": str(raw.get("title") or ""),
        "body": str(raw.get("body") or ""),
        "state": str(raw.get("state") or "").lower(),
        "url": url,
        "labels": _label_names(raw.get("labels")),
        "assignees": sorted(
            str(item.get("login"))
            for item in assignees
            if isinstance(item, Mapping) and item.get("login")
        ),
        "comments": [
            {
                "body": str(item.get("body") or ""),
                "user": str((item.get("user") or {}).get("login") or ""),
            }
            for item in comments
            if isinstance(item, Mapping)
        ],
    }


def _normalize_pr(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub pull request row onto the stable correlation shape."""
    head = raw.get("head") if isinstance(raw.get("head"), Mapping) else {}
    return {
        "number": int(raw.get("number", 0)),
        "title": str(raw.get("title") or ""),
        "body": str(raw.get("body") or ""),
        "state": str(raw.get("state") or "").lower(),
        "url": str(raw.get("html_url") or raw.get("url") or ""),
        "merged_at": str(raw.get("merged_at") or ""),
        "head_ref": str(head.get("ref") or raw.get("head_ref") or ""),
    }


def paginate_rest(
    path: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    per_page: int = PER_PAGE,
    runner: Runner | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a bounded REST collection and report truncation instead of guessing."""
    if max_pages <= 0 or per_page <= 0:
        raise ValueError("max_pages and per_page must be positive")
    run = _runner_or_default(runner)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    pages_read = 0
    truncated = False
    for page in range(1, max_pages + 1):
        separator = "&" if "?" in path else "?"
        endpoint = f"{path}{separator}per_page={per_page}&page={page}"
        payload, error = _parse_json(run(["api", endpoint], None), what=endpoint)
        if error:
            errors.append(error)
            break
        if not isinstance(payload, list):
            errors.append(f"{endpoint} returned a non-list payload")
            break
        pages_read += 1
        rows.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < per_page:
            break
    else:
        truncated = True
    return rows, {
        "pages_read": pages_read,
        "per_page": per_page,
        "page_budget": max_pages,
        "row_count": len(rows),
        "truncated": truncated,
        "errors": errors,
    }


def discover_open_issues(
    repo: str = DEFAULT_REPO,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Runner | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover canonical open issues, filtering pull requests from the issues endpoint."""
    rows, meta = paginate_rest(
        f"repos/{repo}/issues?state=open",
        max_pages=max_pages,
        runner=runner,
    )
    issues = [
        _normalize_issue(row)
        for row in rows
        if "pull_request" not in row and "/issues/" in str(row.get("html_url") or "")
    ]
    return sorted(issues, key=lambda item: item["number"]), {**meta, "row_count": len(issues)}


def discover_issue_comments(
    repo: str,
    issue_number: int,
    *,
    max_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    runner: Runner | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read one complete, bounded issue comment thread through REST."""
    rows, meta = paginate_rest(
        f"repos/{repo}/issues/{issue_number}/comments",
        max_pages=max_pages,
        runner=runner,
    )
    comments = [
        {
            "body": str(row.get("body") or ""),
            "user": str((row.get("user") or {}).get("login") or ""),
        }
        for row in rows
    ]
    return comments, {**meta, "row_count": len(comments)}


def attach_issue_comments(
    repo: str,
    issues: list[dict[str, Any]],
    *,
    max_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Attach bounded REST comment threads to issues and aggregate read status."""
    metadata: dict[str, Any] = {
        "available": True,
        "issue_count": len(issues),
        "pages_read": 0,
        "row_count": 0,
        "truncated": False,
        "errors": [],
    }
    for issue in issues:
        comments, comment_meta = discover_issue_comments(
            repo,
            int(issue["number"]),
            max_pages=max_pages,
            runner=runner,
        )
        issue["comments"] = comments
        metadata["pages_read"] += int(comment_meta.get("pages_read", 0))
        metadata["row_count"] += len(comments)
        metadata["truncated"] = bool(metadata["truncated"] or comment_meta.get("truncated"))
        metadata["errors"].extend(comment_meta.get("errors", []))
    return metadata


def discover_pull_requests(
    repo: str = DEFAULT_REPO,
    *,
    state: str = "open",
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Runner | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover open or closed pull requests through bounded REST pagination."""
    if state not in {"open", "closed"}:
        raise ValueError("state must be open or closed")
    rows, meta = paginate_rest(
        f"repos/{repo}/pulls?state={state}&sort=updated&direction=desc",
        max_pages=max_pages,
        runner=runner,
    )
    prs = [_normalize_pr(row) for row in rows if "/pull/" in str(row.get("html_url") or "")]
    return sorted(prs, key=lambda item: item["number"]), {**meta, "row_count": len(prs)}


def discover_repository_labels(
    repo: str = DEFAULT_REPO,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Runner | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Read existing repository labels; never infer a label that was not returned."""
    rows, meta = paginate_rest(f"repos/{repo}/labels", max_pages=max_pages, runner=runner)
    labels = {
        str(row["name"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    return labels, {**meta, "row_count": len(labels)}


def discover_claims(
    remote: str = DEFAULT_REMOTE,
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Discover remote issue-claim branches without claiming or changing anything."""
    run = command_runner or _run_command
    command = ["git", "ls-remote", "--heads", remote, "refs/heads/agent-claims/issue-*"]
    result = run(command)
    if result.returncode != 0:
        return {}, {"available": False, "errors": [(result.stderr or result.stdout).strip()]}
    claims: dict[int, dict[str, Any]] = {}
    prefix = "refs/heads/agent-claims/issue-"
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].startswith(prefix):
            continue
        suffix = parts[1][len(prefix) :]
        if suffix.isdigit():
            claims[int(suffix)] = {"claimed": True, "sha": parts[0], "ref": parts[1]}
    return claims, {"available": True, "count": len(claims), "errors": []}


def discover_worktrees(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Discover local worktrees and expose branch/path text for issue correlation."""
    run = command_runner or _run_command
    command = ["git", "worktree", "list", "--porcelain"]
    result = run(command)
    if result.returncode != 0:
        return [], {"available": False, "errors": [(result.stderr or result.stdout).strip()]}
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in (result.stdout or "").splitlines() + [""]:
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line.removeprefix("worktree ")}
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch ")
        elif not line and current:
            worktrees.append(current)
            current = {}
    return worktrees, {"available": True, "count": len(worktrees), "errors": []}


def discover_jobs(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Discover visible SLURM jobs when squeue is available.

    A missing squeue is normal on a non-SLURM machine.  For an issue that
    explicitly requires SLURM, the unavailable inventory is still surfaced so
    the classifier can fail closed instead of declaring readiness.
    """
    run = command_runner or _run_command
    command = ["squeue", "--noheader", "--format=%i|%j|%T|%S"]
    result = run(command)
    if result.returncode != 0:
        return [], {"available": False, "errors": [(result.stderr or result.stdout).strip()]}
    jobs: list[dict[str, str]] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            jobs.append(
                {
                    "id": parts[0].strip(),
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "start": parts[3].strip(),
                }
            )
    return jobs, {"available": True, "count": len(jobs), "errors": []}


def discover_inventory(
    repo: str = DEFAULT_REPO,
    *,
    remote: str = DEFAULT_REMOTE,
    max_pages: int = DEFAULT_MAX_PAGES,
    include_comments: bool = False,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    runner: Runner | None = None,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Build the complete read-only inventory consumed by the shared classifier."""
    issues, issue_meta = discover_open_issues(repo, max_pages=max_pages, runner=runner)
    comment_meta = (
        attach_issue_comments(
            repo,
            issues,
            max_pages=max_comment_pages,
            runner=runner,
        )
        if include_comments
        else {"available": False, "reason": "not_requested", "errors": []}
    )
    open_prs, open_pr_meta = discover_pull_requests(
        repo, state="open", max_pages=max_pages, runner=runner
    )
    closed_prs, closed_pr_meta = discover_pull_requests(
        repo, state="closed", max_pages=max_pages, runner=runner
    )
    labels, label_meta = discover_repository_labels(repo, max_pages=max_pages, runner=runner)
    claims, claim_meta = discover_claims(remote, command_runner=command_runner)
    worktrees, worktree_meta = discover_worktrees(command_runner=command_runner)
    jobs, job_meta = discover_jobs(command_runner=command_runner)
    return {
        "repo": repo,
        "remote": remote,
        "issues": issues,
        "open_prs": open_prs,
        "merged_prs": [pr for pr in closed_prs if pr.get("merged_at")],
        "labels": sorted(labels),
        "claims": claims,
        "worktrees": worktrees,
        "jobs": jobs,
        "inventory": {
            "issues": issue_meta,
            "comments": comment_meta,
            "open_prs": open_pr_meta,
            "closed_prs": closed_pr_meta,
            "labels": label_meta,
            "claims": claim_meta,
            "worktrees": worktree_meta,
            "jobs": job_meta,
        },
    }


def _text_for_issue(issue: Mapping[str, Any]) -> str:
    """Join body and comments for evidence matching without inventing content."""
    parts = [str(issue.get("title") or ""), str(issue.get("body") or "")]
    comments = issue.get("comments")
    if isinstance(comments, Sequence) and not isinstance(comments, (str, bytes)):
        for comment in comments:
            if isinstance(comment, Mapping):
                parts.append(str(comment.get("body") or ""))
            elif isinstance(comment, str):
                parts.append(comment)
    return "\n".join(parts)


def _gate_evidence(text: str) -> list[dict[str, str]]:
    """Return current, issue-local provenance, rights, compute, and input gates."""
    evidence: list[dict[str, str]] = []
    lines = [" ".join(line.lower().split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    gate_context = re.compile(
        r"\b(?:hard[- ]?)?(?:blocked|gated)\s+(?:on|by|until|pending)\b"
        r"|\b(?:is|are|remains?|currently|still)\s+(?:hard[- ]?)?blocked\b"
        r"|\b(?:cannot|can't|unable to|not able to|not authorized to|not dispatchable)\b"
        r"[^|.;]{0,100}\b(?:submit|launch|run|proceed|stage|access|publish|release|verify|execute)\b"
        r"|\b(?:required|needed)\b[^|.;]{0,100}\b(?:missing|unavailable|pending|unset)\b"
        r"|\b(?:currently|availability:)\s*(?:missing|unavailable|unset)\b",
        re.IGNORECASE,
    )
    direct_blockers = {
        "blocked",
        "missing",
        "unavailable",
        "mismatch",
        "incompatible",
        "unsupported",
        "not available",
        "not granted",
        "cannot verify",
        "cannot publish",
        "cannot release",
        "cannot submit",
        "no allocation",
    }
    report_pattern = re.compile(
        r"\b\d+\s+of\s+\d+\b|\bgate-blocked\b|\bstate:(?:ready|running|blocked)\b"
        r"|\b(?:issues|rows|labels)\b.{0,60}\b(?:blocked|missing|unavailable)\b",
        re.IGNORECASE,
    )
    rule_pattern = re.compile(
        r"\bfail(?:s|ed)?\s+closed\b|\b(?:records?|rows?|evidence|claims?|outcomes?|"
        r"metadata|protocol|predeclared|execution|validates?|rejects?)\b"
        r"[^|.;]{0,50}\b(?:missing|unavailable|fallback|degraded|invalid)\b",
        re.IGNORECASE,
    )
    generic_gate_pattern = re.compile(
        r"\b(?:remains?|currently|is|are)?\s*(?:hard[- ]?)?blocked\s+"
        r"(?:on|by|until|pending)\b|\bhard[- ]gated\s+(?:on|by|until)\b",
        re.IGNORECASE,
    )
    for kind, (topics, blockers) in BLOCKER_TERMS.items():
        for line in lines:
            topic = next((term for term in topics if term in line), None)
            blocker = next((term for term in blockers if term in line), None)
            if not topic:
                continue
            def near(left: str, right: str) -> bool:
                return bool(
                    re.search(
                        rf"\b{re.escape(left)}\b[^|.;]{{0,80}}\b{re.escape(right)}\b",
                        line,
                    )
                    or re.search(
                        rf"\b{re.escape(right)}\b[^|.;]{{0,80}}\b{re.escape(left)}\b",
                        line,
                    )
                )

            report_line = bool(report_pattern.search(line))
            conditional_line = bool(re.search(r"\b(?:if|unless|when|otherwise)\b", line))
            explicit_gate = bool(gate_context.search(line))
            gate_related = any(
                near(topic, token)
                for token in ("blocked", "gated", "pending", "missing", "unavailable")
            )
            direct_status = bool(blocker and blocker in direct_blockers and near(topic, blocker))
            rule_line = bool(rule_pattern.search(line))
            # A bare "quota exhausted" or similar topic echo is not enough;
            # it must be attached to an explicit current gate.  Report tables
            # and conditional acceptance rules describe other records or
            # future failures, not the issue's present state.
            current_gate = (explicit_gate and gate_related) or (
                direct_status and not conditional_line and not rule_line
            )
            if kind == "external-input" and not gate_related:
                current_gate = False
            if kind == "external-input" and blocker == "missing" and not explicit_gate:
                current_gate = False
            if current_gate and not report_line:
                detail = blocker or "explicit gate"
                evidence.append(
                    {
                        "kind": kind,
                        "text": f"{topic} evidence includes {detail}",
                    }
                )
                break
    for line in lines:
        if generic_gate_pattern.search(line) and not report_pattern.search(line):
            evidence.append({"kind": "blocked", "text": "explicit current blocker"})
            break
    return evidence


def _decision_evidence(text: str, labels: set[str]) -> list[str]:
    """Return decision-gate evidence from explicit labels or issue text."""
    evidence = ["decision-required label present"] if DECISION_LABEL in labels else []
    lines = [" ".join(line.split()) for line in text.splitlines()]
    resolution_pattern = re.compile(
        r"\b(?:decision-required(?:\s+label)?|decision|gate|deferral|approval)\b"
        r"[^.;]{0,100}\b(?:removed|resolved|settled|already exists|no longer needed|"
        r"not pending|remove|waiv(?:ed|ing)|approved|authorized|selected|made|deferred|"
        r"reaffirmed|reaffirm|in force)\b"
        r"|\bchoose\s+option\s+\([a-z]\)(?!\w)"
        r"|\b(?:remove|removed)\s+decision-required\b"
        r"|\breaffirm(?:ed|ation)?\b"
        r"|\bapprove\s+(?!or\s+waive\b)[^.;]{0,100}\b(?:nominal|exact|clean|"
        r"protocol|campaign|option)\b",
        re.IGNORECASE,
    )
    last_resolution = max(
        (index for index, line in enumerate(lines) if resolution_pattern.search(line)),
        default=-1,
    )
    for index, line in enumerate(lines):
        if not line:
            continue
        for pattern in DECISION_PATTERNS:
            match = pattern.search(line)
            if not match or index <= last_resolution:
                continue
            prefix = line[: match.start()]
            if re.search(r"\b(?:no|not|never|without)\b[^.;]{0,80}$", prefix, re.IGNORECASE):
                continue
            if match:
                evidence.append(f"issue text: {match.group(0).strip()[:180]}")
                break
    return evidence


def _ready_evidence(body: str) -> list[str]:
    """Return only concrete acceptance/validation signals supporting readiness."""
    evidence: list[str] = []
    if READY_HEADING_PATTERN.search(body):
        evidence.append("acceptance or validation heading present")
    if CHECKBOX_PATTERN.search(body) and not UNCHECKED_CHECKBOX_PATTERN.search(body):
        evidence.append("all issue checkboxes are marked complete")
    if re.search(r"(?im)^\s*(?:command|entry point|validation)\s*:", body):
        evidence.append("explicit command or validation entry point present")
    return evidence


def _active_records(
    issue_number: int,
    *,
    open_prs: Iterable[Mapping[str, Any]],
    claims: Mapping[int, Mapping[str, Any]],
    worktrees: Iterable[Mapping[str, Any]],
    jobs: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Correlate active execution records with one issue number."""
    prs = [
        dict(pr)
        for pr in open_prs
        if issue_number in _issue_ref_numbers(pr.get("title"), pr.get("body"), pr.get("head_ref"))
    ]
    local_worktrees = [
        dict(row)
        for row in worktrees
        if issue_number in _issue_ref_numbers(row.get("path"), row.get("branch"))
    ]
    active_jobs = [
        dict(row)
        for row in jobs
        if issue_number
        in _issue_ref_numbers(row.get("name"), row.get("command"), row.get("job_name"))
    ]
    claim = dict(claims[issue_number]) if issue_number in claims else {}
    return {
        "open_prs": prs,
        "claims": [claim] if claim else [],
        "worktrees": local_worktrees,
        "jobs": active_jobs,
    }


def _merged_records(
    issue_number: int, merged_prs: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return merged PRs with explicit issue references in title/body/branch."""
    return [
        dict(pr)
        for pr in merged_prs
        if issue_number in _issue_ref_numbers(pr.get("title"), pr.get("body"), pr.get("head_ref"))
    ]


def closure_evidence(
    issue: Mapping[str, Any],
    *,
    merged_prs: Sequence[Mapping[str, Any]],
    open_issue_numbers: set[int] | None = None,
) -> dict[str, Any]:
    """Evaluate the narrow, documented conditions under which autonomous close is safe."""
    body = str(issue.get("body") or "")
    number = int(issue.get("number", 0))
    linked = _merged_records(number, merged_prs)
    if not linked:
        return {"eligible": False, "reason": "no merged issue-linked PR", "merged_prs": []}
    if PARENT_TITLE_PATTERN.search(str(issue.get("title") or "")):
        if not PARENT_CLOSE_PATTERN.search(body):
            return {
                "eligible": False,
                "reason": "parent issue lacks documented all-children close condition",
                "merged_prs": [pr.get("number") for pr in linked],
            }
        child_numbers = _issue_ref_numbers(body)
        child_numbers.discard(number)
        if open_issue_numbers is None or child_numbers & open_issue_numbers:
            return {
                "eligible": False,
                "reason": "documented parent close condition is not proven by the open inventory",
                "merged_prs": [pr.get("number") for pr in linked],
                "child_issues": sorted(child_numbers),
            }
        return {
            "eligible": True,
            "reason": "documented parent close condition and no referenced child remains open",
            "merged_prs": [pr.get("number") for pr in linked],
            "child_issues": sorted(child_numbers),
        }
    explicit = bool(EXPLICIT_MERGED_CLOSE_PATTERN.search(body))
    checked = bool(CHECKBOX_PATTERN.search(body)) and not bool(
        UNCHECKED_CHECKBOX_PATTERN.search(body)
    )
    if explicit or checked:
        reason = (
            "explicit merged-PR close condition"
            if explicit
            else "all acceptance checkboxes complete with merged issue-linked PR"
        )
        return {
            "eligible": True,
            "reason": reason,
            "merged_prs": [pr.get("number") for pr in linked],
        }
    return {
        "eligible": False,
        "reason": "merged work exists but completion condition is not documented",
        "merged_prs": [pr.get("number") for pr in linked],
    }


def _available(label: str, available_labels: set[str] | None) -> bool:
    """Allow a mutation only when the repository label inventory proves it exists."""
    return available_labels is not None and label in available_labels


def _mutation(
    operation: str,
    issue_number: int,
    *,
    value: str | None,
    reason: str,
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a stable mutation row."""
    return {
        "operation": operation,
        "issue": issue_number,
        "value": value,
        "reason": reason,
        "evidence": list(evidence),
    }


def _state_winner(
    state_labels: set[str],
    *,
    blocker: bool,
    external_blocker: bool,
    active: bool,
    ready: bool,
) -> str | None:
    """Choose a state only when repository evidence makes the choice unambiguous."""
    if blocker:
        return "state:blocked-external-input" if external_blocker else "state:blocked"
    if active:
        return "state:running"
    if ready:
        return "state:ready"
    if len(state_labels) == 1:
        return next(iter(state_labels))
    return None


@dataclass(frozen=True)
class Classification:
    """Normalized classification and safe operations for one open issue."""

    issue: int
    classification: str
    state_labels: tuple[str, ...]
    execution_state_labels: tuple[str, ...]
    resource_labels: tuple[str, ...]
    type_labels: tuple[str, ...]
    evidence_labels: tuple[str, ...]
    active: dict[str, list[dict[str, Any]]]
    blocker_evidence: tuple[dict[str, str], ...]
    decision_required: bool
    decision_evidence: tuple[str, ...]
    readiness_evidence: tuple[str, ...]
    closure: dict[str, Any]
    mutations: tuple[dict[str, Any], ...]
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible classification payload."""
        return {
            "issue": f"#{self.issue}",
            "number": self.issue,
            "classification": self.classification,
            "state_labels": list(self.state_labels),
            "execution_state_labels": list(self.execution_state_labels),
            "resource_labels": list(self.resource_labels),
            "type_labels": list(self.type_labels),
            "evidence_labels": list(self.evidence_labels),
            "active_evidence": self.active,
            "blocker_evidence": list(self.blocker_evidence),
            "decision_required": self.decision_required,
            "decision_evidence": list(self.decision_evidence),
            "readiness_evidence": list(self.readiness_evidence),
            "closure_evidence": self.closure,
            "mutations": list(self.mutations),
            "findings": list(self.findings),
        }


def classify_issue(
    issue: Mapping[str, Any],
    *,
    open_prs: Iterable[Mapping[str, Any]] = (),
    merged_prs: Iterable[Mapping[str, Any]] = (),
    claims: Mapping[int, Mapping[str, Any]] | None = None,
    worktrees: Iterable[Mapping[str, Any]] = (),
    jobs: Iterable[Mapping[str, Any]] = (),
    job_inventory_available: bool = True,
    open_issue_numbers: set[int] | None = None,
    available_labels: set[str] | None = None,
) -> Classification:
    """Classify one issue and plan only evidence-supported autonomous repairs."""
    number = int(issue.get("number", 0))
    labels = set(_label_names(issue.get("labels")))
    body = str(issue.get("body") or "")
    text = _text_for_issue(issue)
    state_labels = tuple(sorted(label for label in labels if label.startswith(STATE_PREFIX)))
    execution_state_labels = tuple(
        sorted(label for label in labels if label in EXECUTION_STATE_LABELS)
    )
    resource_labels = tuple(sorted(label for label in labels if label.startswith(RESOURCE_PREFIX)))
    type_labels = tuple(sorted(label for label in labels if label.startswith(TYPE_PREFIX)))
    evidence_labels = tuple(sorted(label for label in labels if label.startswith(EVIDENCE_PREFIX)))
    claims = claims or {}
    findings: list[str] = []

    active = _active_records(
        number,
        open_prs=open_prs,
        claims=claims,
        worktrees=worktrees,
        jobs=jobs,
    )
    active_now = any(active.values())
    blocker_evidence = _gate_evidence(text)
    if "state:blocked-external-input" in labels:
        blocker_evidence.append(
            {"kind": "external-input", "text": "state:blocked-external-input label"}
        )
    if "state:blocked" in labels:
        blocker_evidence.append({"kind": "blocked", "text": "state:blocked label"})
    if "evidence:blocked" in labels:
        blocker_evidence.append({"kind": "blocked", "text": "evidence:blocked label"})
    job_inventory_uncertain = "resource:slurm" in labels and not job_inventory_available
    if job_inventory_uncertain:
        findings.append(
            "SLURM job inventory unavailable; preserve this issue and do not promote it to ready"
        )

    decision_evidence = _decision_evidence(text, labels)
    if len(type_labels) > 1:
        decision_evidence.append("multiple mutually-exclusive type labels present")
    readiness_evidence = _ready_evidence(body)
    gate_blocked = bool(blocker_evidence)
    decision_required = bool(decision_evidence)
    ready = (
        bool(readiness_evidence)
        and not gate_blocked
        and not decision_required
        and not job_inventory_uncertain
    )
    closure = closure_evidence(
        issue,
        merged_prs=list(merged_prs),
        open_issue_numbers=open_issue_numbers,
    )
    execution_state_set = set(execution_state_labels)
    stale_running = "state:running" in execution_state_set and not active_now
    if stale_running:
        findings.append("state:running has no currently observed active record; preserved")
    state_set = execution_state_set
    ready = ready and not stale_running
    external_blocker = any(item["kind"] == "external-input" for item in blocker_evidence)
    winner = _state_winner(
        state_set,
        blocker=gate_blocked,
        external_blocker=external_blocker,
        active=active_now,
        ready=ready,
    )
    mutations: list[dict[str, Any]] = []

    if len(state_set) > 1:
        if winner is None:
            decision_required = True
            decision_evidence.append("multiple state labels without decisive repository evidence")
        else:
            for label in sorted(state_set):
                if label != winner:
                    if _available(label, available_labels):
                        mutations.append(
                            _mutation(
                                "remove_label",
                                number,
                                value=label,
                                reason=f"remove contradictory state label; evidence selects {winner}",
                                evidence=[
                                    *(item["text"] for item in blocker_evidence),
                                    *readiness_evidence,
                                ],
                            )
                        )
                    else:
                        findings.append(f"cannot remove unavailable label {label}")

    if winner and winner not in state_set and _available(winner, available_labels):
        for label in sorted(state_set):
            if _available(label, available_labels):
                mutations.append(
                    _mutation(
                        "remove_label",
                        number,
                        value=label,
                        reason=f"replace stale state label; evidence selects {winner}",
                        evidence=[
                            *(item["text"] for item in blocker_evidence),
                            *readiness_evidence,
                            "active PR/claim/worktree/job evidence" if active_now else "",
                        ],
                    )
                )
            else:
                findings.append(f"cannot remove unavailable label {label}")
        mutations.append(
            _mutation(
                "add_label",
                number,
                value=winner,
                reason="repair missing state label from repository evidence",
                evidence=[
                    *(item["text"] for item in blocker_evidence),
                    *readiness_evidence,
                    "active PR/claim/worktree/job evidence" if active_now else "",
                ],
            )
        )
    elif winner and winner not in state_set:
        findings.append(f"cannot add unavailable label {winner}")

    # A type mirror is safe only for a valid, complete archetype metadata block
    # and only when no type label already exists.  We do not repair malformed
    # metadata or choose among competing types.
    metadata = audit_archetype_metadata(body)
    parsed_metadata = metadata.parsed_metadata
    archetype = parsed_metadata.get("archetype") if isinstance(parsed_metadata, Mapping) else None
    target_type = SAFE_ARCHETYPE_LABEL_MAP.get(archetype) if isinstance(archetype, str) else None
    if not type_labels and not metadata.findings and target_type:
        if _available(target_type, available_labels):
            mutations.append(
                _mutation(
                    "add_label",
                    number,
                    value=target_type,
                    reason="mirror complete issue archetype metadata",
                    evidence=[f"archetype metadata: {archetype}"],
                )
            )
        else:
            findings.append(f"cannot add unavailable label {target_type}")

    # Explicit decision gates always win over readiness and are never guessed
    # away.  A proven blocker is visible through state:blocked when that label
    # exists, but a maintainer decision is not converted into an answer.
    if (
        decision_required
        and DECISION_LABEL not in labels
        and _available(DECISION_LABEL, available_labels)
    ):
        mutations.append(
            _mutation(
                "add_label",
                number,
                value=DECISION_LABEL,
                reason="make an explicit maintainer decision gate visible",
                evidence=decision_evidence,
            )
        )
    if gate_blocked and not state_set.intersection(
        {"state:blocked", "state:blocked-external-input"}
    ):
        target = (
            "state:blocked-external-input"
            if any(item["kind"] == "external-input" for item in blocker_evidence)
            else "state:blocked"
        )
        if _available(target, available_labels):
            mutations.append(
                _mutation(
                    "add_label",
                    number,
                    value=target,
                    reason="record a proven provenance, rights, compute, or external-input gate",
                    evidence=[item["text"] for item in blocker_evidence],
                )
            )
        else:
            findings.append(f"cannot add unavailable blocker label {target}")

    # Never promote an issue to ready while any active execution record or
    # unresolved decision exists.  This guard is intentionally redundant with
    # the winner calculation because it protects future rule additions.
    if ready and not active_now and not decision_required and not gate_blocked:
        if "state:ready" not in state_set and _available("state:ready", available_labels):
            mutations.append(
                _mutation(
                    "add_label",
                    number,
                    value="state:ready",
                    reason="issue has explicit acceptance/validation evidence and no active gate",
                    evidence=readiness_evidence,
                )
            )

    classification = (
        "decision-required"
        if decision_required
        else "blocked"
        if gate_blocked
        else ("running" if active_now else "ready" if ready else "unclassified")
    )
    if closure.get("eligible") and str(issue.get("state") or "").lower() == "open":
        if not decision_required and not gate_blocked and not active["open_prs"]:
            mutations.append(
                _mutation(
                    "close_issue",
                    number,
                    value=None,
                    reason="documented closure condition is proven by merged work",
                    evidence=[
                        closure.get("reason", ""),
                        *(f"merged PR #{pr}" for pr in closure.get("merged_prs", [])),
                    ],
                )
            )
            classification = "complete"

    # Preserve a pre-existing running state when discovery has no active
    # evidence: stale state is uncertain, not permission to promote to ready.
    if stale_running and classification == "unclassified":
        classification = "running"

    # Multiple rules can independently notice the same missing state or
    # blocker. Keep one stable operation so a batch cannot apply duplicate
    # writes or report an empty evidence item.
    unique_mutations: list[dict[str, Any]] = []
    seen_mutations: set[tuple[str, int, str | None]] = set()
    for mutation in mutations:
        key = (
            str(mutation.get("operation")),
            int(mutation.get("issue", number)),
            mutation.get("value") if isinstance(mutation.get("value"), str) else None,
        )
        if key in seen_mutations:
            continue
        seen_mutations.add(key)
        mutation["evidence"] = [item for item in mutation.get("evidence", []) if item]
        unique_mutations.append(mutation)

    return Classification(
        issue=number,
        classification=classification,
        state_labels=state_labels,
        execution_state_labels=execution_state_labels,
        resource_labels=resource_labels,
        type_labels=type_labels,
        evidence_labels=evidence_labels,
        active=active,
        blocker_evidence=tuple(blocker_evidence),
        decision_required=decision_required,
        decision_evidence=tuple(decision_evidence),
        readiness_evidence=tuple(readiness_evidence),
        closure=closure,
        mutations=tuple(unique_mutations),
        findings=tuple(findings),
    )


def build_audit_plan(
    inventory: Mapping[str, Any],
    *,
    mode: str = "autonomous",
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
) -> dict[str, Any]:
    """Build the shared issue_audit_plan.v1 from a read-only inventory."""
    if mode not in {"autonomous", "interactive"}:
        raise ValueError("mode must be autonomous or interactive")
    issues = [item for item in inventory.get("issues", []) if isinstance(item, Mapping)]
    open_prs = [item for item in inventory.get("open_prs", []) if isinstance(item, Mapping)]
    merged_prs = [item for item in inventory.get("merged_prs", []) if isinstance(item, Mapping)]
    claims = inventory.get("claims") if isinstance(inventory.get("claims"), Mapping) else {}
    worktrees = [item for item in inventory.get("worktrees", []) if isinstance(item, Mapping)]
    jobs = [item for item in inventory.get("jobs", []) if isinstance(item, Mapping)]
    available_labels = set(_label_names(inventory.get("labels")))
    job_meta = inventory.get("inventory", {}).get("jobs", {})
    job_available = bool(job_meta.get("available", True)) if isinstance(job_meta, Mapping) else True
    open_numbers = {int(item.get("number", 0)) for item in issues}
    classifications: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for issue in sorted(issues, key=lambda item: int(item.get("number", 0))):
        classified = classify_issue(
            issue,
            open_prs=open_prs,
            merged_prs=merged_prs,
            claims=claims,
            worktrees=worktrees,
            jobs=jobs,
            job_inventory_available=job_available,
            open_issue_numbers=open_numbers,
            available_labels=available_labels,
        )
        row = {
            "number": int(issue.get("number", 0)),
            "title": str(issue.get("title") or ""),
            "url": str(issue.get("url") or ""),
            **classified.to_dict(),
        }
        classifications.append(row)
        issue_mutations = list(classified.mutations)
        mutations.extend(issue_mutations)
        if classified.decision_required:
            pending.append(
                {
                    "issue": f"#{classified.issue}",
                    "decision_required": True,
                    "question_source": "issue body/comments",
                    "blocking_evidence": "; ".join(classified.decision_evidence)
                    or "decision gate detected",
                    "safe_mutations_applied": [],
                }
            )
    inventory_meta = inventory.get("inventory") or {}
    truncated = [
        name
        for name, meta in inventory_meta.items()
        if isinstance(meta, Mapping)
        and (meta.get("truncated") or meta.get("errors"))
        and not (name == "jobs" and meta.get("available") is False)
    ]
    inventory_uncertainties = [
        name
        for name, meta in inventory_meta.items()
        if isinstance(meta, Mapping) and name == "jobs" and meta.get("available") is False
    ]
    if len(mutations) > max_mutations:
        mutations = mutations[:max_mutations]
        truncated.append("mutation_budget")
    return {
        "schema": PLAN_SCHEMA,
        "repo": str(inventory.get("repo") or DEFAULT_REPO),
        "mode": mode,
        "project5": {"writes": False, "owner": "gh-issue-sequencer"},
        "label_policy": {
            "create_missing": False,
            "mutually_exclusive": [sorted(EXECUTION_STATE_LABELS), TYPE_PREFIX],
            "composable": [RESOURCE_PREFIX, EVIDENCE_PREFIX],
            "preserve_state_qualifiers": True,
        },
        "inventory": inventory.get("inventory", {}),
        "inventory_uncertainties": sorted(set(inventory_uncertainties)),
        "issues": classifications,
        "mutations": mutations,
        "pending_decisions": pending,
        "truncation_or_errors": sorted(set(truncated)),
        "counts": {
            "open_issues": len(classifications),
            "mutations": len(mutations),
            "pending_decisions": len(pending),
            "truncated_or_error_sources": len(set(truncated)),
        },
    }


def label_api_path(repo: str, issue_number: int, label: str) -> str:
    """Return the REST label endpoint with every label character URI-escaped."""
    return f"repos/{repo}/issues/{issue_number}/labels/{quote(label, safe='')}"


def apply_mutations(
    plan: Mapping[str, Any],
    *,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Apply a bounded plan and read back every touched issue.

    The caller is responsible for choosing the skill authority.  This function
    only executes explicit operations in the plan and refuses a truncated plan,
    missing repository, or unsupported operation.
    """
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"expected {PLAN_SCHEMA}")
    if plan.get("truncation_or_errors"):
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": "inventory or mutation plan is incomplete",
            "applied": [],
            "failures": list(plan["truncation_or_errors"]),
            "readback": [],
        }
    mutations = plan.get("mutations")
    if not isinstance(mutations, list):
        raise ValueError("plan mutations must be a list")
    if len(mutations) > max_mutations:
        raise ValueError("plan exceeds mutation budget")
    repo = str(plan.get("repo") or DEFAULT_REPO)
    run = _runner_or_default(runner)
    applied: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    touched: set[int] = set()
    expectations: dict[int, dict[str, set[str] | bool]] = {}
    for mutation in mutations:
        if not isinstance(mutation, Mapping):
            failures.append({"mutation": mutation, "error": "mutation is not an object"})
            continue
        try:
            number = int(mutation["issue"])
            operation = str(mutation["operation"])
            value = mutation.get("value")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"mutation": dict(mutation), "error": f"invalid mutation: {exc}"})
            continue
        if operation == "add_label" and isinstance(value, str) and value:
            endpoint = f"repos/{repo}/issues/{number}/labels"
            result = run(
                ["api", "-X", "POST", endpoint, "--input", "-"],
                json.dumps({"labels": [value]}),
            )
        elif operation == "remove_label" and isinstance(value, str) and value:
            result = run(["api", "-X", "DELETE", label_api_path(repo, number, value)], None)
        elif operation == "close_issue":
            result = run(
                ["api", "-X", "PATCH", f"repos/{repo}/issues/{number}", "--input", "-"],
                json.dumps({"state": "closed"}),
            )
        else:
            failures.append(
                {"mutation": dict(mutation), "error": f"unsupported mutation: {operation}"}
            )
            continue
        if result.returncode != 0:
            failures.append(
                {
                    "mutation": dict(mutation),
                    "error": (result.stderr or result.stdout).strip()
                    or f"exit code {result.returncode}",
                }
            )
            continue
        applied.append(dict(mutation))
        touched.add(number)
        expected = expectations.setdefault(
            number,
            {"add_labels": set(), "remove_labels": set(), "closed": False},
        )
        if operation == "add_label" and isinstance(value, str):
            expected["add_labels"].add(value)  # type: ignore[union-attr]
        elif operation == "remove_label" and isinstance(value, str):
            expected["remove_labels"].add(value)  # type: ignore[union-attr]
        elif operation == "close_issue":
            expected["closed"] = True

    readback: list[dict[str, Any]] = []
    for number in sorted(touched):
        result = run(["api", f"repos/{repo}/issues/{number}"], None)
        payload, error = _parse_json(result, what=f"readback issue {number}")
        if error or not isinstance(payload, Mapping):
            readback.append({"issue": number, "ok": False, "error": error or "invalid payload"})
            continue
        labels = _label_names(payload.get("labels"))
        state = str(payload.get("state") or "").lower()
        expected = expectations[number]
        added = sorted(set(expected["add_labels"]) & set(labels))  # type: ignore[arg-type]
        missing_additions = sorted(set(expected["add_labels"]) - set(labels))  # type: ignore[arg-type]
        removed = sorted(set(expected["remove_labels"]) - set(labels))  # type: ignore[arg-type]
        missing_removals = sorted(set(expected["remove_labels"]) & set(labels))  # type: ignore[arg-type]
        expected_closed = bool(expected["closed"])
        state_ok = not expected_closed or state == "closed"
        row_ok = not missing_additions and not missing_removals and state_ok
        readback.append(
            {
                "issue": number,
                "ok": row_ok,
                "state": state,
                "labels": labels,
                "verified": {
                    "added": added,
                    "missing_additions": missing_additions,
                    "removed": removed,
                    "missing_removals": missing_removals,
                    "closed": state == "closed" if expected_closed else None,
                },
            }
        )
    return {
        "schema": "issue_audit_apply.v1",
        "ok": not failures and all(row.get("ok") for row in readback),
        "applied": applied,
        "failures": failures,
        "readback": readback,
    }


def build_pending_decision_queue(
    plan: Mapping[str, Any],
    *,
    applied_mutations: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return the machine-readable queue consumed by the interactive skill."""
    pending = plan.get("pending_decisions")
    if not isinstance(pending, list):
        return []
    applied_by_issue: dict[str, list[dict[str, Any]]] = {}
    for mutation in applied_mutations:
        if not isinstance(mutation, Mapping):
            continue
        try:
            issue = f"#{int(mutation['issue'])}"
        except (KeyError, TypeError, ValueError):
            continue
        applied_by_issue.setdefault(issue, []).append(dict(mutation))
    queue: list[dict[str, Any]] = []
    for item in pending:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        issue = str(row.get("issue") or "")
        row["safe_mutations_applied"] = applied_by_issue.get(issue, [])
        queue.append(row)
    return queue


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Expose bounded plan and apply operations for skills and CI checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="discover and emit an issue-audit plan")
    plan_parser.add_argument("--repo", default=DEFAULT_REPO)
    plan_parser.add_argument("--remote", default=DEFAULT_REMOTE)
    plan_parser.add_argument("--mode", choices=("autonomous", "interactive"), default="autonomous")
    plan_parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    plan_parser.add_argument(
        "--include-comments",
        action="store_true",
        help="include bounded REST comment threads in the issue evidence inventory",
    )
    plan_parser.add_argument("--max-comment-pages", type=int, default=DEFAULT_MAX_COMMENT_PAGES)
    plan_parser.add_argument("--max-mutations", type=int, default=DEFAULT_MAX_MUTATIONS)
    plan_parser.add_argument("--output", type=Path)
    apply_parser = subparsers.add_parser("apply", help="apply a previously emitted plan")
    apply_parser.add_argument("plan", type=Path)
    apply_parser.add_argument("--max-mutations", type=int, default=DEFAULT_MAX_MUTATIONS)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_audit_plan(
            discover_inventory(
                args.repo,
                remote=args.remote,
                max_pages=args.max_pages,
                include_comments=args.include_comments,
                max_comment_pages=args.max_comment_pages,
            ),
            mode=args.mode,
            max_mutations=args.max_mutations,
        )
        rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 2 if plan["truncation_or_errors"] else 0
    result = apply_mutations(_load_json(args.plan), max_mutations=args.max_mutations)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
