#!/usr/bin/env python3
"""Fail-closed remote-state gate for issue-to-PR publication.

The gate captures the issue, base, remote branch, and local HEAD before an
expensive readiness run.  A later check detects a newly opened covering PR, a
merged PR that superseded the issue, movement of the base or remote branch, and
local HEAD changes.  The ``sync`` command can safely integrate changed remote
refs with ordinary Git merges; it never resets or deletes a worktree.

Issue #7515: the gate also rejects publication when the branch carries
non-``main`` ancestry that is undeclared, mismatched, or whose declared parent
is invalid.  ``evaluate_state`` incorporates the snapshot's recorded
``ancestry`` block (populated by ``collect_live_state`` via
``scripts.dev.stack_ancestry``): a blocking ancestry state
(``undeclared_stack`` / ``mismatched_declaration`` / ``parent_invalidated``)
produces the new ``undeclared_stack_ancestry`` reason and blocks before PR
creation, while a declared stack (``stacked``) is permitted to proceed to
publication but is never independently merge-ready (see
``scripts.dev.pr_loop_policy``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scripts.dev.open_issue_closure_audit import DEFAULT_MAX_PR_PAGES as DEFAULT_MAX_REST_PR_PAGES
from scripts.dev.open_issue_closure_audit import fetch_closed_pr_rows, fetch_open_pr_rows
from scripts.dev.stack_ancestry import (
    BLOCKING_STATES,
    ancestry_state,
    collect_ancestry_facts,
    parse_stack_declaration,
)

SCHEMA = "prepublication_state.v1"
DECISION_SCHEMA = "prepublication_decision.v1"
EXIT_CODES = {
    "ready": 0,
    "refresh-required": 2,
    "superseded": 3,
    "blocked": 4,
}
_CLOSING_REFERENCE_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+"
    r"(?:(?:https?://github\.com/(?P<url_repo>[\w.-]+/[\w.-]+)/issues/)|"
    r"(?:(?P<repo>[\w.-]+/[\w.-]+)?#))?(?P<number>\d+)\b",
    re.IGNORECASE,
)
_SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_REPO_COMPONENT_RE = re.compile(r"^[^/\s]+$")
DEFAULT_TIMEOUT_SECONDS = 120
GRAPHQL_FALLBACK_MARKERS = ("graphql:", "graphql ", "api rate limit")
GRAPHQL_FAIL_CLOSED_MARKERS = (
    "bad credentials",
    "requires authentication",
    "authentication required",
    "resource not accessible by integration",
    "forbidden",
    "permission denied",
    "could not resolve to a repository",
    "could not resolve to an issue",
    "repository not found",
)


class GateError(RuntimeError):
    """Raised when the gate cannot establish a trustworthy remote state."""


def _positive_page_budget(value: str) -> int:
    """Parse a positive REST page budget for the CLI."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("page budget must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("page budget must be a positive integer")
    return parsed


def _effective_page_budget(
    explicit: int | None,
    *,
    snapshot: dict[str, Any] | None = None,
) -> int:
    """Choose a validated explicit or snapshot-recorded REST page budget."""
    value: object = explicit
    if value is None and snapshot is not None:
        value = snapshot.get("rest_pr_page_budget", DEFAULT_MAX_REST_PR_PAGES)
    if value is None:
        value = DEFAULT_MAX_REST_PR_PAGES
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GateError(f"invalid REST PR page budget: {value!r}")
    return value


def _run(
    command: list[str], *, check: bool = True, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run a command without invoking a shell."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"{' '.join(command)}: timed out after {timeout}s") from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise GateError(f"{' '.join(command)}: {detail}")
    return result


def _utc_now() -> str:
    """Return a stable UTC timestamp for state records."""
    return datetime.now(UTC).isoformat()


def _git_output(*args: str) -> str:
    """Run Git and return non-empty trimmed stdout."""
    output = _run(["git", *args]).stdout.strip()
    if not output:
        raise GateError(f"git {' '.join(args)} returned empty output")
    return output


def _git_branch() -> str:
    """Return the current branch, rejecting detached publication attempts."""
    branch = _git_output("branch", "--show-current")
    if not _SAFE_BRANCH_RE.fullmatch(branch) or ".." in branch or "@{" in branch:
        raise GateError(f"invalid or unsafe branch name: {branch!r}")
    return branch


def _tree_state() -> str:
    """Return the tracked/untracked worktree state."""
    status = _run(["git", "status", "--porcelain", "--untracked-files=normal"]).stdout
    return "dirty" if status else "clean"


def _json_command(command: list[str]) -> Any:
    """Run a JSON-producing command and parse its result."""
    result = _run(command, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise GateError(f"{' '.join(command)}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GateError(f"{' '.join(command)} returned invalid JSON: {exc}") from exc


def _issue_number(value: str | int) -> int:
    """Validate and normalize an issue number."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"invalid issue number: {value!r}") from exc
    if number <= 0:
        raise GateError(f"issue number must be positive: {number}")
    return number


def _normalized_state(value: Any) -> str:
    """Normalize a GitHub issue state."""
    return str(value or "").strip().upper()


def _normalize_base_ref(base_ref: str, *, remote: str) -> str:
    """Normalize a bare or remote-qualified base branch to its branch name."""
    normalized = str(base_ref or "").strip()
    if not normalized:
        raise GateError("base ref must not be empty")
    remote_prefix = f"{remote}/"
    if normalized.startswith(remote_prefix):
        normalized = normalized[len(remote_prefix) :]
    if not normalized:
        raise GateError(f"base ref {base_ref!r} does not contain a branch name")
    return normalized


def _repo_slug_from_remote_url(remote_url: str) -> str:
    """Convert a Git remote URL to the GitHub CLI repository argument format."""
    value = str(remote_url or "").strip()
    if not value:
        raise GateError("Git remote URL is empty")

    host: str | None = None
    path = ""
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
    else:
        parsed = urlparse(value)
        host = parsed.hostname
        path = parsed.path

    components = [component for component in path.strip("/").split("/") if component]
    if components and components[-1].endswith(".git"):
        components[-1] = components[-1][:-4]
    if len(components) != 2 or not all(_REPO_COMPONENT_RE.fullmatch(item) for item in components):
        raise GateError(f"Git remote URL must identify an OWNER/REPO pair; got {remote_url!r}")

    normalized_host = str(host or "").lower()
    if normalized_host in {"", "github.com", "www.github.com"}:
        return "/".join(components)
    if not _REPO_COMPONENT_RE.fullmatch(normalized_host):
        raise GateError(f"Git remote URL has an invalid host: {remote_url!r}")
    return "/".join([normalized_host, *components])


def _normalize_repo_argument(repo: str, *, remote: str) -> str:
    """Normalize a GitHub repository slug or a local checkout path."""
    value = str(repo or "").strip()
    if not value:
        raise GateError(
            "repository must be OWNER/REPO (for example, ll7/robot_sf_ll7) "
            "or an existing local checkout path"
        )

    path = Path(value).expanduser()
    explicit_path = value.startswith((".", "/", "~")) or path.exists()
    if not explicit_path and 2 <= value.count("/") + 1 <= 3:
        components = value.split("/")
        if all(_REPO_COMPONENT_RE.fullmatch(component) for component in components):
            return value

    if not explicit_path:
        raise GateError(
            "repository must be OWNER/REPO (for example, ll7/robot_sf_ll7) "
            "or an existing local checkout path"
        )
    if not path.is_dir():
        raise GateError(f"local repository path does not exist or is not a directory: {repo!r}")

    resolved_path = path.resolve()
    result = _run(
        ["git", "-C", str(resolved_path), "remote", "get-url", remote],
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "remote lookup failed"
        raise GateError(
            f"local repository path {repo!r} has no usable Git remote {remote!r}: {detail}; "
            "pass an explicit OWNER/REPO value if the checkout has no GitHub remote"
        )
    try:
        return _repo_slug_from_remote_url(result.stdout)
    except GateError as exc:
        raise GateError(
            f"local repository path {repo!r} has no GitHub-compatible remote {remote!r}: {exc}"
        ) from exc


def _is_graphql_fallback_error(error: GateError) -> bool:
    """Return whether a native GitHub read is eligible for REST fallback."""
    message = str(error).lower()
    if any(marker in message for marker in GRAPHQL_FAIL_CLOSED_MARKERS):
        return False
    return any(marker in message for marker in GRAPHQL_FALLBACK_MARKERS)


def _fetch_refs(*, remote: str, base_ref: str, branch: str) -> tuple[str, str | None]:
    """Refresh the base ref and read the remote publication branch tip."""
    base_ref = _normalize_base_ref(base_ref, remote=remote)
    _run(
        [
            "git",
            "fetch",
            "--no-tags",
            remote,
            f"refs/heads/{base_ref}:refs/remotes/{remote}/{base_ref}",
        ]
    )
    base_sha = _git_output("rev-parse", f"refs/remotes/{remote}/{base_ref}")
    branch_result = _run(
        ["git", "ls-remote", "--heads", remote, f"refs/heads/{branch}"], check=False
    )
    if branch_result.returncode != 0:
        detail = branch_result.stderr.strip() or branch_result.stdout.strip() or "command failed"
        raise GateError(f"git ls-remote --heads {remote} {branch}: {detail}")
    lines = [line.split() for line in branch_result.stdout.splitlines() if line.split()]
    if not lines:
        return base_sha, None
    if len(lines) > 1 or len(lines[0]) < 1:
        raise GateError(f"ambiguous remote branch result for {remote}/{branch}")
    return base_sha, lines[0][0]


def _fetch_remote_branch(*, remote: str, branch: str) -> None:
    """Materialize a changed remote branch as a local remote-tracking ref."""
    _run(
        [
            "git",
            "fetch",
            "--no-tags",
            remote,
            f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
        ]
    )


def _closing_issue_numbers(searchable: str, *, repo: str) -> set[int]:
    """Extract repository-qualified closing references from PR text."""
    closes: set[int] = set()
    for match in _CLOSING_REFERENCE_RE.finditer(searchable):
        qualifier = match.group("url_repo") or match.group("repo")
        if qualifier and qualifier.casefold() != repo.casefold():
            continue
        closes.add(int(match.group("number")))
    return closes


def _normalize_closing_pr_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one REST pull row to the native closing-PR snapshot shape."""
    head = row.get("head") if isinstance(row.get("head"), dict) else {}
    base = row.get("base") if isinstance(row.get("base"), dict) else {}
    merge_sha = row.get("merge_commit_sha")
    return {
        "number": row.get("number"),
        "title": row.get("title"),
        "merged_at": row.get("merged_at"),
        "merge_commit": {"oid": merge_sha} if merge_sha else None,
        "head_ref": head.get("ref"),
        "head_sha": head.get("sha"),
        "base_ref": base.get("ref"),
    }


def _normalize_open_pr_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one open REST pull row to the covering-PR snapshot shape."""
    head = row.get("head") if isinstance(row.get("head"), dict) else {}
    base = row.get("base") if isinstance(row.get("base"), dict) else {}
    return {
        "number": row.get("number"),
        "title": row.get("title"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "is_draft": bool(row.get("draft")),
        "head_ref": head.get("ref"),
        "head_sha": head.get("sha"),
        "base_ref": base.get("ref"),
    }


def _closing_prs(*, repo: str, issue: int) -> list[dict[str, Any]]:
    """Return merged PRs whose body or title closes *issue* explicitly."""
    payload = _json_command(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--search",
            str(issue),
            "--limit",
            "100",
            "--json",
            "number,title,body,mergedAt,mergeCommit,headRefName,headRefOid,baseRefName",
        ]
    )
    if not isinstance(payload, list):
        raise GateError("gh pr list returned a non-list payload")

    matches: list[dict[str, Any]] = []
    for pull_request in payload:
        if not isinstance(pull_request, dict):
            continue
        searchable = "\n".join(str(pull_request.get(field) or "") for field in ("title", "body"))
        if issue not in _closing_issue_numbers(searchable, repo=repo):
            continue
        matches.append(
            {
                "number": pull_request.get("number"),
                "title": pull_request.get("title"),
                "merged_at": pull_request.get("mergedAt"),
                "merge_commit": pull_request.get("mergeCommit"),
                "head_ref": pull_request.get("headRefName"),
                "head_sha": pull_request.get("headRefOid"),
                "base_ref": pull_request.get("baseRefName"),
            }
        )
    return sorted(matches, key=lambda item: str(item.get("merged_at") or ""))


def _open_covering_prs(*, repo: str, issue: int) -> list[dict[str, Any]]:
    """Return open PRs whose body or title explicitly closes *issue*."""
    payload = _json_command(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--search",
            str(issue),
            "--limit",
            "100",
            "--json",
            "number,title,body,createdAt,updatedAt,isDraft,headRefName,headRefOid,baseRefName",
        ]
    )
    if not isinstance(payload, list):
        raise GateError("gh pr list returned a non-list payload")

    matches: list[dict[str, Any]] = []
    for pull_request in payload:
        if not isinstance(pull_request, dict):
            continue
        searchable = "\n".join(str(pull_request.get(field) or "") for field in ("title", "body"))
        if issue not in _closing_issue_numbers(searchable, repo=repo):
            continue
        matches.append(
            {
                "number": pull_request.get("number"),
                "title": pull_request.get("title"),
                "created_at": pull_request.get("createdAt"),
                "updated_at": pull_request.get("updatedAt"),
                "is_draft": bool(pull_request.get("isDraft")),
                "head_ref": pull_request.get("headRefName"),
                "head_sha": pull_request.get("headRefOid"),
                "base_ref": pull_request.get("baseRefName"),
            }
        )
    return sorted(matches, key=lambda item: str(item.get("created_at") or ""))


def _rest_json(
    path: str,
    *,
    context: str,
    gh_api: Any = None,
) -> Any:
    """Read and parse one GitHub REST JSON endpoint."""
    if gh_api is None:
        from scripts.dev.gh_issue_rest import _gh_api

        gh_api = _gh_api
    result = gh_api(path)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise GateError(f"GitHub REST read failed ({context}): {detail or result.returncode}")
    try:
        return json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        raise GateError(f"GitHub REST returned invalid JSON ({context}): {exc.msg}") from exc


def _issue_state_rest(*, repo: str, issue: int, gh_api: Any = None) -> dict[str, Any]:
    """Read issue state and timestamps through the REST issue endpoint."""
    path = f"repos/{repo}/issues/{issue}"
    payload = _rest_json(path, context=path, gh_api=gh_api)
    if not isinstance(payload, dict):
        raise GateError(f"GitHub REST issue payload was not an object ({path})")
    issue_state = _normalized_state(payload.get("state"))
    if issue_state not in {"OPEN", "CLOSED"}:
        raise GateError(f"GitHub REST issue returned unknown state: {issue_state!r}")
    return {
        "state": issue_state,
        "updatedAt": payload.get("updated_at"),
        "closedAt": payload.get("closed_at"),
    }


def _closing_prs_rest(
    *, repo: str, issue: int, max_pages: int = DEFAULT_MAX_REST_PR_PAGES
) -> list[dict[str, Any]]:
    """Find merged PRs closing an issue through a bounded REST inventory."""
    try:
        rows, meta = fetch_closed_pr_rows(
            repo=repo,
            max_pages=max_pages,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GateError(f"GitHub REST merged-PR fallback failed: {exc}") from exc
    if meta.truncated:
        raise GateError(
            "GitHub REST merged-PR inventory is truncated: "
            f"read {meta.row_count} rows in {meta.pages_read}/{meta.page_budget} pages; "
            "raise the REST page budget before publication"
        )

    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("merged_at"):
            continue
        searchable = "\n".join(str(row.get(field) or "") for field in ("title", "body"))
        if issue in _closing_issue_numbers(searchable, repo=repo):
            matches.append(_normalize_closing_pr_row(row))
    return sorted(matches, key=lambda item: str(item.get("merged_at") or ""))


def _open_covering_prs_rest(
    *, repo: str, issue: int, max_pages: int = DEFAULT_MAX_REST_PR_PAGES
) -> list[dict[str, Any]]:
    """Find open PRs covering an issue through a bounded REST inventory."""
    try:
        rows, meta = fetch_open_pr_rows(
            repo=repo,
            max_pages=max_pages,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GateError(f"GitHub REST open-PR fallback failed: {exc}") from exc
    if meta.truncated:
        raise GateError(
            "GitHub REST open-PR inventory is truncated: "
            f"read {meta.row_count} rows in {meta.pages_read}/{meta.page_budget} pages; "
            "raise the REST page budget before publication"
        )

    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        searchable = "\n".join(str(row.get(field) or "") for field in ("title", "body"))
        if issue in _closing_issue_numbers(searchable, repo=repo):
            matches.append(_normalize_open_pr_row(row))
    return sorted(matches, key=lambda item: str(item.get("created_at") or ""))


def collect_live_state(
    *,
    repo: str,
    issue: int,
    branch: str,
    base_ref: str = "main",
    remote: str = "origin",
    max_pr_pages: int = DEFAULT_MAX_REST_PR_PAGES,
    declaration_text: str = "",
) -> dict[str, Any]:
    """Fetch and assemble the live issue, ref, and local worktree state.

    ``declaration_text`` optionally carries the canonical ``## Stack
    Declaration`` text (issue #7515) so the recorded snapshot's ``ancestry``
    block can validate it against the live local git refs.
    """
    max_pr_pages = _effective_page_budget(max_pr_pages)
    base_ref = _normalize_base_ref(base_ref, remote=remote)
    remote_state_sources = {
        "issue": "graphql",
        "closing_prs": "graphql",
        "open_covering_prs": "graphql",
    }
    remote_state_fallbacks: dict[str, str] = {}
    try:
        issue_payload = _json_command(
            [
                "gh",
                "issue",
                "view",
                str(issue),
                "--repo",
                repo,
                "--json",
                "state,updatedAt,closedAt",
            ]
        )
        if not isinstance(issue_payload, dict):
            raise GateError("gh issue view returned a non-object payload")
        issue_state = _normalized_state(issue_payload.get("state"))
        if issue_state not in {"OPEN", "CLOSED"}:
            raise GateError(f"gh issue view returned unknown state: {issue_state!r}")
    except GateError as exc:
        if not _is_graphql_fallback_error(exc):
            raise
        issue_payload = _issue_state_rest(repo=repo, issue=issue)
        remote_state_sources["issue"] = "rest"
        remote_state_fallbacks["issue"] = str(exc)
        issue_state = issue_payload["state"]

    base_sha, remote_branch_sha = _fetch_refs(
        remote=remote,
        base_ref=base_ref,
        branch=branch,
    )
    try:
        closing_prs = _closing_prs(repo=repo, issue=issue)
    except GateError as exc:
        if not _is_graphql_fallback_error(exc):
            raise
        closing_prs = _closing_prs_rest(repo=repo, issue=issue, max_pages=max_pr_pages)
        remote_state_sources["closing_prs"] = "rest"
        remote_state_fallbacks["closing_prs"] = str(exc)
    try:
        open_covering_prs = _open_covering_prs(repo=repo, issue=issue)
    except GateError as exc:
        if not _is_graphql_fallback_error(exc):
            raise
        open_covering_prs = _open_covering_prs_rest(repo=repo, issue=issue, max_pages=max_pr_pages)
        remote_state_sources["open_covering_prs"] = "rest"
        remote_state_fallbacks["open_covering_prs"] = str(exc)

    snapshot = {
        "schema": SCHEMA,
        "kind": "snapshot",
        "captured_at_utc": _utc_now(),
        "repo": repo,
        "issue": issue,
        "issue_state": issue_state,
        "issue_updated_at": issue_payload.get("updatedAt"),
        "issue_closed_at": issue_payload.get("closedAt"),
        "closing_prs": closing_prs,
        "open_covering_prs": open_covering_prs,
        "remote_state_sources": remote_state_sources,
        "remote_state_fallbacks": remote_state_fallbacks,
        "rest_pr_page_budget": max_pr_pages,
        "remote": remote,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "branch": branch,
        "remote_branch_sha": remote_branch_sha,
        "local_head_sha": _git_output("rev-parse", "HEAD"),
        "tree_state": _tree_state(),
    }
    if declaration_text:
        snapshot["stack_declaration"] = declaration_text
    return _record_ancestry(snapshot)


def _sha_fields(snapshot: dict[str, Any]) -> dict[str, str | None]:
    """Return the exact SHAs relevant to publication freshness."""
    return {
        "base_sha": snapshot.get("base_sha"),
        "remote_branch_sha": snapshot.get("remote_branch_sha"),
        "local_head_sha": snapshot.get("local_head_sha"),
    }


def _validate_snapshot_pair(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject state records from another schema before comparing them."""
    if baseline.get("schema") != SCHEMA or baseline.get("kind") != "snapshot":
        raise GateError("baseline is not a prepublication snapshot")
    if current.get("schema") != SCHEMA or current.get("kind") != "snapshot":
        raise GateError("current state is not a prepublication snapshot")


def _new_closing_prs(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Return closing PRs observed after the baseline."""
    baseline_prs = {
        str(item.get("number"))
        for item in baseline.get("closing_prs", [])
        if isinstance(item, dict)
    }
    return [
        item
        for item in current.get("closing_prs", [])
        if isinstance(item, dict) and str(item.get("number")) not in baseline_prs
    ]


def _new_open_covering_prs(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return open covering PRs observed after the baseline."""
    baseline_prs = {
        str(item.get("number"))
        for item in baseline.get("open_covering_prs", [])
        if isinstance(item, dict)
    }
    return [
        item
        for item in current.get("open_covering_prs", [])
        if isinstance(item, dict) and str(item.get("number")) not in baseline_prs
    ]


def _sha_drift(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, dict[str, str | None]]:
    """Return changed base, remote-branch, or local-head SHA fields."""
    drift: dict[str, dict[str, str | None]] = {}
    for field in ("base_sha", "remote_branch_sha", "local_head_sha"):
        before = baseline.get(field)
        after = current.get(field)
        if before != after:
            drift[field] = {"baseline": before, "current": after}
    return drift


def _ancestry_blocking_state(current: dict[str, Any]) -> str | None:
    """Return a blocking ancestry state recorded in a live snapshot, or None.

    The snapshot's ``ancestry`` block is written by ``collect_live_state`` and
    carries the deterministic ``state`` from ``ancestry_state``.  Only the
    fail-closed states (``undeclared_stack``, ``mismatched_declaration``,
    ``parent_invalidated``) block pre-PR publication; a declared stack is
    permitted to be published but is never independently merge-ready.
    """
    ancestry = current.get("ancestry")
    if not isinstance(ancestry, dict):
        return None
    state = str(ancestry.get("state") or "")
    if state in BLOCKING_STATES:
        return state
    return None


def _record_ancestry(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Populate (or refresh) the snapshot's ``ancestry`` block from local git.

    Runs the deterministic ancestry classifier (issue #7515) against the live
    local git refs of the publication worktree.  The declaration is parsed from
    the snapshot's recorded ``stack_declaration`` text (the PR body that will be
    published, if any).  When the classifier cannot run (missing SHA/ref), the
    snapshot records the failure fail-closed under ``ancestry.error``.
    """
    head_sha = str(snapshot.get("local_head_sha") or "")
    base_ref = str(snapshot.get("base_ref") or "")
    remote = str(snapshot.get("remote") or "origin")
    if not head_sha or not base_ref:
        snapshot["ancestry"] = {"state": "unknown", "error": "snapshot lacks head/base ref"}
        return snapshot
    facts, error = collect_ancestry_facts(
        head_sha=head_sha,
        base_ref=base_ref,
        worktree=Path.cwd(),
        remote=remote,
    )
    if error or facts is None:
        snapshot["ancestry"] = {"state": "unknown", "error": error}
        return snapshot
    declaration_text = str(snapshot.get("stack_declaration") or "")
    declaration, parse_error = parse_stack_declaration(declaration_text)
    if parse_error:
        snapshot["ancestry"] = {"state": "unknown", "error": parse_error}
        return snapshot
    state = ancestry_state(
        head_sha=head_sha,
        base_ref=base_ref,
        main_tip_sha=facts["main_tip_sha"],
        merge_base_sha=facts["merge_base_sha"],
        commits=facts["commits"],
        declaration=declaration,
    )
    state["unexpected_paths"] = facts["changed_paths"]
    snapshot["ancestry"] = state
    return snapshot


def _drift_reason(drift: dict[str, dict[str, str | None]]) -> str:
    """Map changed SHA fields to the canonical publication-freshness reason."""
    if "remote_branch_sha" in drift:
        return "remote_branch_changed"
    if "base_sha" in drift:
        return "base_changed"
    return "local_head_changed"


def evaluate_state(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Classify a baseline/current pair without performing external I/O."""
    _validate_snapshot_pair(baseline, current)

    baseline_issue_state = _normalized_state(baseline.get("issue_state"))
    current_issue_state = _normalized_state(current.get("issue_state"))
    if baseline_issue_state != "OPEN":
        decision = "superseded" if baseline_issue_state == "CLOSED" else "blocked"
        reason = (
            "baseline_issue_closed" if decision == "superseded" else "baseline_issue_state_unknown"
        )
        return _decision(baseline, current, decision=decision, reason=reason)
    if current_issue_state == "CLOSED":
        return _decision(baseline, current, decision="superseded", reason="issue_closed")
    if current_issue_state != "OPEN":
        return _decision(
            baseline,
            current,
            decision="blocked",
            reason="issue_state_unknown",
        )

    new_open_covering_prs = _new_open_covering_prs(baseline, current)
    if new_open_covering_prs:
        return _decision(
            baseline,
            current,
            decision="superseded",
            reason="open_pr_closes_issue",
            extra={"new_open_covering_prs": new_open_covering_prs},
        )

    new_closing_prs = _new_closing_prs(baseline, current)
    if new_closing_prs:
        return _decision(
            baseline,
            current,
            decision="superseded",
            reason="merged_pr_closes_issue",
            extra={"new_closing_prs": new_closing_prs},
        )
    if current.get("tree_state") != "clean":
        return _decision(baseline, current, decision="blocked", reason="dirty_worktree")

    ancestry_state_value = _ancestry_blocking_state(current)
    if ancestry_state_value is not None:
        return _decision(
            baseline,
            current,
            decision="blocked",
            reason="undeclared_stack_ancestry",
            extra={"ancestry": current.get("ancestry")},
        )

    drift = _sha_drift(baseline, current)
    if drift:
        return _decision(
            baseline,
            current,
            decision="refresh-required",
            reason=_drift_reason(drift),
            extra={"drift": drift},
        )
    return _decision(baseline, current, decision="ready", reason="remote_state_unchanged")


def _decision(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    decision: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a decision record containing exact before/after provenance."""
    payload: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "kind": "decision",
        "recorded_at_utc": _utc_now(),
        "decision": decision,
        "reason": reason,
        "repo": current.get("repo"),
        "issue": current.get("issue"),
        "branch": current.get("branch"),
        "remote": current.get("remote"),
        "base_ref": current.get("base_ref"),
        "exact_shas": {
            "baseline": _sha_fields(baseline),
            "current": _sha_fields(current),
        },
        "baseline": baseline,
        "current": current,
    }
    if extra:
        payload.update(extra)
    return payload


def _default_snapshot_path(branch: str) -> Path:
    """Return the ignored local path used for a branch's publication snapshot."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", branch).strip("-") or "detached-head"
    digest = hashlib.sha256(branch.encode("utf-8")).hexdigest()[:8]
    return Path("output/validation/prepublication") / f"{safe}-{digest}.json"


def _decision_path(snapshot_path: Path, explicit: str | None) -> Path:
    """Return the decision record path paired with a snapshot."""
    if explicit:
        return Path(explicit)
    return snapshot_path.with_name(f"{snapshot_path.stem}.decision.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a generated state record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_snapshot(path: Path) -> dict[str, Any]:
    """Load a previously captured snapshot."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read snapshot {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError(f"snapshot {path} is not a JSON object")
    if payload.get("schema") != SCHEMA or payload.get("kind") != "snapshot":
        raise GateError(f"snapshot {path} has an unsupported schema")
    return payload


def _write_error(path: Path | None, error: str) -> dict[str, Any]:
    """Build a fail-closed error record."""
    payload = {
        "schema": DECISION_SCHEMA,
        "kind": "decision",
        "recorded_at_utc": _utc_now(),
        "decision": "blocked",
        "reason": "state_collection_failed",
        "error": error,
    }
    if path is not None:
        _write_json(path, payload)
    return payload


def _integrate_targets(*, remote: str, branch: str, targets: list[str]) -> dict[str, Any]:
    """Merge refreshed remote refs without resetting or deleting local state."""
    if _tree_state() != "clean":
        return {"ok": False, "reason": "dirty_worktree"}
    if any(target.endswith(f"/{branch}") for target in targets):
        _fetch_remote_branch(remote=remote, branch=branch)
    merged: list[str] = []
    for target in targets:
        result = _run(["git", "merge", "--no-edit", target], check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "merge failed"
            abort = _run(["git", "merge", "--abort"], check=False)
            return {
                "ok": False,
                "reason": "integration_conflict",
                "target": target,
                "detail": detail,
                "merged": merged,
                "merge_aborted": abort.returncode == 0,
            }
        merged.append(target)
    return {"ok": True, "merged": merged}


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Fail-closed remote-state gate for issue-to-PR publication."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture a pre-publication baseline")
    capture.add_argument(
        "--repo",
        required=True,
        help=(
            "GitHub repository as OWNER/REPO (for example, ll7/robot_sf_ll7), "
            "or a local checkout path whose Git remote resolves to that repository"
        ),
    )
    capture.add_argument("--issue", type=int, required=True)
    capture.add_argument("--branch")
    capture.add_argument(
        "--base-ref",
        default="main",
        help="base branch name or remote-qualified branch such as origin/main",
    )
    capture.add_argument("--remote", default="origin")
    capture.add_argument(
        "--max-pr-pages",
        type=_positive_page_budget,
        default=DEFAULT_MAX_REST_PR_PAGES,
        help=(
            "maximum REST pages used for merged/open PR fallback discovery "
            f"(default: {DEFAULT_MAX_REST_PR_PAGES})"
        ),
    )
    capture.add_argument("--snapshot-path")
    capture.add_argument(
        "--declaration-text",
        help=(
            "canonical ## Stack Declaration text (parent_pr + parent_head) used to "
            "validate non-main ancestry before publication (issue #7515)"
        ),
    )

    for name, help_text in (
        ("check", "check a baseline against refreshed remote state"),
        ("sync", "integrate changed remote refs and capture a refreshed baseline"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--snapshot-path", required=True)
        command.add_argument("--decision-path")
        command.add_argument(
            "--max-pr-pages",
            type=_positive_page_budget,
            default=None,
            help=(
                "override the REST page budget; otherwise reuse the capture budget "
                "recorded in the snapshot"
            ),
        )
        if name == "sync":
            command.add_argument(
                "--integrate",
                action="store_true",
                help="merge changed base/remote refs into the clean local branch",
            )
    return parser


def _exit_code(decision: str) -> int:
    """Map a decision to a stable shell exit code."""
    return EXIT_CODES.get(decision, EXIT_CODES["blocked"])


def main(argv: list[str] | None = None) -> int:
    """Run the capture, check, or sync command."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            repo = _normalize_repo_argument(args.repo, remote=args.remote)
            issue = _issue_number(args.issue)
            branch = args.branch or _git_branch()
            snapshot_path = (
                Path(args.snapshot_path) if args.snapshot_path else _default_snapshot_path(branch)
            )
            snapshot = collect_live_state(
                repo=repo,
                issue=issue,
                branch=branch,
                base_ref=args.base_ref,
                remote=args.remote,
                max_pr_pages=args.max_pr_pages,
                declaration_text=args.declaration_text or "",
            )
            _write_json(snapshot_path, snapshot)
            decision = "ready" if snapshot["issue_state"] == "OPEN" else "superseded"
            payload = {
                "schema": DECISION_SCHEMA,
                "kind": "capture",
                "recorded_at_utc": _utc_now(),
                "decision": decision,
                "reason": "baseline_captured" if decision == "ready" else "issue_not_open",
                "snapshot_path": str(snapshot_path),
                "snapshot": snapshot,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return _exit_code(decision)

        snapshot_path = Path(args.snapshot_path)
        baseline = _load_snapshot(snapshot_path)
        baseline["base_ref"] = _normalize_base_ref(
            str(baseline["base_ref"]), remote=str(baseline["remote"])
        )
        repo = _normalize_repo_argument(str(baseline["repo"]), remote=str(baseline["remote"]))
        max_pr_pages = _effective_page_budget(args.max_pr_pages, snapshot=baseline)
        decision_path = _decision_path(snapshot_path, args.decision_path)
        current = collect_live_state(
            repo=repo,
            issue=_issue_number(baseline["issue"]),
            branch=str(baseline["branch"]),
            base_ref=str(baseline["base_ref"]),
            remote=str(baseline["remote"]),
            max_pr_pages=max_pr_pages,
        )
        decision = evaluate_state(baseline, current)
        if (
            args.command == "check"
            or decision["decision"] != "refresh-required"
            or not getattr(args, "integrate", False)
        ):
            decision["decision_path"] = str(decision_path)
            _write_json(decision_path, decision)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return _exit_code(str(decision["decision"]))

        drift = decision.get("drift", {})
        targets: list[str] = []
        if "base_sha" in drift:
            targets.append(f"refs/remotes/{baseline['remote']}/{baseline['base_ref']}")
        if "remote_branch_sha" in drift and current.get("remote_branch_sha"):
            targets.append(f"refs/remotes/{baseline['remote']}/{baseline['branch']}")
        integration = _integrate_targets(
            remote=str(baseline["remote"]),
            branch=str(baseline["branch"]),
            targets=targets,
        )
        if not integration.get("ok"):
            decision["integration"] = integration
            decision["decision_path"] = str(decision_path)
            _write_json(decision_path, decision)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return EXIT_CODES["refresh-required"]

        refreshed = collect_live_state(
            repo=repo,
            issue=_issue_number(baseline["issue"]),
            branch=str(baseline["branch"]),
            base_ref=str(baseline["base_ref"]),
            remote=str(baseline["remote"]),
            max_pr_pages=max_pr_pages,
        )
        _write_json(snapshot_path, refreshed)
        refreshed_decision = evaluate_state(refreshed, refreshed)
        refreshed_decision.update(
            {
                "reason": "remote_state_integrated",
                "comparison": "self_snapshot_after_integration",
                "integrated": integration.get("merged", []),
                "snapshot_path": str(snapshot_path),
                "decision_path": str(decision_path),
            }
        )
        _write_json(decision_path, refreshed_decision)
        print(json.dumps(refreshed_decision, indent=2, sort_keys=True))
        return _exit_code(str(refreshed_decision["decision"]))
    except GateError as exc:
        decision_path: Path | None = None
        if args.command in {"check", "sync"}:
            decision_path = _decision_path(Path(args.snapshot_path), args.decision_path)
        payload = _write_error(decision_path, str(exc))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_CODES["blocked"]


if __name__ == "__main__":
    raise SystemExit(main())
