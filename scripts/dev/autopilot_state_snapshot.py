#!/usr/bin/env python3
"""Emit compact goal-autopilot state snapshots.

The snapshot is route evidence for orientation and handoff.  It is not a substitute for fresh
local checks before publishing, labeling, merging, or making benchmark-facing claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.dev import goal_issue_admission
from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev.check_pr_ci_status import _latest_check_runs
from scripts.dev.routed_worker_manifest import SCHEMA_VERSION as ROUTE_MANIFEST_SCHEMA
from scripts.dev.routed_worker_manifest import classify_worker_output

FAILURE_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "error",
    "failure",
    "startup_failure",
    "timed_out",
}
PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}
GENERATED_STATUS_PATHS = (
    ".venv",
    ".opencode",
    "node_modules",
    "output",
    ".pytest_cache",
    "__pycache__",
)
STATUS_LINE_LIMIT = 30
ROUTE_TERMINAL_FAILURES = frozenset(
    {
        "timeout",
        "exception",
        "non_zero_exit",
        "missing_artifact",
        "route_not_started",
        "scope_violation",
        "unavailable",
    }
)
ROUTE_TERMINAL_STATES = ROUTE_TERMINAL_FAILURES | {"none"}
TOKEN_EFFICIENCY_ACTIONS = (
    "refresh this snapshot or the active ledger before reopening full skill/docs context",
    "prefer compact issue/PR/CI helpers before broad gh, git worktree, or rg output",
    "review delegate artifacts in result -> validation -> diffstat order before raw logs",
    "store verbose validation, worker, and CI logs outside the parent thread",
    "record unavailable worker routes and reset times before retrying delegation",
)
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"


@dataclass(frozen=True)
class CommandResult:
    """Captured command result for compact provenance."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _run(command: list[str], *, timeout: int = 30) -> CommandResult:
    """Run a command and capture text output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=tuple(command),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=tuple(command),
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )


def _now_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _command_source(result: CommandResult, *, name: str) -> dict[str, Any]:
    """Return compact command provenance."""
    return {
        "name": name,
        "command": list(result.command),
        "returncode": result.returncode,
    }


def _parse_json_result(result: CommandResult) -> tuple[Any | None, str | None]:
    """Parse a JSON command result."""
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"json_parse_error: {exc}"


def _route_attempt_snapshot(attempt: Any) -> dict[str, Any]:
    """Return compact failure evidence for one routed-worker attempt."""
    if not isinstance(attempt, dict):
        return {
            "attempt_index": None,
            "route": None,
            "returncode": None,
            "failure_class": None,
            "terminal_state": None,
            "run_dir": None,
            "missing_artifacts": [],
            "missing_artifacts_known": False,
            "compact_artifacts_available": False,
            "scope_check": None,
            "aggregation": "inconclusive",
            "aggregation_reason": "malformed_attempt",
            "output_contract": {
                "status": "inconclusive",
                "aggregation": "inconclusive",
                "reason": "malformed_attempt",
                "missing_evidence": ["malformed_attempt"],
            },
            "malformed": True,
        }

    compact = attempt.get("compact_artifacts")
    compact_artifacts_available = isinstance(compact, dict)
    missing_artifacts: list[str] = []
    if compact_artifacts_available:
        for key, value in sorted(compact.items()):
            if not isinstance(value, dict) or value.get("present") is not True:
                missing_artifacts.append(str(key))
    terminal_state = attempt.get("terminal_state")
    output_contract = classify_worker_output(
        attempt,
        terminal_state=terminal_state,
        compact_artifacts=compact if isinstance(compact, dict) else {},
    )
    return {
        "attempt_index": attempt.get("attempt_index"),
        "route": attempt.get("route"),
        "returncode": attempt.get("returncode"),
        "failure_class": attempt.get("failure_class"),
        "terminal_state": terminal_state,
        "terminal_state_known": terminal_state is not None,
        "run_dir": attempt.get("run_dir"),
        "missing_artifacts": missing_artifacts,
        "missing_artifacts_known": compact_artifacts_available,
        "compact_artifacts_available": compact_artifacts_available,
        "scope_check": attempt.get("scope_check"),
        "aggregation": output_contract["aggregation"],
        "aggregation_reason": output_contract["reason"],
        "output_contract": output_contract,
    }


def route_manifest_snapshot(  # noqa: C901, PLR0912 - explicit fail-closed parser states
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Load a compact route-failure handoff without reading raw worker logs."""
    path = Path(manifest_path).resolve(strict=False)
    base = {
        "manifest_path": str(path),
        "route_evidence_only": True,
        "acceptance_state": "not_established",
    }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "unavailable",
            "error": f"route manifest unavailable: {exc}",
            "next_action": "inspect_route_manifest_path_and_route_artifacts",
        }
    if not isinstance(raw, dict):
        return {
            **base,
            "status": "malformed",
            "error": "route manifest must contain a JSON object",
            "next_action": "inspect_route_manifest_path_and_route_artifacts",
        }

    schema = raw.get("schema")
    if schema != ROUTE_MANIFEST_SCHEMA:
        return {
            **base,
            "status": "malformed",
            "schema": schema,
            "error": (f"route manifest schema must be {ROUTE_MANIFEST_SCHEMA!r}; got {schema!r}"),
            "next_action": "inspect_route_manifest_path_and_route_artifacts",
        }

    attempts = raw.get("attempted_routes")
    if not isinstance(attempts, list):
        return {
            **base,
            "status": "malformed",
            "schema": raw.get("schema"),
            "error": "route manifest attempted_routes must be a JSON array",
            "next_action": "inspect_route_manifest_path_and_route_artifacts",
        }
    attempt_rows = [_route_attempt_snapshot(attempt) for attempt in attempts]
    malformed_attempts = [row for row in attempt_rows if row.get("malformed")]
    if malformed_attempts:
        return {
            **base,
            "status": "malformed",
            "schema": schema,
            "route_evidence_only": raw.get("route_evidence_only"),
            "chosen_route": raw.get("chosen_route"),
            "chosen_run_dir": raw.get("chosen_run_dir"),
            "error": "route manifest attempted_routes contains malformed attempt records",
            "next_action": "inspect_route_manifest_path_and_route_artifacts",
        }
    chosen_run_dir = raw.get("chosen_run_dir")
    chosen_terminal_state = raw.get("chosen_terminal_state")
    chosen: dict[str, Any] | None = None
    for row in attempt_rows:
        if row.get("run_dir") == chosen_run_dir:
            chosen = row
            break
    if chosen is None and attempt_rows:
        chosen = attempt_rows[0]
    if chosen_terminal_state is None and chosen is not None:
        chosen_terminal_state = chosen.get("terminal_state")

    terminal_states = [
        row.get("terminal_state") for row in attempt_rows if row.get("terminal_state") is not None
    ]
    if chosen_terminal_state is not None:
        terminal_states.append(chosen_terminal_state)
    invalid_terminal_states = [
        state
        for state in terminal_states
        if not isinstance(state, str) or state not in ROUTE_TERMINAL_STATES
    ]
    if invalid_terminal_states:
        return {
            **base,
            "status": "malformed",
            "schema": schema,
            "route_evidence_only": raw.get("route_evidence_only"),
            "chosen_route": raw.get("chosen_route"),
            "chosen_run_dir": chosen_run_dir,
            "chosen_terminal_state": chosen_terminal_state,
            "error": (
                "route manifest contains unsupported terminal_state values: "
                f"{sorted({repr(state) for state in invalid_terminal_states})!r}"
            ),
            "next_action": "inspect_route_manifest_path_and_route_artifacts",
        }

    missing_terminal_state = any(row.get("terminal_state") is None for row in attempt_rows)
    no_attempts = not attempt_rows
    missing_terminal_state = missing_terminal_state or no_attempts
    chosen_attempt_missing_terminal = chosen is not None and chosen.get("terminal_state") is None
    normalized_attempt_rows = [
        (
            {
                **row,
                "terminal_state": "unavailable",
                "failure_class": "missing_terminal_state",
            }
            if row.get("terminal_state") is None
            else row
        )
        for row in attempt_rows
    ]
    if chosen is not None:
        chosen = next(
            (row for row in normalized_attempt_rows if row.get("run_dir") == chosen.get("run_dir")),
            chosen,
        )
    if chosen_attempt_missing_terminal and chosen is not None:
        chosen_terminal_state = chosen.get("terminal_state")
    elif chosen_terminal_state is None and chosen is not None:
        chosen_terminal_state = chosen.get("terminal_state")

    failed_attempts = [
        row
        for row in normalized_attempt_rows
        if row.get("terminal_state") in ROUTE_TERMINAL_FAILURES
    ]
    incomplete_output_attempts = [
        row for row in normalized_attempt_rows if row.get("aggregation") == "inconclusive"
    ]
    chosen_output_contract = chosen.get("output_contract") if chosen else None
    aggregation = chosen.get("aggregation", "inconclusive") if chosen else "inconclusive"
    aggregation_reason = (
        chosen.get("aggregation_reason", "no_chosen_route") if chosen else "no_chosen_route"
    )
    reported_aggregation = raw.get("aggregation")
    if missing_terminal_state:
        aggregation = "unavailable"
        aggregation_reason = "terminal_state_unknown"
    elif reported_aggregation == "confirmed" and aggregation != "confirmed":
        aggregation = "inconclusive"
        aggregation_reason = "reported_confirmed_without_usable_worker_output"
    snapshot = {
        **base,
        "status": "unavailable" if missing_terminal_state else "ok",
        "schema": schema,
        "route_evidence_only": raw.get("route_evidence_only"),
        "chosen_route": raw.get("chosen_route"),
        "chosen_run_dir": chosen_run_dir,
        "chosen_terminal_state": chosen_terminal_state,
        "chosen_failure_class": chosen.get("failure_class") if chosen else None,
        "chosen_returncode": chosen.get("returncode") if chosen else None,
        "chosen_missing_artifacts": (
            chosen.get("missing_artifacts")
            if chosen and chosen.get("missing_artifacts_known")
            else None
        ),
        "chosen_scope_check": chosen.get("scope_check")
        if chosen
        else raw.get("chosen_scope_check"),
        "aggregation_contract": raw.get("aggregation_contract"),
        "aggregation": aggregation,
        "aggregation_reason": aggregation_reason,
        "chosen_output_contract": chosen_output_contract,
        "failed_attempts": failed_attempts,
        "incomplete_output_attempts": incomplete_output_attempts,
        "attempt_count": len(attempt_rows),
        "next_action": "inspect_parent_diff_and_run_local_validation",
    }
    if no_attempts:
        snapshot["error"] = "route manifest contains no attempted routes"
        snapshot["next_action"] = "inspect_route_manifest_path_and_route_artifacts"
    elif missing_terminal_state:
        snapshot["error"] = "route manifest attempt is missing terminal_state"
        snapshot["next_action"] = "inspect_route_manifest_path_and_route_artifacts"
    return snapshot


def _git_text(command: list[str], *, name: str) -> tuple[str, dict[str, Any], str | None]:
    """Run a git command that should return one line of text."""
    result = _run(command)
    source = _command_source(result, name=name)
    if result.returncode != 0:
        return "", source, (result.stderr or result.stdout).strip() or f"{name} failed"
    return result.stdout.strip(), source, None


def _append_worktree_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Append a parsed worktree row with stable default fields."""
    if not row:
        return
    row.setdefault("branch", "")
    row.setdefault("head_sha", "")
    row.setdefault("bare", False)
    row.setdefault("detached", False)
    rows.append(row)


def _parse_worktree_porcelain(stdout: str) -> list[dict[str, Any]]:
    """Parse `git worktree list --porcelain` into compact worktree rows."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                _append_worktree_row(rows, current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            if current:
                _append_worktree_row(rows, current)
            current = {"path": value}
        elif key == "HEAD":
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached"}:
            current[key] = True
    if current:
        _append_worktree_row(rows, current)
    return rows


def _bounded_lines(text: str, *, limit: int) -> tuple[list[str], bool]:
    """Return non-empty lines capped to a stable limit."""
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[:limit], len(lines) > limit


def _generated_paths_present() -> list[str]:
    """Return known generated roots that exist as files or directories."""
    return [path for path in GENERATED_STATUS_PATHS if Path(path).exists()]


def compact_status_snapshot() -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Return compact local status that avoids generated untracked trees."""
    result = _run(["git", "status", "--short", "--branch", "--untracked-files=no"])
    source = _command_source(result, name="git.status_compact")
    if result.returncode != 0:
        return (
            {
                "ok": False,
                "tracked_or_staged_count": 0,
                "tracked_or_staged": [],
                "tracked_or_staged_truncated": False,
                "generated_paths_present": [],
            },
            source,
            (result.stderr or result.stdout).strip() or "git compact status failed",
        )
    status_lines = [line for line in result.stdout.splitlines() if line.strip()]
    tracked_lines = [line for line in status_lines if not line.startswith("##")]
    lines, truncated = _bounded_lines("\n".join(tracked_lines), limit=STATUS_LINE_LIMIT)
    generated_paths = _generated_paths_present()
    return (
        {
            "ok": True,
            "tracked_or_staged_count": len(tracked_lines),
            "tracked_or_staged": lines,
            "tracked_or_staged_truncated": truncated,
            "generated_paths_present": generated_paths,
            "full_untracked_inventory_omitted": True,
        },
        source,
        None,
    )


def git_snapshot(
    *, include_worktrees: bool, worktree_limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Return compact local git state."""
    sources: list[dict[str, Any]] = []
    errors: list[str] = []

    branch, source, error = _git_text(["git", "branch", "--show-current"], name="git.branch")
    sources.append(source)
    if error:
        errors.append(error)

    head_sha, source, error = _git_text(["git", "rev-parse", "HEAD"], name="git.head")
    sources.append(source)
    if error:
        errors.append(error)

    origin_main_sha, source, error = _git_text(
        ["git", "rev-parse", "--verify", "origin/main^{commit}"],
        name="git.origin_main",
    )
    sources.append(source)
    if error:
        errors.append(error)

    worktrees: list[dict[str, Any]] = []
    if include_worktrees:
        result = _run(["git", "worktree", "list", "--porcelain"])
        sources.append(_command_source(result, name="git.worktrees"))
        if result.returncode == 0:
            worktrees = _parse_worktree_porcelain(result.stdout)
            worktrees.sort(
                key=lambda row: (
                    row.get("branch") != branch,
                    row.get("head_sha") != head_sha,
                    row.get("path", ""),
                )
            )
        else:
            errors.append((result.stderr or result.stdout).strip() or "git worktree list failed")
    worktree_count = len(worktrees)
    worktree_limit = max(0, worktree_limit)
    visible_worktrees = worktrees[:worktree_limit] if worktree_limit else []
    compact_status, status_source, status_error = compact_status_snapshot()
    sources.append(status_source)
    if status_error:
        errors.append(status_error)

    return (
        {
            "branch": branch,
            "head_sha": head_sha,
            "origin_main_sha": origin_main_sha,
            "head_matches_origin_main": bool(
                head_sha and origin_main_sha and head_sha == origin_main_sha
            ),
            "worktree_count": worktree_count,
            "worktrees_truncated": worktree_count > len(visible_worktrees),
            "worktrees": visible_worktrees,
            "compact_status": compact_status,
        },
        sources,
        errors,
    )


def controller_checkpoint(
    *,
    git: dict[str, Any],
    claims: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    prs: list[dict[str, Any]],
    errors: list[str],
    route_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a one-screen resume checkpoint for long Codex controller threads."""
    pr_next_actions = [
        {
            "number": pr.get("number"),
            "state": pr.get("state"),
            "checks": (pr.get("checks") or {}).get("overall"),
            "head_sha": pr.get("head_sha"),
        }
        for pr in prs
    ]
    stale_claims = [
        claim.get("issue") for claim in claims if claim.get("stale_against_origin_main")
    ]
    generated_paths = (git.get("compact_status") or {}).get("generated_paths_present", [])
    route_failures = route_failures or []
    route_failure_count = sum(
        max(
            len(row.get("failed_attempts", [])),
            len(row.get("incomplete_output_attempts", [])),
        )
        for row in route_failures
        if isinstance(row, dict)
    )
    route_incomplete_output_count = sum(
        len(row.get("incomplete_output_attempts", []))
        for row in route_failures
        if isinstance(row, dict)
    )
    next_action = "continue_from_snapshot"
    if errors:
        next_action = "repair_snapshot_errors"
    elif route_failure_count or any(
        isinstance(row, dict) and row.get("status") != "ok" for row in route_failures
    ):
        next_action = "inspect_route_failure_handoff"
    elif stale_claims:
        next_action = "refresh_stale_claims"
    elif any((pr.get("checks") or {}).get("overall") == "failure" for pr in prs):
        next_action = "inspect_failing_pr_checks"
    elif generated_paths:
        next_action = "use_compact_status_only"
    return {
        "route_evidence_only": True,
        "branch": git.get("branch", ""),
        "head_sha": git.get("head_sha", ""),
        "origin_main_sha": git.get("origin_main_sha", ""),
        "tracked_or_staged_count": (git.get("compact_status") or {}).get(
            "tracked_or_staged_count", 0
        ),
        "generated_paths_present": generated_paths,
        "claims": [
            {
                "issue": claim.get("issue"),
                "claimed": claim.get("claimed"),
                "stale": claim.get("stale_against_origin_main"),
            }
            for claim in claims
        ],
        "issue_numbers": [issue.get("number") for issue in issues],
        "prs": pr_next_actions,
        "route_failure_manifest_count": len(route_failures),
        "route_failure_attempt_count": route_failure_count,
        "route_incomplete_output_attempt_count": route_incomplete_output_count,
        "token_efficiency": {
            "parent_output_limit_lines": 200,
            "compact_first": True,
            "recommended_next_steps": list(TOKEN_EFFICIENCY_ACTIONS),
        },
        "next_action": next_action,
    }


def claim_snapshot(
    issue_numbers: list[int], *, remote: str, origin_main_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return compact claim-ref state for issue numbers."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    for issue in issue_numbers:
        claim_ref = f"refs/heads/agent-claims/issue-{issue}"
        result = _run(["git", "ls-remote", "--heads", remote, claim_ref])
        sources.append(_command_source(result, name=f"claim.issue_{issue}"))
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip() or "claim lookup failed"
            rows.append(
                {
                    "issue": issue,
                    "ok": False,
                    "claimed": None,
                    "claim_ref": claim_ref.removeprefix("refs/heads/"),
                    "sha": None,
                    "stale_against_origin_main": None,
                    "error": error,
                }
            )
            errors.append(f"issue {issue}: {error}")
            continue
        sha = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == claim_ref:
                sha = parts[0]
                break
        rows.append(
            {
                "issue": issue,
                "ok": True,
                "claimed": sha is not None,
                "claim_ref": claim_ref.removeprefix("refs/heads/"),
                "sha": sha,
                "stale_against_origin_main": bool(
                    sha and origin_main_sha and sha != origin_main_sha
                ),
                "error": None,
            }
        )
    return rows, sources, errors


def _queue_issue_admission(
    issue: dict[str, Any], *, repo: str = DEFAULT_REPO, remote: str = DEFAULT_REMOTE
) -> dict[str, Any]:
    """Return read-only admission evidence for one queue issue.

    Only ``state:ready`` rows need a live preflight.  Other rows are visibly
    ineligible before an atomic claim could be attempted, so the snapshot
    records that no claim check was needed and avoids spending one GitHub read
    per blocked queue item.
    """
    number = issue.get("number")
    labels = {
        str(label.get("name"))
        for label in issue.get("labels", []) or []
        if isinstance(label, dict) and label.get("name")
    }
    state = str(issue.get("state") or "").strip().upper()
    if state != "OPEN":
        reason = f"issue state is {state or 'unknown'}; skip autonomous claim"
        classification = "closed" if state else "state_unknown"
    elif "state:ready" not in labels:
        reason = "required label 'state:ready' is absent; skip autonomous claim"
        classification = "needs_ready_label"
    elif not isinstance(number, int) or number <= 0:
        reason = "issue number is invalid; skip autonomous claim"
        classification = "error"
    else:
        try:
            payload = goal_issue_admission.admit_issue(
                number,
                repo=repo,
                remote=remote,
                source_ref=goal_issue_admission.DEFAULT_SOURCE_REF,
                check_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "schema": goal_issue_admission.SCHEMA,
                "ok": False,
                "outcome": "error",
                "write_attempted": False,
                "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
                "classification": "error",
                "reasons": [str(exc)],
                "ready": False,
                "write_allowed": False,
                "claim": None,
                "claim_outcome": "unavailable",
            }
        return goal_issue_admission.compact_admission(payload)

    return {
        "schema": goal_issue_admission.SCHEMA,
        "ok": False,
        "outcome": "not_admitted",
        "write_attempted": False,
        "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
        "classification": classification,
        "reasons": [reason],
        "ready": False,
        "write_allowed": False,
        "claim": None,
        "claim_outcome": "not_checked",
    }


def issue_queue_snapshot(
    searches: list[str], *, limit: int, repo: str = DEFAULT_REPO, remote: str = DEFAULT_REMOTE
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Return compact issue rows for one or more GitHub issue searches.

    The fourth tuple element records, per search, whether the bounded
    ``gh issue list --limit`` result may have been silently capped. Downstream
    consumers should treat any ``truncated: true`` marker as incomplete evidence.
    """
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    truncations: list[dict[str, Any]] = []
    seen: set[int] = set()
    for search in searches:
        command = [
            "gh",
            "issue",
            "list",
            "--search",
            search,
            "--limit",
            str(limit),
            "--json",
            "number,title,state,labels,updatedAt,url",
        ]
        result = _run(command)
        sources.append(_command_source(result, name=f"issues.search:{search}"))
        data, error = _parse_json_result(result)
        if error:
            errors.append(f"issue search {search!r}: {error}")
            continue
        if not isinstance(data, list):
            errors.append(
                f"issue search {search!r}: expected JSON array, got {type(data).__name__}"
            )
            continue
        search_rows = data
        row_count = len(search_rows)
        truncated = is_likely_truncated(row_count, limit=limit)
        truncations.append(
            {
                "search": search,
                "truncated": truncated,
                "row_count": row_count,
                "limit": limit,
                "note": (
                    "gh issue list may be capped: got "
                    f"{row_count} rows at --limit {limit}; raise --limit or paginate"
                    if truncated
                    else ""
                ),
            }
        )
        for issue in search_rows:
            number = issue.get("number")
            if not isinstance(number, int) or number in seen:
                continue
            seen.add(number)
            admission = _queue_issue_admission(issue, repo=repo, remote=remote)
            if admission.get("outcome") == "error":
                errors.append(f"issue {number}: admission unavailable")
            labels = issue.get("labels", []) or []
            rows.append(
                {
                    "number": number,
                    "title": issue.get("title", ""),
                    "state": issue.get("state", ""),
                    "labels": sorted(
                        label.get("name", "")
                        for label in labels
                        if isinstance(label, dict) and label.get("name")
                    ),
                    "updated_at": issue.get("updatedAt", ""),
                    "url": issue.get("url", ""),
                    "admission": admission,
                }
            )
    return rows, sources, errors, truncations


def _rollup_conclusion(check: dict[str, Any]) -> str:
    """Return a normalized check conclusion."""
    return str(check.get("conclusion") or check.get("state") or "pending").lower()


def _rollup_status(check: dict[str, Any]) -> str:
    """Return a normalized check status."""
    status = check.get("status")
    if status:
        return str(status).lower()
    state = str(check.get("state") or "").lower()
    if state in {"success", "failure", "error"}:
        return "completed"
    return state or "completed"


def _check_name(check: dict[str, Any]) -> str:
    """Return a compact check name."""
    return str(check.get("name") or check.get("context") or "unknown")


def _checks_summary(rollup: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a PR statusCheckRollup payload after reconciling duplicate reruns."""
    valid_checks = [check for check in rollup if isinstance(check, dict)]
    effective_checks, superseded_count = _latest_check_runs(valid_checks)
    conclusions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for check in effective_checks:
        conclusion = _rollup_conclusion(check)
        status = _rollup_status(check)
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
    failure_count = sum(conclusions.get(conclusion, 0) for conclusion in FAILURE_CONCLUSIONS)
    pending_count = sum(statuses.get(status, 0) for status in PENDING_STATUSES)
    if failure_count:
        overall = "failure"
    elif pending_count or not effective_checks:
        overall = "pending"
    else:
        overall = "success"
    return {
        "overall": overall,
        "total": len(effective_checks),
        "superseded": superseded_count,
        "by_conclusion": conclusions,
        "by_status": statuses,
        "names": sorted({_check_name(check) for check in effective_checks}),
    }


def pr_snapshot(
    pr_numbers: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return compact PR headline state for explicit PR numbers."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    for pr in pr_numbers:
        command = [
            "gh",
            "pr",
            "view",
            str(pr),
            "--json",
            "number,title,state,mergeable,headRefName,headRefOid,statusCheckRollup,url",
        ]
        result = _run(command)
        sources.append(_command_source(result, name=f"pr.{pr}"))
        data, error = _parse_json_result(result)
        if error or not isinstance(data, dict):
            errors.append(f"pr {pr}: {error or 'not a JSON object'}")
            continue
        rollup = data.get("statusCheckRollup", []) or []
        rows.append(
            {
                "number": data.get("number", pr),
                "title": data.get("title", ""),
                "state": data.get("state", ""),
                "mergeable": data.get("mergeable", ""),
                "branch": data.get("headRefName", ""),
                "head_sha": data.get("headRefOid", ""),
                "checks": _checks_summary(rollup if isinstance(rollup, list) else []),
                "url": data.get("url", ""),
            }
        )
    return rows, sources, errors


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    """Build the full snapshot payload."""
    git, sources, errors = git_snapshot(
        include_worktrees=args.include_worktrees,
        worktree_limit=args.worktree_limit,
    )

    claims, claim_sources, claim_errors = claim_snapshot(
        args.claim_issue,
        remote=args.remote,
        origin_main_sha=git.get("origin_main_sha", ""),
    )
    sources.extend(claim_sources)
    errors.extend(claim_errors)

    issues, issue_sources, issue_errors, issue_truncations = issue_queue_snapshot(
        args.issue_search, limit=args.limit
    )
    sources.extend(issue_sources)
    errors.extend(issue_errors)

    prs, pr_sources, pr_errors = pr_snapshot(args.pr)
    sources.extend(pr_sources)
    errors.extend(pr_errors)

    route_failures: list[dict[str, Any]] = []
    for manifest_path in getattr(args, "route_manifest", []):
        route_row = route_manifest_snapshot(manifest_path)
        route_failures.append(route_row)
        if route_row.get("error"):
            errors.append(f"route manifest {manifest_path}: {route_row['error']}")

    checkpoint = controller_checkpoint(
        git=git,
        claims=claims,
        issues=issues,
        prs=prs,
        errors=errors,
        route_failures=route_failures,
    )

    return {
        "schema": "autopilot_state_snapshot.v1",
        "ok": not errors,
        "generated_at_utc": _now_utc(),
        "freshness": {
            "route_evidence_only": True,
            "requires_fresh_check_before_publication": True,
            "branch": git.get("branch", ""),
            "head_sha": git.get("head_sha", ""),
            "origin_main_sha": git.get("origin_main_sha", ""),
        },
        "git": git,
        "claims": claims,
        "issues": issues,
        "issues_truncated_any": any(marker.get("truncated") for marker in issue_truncations),
        "issues_truncated": issue_truncations,
        "prs": prs,
        "route_failures": route_failures,
        "controller_checkpoint": checkpoint,
        "errors": errors,
        "sources": sources,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable JSON snapshot (the default; compatibility flag)",
    )
    parser.add_argument(
        "--issue-search",
        action="append",
        default=[],
        metavar="QUERY",
        help="GitHub issue search query to summarize; may be repeated.",
    )
    parser.add_argument("--limit", type=int, default=20, help="maximum issues per search")
    parser.add_argument(
        "--pr", type=int, action="append", default=[], help="PR number to summarize"
    )
    parser.add_argument(
        "--claim-issue",
        type=int,
        action="append",
        default=[],
        help="issue number whose agent-claim ref should be summarized",
    )
    parser.add_argument(
        "--route-manifest",
        action="append",
        default=[],
        help="route manifest path to summarize as compact failure evidence; may be repeated",
    )
    parser.add_argument("--remote", default="origin", help="git remote used for claim refs")
    parser.add_argument(
        "--include-worktrees",
        action="store_true",
        help="include `git worktree list --porcelain` summary",
    )
    parser.add_argument(
        "--worktree-limit",
        type=int,
        default=20,
        help="maximum worktree rows to include; total count and truncation are always reported",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    payload = build_snapshot(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
