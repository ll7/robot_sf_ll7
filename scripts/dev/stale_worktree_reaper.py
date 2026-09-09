#!/usr/bin/env python3
"""Dry-run stale-worktree reaper that classifies cleanup candidates.

Emits a preservation-aware deletion plan without removing anything by default.
Use --apply to actually remove safe candidates; risky candidates are refused
unless explicit safeguards are satisfied. Apply-time lease checks serialize on
the same repository lifecycle lock as worktree creation and lease mutation.

Default behavior is dry-run only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "stale_worktree_reaper.v1"
# GitHub's pull-request REST payload does not expose an authoritative merge
# method.  The verified path therefore proves exact merged-tree identity and
# ancestry without classifying the historical merge as squash, rebase, or
# regular.  Keep the scope honest in the machine-readable evidence.
VERIFIED_MERGE_MODE = "verified_merged_tree"
DEFAULT_GITHUB_REPO = "ll7/robot_sf_ll7"
DEFAULT_BASE_BRANCH = "main"
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

GitHubTransport = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass(frozen=True, slots=True)
class WorktreeCandidate:
    """A single worktree cleanup candidate."""

    path: str
    branch: str
    head_sha: str
    is_current: bool
    classification: str  # "current", "clean_stale", "risky"
    risk_flags: list[str] = field(default_factory=list)
    preservation_required: str = ""
    verification: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerifiedMergeRequest:
    """Exact inputs required by the opt-in verified merged-tree path."""

    path: str
    branch: str
    head_sha: str
    pr_number: int
    repo: str = DEFAULT_GITHUB_REPO
    base_branch: str = DEFAULT_BASE_BRANCH


@dataclass(frozen=True, slots=True)
class ReaperPlan:
    """Deletion plan output."""

    schema: str
    mode: str  # "dry_run" or "apply"
    total_worktrees: int
    current_worktree: str | None
    candidates: list[WorktreeCandidate]
    deletable: list[str]
    refused: list[str]
    errors: list[str]
    audit_log: list[str]
    verification_mode: str = "default"


def _run_command(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run a command and capture output."""
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def _is_full_sha(value: object) -> bool:
    """Return whether *value* is a complete hexadecimal Git object ID."""
    return isinstance(value, str) and FULL_SHA_RE.fullmatch(value) is not None


def _git_output(
    args: list[str],
    *,
    cwd: str | None,
    description: str,
) -> tuple[str | None, str | None, int]:
    """Run one Git read and return output, refusal reason, and exit code."""
    result = _run_command(args, cwd=cwd)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        return None, f"{description} failed: {detail}", result.returncode
    output = result.stdout.strip()
    if not output:
        return None, f"{description} returned no output", result.returncode
    return output, None, result.returncode


def _github_read(
    args: list[str],
    *,
    transport: GitHubTransport | None,
) -> subprocess.CompletedProcess:
    """Run one allowlisted read-only GitHub command through an injectable transport."""
    if not args or args[0] != "gh" or args[1:2] not in (["api"], ["pr"]):
        return subprocess.CompletedProcess(
            args=args,
            returncode=126,
            stdout="",
            stderr="reaper GitHub transport only permits read commands",
        )
    if args[1:2] == ["pr"] and "list" not in args[2:3]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=126,
            stdout="",
            stderr="reaper GitHub transport only permits gh pr list",
        )
    if args[1:2] == ["api"] and (
        len(args) != 3 or re.fullmatch(r"repos/[^/\s]+/[^/\s]+/pulls/[1-9][0-9]*", args[2]) is None
    ):
        return subprocess.CompletedProcess(
            args=args,
            returncode=126,
            stdout="",
            stderr="reaper GitHub transport only permits PR-detail reads",
        )
    if transport is not None:
        return transport(args)
    return _run_command(args)


def _verified_refusal(
    *,
    path: str,
    branch: str,
    head_sha: str,
    flags: list[str],
    reasons: list[str],
) -> WorktreeCandidate:
    """Build a risky candidate for a failed verified proof."""
    unique_flags = list(dict.fromkeys(flags))
    unique_reasons = list(dict.fromkeys(reasons))
    if "verified_merge_refused" not in unique_flags:
        unique_flags.append("verified_merge_refused")
    reason = "; ".join(unique_reasons) or "required verified-merge proof was unavailable"
    return WorktreeCandidate(
        path=path,
        branch=branch,
        head_sha=head_sha,
        is_current=False,
        classification="risky",
        risk_flags=unique_flags,
        preservation_required=f"verified refusal: {reason}",
    )


def _parse_worktree_porcelain(stdout: str) -> list[dict[str, str]]:  # noqa: C901
    """Parse git worktree list --porcelain output."""
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        key, value = parts
        if key == "worktree":
            if current:
                worktrees.append(current)
            current = {"path": value}
        elif key == "HEAD":
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached"}:
            current[key] = "true"
    if current:
        worktrees.append(current)
    return worktrees


def _is_worktree_dirty(path: str) -> bool:
    """Check if a worktree has uncommitted changes."""
    result = _run_command(["git", "status", "--porcelain"], cwd=path)
    return result.returncode == 0 and bool(result.stdout.strip())


def _has_unpushed_commits(path: str, branch: str) -> bool:
    """Check if a worktree has commits not pushed to its upstream branch."""
    if not branch:
        return True
    upstream = _run_command(["git", "rev-parse", "--abbrev-ref", "@{upstream}"], cwd=path)
    if upstream.returncode != 0 or not upstream.stdout.strip():
        return True
    result = _run_command(
        ["git", "log", f"{upstream.stdout.strip()}..HEAD", "--oneline"],
        cwd=path,
    )
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _has_open_pr(branch: str) -> bool:
    """Check if the branch has an open PR on GitHub."""
    if not branch:
        return False
    result = _run_command(
        ["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number"],
    )
    if result.returncode != 0:
        # A missing executable is a new failure result from _run_command's
        # OSError boundary; retain the reaper's conservative behavior rather
        # than turning an unavailable PR check into deletion eligibility.
        return result.returncode == 127
    try:
        data = json.loads(result.stdout)
        return isinstance(data, list) and len(data) > 0
    except (json.JSONDecodeError, TypeError):
        return False


def _has_ignored_output(path: str) -> bool:
    """Check if a worktree has ignored files (e.g. output/ contents)."""
    result = _run_command(
        ["git", "status", "--ignored", "--short", "-uall"],
        cwd=path,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        if line.startswith("!! "):
            return True
    return False


def _lease_owner_summary(lease: Any) -> str:
    """Return a deterministic task/owner description for cleanup refusal."""
    parts: list[str] = []
    gate_id = getattr(lease, "gate_id", None)
    owner = getattr(lease, "owner", None)
    pr_number = getattr(lease, "pr_number", None)
    expires_at = getattr(lease, "expires_at", None)
    if gate_id:
        parts.append(f"task={gate_id}")
    if owner and owner != gate_id:
        parts.append(f"owner={owner}")
    elif owner and not gate_id:
        parts.append(f"owner={owner}")
    if pr_number is not None:
        parts.append(f"pr=#{pr_number}")
    if expires_at:
        parts.append(f"expires={expires_at}")
    return "; ".join(parts) if parts else "owner/task unknown"


def _worktree_lease_state(path: str) -> tuple[str | None, Any | None]:
    """Return the active/unreadable lease state plus the loaded live lease."""
    try:
        from scripts.dev.pr_gate_lease import lease_path, legacy_lease_path, load_lease

        lease_paths = [lease_path(Path(path))]
        legacy_path = legacy_lease_path()
        if legacy_path not in lease_paths:
            # Older gate processes used one shared lease file. Treat any still-live
            # legacy record as protecting this worktree until that process releases it.
            lease_paths.append(legacy_path)

        for l_path in lease_paths:
            if not l_path.exists():
                continue
            lease = load_lease(l_path)
            if lease is not None and not lease.is_expired():
                return "active", lease
        return None, None
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # File exists but could not be loaded/parsed successfully, or lease
        # discovery failed. Both cases must fail closed.
        return "unreadable", None


def _check_worktree_lease_state(path: str) -> str | None:
    """Check lease state for a worktree.

    Returns:
        "active" if there is an active lease.
        "unreadable" if a lease file exists but cannot be read/parsed.
        None if there is no lease file or if the lease has expired.
    """
    state, _lease = _worktree_lease_state(path)
    return state


def _has_active_pr_gate_lease(path: str) -> bool:
    """Check if a worktree has an active path-scoped worktree lease."""
    return _check_worktree_lease_state(path) == "active"


def _read_worktree_identity(path: str) -> tuple[str | None, str | None, str | None]:
    """Return a worktree's branch, full HEAD, and a failure reason if unreadable."""
    ref_result = _run_command(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=path,
    )
    if ref_result.returncode == 0:
        branch = ref_result.stdout.strip()
        if not branch:
            return None, None, f"worktree identity read returned no branch for {path}"
    elif (
        ref_result.returncode == 1
        and not ref_result.stdout.strip()
        and not ref_result.stderr.strip()
    ):
        # ``symbolic-ref --quiet`` uses exit 1 for a detached HEAD.  That is a
        # safety refusal, not an unreadable worktree.
        branch = None
    else:
        detail = ref_result.stderr.strip() or ref_result.stdout.strip() or "no diagnostic"
        return None, None, f"worktree branch read failed for {path}: {detail}"

    sha, error, _returncode = _git_output(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=path,
        description=f"worktree HEAD read for {path}",
    )
    if error is not None:
        return branch, None, error
    if not _is_full_sha(sha):
        return branch, None, f"worktree HEAD read for {path} was not a full SHA"
    return branch, sha.lower(), None


def _read_commit_ref(ref: str, *, cwd: str, description: str) -> tuple[str | None, str | None]:
    """Resolve a local ref to one complete commit SHA."""
    sha, error, _returncode = _git_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=cwd,
        description=description,
    )
    if error is not None:
        return None, error
    if not _is_full_sha(sha):
        return None, f"{description} returned a non-full commit SHA"
    return sha.lower(), None


def _read_tree_sha(commit_sha: str, *, cwd: str, description: str) -> tuple[str | None, str | None]:
    """Resolve a commit's complete Git tree identity."""
    if not _is_full_sha(commit_sha):
        return None, f"{description} was not given a full commit SHA"
    tree, error, _returncode = _git_output(
        ["git", "rev-parse", "--verify", f"{commit_sha}^{{tree}}"],
        cwd=cwd,
        description=description,
    )
    if error is not None:
        return None, error
    if not _is_full_sha(tree):
        return None, f"{description} returned a non-full tree SHA"
    return tree.lower(), None


def _exact_registered_path(requested: str, registered: str) -> tuple[bool, str | None]:
    """Require a canonical, non-symlink path string equal to the registered path."""
    requested_path = Path(requested)
    registered_path = Path(registered)
    if not requested_path.is_absolute():
        return False, "verified mode requires an absolute target path"
    if not registered_path.is_absolute():
        return False, "registered worktree path was not absolute"
    if requested_path != registered_path:
        return False, "target path is not the exact registered worktree path"
    if requested_path.is_symlink():
        return False, "verified mode refuses a symlink target path"
    try:
        resolved = requested_path.resolve(strict=True)
    except OSError as exc:
        return False, f"target path could not be resolved: {exc}"
    if resolved != requested_path:
        return False, "verified mode refuses a path alias or symlink target"
    if not requested_path.is_dir():
        return False, "target worktree path is missing or not a directory"
    return True, None


def _read_verified_registration(
    request: VerifiedMergeRequest,
    *,
    path: str,
    current_path: str,
) -> tuple[list[str], list[str]]:
    """Re-read the exact target/current registration before verified removal."""
    result = _run_command(["git", "worktree", "list", "--porcelain"], cwd=current_path)
    if result.returncode != 0:
        return ["verified_merge_refused"], ["cannot re-read registered worktrees"]

    worktrees = _parse_worktree_porcelain(result.stdout)
    target_rows = [worktree for worktree in worktrees if worktree.get("path") == path]
    if len(target_rows) != 1:
        return ["verified_merge_refused"], ["target worktree is no longer exactly registered"]

    try:
        current_path_resolved = Path(current_path).resolve()
        current_rows = [
            worktree
            for worktree in worktrees
            if Path(worktree.get("path", "")).resolve() == current_path_resolved
        ]
    except (OSError, RuntimeError, ValueError):
        return ["verified_merge_refused"], ["registered worktree paths could not be resolved"]
    if len(current_rows) != 1:
        return ["verified_merge_refused"], ["current worktree is no longer registered"]

    target_row = target_rows[0]
    if target_row.get("branch") != request.branch:
        return ["verified_merge_refused"], ["registered target branch drifted"]
    if target_row.get("head_sha", "").casefold() != request.head_sha.casefold():
        return ["verified_merge_refused"], ["registered target head drifted"]
    return [], []


def _read_strict_cleanliness(path: str) -> tuple[list[str], list[str]]:
    """Read dirty/untracked and ignored state, failing closed on read errors."""
    flags: list[str] = []
    reasons: list[str] = []

    status = _run_command(["git", "status", "--porcelain"], cwd=path)
    if status.returncode != 0:
        flags.append("unreadable_worktree_status")
        detail = status.stderr.strip() or status.stdout.strip() or "no diagnostic"
        reasons.append(f"worktree cleanliness read failed: {detail}")
    elif status.stdout.strip():
        flags.append("dirty")
        reasons.append("worktree has dirty or untracked content")

    ignored = _run_command(
        ["git", "status", "--ignored", "--short", "-uall"],
        cwd=path,
    )
    if ignored.returncode != 0:
        flags.append("unreadable_ignored_state")
        detail = ignored.stderr.strip() or ignored.stdout.strip() or "no diagnostic"
        reasons.append(f"ignored-state read failed: {detail}")
    elif any(line.startswith("!! ") for line in ignored.stdout.splitlines()):
        flags.append("ignored_output")
        reasons.append("worktree has ignored content")

    return flags, reasons


def _read_strict_open_pr_state(
    branch: str,
    *,
    repo: str,
    transport: GitHubTransport | None,
) -> tuple[bool, str | None, list[int]]:
    """Read open PR coverage without treating transport failure as no PR."""
    result = _github_read(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number",
        ],
        transport=transport,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        return False, f"open-PR lookup failed: {detail}", []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return False, f"open-PR lookup returned invalid JSON: {exc}", []
    if not isinstance(payload, list):
        return False, "open-PR lookup returned a non-list payload", []

    numbers: list[int] = []
    for row in payload:
        if not isinstance(row, dict) or isinstance(row.get("number"), bool):
            return False, "open-PR lookup returned an ambiguous row", []
        number = row.get("number")
        if not isinstance(number, int) or number < 1:
            return False, "open-PR lookup returned a malformed PR number", []
        numbers.append(number)
    return bool(numbers), None, numbers


def _read_config_value(path: str, key: str) -> tuple[str | None, str | None, bool]:
    """Read one Git config value, distinguishing a missing key from an error."""
    result = _run_command(["git", "config", "--get", key], cwd=path)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None, None, False
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        return None, f"Git config read for {key!r} failed: {detail}", True
    value = result.stdout.strip()
    if not value:
        return None, f"Git config read for {key!r} returned no value", True
    return value, None, True


def _verify_deleted_origin_upstream(
    path: str,
    branch: str,
) -> tuple[dict[str, str] | None, list[str], list[str]]:
    """Prove that a configured origin upstream is absent, without recreating it."""
    remote, error, configured = _read_config_value(path, f"branch.{branch}.remote")
    if error is not None:
        return None, [error], ["verified_merge_refused"]
    if not configured or remote != "origin":
        return (
            None,
            [
                "verified mode requires a configured branch upstream on origin; "
                "never-published or non-origin branches remain refused"
            ],
            ["unpushed_commits"],
        )

    merge_ref, error, configured = _read_config_value(path, f"branch.{branch}.merge")
    if error is not None:
        return None, [error], ["verified_merge_refused"]
    expected_merge_ref = f"refs/heads/{branch}"
    if not configured or merge_ref != expected_merge_ref:
        return (
            None,
            [f"verified mode requires the origin upstream ref to match {expected_merge_ref!r}"],
            ["unpushed_commits"],
        )

    upstream_result = _run_command(
        ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
        cwd=path,
    )
    if upstream_result.returncode == 0 and upstream_result.stdout.strip():
        return (
            None,
            [
                "origin upstream still resolves; verified mode only discharges "
                "the missing-upstream risk"
            ],
            ["unpushed_commits"],
        )
    if upstream_result.returncode == 0:
        return None, ["upstream lookup returned no ref"], ["verified_merge_refused"]
    upstream_detail = upstream_result.stderr.strip().casefold()
    expected_missing = (
        "no such branch" in upstream_detail
        or "no upstream configured" in upstream_detail
        or ("upstream branch" in upstream_detail and "not found" in upstream_detail)
        or "ambiguous argument '@{upstream}'" in upstream_detail
        or "ambiguous argument '@{u}'" in upstream_detail
    )
    if not expected_missing:
        detail = upstream_result.stderr.strip() or upstream_result.stdout.strip() or "no diagnostic"
        return None, [f"upstream lookup failed ambiguously: {detail}"], ["verified_merge_refused"]

    tracking_ref = f"refs/remotes/origin/{branch}"
    tracking_result = _run_command(
        ["git", "show-ref", "--verify", "--quiet", tracking_ref],
        cwd=path,
    )
    if tracking_result.returncode == 0:
        return (
            None,
            [f"configured origin tracking ref still exists: {tracking_ref}"],
            ["unpushed_commits"],
        )
    if tracking_result.returncode != 1:
        detail = tracking_result.stderr.strip() or tracking_result.stdout.strip() or "no diagnostic"
        return (
            None,
            [f"origin tracking-ref absence check failed: {detail}"],
            ["verified_merge_refused"],
        )
    return (
        {
            "tracking_remote": "origin",
            "tracking_ref": tracking_ref,
            "upstream_resolution": "missing",
        },
        [],
        [],
    )


def _read_verified_pr_metadata(
    request: VerifiedMergeRequest,
    *,
    transport: GitHubTransport | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read and validate one fresh authoritative merged-PR REST payload."""
    result = _github_read(
        ["gh", "api", f"repos/{request.repo}/pulls/{request.pr_number}"],
        transport=transport,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        return None, f"merged PR metadata lookup failed: {detail}"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, f"merged PR metadata returned invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "merged PR metadata returned a non-object payload"
    if payload.get("number") != request.pr_number:
        return None, "merged PR metadata returned a different PR number"

    base = payload.get("base")
    head = payload.get("head")
    if not isinstance(base, dict) or not isinstance(head, dict):
        return None, "merged PR metadata has incomplete base/head objects"

    reference_error = _validate_verified_pr_references(request, base=base, head=head)
    if reference_error is not None:
        return None, reference_error
    state_fields, state_error = _read_verified_pr_state_fields(payload, base=base)
    if state_error is not None:
        return None, state_error
    assert state_fields is not None
    observed_head_sha, merge_commit_sha, base_sha, merged_at = state_fields

    return (
        {
            "pr_number": request.pr_number,
            "repo": request.repo,
            "base_branch": request.base_branch,
            "pr_base_sha": base_sha,
            "branch": request.branch,
            "head_sha": observed_head_sha,
            "merge_commit_sha": merge_commit_sha,
            "merged_at": merged_at,
        },
        None,
    )


def _validate_verified_pr_references(
    request: VerifiedMergeRequest,
    *,
    base: dict[str, Any],
    head: dict[str, Any],
) -> str | None:
    """Validate repository, base, branch, and exact head fields from PR metadata."""
    base_repo = base.get("repo")
    head_repo = head.get("repo")
    if (
        not isinstance(base_repo, dict)
        or not isinstance(head_repo, dict)
        or base_repo.get("full_name") != request.repo
        or head_repo.get("full_name") != request.repo
    ):
        return "merged PR metadata does not identify the requested repository"
    if base.get("ref") != request.base_branch:
        return f"PR is not merged to {request.base_branch!r}"
    if head.get("ref") != request.branch:
        return "merged PR metadata branch does not match the local branch"

    observed_head_sha = head.get("sha")
    if not _is_full_sha(observed_head_sha):
        return "merged PR metadata has no full head SHA"
    if observed_head_sha.casefold() != request.head_sha.casefold():
        return "merged PR metadata head SHA does not match the expected local head"
    return None


def _read_verified_pr_state_fields(
    payload: dict[str, Any],
    *,
    base: dict[str, Any],
) -> tuple[tuple[str, str, str, str] | None, str | None]:
    """Validate merged-state fields and return normalized PR identity values."""
    if payload.get("state") not in {"closed", "merged"}:
        return None, f"PR is not closed/merged (state={payload.get('state')!r})"
    merged_at = payload.get("merged_at")
    if not isinstance(merged_at, str) or not merged_at.strip():
        return None, "merged PR metadata has no merge timestamp"
    base_sha = base.get("sha")
    if not _is_full_sha(base_sha):
        return None, "merged PR metadata has no full base SHA"
    merge_commit_sha = payload.get("merge_commit_sha")
    if not _is_full_sha(merge_commit_sha):
        return None, "merged PR metadata has no full merge commit SHA"
    head = payload.get("head")
    if not isinstance(head, dict) or not _is_full_sha(head.get("sha")):
        return None, "merged PR metadata has no full head SHA"
    return (
        (
            str(head["sha"]).lower(),
            str(merge_commit_sha).lower(),
            str(base_sha).lower(),
            merged_at.strip(),
        ),
        None,
    )


def _verified_worktree_state(  # noqa: C901, PLR0912, PLR0915 - ordered fail-closed proof gates
    request: VerifiedMergeRequest,
    *,
    path: str,
    current_path: str,
    transport: GitHubTransport | None,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Collect all proof inputs for one verified candidate.

    This function is deliberately usable both during planning and while the
    lifecycle lock is held immediately before removal.  It performs reads only
    and returns no permission to mutate refs or GitHub state.
    """
    flags: list[str] = []
    reasons: list[str] = []

    registration_flags, registration_reasons = _read_verified_registration(
        request,
        path=path,
        current_path=current_path,
    )
    if registration_reasons:
        return None, registration_flags, registration_reasons

    exact, path_reason = _exact_registered_path(request.path, path)
    if not exact:
        return None, ["verified_merge_refused"], [path_reason or "target path mismatch"]

    target_path = Path(path)
    current_resolved = Path(current_path).resolve()
    if target_path.resolve() == current_resolved:
        return None, ["current_worktree"], ["current worktree is never eligible for removal"]

    if not _is_full_sha(request.head_sha):
        return (
            None,
            ["verified_merge_refused"],
            ["expected head SHA must be a full 40-character SHA"],
        )
    if not request.branch or request.branch != request.branch.strip():
        return None, ["verified_merge_refused"], ["verified branch must be a non-empty exact name"]
    if request.pr_number < 1:
        return None, ["verified_merge_refused"], ["verified PR number must be positive"]

    target_branch, target_sha, identity_error = _read_worktree_identity(path)
    if identity_error is not None:
        return None, ["verified_merge_refused"], [identity_error]
    if target_branch is None:
        return None, ["detached_head"], ["detached worktrees cannot satisfy verified branch proof"]
    if target_branch != request.branch:
        return None, ["branch_drift"], ["checked-out branch does not match the expected branch"]
    if target_sha is None or target_sha.casefold() != request.head_sha.casefold():
        return None, ["head_drift"], ["worktree HEAD does not match the expected full head SHA"]

    local_branch_sha, branch_error = _read_commit_ref(
        f"refs/heads/{request.branch}",
        cwd=current_path,
        description="local branch ref read",
    )
    if branch_error is not None:
        return None, ["verified_merge_refused"], [branch_error]
    if local_branch_sha != request.head_sha.casefold():
        return None, ["branch_drift"], ["local branch ref does not match the expected head SHA"]

    cleanliness_flags, cleanliness_reasons = _read_strict_cleanliness(path)
    flags.extend(cleanliness_flags)
    reasons.extend(cleanliness_reasons)

    lease_state, lease = _worktree_lease_state(path)
    if lease_state == "active":
        flags.append("active_pr_gate_lease")
        reasons.append(f"worktree has an active lease ({_lease_owner_summary(lease)})")
    elif lease_state == "unreadable":
        flags.append("unreadable_pr_gate_lease")
        reasons.append("worktree lease could not be read")

    open_pr, open_pr_error, open_pr_numbers = _read_strict_open_pr_state(
        request.branch,
        repo=request.repo,
        transport=transport,
    )
    if open_pr_error is not None:
        flags.append("unreadable_open_pr_state")
        reasons.append(open_pr_error)
    elif open_pr:
        flags.append("open_pr")
        joined = ", ".join(f"#{number}" for number in open_pr_numbers)
        reasons.append(f"branch has open PR coverage: {joined}")

    if flags:
        return None, flags, reasons

    upstream_evidence, upstream_reasons, upstream_flags = _verify_deleted_origin_upstream(
        path,
        request.branch,
    )
    if upstream_reasons or upstream_flags:
        return None, upstream_flags, upstream_reasons
    assert upstream_evidence is not None

    current_branch, current_sha, current_error = _read_worktree_identity(current_path)
    if current_error is not None:
        return None, ["verified_merge_refused"], [current_error]
    if current_branch != request.base_branch:
        return (
            None,
            ["verified_merge_refused"],
            [f"current checkout is not the {request.base_branch!r} worktree"],
        )
    if current_sha is None:
        return None, ["verified_merge_refused"], ["current main checkout has no full HEAD SHA"]

    local_main_sha, main_error = _read_commit_ref(
        f"refs/heads/{request.base_branch}",
        cwd=current_path,
        description="local main ref read",
    )
    if main_error is not None:
        return None, ["verified_merge_refused"], [main_error]
    remote_main_sha, remote_main_error = _read_commit_ref(
        f"refs/remotes/origin/{request.base_branch}",
        cwd=current_path,
        description="remote main ref read",
    )
    if remote_main_error is not None:
        return None, ["verified_merge_refused"], [remote_main_error]
    if current_sha != local_main_sha or current_sha != remote_main_sha:
        return (
            None,
            ["verified_merge_refused"],
            [
                "current main identity is stale or inconsistent: "
                f"HEAD={current_sha}, local={local_main_sha}, remote={remote_main_sha}"
            ],
        )

    pr_metadata, pr_error = _read_verified_pr_metadata(request, transport=transport)
    if pr_error is not None:
        return None, ["verified_merge_refused"], [pr_error]
    assert pr_metadata is not None

    merge_sha = str(pr_metadata["merge_commit_sha"])
    merge_commit_sha, merge_error = _read_commit_ref(
        merge_sha,
        cwd=current_path,
        description="PR merge commit read",
    )
    if merge_error is not None:
        return None, ["verified_merge_refused"], [merge_error]
    assert merge_commit_sha is not None
    if merge_commit_sha != merge_sha.casefold():
        return (
            None,
            ["verified_merge_refused"],
            ["locally resolved PR merge commit does not match authoritative metadata"],
        )

    target_tree, target_tree_error = _read_tree_sha(
        target_sha,
        cwd=current_path,
        description="worktree HEAD tree read",
    )
    if target_tree_error is not None:
        return None, ["verified_merge_refused"], [target_tree_error]
    merge_tree, merge_tree_error = _read_tree_sha(
        merge_commit_sha,
        cwd=current_path,
        description="PR merge commit tree read",
    )
    if merge_tree_error is not None:
        return None, ["verified_merge_refused"], [merge_tree_error]
    main_tree, main_tree_error = _read_tree_sha(
        current_sha,
        cwd=current_path,
        description="current main tree read",
    )
    if main_tree_error is not None:
        return None, ["verified_merge_refused"], [main_tree_error]
    assert target_tree is not None and merge_tree is not None and main_tree is not None
    if not (target_tree == merge_tree == main_tree):
        return (
            None,
            ["verified_merge_refused"],
            [
                "complete Git trees differ: "
                f"worktree={target_tree}, merge={merge_tree}, main={main_tree}"
            ],
        )

    ancestor = _run_command(
        ["git", "merge-base", "--is-ancestor", merge_commit_sha, current_sha],
        cwd=current_path,
    )
    if ancestor.returncode != 0:
        detail = (
            ancestor.stderr.strip() or ancestor.stdout.strip() or "merge commit is not an ancestor"
        )
        return None, ["verified_merge_refused"], [f"merge ancestry proof failed: {detail}"]

    evidence: dict[str, Any] = {
        "mode": VERIFIED_MERGE_MODE,
        "merge_method_scope": "method_agnostic_exact_tree",
        "path": request.path,
        "current_path": str(Path(current_path).resolve()),
        "repo": request.repo,
        "pr_number": request.pr_number,
        "base_branch": request.base_branch,
        "branch": request.branch,
        "expected_head_sha": request.head_sha.lower(),
        "head_sha": target_sha,
        "merge_commit_sha": merge_commit_sha,
        "main_sha": current_sha,
        "remote_main_sha": remote_main_sha,
        "worktree_tree_sha": target_tree,
        "merge_tree_sha": merge_tree,
        "main_tree_sha": main_tree,
        "risk_discharged": "missing origin upstream only",
        "merged_at": pr_metadata["merged_at"],
        "pr_base_sha": pr_metadata["pr_base_sha"],
        **upstream_evidence,
    }
    return evidence, [], []


def _classify_verified_worktree(
    *,
    path: str,
    branch: str,
    head_sha: str,
    current_path: str,
    request: VerifiedMergeRequest,
    github_transport: GitHubTransport | None,
) -> WorktreeCandidate:
    """Classify one row through the strict verified proof path."""
    if branch != request.branch:
        return _verified_refusal(
            path=path,
            branch=branch,
            head_sha=head_sha,
            flags=["detached_head" if not branch else "branch_drift"],
            reasons=["registered worktree branch does not match the expected branch"],
        )
    if head_sha.casefold() != request.head_sha.casefold():
        return _verified_refusal(
            path=path,
            branch=branch,
            head_sha=head_sha,
            flags=["head_drift"],
            reasons=["registered worktree HEAD does not match the expected full head SHA"],
        )
    verification, flags, reasons = _verified_worktree_state(
        request,
        path=path,
        current_path=current_path,
        transport=github_transport,
    )
    if verification is None:
        return _verified_refusal(
            path=path,
            branch=branch,
            head_sha=head_sha,
            flags=flags,
            reasons=reasons,
        )
    return WorktreeCandidate(
        path=path,
        branch=branch,
        head_sha=head_sha,
        is_current=False,
        classification="clean_stale",
        verification=verification,
    )


def classify_worktree(  # noqa: C901 - preserves the legacy gate order beside verified mode
    *,
    path: str,
    branch: str,
    head_sha: str,
    current_path: str,
    skip_pr_check: bool = False,
    verified_request: VerifiedMergeRequest | None = None,
    github_transport: GitHubTransport | None = None,
) -> WorktreeCandidate:
    """Classify a single worktree as current, clean_stale, or risky."""
    is_current = Path(path).resolve() == Path(current_path).resolve()
    if is_current:
        return WorktreeCandidate(
            path=path,
            branch=branch,
            head_sha=head_sha,
            is_current=True,
            classification="current",
            preservation_required="current worktree",
        )

    if verified_request is not None:
        return _classify_verified_worktree(
            path=path,
            branch=branch,
            head_sha=head_sha,
            current_path=current_path,
            request=verified_request,
            github_transport=github_transport,
        )

    risk_flags: list[str] = []

    if _is_worktree_dirty(path):
        risk_flags.append("dirty")

    if _has_unpushed_commits(path, branch):
        risk_flags.append("unpushed_commits")

    if skip_pr_check and branch:
        risk_flags.append("pr_check_skipped")
    elif _has_open_pr(branch):
        risk_flags.append("open_pr")

    if _has_ignored_output(path):
        risk_flags.append("ignored_output")

    if _has_active_pr_gate_lease(path):
        risk_flags.append("active_pr_gate_lease")
    elif _check_worktree_lease_state(path) == "unreadable":
        risk_flags.append("unreadable_pr_gate_lease")

    if risk_flags:
        return WorktreeCandidate(
            path=path,
            branch=branch,
            head_sha=head_sha,
            is_current=False,
            classification="risky",
            risk_flags=risk_flags,
            preservation_required=f"risky: {', '.join(risk_flags)}",
        )

    return WorktreeCandidate(
        path=path,
        branch=branch,
        head_sha=head_sha,
        is_current=False,
        classification="clean_stale",
    )


def _make_verified_merge_request(  # noqa: C901 - all verified inputs are validated together
    *,
    target_path: str | None,
    verified_pr_number: int | None,
    verified_branch: str | None,
    verified_head_sha: str | None,
    repo: str,
    base_branch: str,
    skip_pr_check: bool,
    limit: int,
) -> tuple[VerifiedMergeRequest | None, str | None]:
    """Validate the all-or-nothing CLI/API inputs for verified mode."""
    supplied = (verified_pr_number, verified_branch, verified_head_sha)
    if not any(value is not None for value in supplied):
        return None, None
    if target_path is None:
        return None, "verified mode requires --path for one exact registered worktree"
    if skip_pr_check:
        return None, "verified mode cannot be combined with --skip-pr-check"
    if limit:
        return None, "verified mode cannot be combined with --limit"
    if not all(value is not None for value in supplied):
        return (
            None,
            "verified mode requires --verified-merged-pr, --verified-branch, and "
            "--verified-head-sha together",
        )
    if isinstance(verified_pr_number, bool) or not isinstance(verified_pr_number, int):
        return None, "verified PR number must be a positive integer"
    if verified_pr_number < 1:
        return None, "verified PR number must be positive"
    assert verified_branch is not None and verified_head_sha is not None
    if not verified_branch or verified_branch != verified_branch.strip():
        return None, "verified branch must be a non-empty exact branch name"
    if not _is_full_sha(verified_head_sha):
        return None, "verified head SHA must be a full 40-character SHA"
    if (
        not isinstance(repo, str)
        or repo.count("/") != 1
        or any(not part for part in repo.split("/"))
        or repo != repo.strip()
    ):
        return None, "GitHub repository must be an exact owner/name value"
    if base_branch != DEFAULT_BASE_BRANCH:
        return None, f"verified mode only supports the {DEFAULT_BASE_BRANCH!r} base branch"
    return (
        VerifiedMergeRequest(
            path=target_path,
            branch=verified_branch,
            head_sha=verified_head_sha.lower(),
            pr_number=verified_pr_number,
            repo=repo,
            base_branch=base_branch,
        ),
        None,
    )


def build_plan(  # noqa: C901, PLR0912, PLR0913, PLR0915 - CLI/planning contract is explicit
    *,
    current_path: str | None = None,
    skip_pr_check: bool = False,
    limit: int = 0,
    target_path: str | None = None,
    verified_pr_number: int | None = None,
    verified_branch: str | None = None,
    verified_head_sha: str | None = None,
    repo: str = DEFAULT_GITHUB_REPO,
    base_branch: str = DEFAULT_BASE_BRANCH,
    github_transport: GitHubTransport | None = None,
) -> ReaperPlan:
    """Build a deletion plan from current repository worktrees."""
    errors: list[str] = []
    if current_path is None:
        current_path = str(Path.cwd().resolve())
    else:
        current_path = str(Path(current_path).resolve())

    verified_request, verified_error = _make_verified_merge_request(
        target_path=target_path,
        verified_pr_number=verified_pr_number,
        verified_branch=verified_branch,
        verified_head_sha=verified_head_sha,
        repo=repo,
        base_branch=base_branch,
        skip_pr_check=skip_pr_check,
        limit=limit,
    )
    verification_mode = (
        VERIFIED_MERGE_MODE if verified_request is not None or verified_error else "default"
    )
    if verified_error is not None:
        return ReaperPlan(
            schema=SCHEMA_VERSION,
            mode="dry_run",
            total_worktrees=0,
            current_worktree=None,
            candidates=[],
            deletable=[],
            refused=[],
            errors=[verified_error],
            audit_log=[f"verified mode refused: {verified_error}"],
            verification_mode=verification_mode,
        )

    result = _run_command(
        ["git", "worktree", "list", "--porcelain"],
        cwd=current_path,
    )
    if result.returncode != 0:
        errors.append("failed to list worktrees")
        return ReaperPlan(
            schema=SCHEMA_VERSION,
            mode="dry_run",
            total_worktrees=0,
            current_worktree=None,
            candidates=[],
            deletable=[],
            refused=[],
            errors=errors,
            audit_log=["failed to list worktrees"],
            verification_mode=verification_mode,
        )

    parsed = _parse_worktree_porcelain(result.stdout)
    current_worktree = None
    candidates: list[WorktreeCandidate] = []
    audit_log: list[str] = []

    if verified_request is not None:
        current_rows = [
            wt
            for wt in parsed
            if wt.get("path") and Path(wt["path"]).resolve() == Path(current_path).resolve()
        ]
        if current_rows:
            current_worktree = current_rows[0].get("path")
        else:
            errors.append("verified mode requires the current main worktree to be registered")

        worktrees_to_check = [wt for wt in parsed if wt.get("path") == verified_request.path]
        if not worktrees_to_check:
            errors.append(
                f"target worktree is not registered at the exact requested path: "
                f"{verified_request.path}"
            )
            audit_log.append(
                f"verified target path not registered exactly: {verified_request.path}"
            )
        elif errors:
            audit_log.append("verified mode refused before candidate validation")
        else:
            wt = worktrees_to_check[0]
            wt_path = wt.get("path", "")
            branch = wt.get("branch", "")
            head_sha = wt.get("head_sha", "")
            candidate = classify_worktree(
                path=wt_path,
                branch=branch,
                head_sha=head_sha,
                current_path=current_path,
                verified_request=verified_request,
                github_transport=github_transport,
            )
            candidates.append(candidate)
            audit_log.append(
                f"classified {wt_path} as {candidate.classification}"
                + (f" ({', '.join(candidate.risk_flags)})" if candidate.risk_flags else "")
            )
            if candidate.verification:
                audit_log.append(
                    "verified missing-upstream risk discharged: "
                    f"PR #{candidate.verification['pr_number']} head "
                    f"{candidate.verification['head_sha']} merge "
                    f"{candidate.verification['merge_commit_sha']} main "
                    f"{candidate.verification['main_sha']} tree "
                    f"{candidate.verification['main_tree_sha']}"
                )
            elif candidate.preservation_required:
                audit_log.append(candidate.preservation_required)

        deletable = [c.path for c in candidates if c.classification == "clean_stale"]
        refused = [c.path for c in candidates if c.classification == "risky"]
        return ReaperPlan(
            schema=SCHEMA_VERSION,
            mode="dry_run",
            total_worktrees=len(parsed),
            current_worktree=current_worktree,
            candidates=candidates,
            deletable=deletable,
            refused=refused,
            errors=errors,
            audit_log=audit_log,
            verification_mode=verification_mode,
        )

    worktrees_to_check = parsed
    if target_path is not None:
        target = Path(target_path).resolve()
        worktrees_to_check = [
            wt for wt in parsed if wt.get("path") and Path(wt["path"]).resolve() == target
        ]
        if not worktrees_to_check:
            errors.append(f"target worktree is not registered: {target}")
            audit_log.append(f"target worktree not registered: {target}")
    elif limit > 0:
        worktrees_to_check = parsed[:limit]

    for wt in worktrees_to_check:
        wt_path = wt.get("path", "")
        branch = wt.get("branch", "")
        head_sha = wt.get("head_sha", "")

        candidate = classify_worktree(
            path=wt_path,
            branch=branch,
            head_sha=head_sha,
            current_path=current_path,
            skip_pr_check=skip_pr_check,
        )
        if candidate.is_current:
            current_worktree = wt_path
        candidates.append(candidate)
        audit_log.append(
            f"classified {wt_path} as {candidate.classification}"
            + (f" ({', '.join(candidate.risk_flags)})" if candidate.risk_flags else "")
        )

    deletable = [c.path for c in candidates if c.classification == "clean_stale"]
    refused = [c.path for c in candidates if c.classification == "risky"]

    return ReaperPlan(
        schema=SCHEMA_VERSION,
        mode="dry_run",
        total_worktrees=len(parsed),
        current_worktree=current_worktree,
        candidates=candidates,
        deletable=deletable,
        refused=refused,
        errors=errors,
        audit_log=audit_log,
        verification_mode=verification_mode,
    )


def _request_from_verified_candidate(
    candidate: WorktreeCandidate,
) -> tuple[VerifiedMergeRequest | None, str | None]:
    """Recover and validate the immutable verification inputs stored in a plan."""
    evidence = candidate.verification
    if evidence.get("mode") != VERIFIED_MERGE_MODE:
        return None, "candidate has no verified merged-tree evidence"
    path = evidence.get("path")
    branch = evidence.get("branch")
    head_sha = evidence.get("expected_head_sha")
    pr_number = evidence.get("pr_number")
    repo = evidence.get("repo")
    base_branch = evidence.get("base_branch")
    if (
        not isinstance(path, str)
        or not isinstance(branch, str)
        or not isinstance(head_sha, str)
        or isinstance(pr_number, bool)
        or not isinstance(pr_number, int)
        or not isinstance(repo, str)
        or not isinstance(base_branch, str)
    ):
        return None, "candidate verification evidence is incomplete or malformed"
    return (
        VerifiedMergeRequest(
            path=path,
            branch=branch,
            head_sha=head_sha,
            pr_number=pr_number,
            repo=repo,
            base_branch=base_branch,
        ),
        None,
    )


def _revalidate_verified_candidate(
    candidate: WorktreeCandidate,
    *,
    github_transport: GitHubTransport | None,
) -> tuple[bool, str]:
    """Repeat the complete verified proof and reject any plan/apply drift."""
    request, request_error = _request_from_verified_candidate(candidate)
    if request_error is not None:
        return False, request_error
    assert request is not None
    current_path = candidate.verification.get("current_path")
    if not isinstance(current_path, str) or not current_path:
        return False, "candidate verification evidence has no current main path"
    evidence, flags, reasons = _verified_worktree_state(
        request,
        path=candidate.path,
        current_path=current_path,
        transport=github_transport,
    )
    if evidence is None:
        detail = "; ".join(reasons) or ", ".join(flags) or "required proof was refused"
        return False, f"verified plan revalidation refused: {detail}"
    if evidence != candidate.verification:
        return False, "verified plan evidence drifted before removal"
    return True, "verified plan revalidated under lifecycle lock"


def _attempt_candidate_removal(
    candidate: WorktreeCandidate,
    *,
    github_transport: GitHubTransport | None = None,
) -> tuple[str, str, str | None]:
    """Remove one candidate under the lifecycle lock after rechecking its lease."""
    from scripts.dev.pr_gate_lease import worktree_lifecycle_lock

    try:
        # Lease acquisition/release and worktree creation use this same lock.
        # Re-read the lease while the lock is held and keep it held through
        # `git worktree remove`; this closes the plan->apply TOCTOU window.
        with worktree_lifecycle_lock():
            lease_state, lease = _worktree_lease_state(candidate.path)
            if lease_state == "active":
                return (
                    "refused",
                    "refused live lease candidate "
                    f"{candidate.path} ({_lease_owner_summary(lease)})",
                    None,
                )
            if lease_state == "unreadable":
                return (
                    "refused",
                    f"refused unreadable lease candidate {candidate.path} (owner/task unknown)",
                    None,
                )

            if candidate.verification:
                verified, detail = _revalidate_verified_candidate(
                    candidate,
                    github_transport=github_transport,
                )
                if not verified:
                    return (
                        "refused",
                        f"refused verified candidate {candidate.path}: {detail}",
                        None,
                    )

            repo_cwd = (
                candidate.verification.get("current_path") if candidate.verification else None
            )
            result = _run_command(
                ["git", "worktree", "remove", candidate.path],
                cwd=repo_cwd if isinstance(repo_cwd, str) else None,
            )
    except RuntimeError as exc:
        return (
            "error",
            f"refused lifecycle-lock failure {candidate.path}",
            f"failed lifecycle guard for {candidate.path}: {exc}",
        )

    if result.returncode != 0:
        return (
            "error",
            f"failed to remove {candidate.path}",
            f"failed to remove {candidate.path}: {result.stderr.strip()}",
        )
    return "removed", f"removed {candidate.path}", None


def apply_deletions(
    plan: ReaperPlan,
    *,
    force: bool = False,
    github_transport: GitHubTransport | None = None,
) -> ReaperPlan:
    """Apply safe deletions, atomically refusing leases acquired after planning."""
    errors = list(plan.errors)

    if plan.errors:
        return plan

    deletable = list(plan.deletable)
    refused = list(plan.refused)
    audit_log = list(plan.audit_log)

    for candidate in plan.candidates:
        if candidate.classification != "clean_stale":
            if candidate.classification == "risky":
                audit_log.append(f"refused risky candidate {candidate.path}")
            continue
        if candidate.is_current:
            refused.append(candidate.path)
            audit_log.append(f"refused current worktree {candidate.path}")
            continue

        outcome, audit_event, error = _attempt_candidate_removal(
            candidate,
            github_transport=github_transport,
        )
        audit_log.append(audit_event)
        deletable = [path for path in deletable if path != candidate.path]
        if error is not None:
            errors.append(error)
        if outcome != "removed" and candidate.path not in refused:
            refused.append(candidate.path)

    return ReaperPlan(
        schema=plan.schema,
        mode="apply",
        total_worktrees=plan.total_worktrees,
        current_worktree=plan.current_worktree,
        candidates=plan.candidates,
        deletable=deletable,
        refused=refused,
        errors=errors,
        audit_log=audit_log,
        verification_mode=plan.verification_mode,
    )


def _append_candidate_lines(lines: list[str], candidates: list[WorktreeCandidate]) -> None:
    """Append human-readable candidate rows."""
    for c in candidates:
        if c.classification == "current":
            tag = "[CURRENT]"
        elif c.classification == "clean_stale":
            tag = "[DELETABLE]"
        else:
            tag = f"[RISKY: {', '.join(c.risk_flags)}]"
        lines.append(f"  {tag} {c.branch or 'detached'}: {c.path}")
        if c.verification:
            lines.append(
                "    Verified: "
                f"PR #{c.verification['pr_number']}, head {c.verification['head_sha']}, "
                f"merge {c.verification['merge_commit_sha']}, "
                f"main {c.verification['main_sha']}, "
                f"tree {c.verification['main_tree_sha']}"
            )


def _append_path_section(lines: list[str], title: str, paths: list[str]) -> None:
    """Append a titled path list when non-empty."""
    if paths:
        lines.append("")
        lines.append(f"  {title} ({len(paths)}):")
        for p in paths:
            lines.append(f"    - {p}")


def _append_text_section(lines: list[str], title: str, values: list[str]) -> None:
    """Append a titled text list when non-empty."""
    if values:
        lines.append("")
        lines.append(f"  {title}:")
        for value in values:
            lines.append(f"    - {value}")


def format_human(plan: ReaperPlan) -> str:
    """Format plan as human-readable text."""
    lines = [
        f"Stale Worktree Reaper (schema: {plan.schema})",
        f"  Mode: {plan.mode}",
        f"  Verification: {plan.verification_mode}",
        f"  Total worktrees: {plan.total_worktrees}",
        f"  Current: {plan.current_worktree or 'N/A'}",
        "",
    ]

    _append_candidate_lines(lines, plan.candidates)
    _append_path_section(lines, "Deletable", plan.deletable)
    _append_path_section(lines, "Refused", plan.refused)
    _append_text_section(lines, "Errors", plan.errors)
    _append_text_section(lines, "Audit log", plan.audit_log)

    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove safe worktrees (risky candidates still refused)",
    )
    parser.add_argument(
        "--skip-pr-check",
        action="store_true",
        help="Skip GitHub PR lookup (faster, offline-safe)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum worktrees to consider (0 = all)",
    )
    parser.add_argument(
        "--path",
        help="Restrict planning/apply to one exact registered worktree path",
    )
    parser.add_argument(
        "--verified-merged-pr",
        "--verified-pr",
        dest="verified_pr_number",
        type=int,
        help=(
            "Opt into exact verified merged-tree cleanup for one --path; requires "
            "--verified-branch and --verified-head-sha"
        ),
    )
    parser.add_argument(
        "--verified-branch",
        dest="verified_branch",
        help="Exact local branch required by --verified-merged-pr",
    )
    parser.add_argument(
        "--verified-head-sha",
        dest="verified_head_sha",
        help="Exact full local head SHA required by --verified-merged-pr",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_GITHUB_REPO,
        help=f"GitHub owner/repository for verified metadata (default: {DEFAULT_GITHUB_REPO})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    try:
        plan = build_plan(
            skip_pr_check=args.skip_pr_check,
            limit=args.limit,
            target_path=args.path,
            verified_pr_number=args.verified_pr_number,
            verified_branch=args.verified_branch,
            verified_head_sha=args.verified_head_sha,
            repo=args.repo,
        )
    except Exception as exc:
        print(f"ERROR building reaper plan: {exc}", file=sys.stderr)
        return 1

    if args.apply:
        plan = apply_deletions(plan)

    if args.json:
        output = asdict(plan)
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(format_human(plan))

    return 0 if not plan.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
