#!/usr/bin/env python3
"""Refresh a docstring-baseline-conflicting PR branch after sibling merges (#9057).

Sibling ``docs(test-docstrings)`` PRs from epic #8876 all rewrite
``scripts/validation/docstring_todo_baseline.json``. When one sibling merges,
the others conflict on that single file and need an ad-hoc refresh. This helper
performs the deterministic sequence in one command:

1. verify the chosen worktree is clean and matches the requested PR/branch,
2. fetch the base ref and merge it,
3. when the only conflict is the tracked baseline, take the incoming version,
4. regenerate the baseline, then require ``ratchet`` and ``verify-baseline
   --check-working-tree`` to pass,
5. commit the refresh and push it (``--no-push`` stops before publication).

It fails closed and aborts an in-progress merge when the worktree is dirty, the
selected PR/branch does not match the checkout, a conflict touches any file
other than the baseline, or validation fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = "refresh_docstring_baseline_conflict.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_BASE_REF = "origin/main"
DEFAULT_TIMEOUT_SECONDS = 300
BASELINE_PATH = Path("scripts/validation/docstring_todo_baseline.json")
CHECK_SCRIPT = Path("scripts/validation/check_docstring_todos.py")
MERGE_COMMIT_MESSAGE = "merge: refresh docstring baseline after sibling merge (#9057)"
REFRESH_COMMIT_MESSAGE = "chore(validation): regenerate docstring todo baseline (#9057)"


class RefreshError(RuntimeError):
    """A deterministic, user-actionable refresh failure."""


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess without leaking a shell or raising on failure."""
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _git(
    worktree: Path,
    *args: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run a git command that must succeed and return its stripped stdout."""
    result = _run(["git", *args], cwd=worktree, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RefreshError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _resolve_worktree(value: str | Path) -> Path:
    """Resolve and validate that the target is a git worktree root."""
    worktree = Path(value).resolve(strict=False)
    if not worktree.is_dir():
        raise RefreshError(f"worktree is not an existing directory: {value}")
    top_level = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
    if top_level != worktree:
        raise RefreshError(f"git top-level {top_level} does not match worktree {worktree}")
    return worktree


def _current_branch(worktree: Path) -> str:
    """Return the checked-out branch or fail closed on a detached HEAD."""
    result = _run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=worktree)
    if result.returncode != 0 or not result.stdout.strip():
        raise RefreshError("refresh requires a checkout on a branch, not a detached HEAD")
    return result.stdout.strip()


def _assert_clean(worktree: Path) -> None:
    """Refuse to refresh over tracked local modifications."""
    status = _git(worktree, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RefreshError(
            "worktree has uncommitted tracked changes; commit, stash, or discard them first:\n"
            f"{status}"
        )


def _verify_pr(repo: str, pr: int, branch: str) -> dict[str, str]:
    """Verify the PR is open and its head branch is the current checkout."""
    result = _run(
        ["gh", "pr", "view", str(pr), "--repo", repo, "--json", "state,headRefName"],
        cwd=Path.cwd(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RefreshError(f"gh pr view {pr} failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RefreshError(f"gh pr view {pr} did not return JSON") from exc
    state = str(payload.get("state", ""))
    head_ref = str(payload.get("headRefName", ""))
    if state != "OPEN":
        raise RefreshError(f"PR {pr} is {state or 'unknown'}, not OPEN")
    if head_ref != branch:
        raise RefreshError(
            f"PR {pr} head branch '{head_ref}' does not match checkout branch '{branch}'"
        )
    return {"state": state, "head_ref": head_ref}


def _split_base_ref(base_ref: str) -> tuple[str, str]:
    """Split ``origin/main`` into a fetch remote and ref name."""
    if "/" in base_ref:
        remote, ref = base_ref.split("/", 1)
        if remote and ref:
            return remote, ref
    return "origin", base_ref


def _merge_base(worktree: Path, base_ref: str) -> list[str]:
    """Merge the base ref and return the unmerged (conflicting) paths."""
    result = _run(["git", "merge", "--no-edit", base_ref], cwd=worktree)
    if result.returncode == 0:
        return []
    unmerged = _git(worktree, "diff", "--name-only", "--diff-filter=U")
    paths = [line for line in unmerged.splitlines() if line.strip()]
    if not paths:
        detail = (result.stderr or result.stdout).strip()
        raise RefreshError(f"git merge {base_ref} failed without conflict paths: {detail}")
    return paths


def _abort_merge(worktree: Path) -> bool:
    """Best-effort merge abort that never masks the original failure."""
    result = _run(["git", "merge", "--abort"], cwd=worktree)
    return result.returncode == 0


def _resolve_baseline_conflict(worktree: Path, conflicts: list[str]) -> None:
    """Take the incoming baseline version when it is the only conflict."""
    unexpected = [path for path in conflicts if path != str(BASELINE_PATH)]
    if unexpected:
        raise RefreshError(
            "conflict touches files other than the docstring baseline; resolve manually: "
            + ", ".join(unexpected)
        )
    _git(worktree, "checkout", "--theirs", "--", str(BASELINE_PATH))
    _git(worktree, "add", "--", str(BASELINE_PATH))


def _validator_command(mode: str, base_ref: str) -> list[str]:
    """Build the validation command for one docstring-checker mode."""
    command = [sys.executable, str(CHECK_SCRIPT), "--mode", mode]
    if mode == "verify-baseline":
        command.extend(["--base", base_ref, "--check-working-tree"])
    return command


def _run_validators(worktree: Path, base_ref: str) -> dict[str, dict[str, object]]:
    """Run write-baseline, ratchet, and verify-baseline; fail closed on any error."""
    evidence: dict[str, dict[str, object]] = {}
    for mode in ("write-baseline", "ratchet", "verify-baseline"):
        result = _run(_validator_command(mode, base_ref), cwd=worktree)
        detail = (result.stdout + result.stderr).strip().splitlines()
        evidence[mode] = {
            "returncode": result.returncode,
            "summary": detail[-1] if detail else "",
        }
        if result.returncode != 0:
            raise RefreshError(
                f"docstring validation '{mode}' failed (exit {result.returncode}); not pushing"
            )
    return evidence


def _commit_refresh(worktree: Path, merge_in_progress: bool, baseline_changed: bool) -> str | None:
    """Commit the merge and the regenerated baseline, returning the new head."""
    if merge_in_progress:
        _git(worktree, "commit", "-m", MERGE_COMMIT_MESSAGE)
    if baseline_changed:
        _git(worktree, "add", "--", str(BASELINE_PATH))
        _git(worktree, "commit", "-m", REFRESH_COMMIT_MESSAGE)
    return _git(worktree, "rev-parse", "HEAD") or None


def refresh(
    *,
    worktree: Path,
    repo: str,
    pr: int | None,
    branch: str | None,
    base_ref: str,
    no_push: bool,
    dry_run: bool,
) -> dict[str, object]:
    """Run the deterministic refresh and return its compact evidence payload."""
    resolved = _resolve_worktree(worktree)
    current_branch = _current_branch(resolved)
    if branch is not None and branch != current_branch:
        raise RefreshError(
            f"requested branch '{branch}' does not match checkout branch '{current_branch}'"
        )
    payload: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "ok": False,
        "worktree": str(resolved),
        "branch": current_branch,
        "base_ref": base_ref,
        "pr": pr,
        "dry_run": dry_run,
    }
    if pr is not None:
        payload["pr_identity"] = _verify_pr(repo, pr, current_branch)
    _assert_clean(resolved)
    fetch_remote, fetch_ref = _split_base_ref(base_ref)
    _git(resolved, "fetch", fetch_remote, fetch_ref)
    if dry_run:
        payload.update(ok=True, status="dry_run", next_action="rerun without --dry-run")
        return payload

    start_head = _git(resolved, "rev-parse", "HEAD")
    conflicts = _merge_base(resolved, base_ref)
    merge_in_progress = bool(conflicts)
    payload["conflicts"] = conflicts
    if conflicts:
        try:
            _resolve_baseline_conflict(resolved, conflicts)
        except RefreshError:
            payload["aborted"] = _abort_merge(resolved)
            raise
    try:
        validations = _run_validators(resolved, base_ref)
        baseline_changed = bool(_git(resolved, "status", "--porcelain", "--", str(BASELINE_PATH)))
        head_sha = _commit_refresh(resolved, merge_in_progress, baseline_changed)
    except RefreshError:
        if merge_in_progress:
            payload["aborted"] = _abort_merge(resolved)
        raise
    payload["validations"] = validations
    payload["baseline_changed"] = baseline_changed
    payload["start_head"] = start_head
    payload["head_sha"] = head_sha
    if no_push:
        payload.update(ok=True, status="refreshed", pushed=False)
        return payload
    push = _run(
        ["git", "push", "origin", f"HEAD:refs/heads/{current_branch}"],
        cwd=resolved,
    )
    if push.returncode != 0:
        detail = (push.stderr or push.stdout).strip()
        raise RefreshError(f"refresh validated locally but push failed: {detail}")
    payload.update(ok=True, status="refreshed", pushed=True)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", default=".", help="child PR worktree (default: cwd)")
    parser.add_argument("--pr", type=int, help="expected open PR number")
    parser.add_argument("--branch", help="expected branch name (default: current checkout)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="PR repository for --pr checks")
    parser.add_argument("--base", default=DEFAULT_BASE_REF, help="base ref to merge")
    parser.add_argument("--no-push", action="store_true", help="refresh locally without pushing")
    parser.add_argument("--dry-run", action="store_true", help="check preconditions only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the refresh helper and print one compact JSON payload."""
    args = _parse_args(argv)
    try:
        payload = refresh(
            worktree=Path(args.worktree),
            repo=args.repo,
            pr=args.pr,
            branch=args.branch,
            base_ref=args.base,
            no_push=args.no_push,
            dry_run=args.dry_run,
        )
    except (RefreshError, OSError, subprocess.TimeoutExpired) as exc:
        payload = {"schema": SCHEMA_VERSION, "ok": False, "error": str(exc)}
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
