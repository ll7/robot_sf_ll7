#!/usr/bin/env python3
"""Detect undeclared stacked ancestry before PR review (issue #7515).

A branch may silently carry commits and changed paths from another issue's
non-``main`` ancestry (the #7308/#7309 contaminated-history failure, replaced by
the clean #7389/#7390 PRs).  Today nothing fails closed before PR creation or
``merge-ready`` when that ancestry is undeclared.  This module computes the live
``origin/main`` merge base, enumerates the commits and paths introduced through
non-``main`` ancestry, and classifies the branch against one machine-readable
stack declaration.

Canonical stack declaration
===========================

One and only one format is accepted.  A PR body (or the branch's declaration
text, for pre-PR branches) may carry a ``## Stack Declaration`` section::

    ## Stack Declaration
    parent_pr: #1234
    parent_head: 0123456789abcdef0123456789abcdef01234567

``parent_pr`` binds the parent pull-request number (required) and
``parent_head`` binds the *expected parent head SHA* the branch was created from
(required, full 40-hex).  The parser is a small pure function
(``parse_stack_declaration``); malformed declarations are rejected fail-closed
(``None`` plus a reason) rather than silently ignored.

Classification (pure, unit-testable)
====================================

``ancestry_state(...)`` returns one of:

- ``clean`` — the head has no commits beyond current ``origin/main`` (already
  merged, or an ancestor), or the live merge base equals the live ``origin/main``
  tip, so the branch was created from current ``main`` and only its intended
  commits are present.  Passes before PR creation and before ``merge-ready``.
- ``stacked`` — non-``main`` ancestry is present and the declaration binds the
  exact parent PR and parent head SHA that produced the actual ancestry.
  Classified ``stacked_not_independently_mergeable``: never independently
  merge-ready until the parent merges and the child is re-evaluated against
  current ``main`` (then only its residual diff remains, i.e. ``clean``).
- ``undeclared_stack`` — non-``main`` ancestry is present with no declaration.
  FAILS before PR creation and before ``merge-ready``.
- ``mismatched_declaration`` — a declaration is present but the declared parent
  PR/head does not match the actual ancestry.  FAILS closed.
- ``parent_invalidated`` — the declared parent is closed-unmerged, rewritten, or
  its head changed (or its live state cannot be verified).  FAILS closed /
  invalidates the prior declaration.
- ``parent_merged`` — the declared parent merged; the child must be re-evaluated
  against current ``main``.  After re-evaluation only the residual diff remains
  and the state becomes ``clean``.

The module deliberately adds no parallel ownership or gate system: the decision
values flow into the existing pre-PR gate
(``scripts/dev/check_prepublication_state.py``), the readiness classifier
(``scripts/dev/pr_loop_policy.py``), and the merge gate
(``scripts/dev/merge_queue_gate.py``) through the same fail-closed reason
vocabulary those modules already use.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_SCHEMA = "stack_ancestry.v1"
_DEFAULT_REMOTE = "origin"
_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_PARENT_PR_RE = re.compile(r"^parent_pr\s*:\s*#?(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_PARENT_HEAD_RE = re.compile(
    r"^parent_head\s*:\s*([0-9a-fA-F]{40})\s*$", re.IGNORECASE | re.MULTILINE
)
_SECTION_HEADING_RE = re.compile(r"^##\s+Stack\s+Declaration\s*$", re.IGNORECASE | re.MULTILINE)
_NEXT_SECTION_HEADING_RE = re.compile(r"^##\s+.*$", re.MULTILINE)

# States that fail closed before PR creation and before ``merge-ready``.
BLOCKING_STATES = frozenset({"undeclared_stack", "mismatched_declaration", "parent_invalidated"})
# States that must never be independently merged.
NOT_INDEPENDENTLY_MERGEABLE_STATES = frozenset({"stacked", "parent_merged"})

# Valid parent PR lifecycle states used by the fail-closed validator.
_VALID_PARENT_STATES = frozenset({"open", "closed"})


@dataclass(frozen=True, slots=True)
class StackDeclaration:
    """A parsed machine-readable stack declaration.

    ``parent_pr`` is the parent pull-request number and ``parent_head`` the
    expected parent head SHA (lowercased full 40-hex) the branch was created
    from.
    """

    parent_pr: int
    parent_head: str


def _extract_declaration_section(text: str) -> str:
    """Return the text of the first ``## Stack Declaration`` section, or ''."""
    if not isinstance(text, str) or not text:
        return ""
    match = _SECTION_HEADING_RE.search(text)
    if not match:
        return ""
    section = text[match.end() :]
    next_heading = _NEXT_SECTION_HEADING_RE.search(section)
    return section[: next_heading.start()] if next_heading else section


def parse_stack_declaration(text: str) -> tuple[StackDeclaration | None, str | None]:
    """Parse the canonical stack declaration from PR/branch text.

    Returns ``(declaration, None)`` for a well-formed declaration, or
    ``(None, reason)`` fail-closed for a malformed one.  A missing section is
    *not* an error: it simply means no declaration is present
    (``(None, None)``), which the classifier treats as ``undeclared_stack`` when
    non-``main`` ancestry exists.

    Both ``parent_pr`` and ``parent_head`` are required; the head must be the
    full 40-hex SHA.  A section carrying only one of the two fields is rejected
    (a partial declaration must not silently weaken the gate).
    """
    section = _extract_declaration_section(text)
    if not section:
        return None, None
    pr_match = _PARENT_PR_RE.search(section)
    head_match = _PARENT_HEAD_RE.search(section)
    if pr_match is None or head_match is None:
        return None, "stack declaration requires both parent_pr and parent_head"
    parent_pr = int(pr_match.group(1))
    if parent_pr < 1:
        return None, "stack declaration parent_pr must be positive"
    parent_head = head_match.group(1).lower()
    return StackDeclaration(parent_pr=parent_pr, parent_head=parent_head), None


def _sha_matches(expected: str, actual: str) -> bool:
    """Return whether two SHA strings identify the same commit."""
    if not expected or not actual:
        return False
    return expected.lower() == actual.lower()


def _unexpected_commits(*, main_tip_sha: str, merge_base_sha: str, commits: list[str]) -> list[str]:
    """Return commits introduced through non-``main`` ancestry.

    When the merge base equals the live ``origin/main`` tip, the branch was
    created from current ``main`` and every ``git log main..head`` commit is the
    branch's own intended work (nothing unexpected).  Otherwise the full
    ``main..head`` enumeration is the non-``main`` ancestry: it contains commits
    from other branches/PRs that were never merged to ``main`` plus the branch's
    own commits, and the caller must resolve it via declaration or rebase.
    """
    if _sha_matches(merge_base_sha, main_tip_sha):
        return []
    return [str(commit) for commit in commits if str(commit)]


def _parent_head_in_ancestry(parent_head: str, *, merge_base_sha: str, commits: list[str]) -> bool:
    """Return whether the declared parent head produced the actual ancestry.

    A declaration matches when the declared parent head is the merge base (the
    commit the branch was created from) or is itself one of the non-``main``
    ancestry commits.  ``git log --oneline`` lines carry abbreviated SHAs, so a
    commit line whose leading hex prefix matches the declared head also counts
    as the same commit.
    """
    if _sha_matches(parent_head, merge_base_sha):
        return True
    declared = parent_head.lower()
    for commit in commits:
        if _sha_matches(parent_head, commit):
            return True
        head = commit.split(" ", 1)[0].lower()
        if head and head.isalnum() and declared.startswith(head) and len(head) >= 7:
            return True
    return False


def _parent_pr_in_commits(parent_pr: int, commits: list[str]) -> bool:
    """Return whether any ancestry commit references the declared parent PR.

    Best-effort evidence used only to distinguish ``mismatched_declaration``
    from a head-only mismatch: PR-merged commits conventionally carry a
    ``(#<number>)`` trailer in their subject.
    """
    needle = f"#{parent_pr}"
    return any(needle in commit for commit in commits)


def ancestry_state(  # noqa: PLR0913 - explicit pure-classifier inputs (issue #7515)
    *,
    head_sha: str,
    base_ref: str,
    main_tip_sha: str,
    merge_base_sha: str,
    commits: list[str],
    declaration: StackDeclaration | None = None,
    parent_state: str = "",
    parent_merged: bool = False,
    parent_head_changed: bool = False,
) -> dict[str, Any]:
    """Classify branch ancestry into a deterministic, fail-closed state.

    Pure function: every input is explicit; no network or filesystem access.
    ``commits`` is the full ``git log --oneline <main>..<head>`` output list
    (each entry may carry a subject with a ``(#NNN)`` trailer).  ``parent_state``
    is the live parent PR state (``open``/``closed``/``unknown``),
    ``parent_merged`` whether the parent was merged, and
    ``parent_head_changed`` whether the parent's live head no longer equals the
    declared head.

    Returns a dict with ``state`` plus diagnostic fields consumed by the CLI and
    the pre-PR gate (issue #7515 item 5).
    """
    commits = [str(commit) for commit in commits if str(commit)]
    base_is_main_tip = _sha_matches(merge_base_sha, main_tip_sha)
    unexpected = _unexpected_commits(
        main_tip_sha=main_tip_sha, merge_base_sha=merge_base_sha, commits=commits
    )

    diagnostic: dict[str, Any] = {
        "head_sha": head_sha,
        "base_ref": base_ref,
        "merge_base_sha": merge_base_sha,
        "main_tip_sha": main_tip_sha,
        "unexpected_commits": unexpected,
        "unexpected_paths": [],
        "declared_parent": None if declaration is None else declaration.parent_pr,
        "declared_parent_head": None if declaration is None else declaration.parent_head,
        "parent_state": str(parent_state or "").lower(),
        "parent_merged": parent_merged,
        "parent_head_changed": parent_head_changed,
        "remediation": "",
    }

    # A head with no commits beyond current main (already merged, or an ancestor
    # of main) has nothing to review: pass.  A head whose merge base equals the
    # live main tip was created from current main, so every ``main..head``
    # commit is the branch's own intended work: pass.  A stale declaration on
    # such a branch is harmless cleanup, not a blocker.
    if not commits or base_is_main_tip:
        return {"state": "clean", **diagnostic}

    # Non-main ancestry is present: ``git log main..head`` is non-empty while
    # the live merge base is an older commit than current ``origin/main``.  The
    # enumeration contains commits that were never merged to main, so a
    # machine-readable declaration is required (issue #7515).
    if declaration is None:
        diagnostic["remediation"] = (
            "branch carries commits not reachable from origin/main with no stack "
            "declaration; rebase the intended work onto origin/main "
            "(git rebase --onto origin/main <merge-base> <branch>) and keep only the "
            "intended commits, or add the canonical ## Stack Declaration binding the "
            "parent PR and parent head SHA"
        )
        return {"state": "undeclared_stack", **diagnostic}

    # Parent lifecycle failures invalidate the declaration before content checks.
    # An empty ``parent_state`` means the lifecycle was not evaluated (the pre-PR
    # gate records local git ancestry only); it is not evidence of a broken
    # parent, so the declaration is still validated on the head anchor.  An
    # explicit ``"unknown"`` (parent lookup attempted and failed, as in the
    # ``check-ancestry`` CLI) fails closed.
    state = str(parent_state or "").strip().lower()
    if state == "unknown":
        diagnostic["remediation"] = (
            "cannot verify the declared parent PR; verify it exists and re-declare "
            "with its current head, or rebase the branch onto origin/main"
        )
        diagnostic["parent_state"] = "unknown"
        return {"state": "parent_invalidated", **diagnostic}
    if state not in _VALID_PARENT_STATES:
        state = ""  # unevaluated: not a lifecycle failure signal
    if parent_merged:
        diagnostic["remediation"] = (
            "parent merged; re-evaluate the child against current main (rebase onto "
            "origin/main) so only its residual diff remains"
        )
        return {"state": "parent_merged", **diagnostic}
    if state == "closed":
        diagnostic["remediation"] = (
            "declared parent is closed unmerged; rebase the branch onto origin/main "
            "and remove the stale declaration"
        )
        return {"state": "parent_invalidated", **diagnostic}
    if parent_head_changed:
        diagnostic["remediation"] = (
            "declared parent head changed (parent rewritten); re-declare with the "
            "parent's current head or rebase onto origin/main"
        )
        return {"state": "parent_invalidated", **diagnostic}

    if not _parent_head_in_ancestry(
        declaration.parent_head, merge_base_sha=merge_base_sha, commits=unexpected
    ):
        if _parent_pr_in_commits(declaration.parent_pr, unexpected):
            diagnostic["remediation"] = (
                "declared parent head does not match the actual parent ancestry; "
                "re-declare with the parent's exact head SHA"
            )
        else:
            diagnostic["remediation"] = (
                "declared parent PR does not match the actual ancestry; re-declare "
                "with the real parent PR and head, or rebase onto origin/main"
            )
        return {"state": "mismatched_declaration", **diagnostic}

    diagnostic["remediation"] = (
        "valid stack: classified stacked_not_independently_mergeable until the parent "
        "merges and this branch is re-evaluated against current main"
    )
    return {"state": "stacked", **diagnostic}


def _changed_paths(
    worktree: Path,
    *,
    main_ref: str,
    head_sha: str,
    git_runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[str] | None, str | None]:
    """Return changed paths introduced by the non-``main`` ancestry commits."""
    if git_runner is None:
        git_runner = _default_git
    result = git_runner(["diff", "--name-only", f"{main_ref}...{head_sha}"], worktree)
    if result.returncode != 0:
        return None, result.stderr.strip() or "git diff --name-only failed"
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(set(paths)), None


def _default_git(args: list[str], worktree: Path) -> subprocess.CompletedProcess[str]:
    """Run one git command in an explicit worktree."""
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def collect_ancestry_facts(
    *,
    head_sha: str,
    base_ref: str,
    worktree: Path,
    remote: str = _DEFAULT_REMOTE,
    main_ref: str | None = None,
    git_runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Compute live ancestry facts for one branch head.

    Runs bounded local git commands in ``worktree`` (the remote-tracking refs
    must already be fetched; ``stacked_prs check-ancestry`` fetches first).  All
    inputs are validated before any command runs.  ``git_runner`` is injectable
    for deterministic tests (default: real ``git -C`` subprocess calls).
    Returns ``(facts, None)`` or ``(None, error)`` fail-closed.
    """
    if not _FULL_SHA_RE.fullmatch(str(head_sha or "")):
        return None, "head SHA must be the full 40-hex SHA"
    base_ref = str(base_ref or "").strip()
    if not base_ref:
        return None, "base ref must not be empty"
    remote = str(remote or "").strip() or _DEFAULT_REMOTE
    main_ref = main_ref or f"refs/remotes/{remote}/main"
    if git_runner is None:
        git_runner = _default_git

    main_tip = git_runner(["rev-parse", main_ref], worktree)
    if main_tip.returncode != 0:
        return None, main_tip.stderr.strip() or f"cannot resolve {main_ref}"
    main_tip_sha = main_tip.stdout.strip()
    if not _FULL_SHA_RE.fullmatch(main_tip_sha):
        return None, f"unexpected main tip SHA from {main_ref}: {main_tip_sha!r}"

    merge_base = git_runner(["merge-base", main_ref, head_sha], worktree)
    if merge_base.returncode != 0:
        return None, merge_base.stderr.strip() or f"git merge-base {main_ref} {head_sha} failed"
    merge_base_sha = merge_base.stdout.strip()
    if not _FULL_SHA_RE.fullmatch(merge_base_sha):
        return None, f"unexpected merge-base SHA: {merge_base_sha!r}"

    log = git_runner(["log", "--oneline", f"{main_ref}..{head_sha}"], worktree)
    if log.returncode != 0:
        return None, log.stderr.strip() or "git log failed"
    commits = [line for line in log.stdout.splitlines() if line.strip()]

    paths, path_error = _changed_paths(
        worktree,
        main_ref=main_ref,
        head_sha=head_sha,
        git_runner=git_runner,
    )
    if path_error:
        return None, path_error

    return {
        "schema": _SCHEMA,
        "head_sha": head_sha,
        "base_ref": base_ref,
        "remote": remote,
        "main_ref": main_ref,
        "main_tip_sha": main_tip_sha,
        "merge_base_sha": merge_base_sha,
        "commits": commits,
        "changed_paths": paths,
    }, None


def render_diagnostics(state: dict[str, Any]) -> list[str]:
    """Render the deterministic diagnostic block (issue #7515 item 5).

    Prints the actual base ref, merge base SHA, unexpected commit list,
    unexpected path list, declared parent (or "none"), and a remediation
    command or procedure.
    """
    lines = [
        f"ancestry state: {state.get('state', 'unknown')}",
        f"base ref: {state.get('base_ref') or '(none)'}",
        f"merge base: {state.get('merge_base_sha') or '(unknown)'}",
        f"main tip: {state.get('main_tip_sha') or '(unknown)'}",
    ]
    commits = state.get("unexpected_commits") or []
    paths = state.get("unexpected_paths") or []
    if commits:
        lines.append("unexpected commits:")
        lines.extend(f"  {commit}" for commit in commits)
    else:
        lines.append("unexpected commits: (none)")
    if paths:
        lines.append("unexpected paths:")
        lines.extend(f"  {path}" for path in paths)
    else:
        lines.append("unexpected paths: (none)")
    declared = state.get("declared_parent")
    declared_head = state.get("declared_parent_head")
    lines.append(f"declared parent: {declared if declared is not None else 'none'}")
    lines.append(f"declared parent head: {declared_head if declared_head is not None else 'none'}")
    remediation = state.get("remediation")
    if remediation:
        lines.append(f"remediation: {remediation}")
    return lines


def remediation_command(*, parent_head: str, branch: str) -> str:
    """Return the deterministic rebase remediation command (issue #7515 item 5).

    ``parent_head`` may be the declared parent head or the actual merge base;
    ``branch`` is the local branch name.  When no parent head is available the
    command degrades to a plain onto-``origin/main`` rebase.
    """
    if parent_head:
        return f"git rebase --onto origin/main {parent_head} {branch}"
    return f"git rebase --onto origin/main {branch}"
