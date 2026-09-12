#!/usr/bin/env python3
"""Prune stale remote code branches and released claim refs (issue #9087).

Remote heads accumulate historical work branches and ``agent-claims/issue-<n>``
refs whose issues are already closed. Auto-delete on merge covers recent PR
branches, but there is no repository-owned sweep for the backlog. This command
classifies every remote head with stable reason codes and deletes only the two
safe classes, dry-run by default:

- ``merged_code_branch``: the tip is already an ancestor of the main ref and no
  open PR references the head.
- ``claim_ref_closed_issue``: an ``agent-claims/issue-<n>`` ref whose issue is
  closed.

Everything else is kept, including protected refs, open-PR heads, claim refs
whose issue is open or unresolved, unmerged branches, and any ref whose state
could not be determined (fail closed). Deletion is bounded by ``--limit`` and
each deletion is recorded in the report.

Usage::

    uv run python scripts/dev/prune_stale_remote_branches.py
    uv run python scripts/dev/prune_stale_remote_branches.py --apply --limit 25 --report /tmp/prune.json

Exit codes:
    0 - scan completed (dry run or all requested deletions succeeded)
    1 - one or more deletions failed
    2 - the sweep could not classify safely (remote read failure)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPORT_SCHEMA = "stale_remote_branch_report.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
MAIN_BRANCH = "main"
PROTECTED_BRANCH_PREFIXES = ("release/", "release-")
CLAIM_REF_RE = re.compile(r"^refs/heads/agent-claims/issue-(\d+)$")

DELETE_MERGED = "merged_code_branch"
DELETE_CLAIM_CLOSED = "claim_ref_closed_issue"
DELETE_REASONS = (DELETE_MERGED, DELETE_CLAIM_CLOSED)
KEEP_OPEN_PR = "open_pr_head"
KEEP_OPEN_CLAIM = "open_issue_claim"
KEEP_UNMERGED = "unmerged_branch"
KEEP_PROTECTED = "protected_ref"
KEEP_PROBE_ERROR = "probe_error"


class RemoteProbe:
    """Read/write access to the remote heads a sweep needs."""

    def list_heads(self) -> dict[str, str]:  # pragma: no cover - protocol
        """Return ``{ref, sha}`` for every remote head."""
        raise NotImplementedError

    def is_ancestor(self, sha: str) -> bool | None:  # pragma: no cover - protocol
        """Return whether *sha* is an ancestor of the main ref (None on error)."""
        raise NotImplementedError

    def open_pr_heads(self) -> set[str] | None:  # pragma: no cover - protocol
        """Return short head ref names with an open PR (None on error)."""
        raise NotImplementedError

    def issue_state(self, number: int) -> str | None:  # pragma: no cover - protocol
        """Return ``open``/``closed`` for one issue number (None on error)."""
        raise NotImplementedError

    def delete_head(self, ref: str) -> tuple[bool, str]:  # pragma: no cover - protocol
        """Delete one remote head; return ``(ok, error_message)``."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HeadRow:
    """One classified remote head."""

    ref: str
    sha: str
    reason: str

    @property
    def action(self) -> str:
        """Return ``delete`` for classified candidates and ``keep`` otherwise."""
        return "delete" if self.reason in DELETE_REASONS else "keep"

    def to_dict(self) -> dict[str, str]:
        """Return the stable report representation of this row."""
        return {"ref": self.ref, "sha": self.sha, "reason": self.reason, "action": self.action}


def _short_name(ref: str) -> str:
    return ref.removeprefix("refs/heads/")


def _is_protected(ref: str) -> bool:
    short = _short_name(ref)
    return short == MAIN_BRANCH or short.startswith(PROTECTED_BRANCH_PREFIXES)


def classify_heads(
    heads: dict[str, str],
    *,
    is_ancestor,
    open_pr_heads: set[str],
    issue_state,
) -> list[HeadRow]:
    """Classify every head with a stable reason code; keep on any uncertainty."""
    rows: list[HeadRow] = []
    for ref, sha in sorted(heads.items()):
        if _is_protected(ref):
            rows.append(HeadRow(ref, sha, KEEP_PROTECTED))
            continue
        claim_match = CLAIM_REF_RE.match(ref)
        if claim_match is not None:
            state = issue_state(int(claim_match.group(1)))
            if state == "closed":
                rows.append(HeadRow(ref, sha, DELETE_CLAIM_CLOSED))
            elif state == "open":
                rows.append(HeadRow(ref, sha, KEEP_OPEN_CLAIM))
            else:
                rows.append(HeadRow(ref, sha, KEEP_PROBE_ERROR))
            continue
        if _short_name(ref) in open_pr_heads:
            rows.append(HeadRow(ref, sha, KEEP_OPEN_PR))
            continue
        ancestor = is_ancestor(sha)
        if ancestor is None:
            rows.append(HeadRow(ref, sha, KEEP_PROBE_ERROR))
        elif ancestor:
            rows.append(HeadRow(ref, sha, DELETE_MERGED))
        else:
            rows.append(HeadRow(ref, sha, KEEP_UNMERGED))
    return rows


def build_report(
    rows: Sequence[HeadRow], *, repo: str, main_ref: str, deletions: Sequence[dict] = ()
) -> dict:
    """Build the deterministic sweep report."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.reason] = counts.get(row.reason, 0) + 1
    candidates = [row.to_dict() for row in rows if row.action == "delete"]
    return {
        "schema": REPORT_SCHEMA,
        "repo": repo,
        "main_ref": main_ref,
        "head_count": len(rows),
        "candidate_count": len(candidates),
        "keep_count": len(rows) - len(candidates),
        "reason_counts": dict(sorted(counts.items())),
        "candidates": candidates,
        "heads": [row.to_dict() for row in rows],
        "deletions": list(deletions),
    }


def run_scan(probe: RemoteProbe, *, repo: str, main_ref: str) -> dict:
    """Scan the remote and return a dry-run report, or exit 2 when unreadable."""
    heads = probe.list_heads()
    open_pr_heads = probe.open_pr_heads()
    if open_pr_heads is None:
        print("error: could not read open PR heads; refusing to classify", file=sys.stderr)
        raise SystemExit(2)
    rows = classify_heads(
        heads,
        is_ancestor=probe.is_ancestor,
        open_pr_heads=open_pr_heads,
        issue_state=probe.issue_state,
    )
    return build_report(rows, repo=repo, main_ref=main_ref)


def apply_deletions(probe: RemoteProbe, report: dict, *, limit: int) -> dict:
    """Delete up to *limit* candidates and return the updated report."""
    deletions: list[dict] = []
    for candidate in report["candidates"][:limit]:
        ok, error = probe.delete_head(candidate["ref"])
        deletions.append({"ref": candidate["ref"], "ok": ok, "error": error})
    updated = dict(report)
    updated["deletions"] = deletions
    updated["deleted_count"] = sum(1 for item in deletions if item["ok"])
    updated["failed_count"] = sum(1 for item in deletions if not item["ok"])
    return updated


class GitGhProbe:
    """RemoteProbe backed by ``git`` and ``gh``."""

    def __init__(self, *, repo: str, remote: str, main_ref: str) -> None:
        """Initialize the probe with repository, remote, and main ref."""
        self.repo = repo
        self.remote = remote
        self.main_ref = main_ref

    def _run(self, args: Sequence[str]) -> tuple[int, str]:
        completed = subprocess.run(list(args), capture_output=True, text=True, check=False)
        return completed.returncode, completed.stdout.strip()

    def list_heads(self) -> dict[str, str]:
        """Return ``{ref, sha}`` for every remote head."""
        code, output = self._run(["git", "ls-remote", "--heads", self.remote])
        if code != 0:
            print("error: git ls-remote failed", file=sys.stderr)
            raise SystemExit(2)
        heads: dict[str, str] = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                heads[parts[1]] = parts[0]
        return heads

    def is_ancestor(self, sha: str) -> bool | None:
        """Return whether *sha* is an ancestor of the main ref."""
        code, _ = self._run(["git", "merge-base", "--is-ancestor", sha, self.main_ref])
        if code == 0:
            return True
        if code == 1:
            return False
        return None

    def open_pr_heads(self) -> set[str] | None:
        """Return short head ref names with an open PR."""
        code, output = self._run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                self.repo,
                "--state",
                "open",
                "--limit",
                "1000",
                "--json",
                "headRefName",
            ]
        )
        if code != 0:
            return None
        try:
            items = json.loads(output)
        except json.JSONDecodeError:
            return None
        return {str(item["headRefName"]) for item in items if isinstance(item, dict)}

    def issue_state(self, number: int) -> str | None:
        """Return ``open``/``closed`` for one issue number."""
        code, output = self._run(
            ["gh", "api", f"repos/{self.repo}/issues/{number}", "--jq", ".state"]
        )
        if code != 0:
            return None
        return output.strip() or None

    def delete_head(self, ref: str) -> tuple[bool, str]:
        """Delete one remote head and return ``(ok, error_message)``."""
        code, output = self._run(["git", "push", self.remote, "--delete", _short_name(ref)])
        if code == 0:
            return True, ""
        return False, output or f"git push --delete failed with exit code {code}"


def _write_report(report: dict, path: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repository ({DEFAULT_REPO}).")
    parser.add_argument("--remote", default="origin", help="Git remote name (default: origin).")
    parser.add_argument("--main-ref", default="origin/main", help="Main ref to compare against.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete classified candidates (dry run by default).",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum deletions per run.")
    parser.add_argument("--report", type=Path, default=None, help="Write the report to this path.")
    return parser


def main(argv: Sequence[str] | None = None, *, probe: RemoteProbe | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    active_probe = probe or GitGhProbe(repo=args.repo, remote=args.remote, main_ref=args.main_ref)
    report = run_scan(active_probe, repo=args.repo, main_ref=args.main_ref)
    exit_code = 0
    if args.apply:
        report = apply_deletions(active_probe, report, limit=max(args.limit, 0))
        exit_code = 1 if report.get("failed_count") else 0
    _write_report(report, args.report)
    summary = (
        f"heads={report['head_count']} candidates={report['candidate_count']} "
        f"kept={report['keep_count']}"
    )
    if args.apply:
        summary += (
            f" deleted={report.get('deleted_count', 0)} failed={report.get('failed_count', 0)}"
        )
    print(summary, file=sys.stderr)
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
