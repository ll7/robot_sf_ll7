#!/usr/bin/env python3
"""Superseded-draft scanner — deterministic close-candidate report for zombie draft PRs.

Draft PRs whose original purpose has been fulfilled by other work can linger
indefinitely as "zombies".  This scanner identifies candidates by applying three
deterministic rules, emits a structured JSON report, and supports a ``--check``
flag for CI gates.

What it does (issue #5393)
--------------------------
For each open DRAFT PR, flag as a close-candidate when ANY of these hold:

1. **Linked issue closed** (hard): the PR references an issue (``Closes`` /
   ``Refs #N`` in body) that is now CLOSED.
2. **Superceded by merged PR** (hard): a MERGED PR claims ``Closes #N`` for the
   same issue number the draft references.
3. **Stale + superseded files** (weak, report-only): every file the draft
   touches has been modified on ``main`` since the draft's last commit, AND the
   draft is older than 48 h.

Design notes
------------
- NO auto-closing. The report feeds the gate/orchestrator who closes with a
  citation.
- ``--check`` exits nonzero when hard candidates (rules 1-2) exist.
- ``--markdown`` emits a human-readable summary suitable for GitHub comments.
- The scanner is read-only against GitHub; it never mutates PR state.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.dev._gh_pagination import is_likely_truncated

DEFAULT_REPO = "ll7/robot_sf_ll7"
STALE_HOURS = 48
ISSUE_REF_RE = re.compile(r"(?:Closes|Refs|Fixes|#)\s*(\d+)")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class DraftPr:
    """One open draft PR pulled from the GitHub API."""

    number: int
    title: str
    body: str
    url: str
    created_at: str
    updated_at: str
    files: list[str]

    def __init__(
        self,
        *,
        number: int,
        title: str,
        body: str,
        url: str,
        created_at: str,
        updated_at: str,
        files: list[str] | None = None,
    ) -> None:
        """Initialize DraftPr."""
        self.number = number
        self.title = title
        self.body = body
        self.url = url
        for field_name, value in (("created_at", created_at), ("updated_at", updated_at)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty ISO timestamp")
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid {field_name} timestamp: {value!r}") from exc
        self.created_at = created_at
        self.updated_at = updated_at
        self.files = files or []

    def linked_issue_numbers(self) -> list[int]:
        """Extract issue numbers referenced in body via Closes/Refs/Fixes/#."""
        nums: set[int] = set()
        for m in ISSUE_REF_RE.finditer(self.body or ""):
            nums.add(int(m.group(1)))
        return sorted(nums)

    @property
    def age(self) -> timedelta:
        """Time since creation."""
        dt = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        return datetime.now(dt.tzinfo) - dt

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable PR summary."""
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "linked_issues": self.linked_issue_numbers(),
            "file_count": len(self.files),
        }


class SupersededCandidate:
    """A draft PR flagged as a close candidate."""

    def __init__(
        self,
        *,
        pr: DraftPr,
        rules: list[str],
        evidence: list[str],
    ) -> None:
        """Initialize SupersededCandidate."""
        self.pr = pr
        self.rules = rules  # e.g. ["linked_issue_closed"]
        self.evidence = evidence

    @property
    def has_hard_rule(self) -> bool:
        """Return True when a hard close-candidate rule fired."""
        hard = {"linked_issue_closed", "superseded_by_merged_pr"}
        return bool(set(self.rules) & hard)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate summary."""
        return {
            "pr": self.pr.to_payload(),
            "rules": self.rules,
            "evidence": self.evidence,
            "hard": self.has_hard_rule,
        }


# ---------------------------------------------------------------------------
# Low-level GitHub CLI helpers
# ---------------------------------------------------------------------------


def _run_json(command: list[str], *, default: Any = None) -> Any:
    """Run a gh/gh-search command and parse JSON output."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' was not found; install gh or add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f": {stderr}" if stderr else ""
        raise RuntimeError(f"GitHub CLI command failed ({' '.join(command)}){details}") from exc

    stdout = result.stdout.strip()
    if not stdout:
        return default if default is not None else []
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse gh JSON output: {exc.msg}") from exc


def fetch_draft_prs(*, repo: str, limit: int) -> tuple[list[DraftPr], bool]:
    """Fetch all open draft PRs for a repo.

    Returns (draft_prs, truncated).
    """
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--draft",
        "--json",
        "number,title,body,url,createdAt,updatedAt",
        "--limit",
        str(limit),
    ]
    raw = _run_json(cmd)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list from draft PR fetch, got {type(raw).__name__}")

    truncated = is_likely_truncated(len(raw), limit=limit)
    prs: list[DraftPr] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            prs.append(
                DraftPr(
                    number=int(row["number"]),
                    title=str(row.get("title", "")),
                    body=str(row.get("body", "") or ""),
                    url=str(row.get("url", "")),
                    created_at=str(row.get("createdAt", "")),
                    updated_at=str(row.get("updatedAt", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return prs, truncated


def fetch_pr_files(*, repo: str, pr_number: int) -> list[str]:
    """Fetch file paths changed in a PR."""
    cmd = [
        "gh",
        "pr",
        "diff",
        str(pr_number),
        "--repo",
        repo,
        "--name-only",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        return lines
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def fetch_issue_state(*, repo: str, number: int) -> str | None:
    """Return ``CLOSED`` or ``OPEN`` for an issue number.

    Raises:
        RuntimeError: If GitHub cannot return a valid issue state.
    """
    cmd = [
        "gh",
        "issue",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "state",
    ]
    payload = _run_json(cmd)
    if not isinstance(payload, dict):
        raise RuntimeError(f"GitHub returned an invalid state payload for issue #{number}")
    state = str(payload.get("state", "")).upper()
    if state not in {"CLOSED", "OPEN"}:
        raise RuntimeError(f"GitHub returned an invalid state for issue #{number}: {state!r}")
    return state


def fetch_merged_prs_for_issue(
    *,
    repo: str,
    issue_number: int,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Find merged PRs that claim 'Closes #N' for a given issue."""
    cmd = [
        "gh",
        "search",
        "prs",
        f"#{issue_number}",
        "--repo",
        repo,
        "--state",
        "closed",
        "--merged",
        "--json",
        "number,title,url,body",
        "--limit",
        str(limit),
    ]
    raw = _run_json(cmd)
    if not isinstance(raw, list):
        return []

    results: list[dict[str, Any]] = []
    pattern = re.compile(rf"Closes\s+#{re.escape(str(issue_number))}\b", re.IGNORECASE)
    for row in raw:
        if not isinstance(row, dict):
            continue
        body = str(row.get("body", "") or "")
        if pattern.search(body):
            results.append(
                {
                    "number": int(row.get("number", 0)),
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                }
            )
    return results


def get_modified_files_on_main_since(
    *,
    repo: str,
    pr_number: int,
    branch: str = "main",
) -> list[str]:
    """Return files modified on ``branch`` since the draft PR's last commit.

    Uses ``gh pr diff`` to find the merge base, then ``git log`` on the branch.
    This is a best-effort helper; failures return an empty list (conservative).
    """
    # Get the draft PR's last commit SHA
    cmd_sha = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "commits",
    ]
    try:
        payload = _run_json(cmd_sha)
        if not isinstance(payload, dict):
            return []
        commits = payload.get("commits", [])
        if not commits:
            return []
        last_sha = None
        for c in reversed(commits):
            if isinstance(c, dict) and isinstance(c.get("oid"), str):
                last_sha = c["oid"]
                break
        if not last_sha:
            return []
    except RuntimeError:
        return []

    # Find files changed on branch since last_sha was its tip
    # We use git log with --first-parent to stay on branch history
    cmd_files = [
        "git",
        "log",
        f"{last_sha}..{branch}",
        "--first-parent",
        "--name-only",
        "--pretty=",
    ]
    try:
        result = subprocess.run(
            cmd_files,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = {line.strip() for line in result.stdout.strip().splitlines() if line.strip()}
        return sorted(lines)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------


def evaluate_rules(
    draft: DraftPr,
    *,
    repo: str,
    get_issue_state: Any = fetch_issue_state,
    get_merged_prs: Any = fetch_merged_prs_for_issue,
    get_modified_files: Any = get_modified_files_on_main_since,
    merged_pr_limit: int = 30,
) -> tuple[list[str], list[str]]:
    """Evaluate all rules for one draft PR.

    Returns (rules_triggered, evidence_lines).
    """
    rules: list[str] = []
    evidence: list[str] = []
    linked = draft.linked_issue_numbers()

    # Rule 1: linked issue closed
    for issue_num in linked:
        state = get_issue_state(repo=repo, number=issue_num)
        if state == "CLOSED":
            rules.append("linked_issue_closed")
            evidence.append(f"Rule 1: linked issue #{issue_num} is CLOSED")

    # Rule 2: superseded by merged PR
    for issue_num in linked:
        merged = get_merged_prs(
            repo=repo,
            issue_number=issue_num,
            limit=merged_pr_limit,
        )
        for pr_info in merged:
            rules.append("superseded_by_merged_pr")
            evidence.append(
                f"Rule 2: merged PR #{pr_info['number']} ({pr_info['title']}) "
                f"claims 'Closes #{issue_num}'"
            )

    # Rule 3: stale + all files modified on main (weak, report-only)
    if draft.age > timedelta(hours=STALE_HOURS):
        draft_files = set(draft.files) if draft.files else set()
        if draft_files:
            modified = set(get_modified_files(repo=repo, pr_number=draft.number))
            if modified and draft_files.issubset(modified):
                rules.append("stale_all_files_modified_on_main")
                evidence.append(
                    f"Rule 3: draft is {int(draft.age.total_seconds() // 3600)}h old; "
                    f"all {len(draft_files)} file(s) modified on main since last commit"
                )

    return rules, evidence


def scan_drafts(
    draft_prs: list[DraftPr],
    *,
    repo: str,
    get_issue_state: Any = fetch_issue_state,
    get_merged_prs: Any = fetch_merged_prs_for_issue,
    get_modified_files: Any = get_modified_files_on_main_since,
) -> list[SupersededCandidate]:
    """Run all rules over all draft PRs and return candidates."""
    candidates: list[SupersededCandidate] = []
    for draft in draft_prs:
        rules, evidence = evaluate_rules(
            draft,
            repo=repo,
            get_issue_state=get_issue_state,
            get_merged_prs=get_merged_prs,
            get_modified_files=get_modified_files,
        )
        if rules:
            candidates.append(
                SupersededCandidate(
                    pr=draft,
                    rules=rules,
                    evidence=evidence,
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def build_report(
    *,
    repo: str,
    candidates: list[SupersededCandidate],
    scanned_count: int,
    truncated: bool = False,
) -> dict[str, Any]:
    """Build the machine-readable JSON report."""
    hard_candidates = [c for c in candidates if c.has_hard_rule]
    return {
        "schema": "superseded_draft_scanner.v1",
        "ok": not hard_candidates,
        "read_only": True,
        "repo": repo,
        "scanned_drafts": scanned_count,
        "truncated": truncated,
        "candidate_count": len(candidates),
        "hard_candidate_count": len(hard_candidates),
        "candidates": [c.to_payload() for c in candidates],
        "failure_summary": {
            "reason": "superseded_draft_candidates_found",
            "hard_count": len(hard_candidates),
            "total_count": len(candidates),
        }
        if hard_candidates
        else None,
    }


def build_markdown(report: dict[str, Any]) -> str:
    """Build a human-readable markdown summary from a report dict."""
    lines: list[str] = []
    lines.append("## Superseded Draft PR Scan")
    lines.append("")

    if report.get("truncated"):
        lines.append("WARNING: results may be truncated (hit gh search limit).")
        lines.append("")

    lines.append(f"**Repo**: {report['repo']}")
    lines.append(f"**Drafts scanned**: {report['scanned_drafts']}")
    lines.append(f"**Candidates**: {report['candidate_count']}")
    lines.append(f"**Hard (rules 1-2)**: {report['hard_candidate_count']}")
    lines.append("")

    candidates = report.get("candidates", [])
    if not candidates:
        lines.append("No superseded draft candidates found.")
        lines.append("")
        return "\n".join(lines)

    for c in candidates:
        pr = c["pr"]
        hard_tag = " [HARD]" if c["hard"] else ""
        rules_str = ", ".join(c["rules"])
        lines.append(f"### PR #{pr['number']}: {pr['title']} ({pr['url']}){hard_tag}")
        lines.append("")
        lines.append(f"**Rules**: {rules_str}")
        lines.append(f"**Age**: {pr['created_at']}")
        lines.append(f"**Linked issues**: {pr['linked_issues'] or '(none)'}")
        lines.append("")
        for ev in c["evidence"]:
            lines.append(f"- {ev}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub repository as OWNER/REPO (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max draft PRs to scan (default: 100).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero when hard close-candidates exist (rules 1-2).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a markdown summary to stderr in addition to JSON on stdout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON report to this path instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    repo = args.repo

    try:
        draft_prs, truncated = fetch_draft_prs(repo=repo, limit=args.limit)

        # Enrich with file lists for Rule 3
        enriched: list[DraftPr] = []
        for pr in draft_prs:
            files = fetch_pr_files(repo=repo, pr_number=pr.number)
            pr.files = files
            enriched.append(pr)

        candidates = scan_drafts(
            enriched,
            repo=repo,
            get_issue_state=fetch_issue_state,
            get_merged_prs=fetch_merged_prs_for_issue,
            get_modified_files=get_modified_files_on_main_since,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        error_report = {
            "schema": "superseded_draft_scanner.v1",
            "ok": False,
            "read_only": True,
            "repo": repo,
            "scanned_drafts": 0,
            "candidate_count": 0,
            "hard_candidate_count": 0,
            "candidates": [],
            "error": str(exc),
        }
        serialized = json.dumps(error_report, indent=2, sort_keys=True)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        else:
            print(serialized)
        if args.markdown:
            print(f"Scanner failed: {exc}", file=sys.stderr)
        return 2

    report = build_report(
        repo=repo,
        candidates=candidates,
        scanned_count=len(draft_prs),
        truncated=truncated,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.markdown:
        md = build_markdown(report)
        print(md, file=sys.stderr)

    if args.check:
        hard = sum(1 for c in candidates if c.has_hard_rule)
        if hard:
            print(
                f"FAIL: {hard} hard close-candidate(s) found among {len(candidates)} candidate(s).",
                file=sys.stderr,
            )
            return 1

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
