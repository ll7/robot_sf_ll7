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

REST-first merged-PR inventory (issue #8927)
--------------------------------------------
Rule 2 previously resolved each draft reference with a ``gh search prs``
request, which draws on the low-volume, shared search bucket (30 requests per
minute) and failed closed with HTTP 403 while core REST quota was healthy.  The
scanner now reads closed pull requests once through a bounded
``gh api repos/<repo>/pulls?state=closed`` pass, filters to merged pull
requests locally, and builds the reference index in-process.  Search is no
longer on the default path.

If that REST inventory cannot be completed (transport failure, API error, or
page-budget truncation), the report carries ``quota_degraded: true`` with a
``degraded_reason`` and a ``merged_pr_inventory`` metadata block, and the
process exits nonzero even in ``--check`` mode, so unknown Rule 2 coverage is
never mistaken for a clean "no candidates" result.  ``--max-pr-pages`` raises
the bounded page budget.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev._gh_rest import run_gh_api_or_raise as _gh_api_get

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

DEFAULT_REPO = "ll7/robot_sf_ll7"
STALE_HOURS = 48
ISSUE_REF_RE = re.compile(r"(?:Closes|Refs|Fixes|#)\s*(\d+)")
CLOSES_REF_RE = re.compile(r"Closes\s+#(\d+)\b", re.IGNORECASE)
PER_PAGE = 100
# Bounded page budget for the closed-PR REST inventory.  Pagination stops early
# at the true end of history; exhausting this budget is reported as degraded
# coverage rather than being treated as a complete inventory.
DEFAULT_MAX_PR_PAGES = 80


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


class ReferenceLookupError(RuntimeError):
    """An individual issue/PR reference could not be resolved.

    This is distinct from a GitHub/API failure.  The scanner can report one
    unresolved reference and continue evaluating the other drafts, while
    authentication and transport failures still fail the whole read-only
    report closed.
    """


# ---------------------------------------------------------------------------
# Low-level GitHub CLI helpers
# ---------------------------------------------------------------------------


def _run_json(command: list[str], *, default: Any = None) -> Any:
    """Run a gh command and parse JSON output."""
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


def _is_unresolvable_reference_error(error: RuntimeError) -> bool:
    """Return whether a GitHub lookup failed because the number is unknown."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "could not resolve to an issue",
            "could not resolve to a pullrequest",
            "no pull requests found",
        )
    )


def fetch_issue_state(*, repo: str, number: int) -> str | None:
    """Return the state for an issue or pull-request number.

    GitHub CLI resolves pull-request numbers through ``gh issue view`` too and
    returns ``MERGED`` for a merged pull request.  Keeping that state explicit
    lets rule evaluation classify a draft that references a merged PR as
    superseded instead of treating the reference as malformed.

    Raises:
        ReferenceLookupError: If GitHub returns an unexpected reference state.
        RuntimeError: If GitHub cannot return a valid state payload.
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
    try:
        payload = _run_json(cmd)
    except RuntimeError as exc:
        if _is_unresolvable_reference_error(exc):
            raise ReferenceLookupError(
                f"GitHub could not resolve issue/PR #{number}: {exc}"
            ) from exc
        raise
    if not isinstance(payload, dict):
        raise RuntimeError(f"GitHub returned an invalid state payload for issue #{number}")
    state = str(payload.get("state", "")).upper()
    if state not in {"CLOSED", "OPEN", "MERGED"}:
        raise ReferenceLookupError(
            f"GitHub returned an unresolvable state for issue/PR #{number}: {state!r}"
        )
    return state


def _is_merged_pr_row(row: dict[str, Any]) -> bool:
    """Return whether a REST pull-request row represents a merged pull request.

    The ``pulls?state=closed`` list endpoint returns both merged and
    closed-unmerged pull requests.  The list payload does not populate the
    ``merged`` boolean (it comes back as ``null``), so a non-null ``merged_at``
    timestamp is the reliable merged signal there; an explicit boolean is
    honored when a caller supplies the single-PR payload shape.
    """
    merged = row.get("merged")
    if merged is True:
        return True
    if merged is False:
        return False
    return bool(row.get("merged_at"))


def _normalize_merged_pr_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a REST pull payload onto the merged-PR row shape rule 2 consumes."""
    return {
        "number": int(row.get("number", 0)),
        "title": str(row.get("title", "")),
        "url": str(row.get("html_url", "") or row.get("url", "") or ""),
        "body": str(row.get("body", "") or ""),
    }


def fetch_merged_pr_inventory(
    *,
    repo: str,
    max_pages: int = DEFAULT_MAX_PR_PAGES,
    per_page: int = PER_PAGE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read merged pull requests once through a bounded core-REST inventory.

    Closed pull requests are read page by page from
    ``repos/<repo>/pulls?state=closed`` (core quota, never the rate-limited
    search endpoint), merged rows are filtered locally, and pagination stops as
    soon as a page returns fewer than ``per_page`` rows.  Exhausting the page
    budget with full pages marks the inventory as potentially truncated so the
    caller reports degraded coverage instead of an authoritative empty result.

    Returns ``(merged_pr_rows, inventory_metadata)``.
    """
    if max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    rows: list[dict[str, Any]] = []
    pages_read = 0
    for page in range(1, max_pages + 1):
        path = (
            f"repos/{repo}/pulls?state=closed&sort=updated&direction=desc"
            f"&per_page={per_page}&page={page}"
        )
        result = _gh_api_get(path)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip() or (
                f"exit code {result.returncode}"
            )
            raise RuntimeError(f"GitHub REST read failed ({path}): {detail}")
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from GitHub REST ({path}): {exc.msg}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list from GitHub REST ({path})")
        page_rows = [row for row in payload if isinstance(row, dict)]
        rows.extend(page_rows)
        pages_read = page
        if len(page_rows) < per_page:
            break

    merged_rows = [_normalize_merged_pr_row(row) for row in rows if _is_merged_pr_row(row)]
    truncated = is_likely_truncated(len(rows), limit=max_pages * per_page)
    metadata: dict[str, Any] = {
        "mode": "rest",
        "source": f"repos/{repo}/pulls?state=closed",
        "closed_rows": len(rows),
        "merged_count": len(merged_rows),
        "pages_read": pages_read,
        "per_page": per_page,
        "page_budget": max_pages,
        "truncated": truncated,
    }
    return merged_rows, metadata


def build_merged_pr_index(
    merged_pr_rows: list[dict[str, Any]],
    issue_numbers: Iterable[int],
) -> dict[int, list[dict[str, Any]]]:
    """Build the issue-reference index from one flat merged-PR inventory.

    Each merged PR body is scanned once for ``Closes #N`` references; matches
    for the requested issue numbers are grouped in-process so rule 2 never
    issues a per-reference search request.  Index rows are ordered by PR number
    for deterministic reports.
    """
    index: dict[int, list[dict[str, Any]]] = {number: [] for number in sorted(set(issue_numbers))}
    wanted = set(index)
    if not wanted:
        return index
    for row in sorted(merged_pr_rows, key=lambda item: int(item.get("number", 0))):
        body = str(row.get("body", "") or "")
        referenced = {int(match.group(1)) for match in CLOSES_REF_RE.finditer(body)}
        for issue_number in sorted(referenced & wanted):
            index[issue_number].append(
                {
                    "number": int(row.get("number", 0)),
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                }
            )
    return index


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


def _resolve_reference_state(
    draft: DraftPr,
    *,
    repo: str,
    number: int,
    get_issue_state: Any,
    warnings: list[dict[str, Any]] | None,
) -> tuple[str | None, bool]:
    """Resolve one reference, returning ``(state, unresolved)``."""
    try:
        return get_issue_state(repo=repo, number=number), False
    except ReferenceLookupError as exc:
        if warnings is not None:
            warnings.append(
                {
                    "type": "skipped_reference",
                    "draft_pr": draft.number,
                    "reference": number,
                    "reason": str(exc),
                }
            )
        return None, True


def _append_reference_state_rule(
    state: str | None,
    *,
    issue_num: int,
    rules: list[str],
    evidence: list[str],
) -> None:
    """Append the hard rule represented by a resolved reference state."""
    if state == "CLOSED":
        rules.append("linked_issue_closed")
        evidence.append(f"Rule 1: linked issue #{issue_num} is CLOSED")
    elif state == "MERGED":
        rules.append("superseded_by_merged_pr")
        evidence.append(f"Rule 2: reference #{issue_num} resolves to a MERGED pull request")


def _merged_prs_for_issue(
    issue_number: int,
    *,
    repo: str,
    merged_pr_index: Mapping[int, list[dict[str, Any]]] | None,
    get_merged_prs: Any,
    limit: int,
) -> list[dict[str, Any]]:
    """Resolve rule-2 merged PRs from the pre-built index or an injected lookup.

    Exactly one source must be available; refusing to guess coverage keeps a
    missing inventory from silently reading as "no merged PR matches".
    """
    if merged_pr_index is not None:
        return merged_pr_index.get(issue_number, [])
    if get_merged_prs is not None:
        return get_merged_prs(repo=repo, issue_number=issue_number, limit=limit)
    raise RuntimeError("rule 2 requires a merged-PR index or lookup; refusing to guess coverage")


def evaluate_rules(
    draft: DraftPr,
    *,
    repo: str,
    get_issue_state: Any = fetch_issue_state,
    get_merged_prs: Any = None,
    get_modified_files: Any = get_modified_files_on_main_since,
    merged_pr_index: Mapping[int, list[dict[str, Any]]] | None = None,
    merged_pr_limit: int = 30,
    warnings: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """Evaluate all rules for one draft PR.

    Rule 2 resolves references from the pre-built ``merged_pr_index`` when one
    is supplied (the REST-first default path).  The ``get_merged_prs`` callable
    remains injectable for focused tests; callers must supply exactly one of
    the two so unknown coverage can never be silently treated as "no match".

    Returns (rules_triggered, evidence_lines).
    """
    rules: list[str] = []
    evidence: list[str] = []
    linked = draft.linked_issue_numbers()
    unresolved_references: set[int] = set()

    # Rule 1: linked issue closed
    for issue_num in linked:
        state, unresolved = _resolve_reference_state(
            draft,
            repo=repo,
            number=issue_num,
            get_issue_state=get_issue_state,
            warnings=warnings,
        )
        if unresolved:
            unresolved_references.add(issue_num)
            continue
        _append_reference_state_rule(
            state,
            issue_num=issue_num,
            rules=rules,
            evidence=evidence,
        )

    # Rule 2: superseded by merged PR
    for issue_num in linked:
        if issue_num in unresolved_references:
            continue
        merged = _merged_prs_for_issue(
            issue_num,
            repo=repo,
            merged_pr_index=merged_pr_index,
            get_merged_prs=get_merged_prs,
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
    get_merged_prs: Any = None,
    get_modified_files: Any = get_modified_files_on_main_since,
    merged_pr_index: Mapping[int, list[dict[str, Any]]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
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
            merged_pr_index=merged_pr_index,
            warnings=warnings,
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
    warnings: list[dict[str, Any]] | None = None,
    quota_degraded: bool = False,
    degraded_reason: str | None = None,
    merged_pr_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable JSON report.

    ``quota_degraded`` marks a merged-PR REST inventory that could not be
    completed; the report then fails closed (``ok: false``) even with zero
    candidates so unknown rule-2 coverage is never read as a clean result.
    """
    hard_candidates = [c for c in candidates if c.has_hard_rule]
    if hard_candidates:
        failure_summary: dict[str, Any] | None = {
            "reason": "superseded_draft_candidates_found",
            "hard_count": len(hard_candidates),
            "total_count": len(candidates),
        }
    elif quota_degraded:
        failure_summary = {
            "reason": "merged_pr_inventory_degraded",
            "hard_count": 0,
            "total_count": len(candidates),
        }
    else:
        failure_summary = None
    return {
        "schema": "superseded_draft_scanner.v1",
        "ok": not hard_candidates and not quota_degraded,
        "read_only": True,
        "repo": repo,
        "scanned_drafts": scanned_count,
        "truncated": truncated,
        "quota_degraded": quota_degraded,
        "degraded_reason": degraded_reason,
        "merged_pr_inventory": merged_pr_inventory or {},
        "candidate_count": len(candidates),
        "hard_candidate_count": len(hard_candidates),
        "candidates": [c.to_payload() for c in candidates],
        "warnings": list(warnings or []),
        "failure_summary": failure_summary,
    }


def build_markdown(report: dict[str, Any]) -> str:
    """Build a human-readable markdown summary from a report dict."""
    lines: list[str] = []
    lines.append("## Superseded Draft PR Scan")
    lines.append("")

    if report.get("truncated"):
        lines.append("WARNING: results may be truncated (hit gh search limit).")
        lines.append("")

    if report.get("quota_degraded"):
        reason = report.get("degraded_reason") or "merged-PR inventory incomplete"
        lines.append(
            f"WARNING: merged-PR inventory degraded ({reason}); rule 2 coverage is unknown."
        )
        lines.append("")

    lines.append(f"**Repo**: {report['repo']}")
    lines.append(f"**Drafts scanned**: {report['scanned_drafts']}")
    lines.append(f"**Candidates**: {report['candidate_count']}")
    lines.append(f"**Hard (rules 1-2)**: {report['hard_candidate_count']}")
    lines.append("")

    warnings = report.get("warnings", [])
    if warnings:
        lines.append(f"**Warnings**: {len(warnings)} individual reference(s) skipped")
        for warning in warnings:
            lines.append(
                f"- PR #{warning.get('draft_pr')}: reference #{warning.get('reference')} "
                f"skipped: {warning.get('reason', 'unresolved reference')}"
            )
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
        help=(
            "Exit nonzero when hard close-candidates exist (rules 1-2) or the "
            "merged-PR inventory is degraded."
        ),
    )
    parser.add_argument(
        "--max-pr-pages",
        type=int,
        default=DEFAULT_MAX_PR_PAGES,
        help=(
            "Maximum REST pages of closed PRs to read for the merged-PR inventory "
            f"(each {PER_PAGE} rows, default {DEFAULT_MAX_PR_PAGES}); exhausting the "
            "budget reports quota_degraded."
        ),
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


def _resolve_merged_pr_index(
    *,
    repo: str,
    referenced_issues: list[int],
    max_pages: int,
) -> tuple[dict[int, list[dict[str, Any]]], bool, str | None, dict[str, Any]]:
    """Build the rule-2 index from the REST inventory, marking degraded coverage.

    Returns ``(index, quota_degraded, degraded_reason, inventory_metadata)``.
    An unavailable or page-budget-truncated inventory yields an empty index and
    a degraded status so the caller fails closed instead of reporting a clean
    empty candidate set.
    """
    if not referenced_issues:
        return {}, False, None, {"mode": "rest", "skipped": "no_linked_references"}
    try:
        merged_pr_rows, metadata = fetch_merged_pr_inventory(repo=repo, max_pages=max_pages)
    except (OSError, RuntimeError, ValueError) as exc:
        return (
            {},
            True,
            f"merged-PR REST inventory unavailable: {exc}",
            {"mode": "rest", "error": str(exc)},
        )
    if metadata.get("truncated"):
        reason = (
            "merged-PR REST inventory truncated at "
            f"{metadata.get('page_budget')} pages; raise --max-pr-pages"
        )
        return build_merged_pr_index(merged_pr_rows, referenced_issues), True, reason, metadata
    return build_merged_pr_index(merged_pr_rows, referenced_issues), False, None, metadata


def _write_json_report(report: dict[str, Any], *, output: str | None) -> None:
    """Write a JSON report to ``output`` or stdout."""
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


def _emit_error_report(
    *,
    repo: str,
    error: str,
    output: str | None,
    markdown: bool,
    warnings: list[dict[str, Any]],
) -> int:
    """Write the deterministic error report JSON and optional markdown note."""
    error_report = {
        "schema": "superseded_draft_scanner.v1",
        "ok": False,
        "read_only": True,
        "repo": repo,
        "scanned_drafts": 0,
        "candidate_count": 0,
        "hard_candidate_count": 0,
        "candidates": [],
        "warnings": warnings,
        "error": error,
    }
    _write_json_report(error_report, output=output)
    if markdown:
        print(f"Scanner failed: {error}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    repo = args.repo
    warnings: list[dict[str, Any]] = []

    try:
        if args.max_pr_pages < 1:
            raise ValueError(f"--max-pr-pages must be >= 1, got {args.max_pr_pages}")
        draft_prs, truncated = fetch_draft_prs(repo=repo, limit=args.limit)

        # Enrich with file lists for Rule 3
        enriched: list[DraftPr] = []
        for pr in draft_prs:
            files = fetch_pr_files(repo=repo, pr_number=pr.number)
            pr.files = files
            enriched.append(pr)

        referenced_issues = sorted(
            {number for pr in enriched for number in pr.linked_issue_numbers()}
        )
        merged_pr_index, quota_degraded, degraded_reason, inventory_meta = _resolve_merged_pr_index(
            repo=repo,
            referenced_issues=referenced_issues,
            max_pages=args.max_pr_pages,
        )
        candidates = scan_drafts(
            enriched,
            repo=repo,
            get_issue_state=fetch_issue_state,
            merged_pr_index=merged_pr_index,
            get_modified_files=get_modified_files_on_main_since,
            warnings=warnings,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _emit_error_report(
            repo=repo,
            error=str(exc),
            output=args.output,
            markdown=args.markdown,
            warnings=warnings,
        )

    report = build_report(
        repo=repo,
        candidates=candidates,
        scanned_count=len(draft_prs),
        truncated=truncated,
        warnings=warnings,
        quota_degraded=quota_degraded,
        degraded_reason=degraded_reason,
        merged_pr_inventory=inventory_meta,
    )

    _write_json_report(report, output=args.output)

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
        if quota_degraded:
            print(
                "FAIL: merged-PR inventory degraded; --check cannot verify rule-2 coverage.",
                file=sys.stderr,
            )
            return 1

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
