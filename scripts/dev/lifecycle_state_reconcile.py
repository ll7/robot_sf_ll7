#!/usr/bin/env python3
"""Deterministic lifecycle reconciliation for contradictory issue state.

Autonomous candidate selection repeatedly pays triage cost for contradictory
live issue state: ``state:ready`` combined with ``state:running``,
``state:parked``, or ``decision-required``; ``state:running`` with a released
claim and no covering PR; assignments that silently block ready rows; and
closed issues that still carry live routing labels. Individual admission
gates fail closed correctly, but the queue stays polluted.

This planner repairs only evidence-provable contradictions and surfaces the
rest with a stable reason code. Report/plan modes are zero-write. Apply mode
re-reads every row (issue state, labels, body digest, atomic claim, open
covering PRs) immediately before mutating, aborts the row on any drift, and
removes labels only through the canonical REST label helper. Ambiguous rows
are retained untouched with the next authority named.

Closed rows reuse :mod:`scripts.dev.closed_state_label_hygiene` discovery,
confirmation, and fix owners. Open-row classification reuses the shared
:mod:`scripts.dev.issue_state_taxonomy`. The summary block feeds the
goal-autopilot controller receipt so terminal zero-work cannot ignore
unresolved lifecycle drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from scripts.dev import closed_state_label_hygiene as closed_hygiene
from scripts.dev import issue_claim
from scripts.dev import issue_state_taxonomy as taxonomy
from scripts.dev._gh_rest import parse_json, run_gh_api_or_raise
from scripts.dev.gh_pr_label_rest import remove_label

if TYPE_CHECKING:
    from collections.abc import Callable

SCHEMA = "lifecycle_state_reconcile.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 5
DEFAULT_CLOSED_LIMIT = 100

ACTIONS = ("remove_labels", "report_only", "deferred", "no_action")

# Stable reason codes for contradictory lifecycle state. Only the repair
# codes authorize label removal; every other code is surfaced untouched.
REPAIRABLE_REASONS = frozenset(
    {
        "closed_with_live_state_label",
        "stale_running_without_claim_or_pr",
        "evidenced_running_ready_stale",
    }
)

DECISION_REQUIRED_LABEL = "decision-required"


class InventoryIncompleteError(RuntimeError):
    """Raised when a bounded issue inventory cannot prove complete coverage."""

    def __init__(self, *, pages_read: int, max_pages: int) -> None:
        """Record the exhausted page budget for a structured fail-closed result."""
        self.pages_read = pages_read
        self.max_pages = max_pages
        super().__init__(
            f"issue inventory incomplete: max_pages={max_pages} exhausted after a full "
            f"page ({pages_read} pages read); increase --max-pages"
        )


def _label_names(raw: Any) -> list[str]:
    """Normalize REST or normalized label values to a sorted unique name list."""
    names: list[str] = []
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
            elif isinstance(value, dict) and isinstance(value.get("name"), str):
                name = str(value["name"]).strip()
                if name:
                    names.append(name)
    return sorted(set(names))


def _assignee_logins(raw: Any) -> list[str]:
    """Normalize REST assignee values to a sorted unique login list."""
    logins: list[str] = []
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, str) and value.strip():
                logins.append(value.strip())
            elif isinstance(value, dict) and isinstance(value.get("login"), str):
                login = str(value["login"]).strip()
                if login:
                    logins.append(login)
    return sorted(set(logins))


def _body_digest(raw: Any) -> str:
    """Return the SHA-256 digest of an issue body for compare-and-swap checks."""
    body = raw if isinstance(raw, str) else ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _has_unknown_state_labels(labels: set[str]) -> bool:
    """Return whether unclassified ``state:*`` labels are present."""
    return bool(taxonomy.unknown_state_labels(labels))


def _classify_multi_exec(
    *,
    exec_labels: list[str],
    labels: set[str],
    assignees: list[str],
    claim: dict[str, Any] | None,
    covering_prs: list[int] | None,
) -> tuple[str, list[str], str]:
    """Classify rows carrying more than one execution-state label."""
    if set(exec_labels) == {"state:ready", "state:running"}:
        if assignees:
            return (
                "ready_but_assigned",
                [],
                "ready+running with an assignee; ownership is ambiguous, retained",
            )
        if claim is None or covering_prs is None:
            return (
                "ambiguous_do_not_mutate",
                [],
                "ready+running requires claim/PR evidence that was not collected",
            )
        if claim.get("claimed") is True or len(covering_prs) > 0:
            return (
                "evidenced_running_ready_stale",
                ["state:ready"],
                "active running evidence (claim or open covering PR); stale ready removed",
            )
        return (
            "stale_running_without_claim_or_pr",
            ["state:running"],
            "no claim and no open covering PR; stale running removed",
        )
    return (
        "execution_state_conflict",
        [],
        f"multiple execution states {exec_labels}; a human decision is involved, retained",
    )


def _classify_running(
    *,
    labels: set[str],
    assignees: list[str],
    claim: dict[str, Any] | None,
    covering_prs: list[int] | None,
) -> tuple[str, list[str], str]:
    """Classify rows carrying only ``state:running``."""
    if claim is None or covering_prs is None:
        return (
            "ambiguous_do_not_mutate",
            [],
            "running requires claim/PR evidence that was not collected",
        )
    if claim.get("claimed") is True or len(covering_prs) > 0:
        return "consistent", [], "running is evidenced by a claim or an open covering PR"
    if assignees:
        return (
            "ambiguous_do_not_mutate",
            [],
            "unevidenced running with an assignee; ownership is ambiguous, retained",
        )
    if taxonomy.state_qualifier_labels(labels):
        return (
            "ambiguous_do_not_mutate",
            [],
            "unevidenced running with state qualifiers; process signal unclear, retained",
        )
    return (
        "stale_running_without_claim_or_pr",
        ["state:running"],
        "no claim, no open covering PR, no assignees; stale running removed",
    )


def _classify_ready(*, labels: set[str], assignees: list[str]) -> tuple[str, list[str], str]:
    """Classify rows carrying only ``state:ready``."""
    if "state:parked" in labels:
        return (
            "ready_and_parked",
            [],
            "parked work must not be reopened or closed by reconciliation; retained",
        )
    if DECISION_REQUIRED_LABEL in labels:
        return (
            "ready_and_decision_required",
            [],
            "a maintainer decision is pending; retained for the owner",
        )
    if assignees:
        return (
            "ready_but_assigned",
            [],
            "assignee ownership is ambiguous without a claim; retained",
        )
    return "consistent", [], "ready row carries no contradictory lifecycle state"


def _contradiction_signature(
    *,
    labels: set[str],
    assignees: list[str],
    claim: dict[str, Any] | None,
    covering_prs: list[int] | None,
) -> tuple[str, list[str], str]:
    """Classify one open issue row without performing any I/O.

    Returns the stable reason code, the labels eligible for removal (empty
    unless the row is deterministically repairable), and a human rationale.
    ``claim`` and ``covering_prs`` may be ``None`` when the row carries no
    ``state:running`` label and needs no external evidence.
    """
    exec_labels = taxonomy.execution_state_labels(labels)
    if _has_unknown_state_labels(labels):
        return (
            "ambiguous_do_not_mutate",
            [],
            "unclassified state:* labels present; cannot reason about execution semantics",
        )
    if len(exec_labels) > 1:
        return _classify_multi_exec(
            exec_labels=exec_labels,
            labels=labels,
            assignees=assignees,
            claim=claim,
            covering_prs=covering_prs,
        )
    if exec_labels == ["state:running"]:
        return _classify_running(
            labels=labels,
            assignees=assignees,
            claim=claim,
            covering_prs=covering_prs,
        )
    if exec_labels == ["state:ready"]:
        return _classify_ready(labels=labels, assignees=assignees)
    return "consistent", [], "no contradictory execution state"


def plan_row(
    raw_issue: dict[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    claim: dict[str, Any] | None = None,
    covering_prs: list[int] | None = None,
) -> dict[str, Any]:
    """Derive one reconciliation action for a raw open-issue payload."""
    try:
        number = int(raw_issue.get("number"))
    except (TypeError, ValueError):
        number = None
    labels = set(_label_names(raw_issue.get("labels")))
    row: dict[str, Any] = {
        "issue": number,
        "url": raw_issue.get("html_url") or raw_issue.get("url") or "",
        "state": str(raw_issue.get("state") or "").lower(),
        "labels": sorted(labels),
        "assignees": _assignee_logins(raw_issue.get("assignees")),
        "reason_code": "ambiguous_do_not_mutate",
        "action": "deferred",
        "target_labels": [],
        "evidence": {
            "claim_present": None if claim is None else bool(claim.get("claimed")),
            "covering_prs": [] if covering_prs is None else list(covering_prs),
            "body_sha256": _body_digest(raw_issue.get("body")),
        },
        "reason": "",
        "applied": False,
        "applied_labels": [],
    }
    if number is None:
        row["reason"] = "issue number is missing or invalid; cannot address a row"
        return row
    if row["state"] != "open":
        row["reason"] = f"issue state is {row['state'] or 'unknown'}, not open; skipped"
        return row
    if claim is not None and claim.get("ok") is False:
        row["reason"] = "claim read failed; absence of a claim cannot be proven"
        return row
    if _needs_evidence(labels) and (claim is None or covering_prs is None):
        row["reason"] = "running requires claim/PR evidence that was not collected; retry"
        return row
    reason_code, targets, reason = _contradiction_signature(
        labels=labels,
        assignees=row["assignees"],
        claim=claim,
        covering_prs=covering_prs,
    )
    row["reason_code"] = reason_code
    row["reason"] = reason
    if reason_code == "consistent":
        row["action"] = "no_action"
        return row
    if reason_code in REPAIRABLE_REASONS:
        row["action"] = "remove_labels"
        row["target_labels"] = list(targets)
        return row
    row["action"] = "report_only"
    return row


def _read_claim(issue: int, *, remote: str) -> dict[str, Any]:
    """Read atomic claim state through the canonical claim owner."""
    try:
        return issue_claim.status_issue(issue, remote=remote)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"ok": False, "claimed": False, "error": str(exc)}


def _read_covering_prs(issue: int, *, repo: str) -> dict[str, Any]:
    """Read open covering PRs through the canonical coverage owner."""
    try:
        return issue_claim.open_prs_covering_issue(repo=repo, issue_number=issue)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"ok": False, "covering_prs": [], "truncated": False, "error": str(exc)}


def _needs_evidence(labels: set[str]) -> bool:
    """Return whether a row needs claim/PR evidence for classification."""
    return "state:running" in labels


def _fetch_open_state_pages(
    repo: str, *, max_pages: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Page open issues carrying any ``state:*`` label, failing closed on truncation."""
    scanned = 0
    candidates: list[dict[str, Any]] = []
    pages_read = 0
    for page in range(1, max_pages + 1):
        path = f"repos/{repo}/issues?state=open&per_page={PAGE_SIZE}&page={page}"
        result = run_gh_api_or_raise(path)
        payload, error = parse_json(result, what=f"open issue list page {page}")
        if error:
            raise RuntimeError(error)
        if not isinstance(payload, list):
            raise RuntimeError(f"open issue list page {page} was not a JSON list")
        pages_read = page
        scanned += len(payload)
        for entry in payload:
            if not isinstance(entry, dict) or "pull_request" in entry:
                continue
            labels = set(_label_names(entry.get("labels")))
            if not any(label.startswith(taxonomy.STATE_PREFIX) for label in labels):
                continue
            # REST list rows carry no state field; the query already selected open rows.
            entry["state"] = "open"
            candidates.append(entry)
        if len(payload) < PAGE_SIZE:
            break
    else:
        raise InventoryIncompleteError(pages_read=pages_read, max_pages=max_pages)
    inventory = {
        "complete": True,
        "truncated": False,
        "page_size": PAGE_SIZE,
        "max_pages": max_pages,
        "pages_read": pages_read,
        "rows_scanned": scanned,
    }
    return candidates, inventory


def _plan_open_entry(
    entry: dict[str, Any],
    *,
    repo: str,
    read_claim: Callable[..., dict[str, Any]],
    read_covering: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Plan one open candidate, reading claim/PR evidence only for running rows."""
    labels = set(_label_names(entry.get("labels")))
    claim: dict[str, Any] | None = None
    covering: list[int] | None = None
    if _needs_evidence(labels):
        claim = read_claim(int(entry["number"]))
        coverage = read_covering(int(entry["number"]))
        if (
            not isinstance(coverage, dict)
            or coverage.get("ok") is not True
            or coverage.get("truncated") is True
        ):
            claim = {"ok": False, "claimed": False, "error": "covering-PR read unavailable"}
        else:
            covering = [int(pr) for pr in coverage.get("covering_prs", [])]
    return plan_row(entry, repo=repo, claim=claim, covering_prs=covering)


def collect_open_rows(
    repo: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    claim_reader: Callable[..., dict[str, Any]] | None = None,
    covering_reader: Callable[..., dict[str, Any]] | None = None,
    remote: str = DEFAULT_REMOTE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inventory open issues carrying ``state:*`` labels and plan each row.

    Claim and covering-PR evidence is read only for rows carrying
    ``state:running``. Returns planned rows plus inventory metadata.
    """
    if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    read_claim = claim_reader or (lambda issue: _read_claim(issue, remote=remote))
    read_covering = covering_reader or (lambda issue: _read_covering_prs(issue, repo=repo))
    candidates, inventory = _fetch_open_state_pages(repo, max_pages=max_pages)
    rows = [
        _plan_open_entry(entry, repo=repo, read_claim=read_claim, read_covering=read_covering)
        for entry in candidates
    ]
    rows.sort(key=lambda row: (row["issue"] is None, row["issue"] or 0))
    return rows, inventory


def collect_closed_rows(
    repo: str,
    *,
    limit: int = DEFAULT_CLOSED_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover closed issues carrying live routing labels via bounded search.

    Reuses :mod:`scripts.dev.closed_state_label_hygiene` discovery and REST
    confirmation owners. Returns planned rows plus inventory metadata.
    """
    try:
        discovery = closed_hygiene.discover_closed_issues_by_label(
            repo=repo, labels=closed_hygiene.LIVE_STATE_LABELS, limit=limit
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"closed-row discovery failed: {exc}") from exc
    truncated_any = any(
        isinstance(marker, dict) and marker.get("truncated") is True
        for marker in discovery.truncations
    )
    try:
        candidates = closed_hygiene.collect_stale_issues(discovery.rows_by_label, repo=repo)
        confirmed = closed_hygiene.reconcile_stale_issues(repo=repo, candidates=candidates)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"closed-row confirmation failed: {exc}") from exc
    rows = [
        {
            "issue": stale.number,
            "url": stale.url,
            "state": "closed",
            "labels": [],
            "assignees": [],
            "reason_code": "closed_with_live_state_label",
            "action": "remove_labels",
            "target_labels": list(stale.stale_labels),
            "evidence": {
                "claim_present": None,
                "covering_prs": [],
                "body_sha256": "",
            },
            "reason": "closed issues must not carry live routing labels",
            "applied": False,
            "applied_labels": [],
        }
        for stale in confirmed
    ]
    return rows, {
        "complete": not truncated_any,
        "truncated": truncated_any,
        "per_label_limit": limit,
        "labels": list(closed_hygiene.LIVE_STATE_LABELS),
    }


def _live_issue(repo: str, number: int) -> dict[str, Any]:
    """Re-read one issue through the canonical normalized reader."""
    from scripts.dev import issue_implementability

    try:
        return issue_implementability.fetch_live_issue(number, repo=repo)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"live re-read failed before apply: {exc}") from exc


def _collect_apply_evidence(
    *,
    repo: str,
    remote: str,
    number: int,
    live_labels: set[str],
    claim: dict[str, Any] | None,
    covering_prs: list[int] | None,
) -> tuple[dict[str, Any] | None, list[int] | None, str]:
    """Collect live claim/PR evidence for running rows; returns evidence or a deferral reason."""
    if not _needs_evidence(live_labels):
        return None, None, ""
    live_claim = dict(claim) if claim is not None else _read_claim(number, remote=remote)
    if covering_prs is not None:
        live_coverage: dict[str, Any] = {
            "ok": True,
            "covering_prs": list(covering_prs),
            "truncated": False,
        }
    else:
        live_coverage = _read_covering_prs(number, repo=repo)
    if (
        not isinstance(live_coverage, dict)
        or live_coverage.get("ok") is not True
        or live_coverage.get("truncated") is True
    ):
        return None, None, "covering-PR evidence unavailable before apply; re-run report"
    if live_claim.get("ok") is False:
        return None, None, "claim evidence unavailable before apply; re-run the report"
    return live_claim, [int(pr) for pr in live_coverage.get("covering_prs", [])], ""


def _remove_planned_labels(
    *,
    row: dict[str, Any],
    number: int,
    repo: str,
    remover: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Remove each planned label, deferring with partial progress on failure."""
    applied: list[str] = []
    for target in sorted(row["target_labels"]):
        try:
            result = remover(number, target, repo=repo)
        except (OSError, RuntimeError, ValueError) as exc:
            row["action"] = "deferred"
            row["reason"] = f"label removal failed for {target}: {exc}"
            row["applied_labels"] = applied
            return row
        if not isinstance(result, dict) or result.get("status") != "ok":
            row["action"] = "deferred"
            row["reason"] = f"label removal failed for {target}: {result}"
            row["applied_labels"] = applied
            return row
        applied.append(target)
    row["applied"] = True
    row["applied_labels"] = applied
    row["reason"] = f"{row['reason']}; removed {', '.join(applied)}"
    return row


def apply_row(
    row: dict[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    remote: str = DEFAULT_REMOTE,
    live_issue: dict[str, Any] | None = None,
    claim: dict[str, Any] | None = None,
    covering_prs: list[int] | None = None,
    label_remover: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply one planned label removal with exact-item drift checks.

    Re-derives the plan from live state and proceeds only when the reason
    code, target labels, issue state, body digest, claim evidence, and
    covering-PR evidence all match the plan. Injected ``live_issue``,
    ``claim``, and ``covering_prs`` values make the check deterministic in
    fixtures; when omitted they are read live.
    """
    if row.get("action") != "remove_labels" or not row.get("target_labels"):
        return row
    number = row.get("issue")
    if not isinstance(number, int):
        row["action"] = "deferred"
        row["reason"] = "planned issue number is invalid; refusing to apply"
        return row
    remover = label_remover or remove_label
    try:
        live = dict(live_issue) if live_issue is not None else _live_issue(repo, number)
    except RuntimeError as exc:
        row["action"] = "deferred"
        row["reason"] = str(exc)
        return row
    live_state = str(live.get("state") or "").strip().lower()
    if live_state != "open":
        row["action"] = "deferred"
        row["reason"] = f"live issue state is {live_state or 'unknown'}, not open; refusing"
        return row
    live_labels = set(_label_names(live.get("labels")))
    if not set(row["target_labels"]).issubset(live_labels):
        row["action"] = "deferred"
        row["reason"] = "target labels absent from live labels; plan is stale, re-run report"
        return row
    if _body_digest(live.get("body")) != str(row.get("evidence", {}).get("body_sha256") or ""):
        row["action"] = "deferred"
        row["reason"] = "issue body drifted before apply; re-run the report"
        return row
    live_claim, live_covering, defer_reason = _collect_apply_evidence(
        repo=repo,
        remote=remote,
        number=number,
        live_labels=live_labels,
        claim=claim,
        covering_prs=covering_prs,
    )
    if defer_reason:
        row["action"] = "deferred"
        row["reason"] = defer_reason
        return row
    live_plan = plan_row(
        {
            "number": number,
            "state": "open",
            "labels": sorted(live_labels),
            "assignees": _assignee_logins(live.get("assignees")),
            "body": live.get("body") if isinstance(live.get("body"), str) else "",
            "html_url": live.get("html_url") or live.get("url") or "",
        },
        repo=repo,
        claim=live_claim,
        covering_prs=live_covering,
    )
    if live_plan.get("reason_code") != row.get("reason_code") or sorted(
        live_plan.get("target_labels", [])
    ) != sorted(row.get("target_labels", [])):
        row["action"] = "deferred"
        row["reason"] = "live replan differs from the plan; re-run the report"
        return row
    return _remove_planned_labels(row=row, number=number, repo=repo, remover=remover)


def apply_closed_rows(
    rows: list[dict[str, Any]],
    *,
    repo: str = DEFAULT_REPO,
    confirm_closed: Callable[..., bool] | None = None,
    label_remover: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply closed-row label removals through the hygiene fix owner."""
    stale = [
        closed_hygiene.StaleIssue(
            number=int(row["issue"]),
            title="",
            url=str(row.get("url") or ""),
            state="closed",
            stale_labels=tuple(row.get("target_labels", [])),
        )
        for row in rows
        if row.get("action") == "remove_labels" and isinstance(row.get("issue"), int)
    ]
    kwargs: dict[str, Any] = {"repo": repo}
    if confirm_closed is not None:
        kwargs["confirm_closed"] = confirm_closed
    if label_remover is not None:
        kwargs["remove_label"] = label_remover
    outcomes = {
        action["number"]: action
        for action in closed_hygiene.fix_stale_issues(repo=repo, stale_issues=stale, **kwargs)
        if isinstance(action, dict)
    }
    for row in rows:
        outcome = outcomes.get(row.get("issue"))
        if outcome is None:
            continue
        if outcome.get("skipped"):
            row["action"] = "deferred"
            row["reason"] = f"hygiene fix skipped: {outcome.get('reason')}"
        else:
            row["applied"] = True
            row["applied_labels"] = list(outcome.get("removed_labels", []))
            row["reason"] = f"{row['reason']}; removed {', '.join(row['applied_labels'])}"
    return rows


def _summarize(
    rows: list[dict[str, Any]], *, mode: str, origin_main_sha: str | None
) -> dict[str, Any]:
    """Build the versioned reconciliation summary consumed by the controller."""
    actions = {action: sum(1 for row in rows if row.get("action") == action) for action in ACTIONS}
    reason_counts: dict[str, int] = {}
    for row in rows:
        code = str(row.get("reason_code") or "unknown")
        reason_counts[code] = reason_counts.get(code, 0) + 1
    repaired = sum(1 for row in rows if row.get("applied") is True)
    unresolved = sum(
        1
        for row in rows
        if str(row.get("reason_code") or "") != "consistent" and row.get("applied") is not True
    )
    return {
        "schema": SCHEMA,
        "mode": mode,
        "origin_main_sha": origin_main_sha,
        "scanned": len(rows),
        "actions": actions,
        "reason_counts": dict(sorted(reason_counts.items())),
        "repaired_count": repaired,
        "unresolved_drift_count": unresolved,
        "rows": rows,
    }


def _resolve_origin_main_sha(explicit: str | None) -> str | None:
    """Resolve the base revision for summary binding without failing the run."""
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "origin/main^{commit}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, ValueError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote for claim reads.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply planned label removals with drift checks; default is a read-only report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum open-issue pages to scan (100 rows per page).",
    )
    parser.add_argument(
        "--closed-limit",
        type=int,
        default=DEFAULT_CLOSED_LIMIT,
        help="Maximum closed-issue rows per live label for search discovery.",
    )
    parser.add_argument(
        "--skip-closed",
        action="store_true",
        help="Scan open rows only; closed-row discovery needs search quota.",
    )
    parser.add_argument(
        "--origin-main-sha",
        default=None,
        help="Base revision recorded in the summary; resolved from git when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the lifecycle reconciliation planner; incomplete evidence fails closed."""
    args = _build_parser().parse_args(argv)
    mode = "apply" if args.apply else "report"
    try:
        open_rows, open_inventory = collect_open_rows(
            args.repo, max_pages=args.max_pages, remote=args.remote
        )
        closed_rows: list[dict[str, Any]] = []
        closed_inventory: dict[str, Any] = {"complete": True, "skipped": True}
        if not args.skip_closed:
            closed_rows, closed_inventory = collect_closed_rows(args.repo, limit=args.closed_limit)
        if args.apply:
            open_rows = [apply_row(row, repo=args.repo, remote=args.remote) for row in open_rows]
            closed_rows = apply_closed_rows(closed_rows, repo=args.repo)
    except (InventoryIncompleteError, RuntimeError, ValueError) as exc:
        print(f"lifecycle reconcile blocked: {exc}", file=sys.stderr)
        return 2
    rows = sorted(
        open_rows + closed_rows,
        key=lambda row: (row.get("issue") is None, row.get("issue") or 0),
    )
    summary = _summarize(
        rows, mode=mode, origin_main_sha=_resolve_origin_main_sha(args.origin_main_sha)
    )
    summary["inventory"] = {"open": open_inventory, "closed": closed_inventory}
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for row in rows:
            status = "APPLIED" if row.get("applied") else row.get("action")
            print(f"#{row.get('issue')} {status} [{row.get('reason_code')}]: {row.get('reason')}")
        print(
            f"lifecycle reconcile {mode}: scanned={summary['scanned']} "
            f"repaired={summary['repaired_count']} "
            f"unresolved={summary['unresolved_drift_count']}"
        )
    deferred = summary["actions"].get("deferred", 0)
    return 1 if deferred else 0


if __name__ == "__main__":
    sys.exit(main())
