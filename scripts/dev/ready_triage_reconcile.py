"""Evidence-based reconciliation for contradictory ready/triage labels on open issues.

``state:ready`` and ``needs-triage`` are mutually exclusive by contract: readiness records that
contract triage already happened, so the blocking triage label must not suppress an admitted issue
(see ``docs/context/issue_audit_contract.md`` and issue #8837). Historical bulk sweeps could still
leave both labels in place, which makes the ready pool look blocked.

This planner finds open issues carrying both labels, recomputes each issue's classification with
the triage label ignored, and derives exactly one action:

- ``remove_triage``: readiness is current (the issue classifies ``ready`` without triage), so the
  stale triage label is removed.
- ``remove_ready``: readiness is stale (the issue classifies as a contract or structural failure
  without triage), so readiness is removed and the triage label is preserved.
- ``report_only``: another blocking label is present, so the row needs manual review.
- ``deferred``: a live re-read or label write failed or drifted; the row is retried on a later run.

``--report`` (default) performs no GitHub mutation. ``--apply`` re-reads each issue immediately
before mutating, aborts the row on drift, and never closes issues, merges pull requests, or edits
Project #5 state.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from scripts.dev import issue_implementability
from scripts.dev._gh_rest import parse_json, run_gh_api_or_raise
from scripts.dev.gh_pr_label_rest import remove_label

SCHEMA = "ready_triage_reconcile.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
READY_LABEL = "state:ready"
TRIAGE_LABEL = "needs-triage"
ACTIONS = ("remove_triage", "remove_ready", "report_only", "deferred")
DEFAULT_MAX_PAGES = 5
PAGE_SIZE = 100
# Classifications that invalidate the readiness decision itself. Anything else (ready,
# needs_compute, external-input block, working/review holds, ...) means readiness is still a
# valid triage decision and only the contradictory triage label is removed.
STALE_READINESS_CLASSIFICATIONS = frozenset(
    {
        "needs_spec",
        "parent",
        "state_conflict",
        "wrong_owner_repo",
        "human_decision",
        "invalid",
    }
)


def _label_names(raw: Any) -> list[str]:
    """Normalize REST or normalized label values to a sorted unique name list."""
    names: list[str] = []
    for value in raw or []:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            names.append(value["name"])
    return sorted({name.strip() for name in names if name.strip()})


def _blocking_labels(labels: set[str]) -> list[str]:
    """Return the blocking workflow labels present on an issue."""
    return sorted(
        label
        for label in labels
        if label in issue_implementability.BLOCKING_LABELS or label.startswith("blocked:")
    )


def list_contradictory_issues(
    repo: str, *, max_pages: int = DEFAULT_MAX_PAGES
) -> list[dict[str, Any]]:
    """Return open issues carrying both the ready and triage labels, via REST."""
    issues: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        path = (
            f"repos/{repo}/issues?state=open&labels={READY_LABEL},{TRIAGE_LABEL}"
            f"&per_page={PAGE_SIZE}&page={page}"
        )
        result = run_gh_api_or_raise(path)
        payload, error = parse_json(result, what=f"issue list page {page}")
        if error:
            raise RuntimeError(error)
        if not isinstance(payload, list):
            raise RuntimeError(f"issue list page {page} was not a JSON list")
        rows = [row for row in payload if isinstance(row, dict) and "pull_request" not in row]
        issues.extend(rows)
        if len(payload) < PAGE_SIZE:
            break
    return issues


def plan_row(raw_issue: dict[str, Any], *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Derive one reconciliation action for a raw issue payload."""
    labels = set(_label_names(raw_issue.get("labels")))
    blocking = _blocking_labels(labels)
    row: dict[str, Any] = {
        "issue": raw_issue.get("number"),
        "url": raw_issue.get("html_url") or raw_issue.get("url") or "",
        "labels": sorted(labels),
        "blocking_labels": blocking,
        "action": "report_only",
        "classification_without_triage": None,
        "reason": "",
        "applied": False,
    }
    if blocking != [TRIAGE_LABEL]:
        row["reason"] = "other blocking labels present; requires manual review"
        return row

    without_triage = dict(raw_issue)
    without_triage["labels"] = [
        label for label in _label_names(raw_issue.get("labels")) if label != TRIAGE_LABEL
    ]
    claim = {"ok": True, "claimed": False, "claim_ref": None, "sha": None}
    try:
        report = issue_implementability.evaluate_issue(without_triage, claim, repository=repo)
    except (TypeError, ValueError) as exc:
        row["action"] = "deferred"
        row["reason"] = f"classification failed: {exc}"
        return row

    classification = report["classification"]
    row["classification_without_triage"] = classification
    if classification == "error":
        row["action"] = "deferred"
        row["reason"] = "classification returned an error state; retry before mutating"
    elif classification in STALE_READINESS_CLASSIFICATIONS:
        row["action"] = "remove_ready"
        row["reason"] = (
            f"readiness is stale; issue classifies as {classification} without the triage label"
        )
    else:
        row["action"] = "remove_triage"
        row["reason"] = (
            "readiness remains a valid triage decision without the triage label "
            f"(classification: {classification})"
        )
    return row


def apply_row(
    row: dict[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    fetch_issue: Callable[..., dict[str, Any]] = issue_implementability.fetch_live_issue,
    label_remover: Callable[..., dict[str, Any]] = remove_label,
) -> dict[str, Any]:
    """Apply one planned label removal with a live drift check, fail closed per row."""
    if row["action"] not in ("remove_triage", "remove_ready"):
        return row
    number = row["issue"]
    try:
        live = fetch_issue(number, repo=repo)
    except (RuntimeError, ValueError) as exc:
        row["action"] = "deferred"
        row["reason"] = f"live re-read failed before apply: {exc}"
        return row

    live_labels = set(_label_names(live.get("labels")))
    if not {READY_LABEL, TRIAGE_LABEL}.issubset(live_labels):
        row["action"] = "deferred"
        row["reason"] = "label drift before apply; re-run the report"
        return row
    if _blocking_labels(live_labels) != [TRIAGE_LABEL]:
        row["action"] = "deferred"
        row["reason"] = "another blocking label appeared before apply; re-run the report"
        return row

    target = TRIAGE_LABEL if row["action"] == "remove_triage" else READY_LABEL
    result = label_remover(number, target, repo=repo)
    if isinstance(result, dict) and result.get("status") == "ok":
        row["applied"] = True
        row["reason"] = f"{row['reason']}; removed {target}"
    else:
        row["action"] = "deferred"
        row["reason"] = f"label removal failed: {result}"
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned label removals; default is a read-only report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the plan as JSON.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum issue-list pages to scan (100 rows per page).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the reconciliation planner; non-zero exit means deferred rows remain."""
    args = _build_parser().parse_args(argv)
    raw_issues = list_contradictory_issues(args.repo, max_pages=args.max_pages)
    rows = [plan_row(issue, repo=args.repo) for issue in raw_issues]
    if args.apply:
        rows = [apply_row(row, repo=args.repo) for row in rows]

    counts = {action: sum(1 for row in rows if row["action"] == action) for action in ACTIONS}
    plan = {
        "schema": SCHEMA,
        "mode": "apply" if args.apply else "report",
        "repo": args.repo,
        "scanned": len(rows),
        "actions": counts,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        for row in rows:
            status = "APPLIED" if row["applied"] else row["action"]
            print(f"#{row['issue']} {status}: {row['reason']}")
        print(f"ready/triage reconcile plan: {counts}")
    return 1 if counts["deferred"] else 0


if __name__ == "__main__":
    sys.exit(main())
