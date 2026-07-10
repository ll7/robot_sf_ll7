#!/usr/bin/env python3
"""Emit a compact issue-batch snapshot for token-efficient goal orchestration."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from scripts.dev._gh_pagination import is_likely_truncated
from scripts.dev.issue_claim import short_claim_ref, status_issue

BODY_EXCERPT_CHARS = 300
DEFAULT_CLAIMABLE_LIMIT = 20
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
BLOCKED_EXTERNAL_INPUT_LABEL = "state:blocked-external-input"
EXTERNAL_RESOURCE_LABEL = "resource:external-data"
EXTERNAL_BLOCKER_LABELS = {
    BLOCKED_EXTERNAL_INPUT_LABEL,
    "blocked",
    "evidence:blocked",
    "state:blocked",
}
PORTFOLIO_CLASSIFICATIONS = {
    "blocked_external_asset",
    "diagnostic_only",
    "executable_now",
    "needs_human_decision",
    "paper_critical",
    "stale_synthesis",
}
UNCLAIMABLE_LABELS = {
    "blocked",
    "decision-required",
    "duplicate",
    "invalid",
    "state:blocked",
    "state:hold",
    "state:running",
    "wontfix",
}


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def expand_issue_numbers(values: list[int], *, expand_range: bool) -> list[int]:
    """Return issue numbers, expanding a two-number range when requested."""
    if expand_range and len(values) == 2 and values[0] < values[1]:
        return list(range(values[0], values[1] + 1))
    return values


def _labels(issue: dict[str, Any]) -> list[str]:
    """Return compact label names from gh issue JSON."""
    return sorted(
        str(label.get("name", ""))
        for label in issue.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    )


def _assignees(issue: dict[str, Any]) -> list[str]:
    """Return compact assignee logins from gh issue JSON."""
    return sorted(
        str(user.get("login", ""))
        for user in issue.get("assignees", [])
        if isinstance(user, dict) and user.get("login")
    )


def _issue_classification(
    *,
    assignees: list[str],
    claim: dict[str, Any],
    labels: list[str],
) -> tuple[str, str]:
    """Return a short claimability classification and rationale."""
    if assignees:
        return "assigned", "assigned; skip auto-claim"
    if _is_blocked_external_issue(labels):
        return "blocked_external", "external input required; omit from default agent queue"
    if any(label in UNCLAIMABLE_LABELS for label in labels):
        return "blocked_label", "label suggests skip in autonomous claim mode"
    if claim.get("ok") is False:
        return "claim_unknown", "unable to read claim state"
    if claim.get("claimed"):
        return "claimed", "already claimed by another worker"
    return "claimable", "open, unassigned, and unclaimed"


def _claim_payload(claim: dict[str, Any]) -> dict[str, Any]:
    """Return the stable claim subset exposed in issue snapshots."""
    return {
        "ok": claim.get("ok"),
        "claimed": claim.get("claimed"),
        "claim_ref": claim.get("claim_ref"),
        "sha": claim.get("sha"),
    }


def _claim_status_payload(
    issue_number: int,
    *,
    remote: str,
    ok: bool,
    claimed: bool | None,
    sha: str | None,
    error: str = "",
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Return an issue-claim status payload matching ``status_issue`` shape."""
    payload: dict[str, Any] = {
        "schema": "issue_claim.v1",
        "action": "status",
        "ok": ok,
        "claimed": claimed,
        "issue": issue_number,
        "remote": remote,
        "claim_ref": short_claim_ref(issue_number),
        "sha": sha,
        "command": command or [],
    }
    if error:
        payload["error"] = error
    return payload


def _batch_claim_statuses(issue_numbers: list[int], *, remote: str) -> dict[int, dict[str, Any]]:
    """Fetch all issue-claim refs once and return status payloads for each issue."""
    unique_numbers = sorted(set(issue_numbers))
    if not unique_numbers:
        return {}

    command = ["git", "ls-remote", "--heads", remote, "refs/heads/agent-claims/issue-*"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
        return {
            number: _claim_status_payload(
                number,
                remote=remote,
                ok=False,
                claimed=None,
                sha=None,
                error=error,
                command=command,
            )
            for number in unique_numbers
        }

    claimed_refs: dict[int, str] = {}
    ref_prefix = "refs/heads/agent-claims/issue-"
    for line in (result.stdout or "").strip().splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].startswith(ref_prefix):
            continue
        issue_text = parts[1].removeprefix(ref_prefix)
        if issue_text.isdigit():
            claimed_refs[int(issue_text)] = parts[0]

    return {
        number: _claim_status_payload(
            number,
            remote=remote,
            ok=True,
            claimed=number in claimed_refs,
            sha=claimed_refs.get(number),
            command=command,
        )
        for number in unique_numbers
    }


def _is_blocked_external_issue(labels: list[str]) -> bool:
    """Return whether labels describe an issue blocked on external assets or input."""
    label_set = set(labels)
    if BLOCKED_EXTERNAL_INPUT_LABEL in label_set:
        return True
    return EXTERNAL_RESOURCE_LABEL in label_set and bool(EXTERNAL_BLOCKER_LABELS & label_set)


def _body_excerpt(body: Any, *, limit: int) -> tuple[str, bool]:
    """Return a whitespace-normalized body excerpt and truncation flag."""
    text = " ".join(str(body or "").split())
    return text[:limit], len(text) > limit


def _recommended_context_pack(number: int, labels: list[str], title: str) -> str:
    """Return a conservative context-pack hint for a worker prompt."""
    label_text = " ".join(labels).lower()
    title_text = title.lower()
    if "docs" in label_text or "doc" in title_text:
        return "docs/context/INDEX.md"
    if "benchmark" in label_text or "benchmark" in title_text:
        return "docs/benchmark_camera_ready.md"
    if "workflow" in label_text or "workflow" in title_text:
        return "docs/context/goal_driven_agent_loops_2026-05-13.md"
    return f"docs/context/issue_{number}* if present, otherwise docs/context/INDEX.md"


def fetch_issue(number: int, *, repo: str, body_limit: int, remote: str) -> dict[str, Any]:
    """Fetch one issue and return a compact orchestration snapshot."""
    result = _gh(
        [
            "issue",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,body,state,labels,url,assignees",
        ]
    )
    if result.returncode != 0:
        return {
            "number": number,
            "status": "error",
            "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
        }
    try:
        issue = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"number": number, "status": "error", "error": f"invalid gh JSON: {exc}"}
    labels = _labels(issue)
    assignees = _assignees(issue)
    claim = status_issue(number, remote=remote)
    classification, reason = _issue_classification(
        assignees=assignees,
        claim=claim,
        labels=labels,
    )
    excerpt, truncated = _body_excerpt(issue.get("body"), limit=body_limit)
    return {
        "number": issue.get("number", number),
        "status": "ok",
        "title": issue.get("title", ""),
        "state": issue.get("state", ""),
        "url": issue.get("url", ""),
        "labels": labels,
        "assignees": assignees,
        "body_excerpt": excerpt,
        "body_truncated": truncated,
        "claim": _claim_payload(claim),
        "classification": classification,
        "reason": reason,
        "linked_prs": [],
        "recommended_context_pack": _recommended_context_pack(
            int(issue.get("number", number)), labels, str(issue.get("title", ""))
        ),
    }


def _snapshot_from_issue_list(
    issue: dict[str, Any], *, remote: str, claim_statuses: dict[int, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build an issue snapshot using preloaded issue fields."""
    try:
        number = int(issue.get("number"))
    except (TypeError, ValueError):
        return {"status": "error", "error": "invalid issue number in gh list payload"}

    labels = _labels(issue)
    assignees = _assignees(issue)
    claim = (
        claim_statuses.get(number)
        if claim_statuses is not None
        else status_issue(number, remote=remote)
    )
    if not isinstance(claim, dict):
        claim = _claim_status_payload(
            number,
            remote=remote,
            ok=False,
            claimed=None,
            sha=None,
            error="claim status unavailable",
        )
    classification, reason = _issue_classification(
        assignees=assignees,
        claim=claim,
        labels=labels,
    )
    return {
        "number": number,
        "status": "ok",
        "title": issue.get("title", ""),
        "state": issue.get("state", ""),
        "url": issue.get("url", ""),
        "labels": labels,
        "assignees": assignees,
        "claim": _claim_payload(claim),
        "body_excerpt": "",
        "body_truncated": False,
        "classification": classification,
        "reason": reason,
        "linked_prs": [],
    }


def snapshot_claimable_issues(
    *,
    repo: str,
    remote: str,
    body_limit: int,
    limit: int,
    include_blocked_external: bool = False,
) -> dict[str, Any]:
    """Return a compact claimable/open issue snapshot."""
    result = _gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            "number,title,state,labels,url,assignees",
        ]
    )
    if result.returncode != 0:
        return {
            "schema": "issue_batch_snapshot.v1",
            "repo": repo,
            "body_excerpt_chars": body_limit,
            "mode": "claimable",
            "issues": [
                {
                    "status": "error",
                    "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
                }
            ],
        }
    try:
        listed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "schema": "issue_batch_snapshot.v1",
            "repo": repo,
            "body_excerpt_chars": body_limit,
            "mode": "claimable",
            "issues": [
                {
                    "status": "error",
                    "error": f"invalid gh JSON: {exc}",
                }
            ],
        }
    if not isinstance(listed, list):
        return {
            "schema": "issue_batch_snapshot.v1",
            "repo": repo,
            "body_excerpt_chars": body_limit,
            "mode": "claimable",
            "issues": [
                {
                    "status": "error",
                    "error": "expected gh issue list JSON array",
                }
            ],
        }

    issue_numbers = [
        int(issue["number"])
        for issue in listed
        if isinstance(issue, dict) and str(issue.get("number", "")).isdigit()
    ]
    claim_statuses = _batch_claim_statuses(issue_numbers, remote=remote)
    snapshots = [
        _snapshot_from_issue_list(issue, remote=remote, claim_statuses=claim_statuses)
        for issue in listed
    ]
    issues = [
        issue
        for issue in snapshots
        if include_blocked_external or issue.get("classification") != "blocked_external"
    ]
    if body_limit <= 0:
        body_limit = BODY_EXCERPT_CHARS
    truncated = is_likely_truncated(len(listed), limit=limit)
    return {
        "schema": "issue_batch_snapshot.v1",
        "repo": repo,
        "body_excerpt_chars": body_limit,
        "mode": "claimable",
        "truncated": truncated,
        "truncation_note": (
            f"gh issue list may be capped: got {len(listed)} rows at --limit {limit}; "
            "raise --limit or paginate"
            if truncated
            else ""
        ),
        "include_blocked_external": include_blocked_external,
        "excluded_counts": {
            "blocked_external": sum(
                1 for issue in snapshots if issue.get("classification") == "blocked_external"
            ),
        },
        "issues": issues,
    }


def _next_monthly_review_date(now: datetime | None = None) -> str:
    """Return a stable monthly review date, using the first UTC day of next month."""
    current = now or datetime.now(UTC)
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    return f"{year:04d}-{month:02d}-01"


def _blocked_external_row(issue: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Return one human-review row for a blocked external-asset issue."""
    labels = _labels(issue)
    label_set = set(labels)
    recommendations: list[str] = []
    if BLOCKED_EXTERNAL_INPUT_LABEL not in label_set:
        recommendations.append(f"add `{BLOCKED_EXTERNAL_INPUT_LABEL}`")
    if "state:ready" in label_set:
        recommendations.append("remove `state:ready`")
    return {
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "url": issue.get("url", ""),
        "labels": labels,
        "owner_type": "external data",
        "human_action": (
            "Stage or document the required external data/asset/license before agent execution."
        ),
        "monthly_review_date": _next_monthly_review_date(now),
        "label_recommendation": "; ".join(recommendations) if recommendations else "none",
    }


def _blocked_external_markdown(rows: list[dict[str, Any]]) -> str:
    """Return a compact Markdown report for human review."""
    lines = [
        "# Blocked External Assets Report",
        "",
        "| Issue | Owner | Human action | Monthly review | Label recommendation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        issue = f"#{row['number']} {row['title']}".replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    issue,
                    row["owner_type"],
                    row["human_action"],
                    row["monthly_review_date"],
                    row["label_recommendation"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: Any) -> str:
    """Return a table-safe compact Markdown cell."""
    value_text = "" if value is None else str(value)
    return value_text.replace("|", "\\|").replace("\n", " ")


def snapshot_blocked_external_issues(
    *, repo: str, report_path: str = "", limit: int, now: datetime | None = None
) -> dict[str, Any]:
    """Return a compact blocked external-assets report."""
    result = _gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--label",
            EXTERNAL_RESOURCE_LABEL,
            "--limit",
            str(limit),
            "--json",
            "number,title,state,labels,url,assignees",
        ]
    )
    if result.returncode != 0:
        rows: list[dict[str, Any]] = []
        errors = [
            {
                "status": "error",
                "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
            }
        ]
    else:
        errors = []
        try:
            listed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            listed = []
            errors = [{"status": "error", "error": f"invalid gh JSON: {exc}"}]
        if not errors and not isinstance(listed, list):
            listed = []
            errors = [
                {
                    "status": "error",
                    "error": "expected gh issue list JSON array",
                }
            ]
        rows = [
            _blocked_external_row(issue, now=now)
            for issue in listed
            if isinstance(issue, dict) and _is_blocked_external_issue(_labels(issue))
        ]
    markdown = _blocked_external_markdown(rows)
    if report_path:
        path = pathlib.Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {
        "schema": "blocked_external_assets_report.v1",
        "repo": repo,
        "mode": "blocked_external_report",
        "recommended_state_label": BLOCKED_EXTERNAL_INPUT_LABEL,
        "rows": rows,
        "row_count": len(rows),
        "report_path": report_path,
        "markdown": markdown,
        "errors": errors,
    }


def _portfolio_owner_type(labels: list[str], classification: str) -> str:
    """Return the likely next owner type for an active issue portfolio row."""
    label_set = set(labels)
    if classification == "blocked_external_asset":
        return "external data"
    if classification in {"needs_human_decision", "stale_synthesis"}:
        return "maintainer"
    if (
        "slurm" in label_set
        or "resource:slurm" in label_set
        or "training" in label_set
        or "state:running" in label_set
    ):
        return "Slurm"
    return "agent"


def _portfolio_classification(
    *, labels: list[str], title: str, assignees: list[str], claim: dict[str, Any]
) -> tuple[str, str]:
    """Classify one open issue for compact active-portfolio routing."""
    label_set = set(labels)
    title_text = title.lower()
    if _is_blocked_external_issue(labels):
        return "blocked_external_asset", "external asset, data, license, or staging input required"
    if (
        "blocked" in label_set
        or "decision-required" in label_set
        or "state:blocked" in label_set
        or "state:hold" in label_set
    ):
        return "needs_human_decision", "maintainer decision label blocks autonomous execution"
    if "evidence:analysis-only" in label_set or "diagnostic" in title_text:
        return (
            "diagnostic_only",
            "analysis or diagnostic evidence only; do not present as benchmark proof",
        )
    if "type:synthesis" in label_set and "state:ready" not in label_set:
        return "stale_synthesis", "synthesis issue lacks ready-state routing"
    if "priority: high" in label_set and (
        "benchmark" in label_set or "research" in label_set or "epic" in label_set
    ):
        return "paper_critical", "high-priority benchmark or research surface"
    if claim.get("ok") is False:
        return "needs_human_decision", "unable to read claim state; skip autonomous claim"
    if assignees or claim.get("claimed"):
        return "needs_human_decision", "assigned or already claimed; skip autonomous claim"
    return "executable_now", "unassigned, unclaimed, and not blocked by labels"


def _portfolio_label_recommendation(labels: list[str], classification: str) -> str:
    """Return label-only recommendations for one portfolio row."""
    label_set = set(labels)
    recommendation_rules = {
        "blocked_external_asset": lambda: [
            (
                BLOCKED_EXTERNAL_INPUT_LABEL not in label_set,
                f"add `{BLOCKED_EXTERNAL_INPUT_LABEL}`",
            ),
            ("state:ready" in label_set, "remove `state:ready`"),
        ],
        "needs_human_decision": lambda: [
            ("decision-required" not in label_set, "add `decision-required`"),
            ("state:ready" in label_set, "remove `state:ready`"),
        ],
        "diagnostic_only": lambda: [
            ("evidence:analysis-only" not in label_set, "add `evidence:analysis-only`")
        ],
        "stale_synthesis": lambda: [
            ("decision-required" not in label_set, "add `decision-required`")
        ],
        "paper_critical": lambda: [("paper-critical" not in label_set, "add `paper-critical`")],
        "executable_now": lambda: [("state:ready" not in label_set, "add `state:ready`")],
    }
    rules = recommendation_rules.get(classification, lambda: [])()
    recommendations = [
        recommendation for should_recommend, recommendation in rules if should_recommend
    ]
    return "; ".join(recommendations) if recommendations else "none"


def _portfolio_next_action(classification: str) -> str:
    """Return one compact next action for a portfolio row."""
    actions = {
        "blocked_external_asset": "Park until the required asset, data, license, or staging note exists.",
        "diagnostic_only": "Keep as diagnostic evidence unless a follow-up benchmark proof issue is opened.",
        "executable_now": "Agent may claim and execute the issue contract.",
        "needs_human_decision": "Maintainer should decide, relabel, or split before agent execution.",
        "paper_critical": "Prioritize for release evidence review and claim-boundary checks.",
        "stale_synthesis": "Refresh, supersede, or close the synthesis before new implementation work.",
    }
    return actions[classification]


def _active_portfolio_row(
    issue: dict[str, Any], *, remote: str, claim_statuses: dict[int, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return one active-portfolio row for an open issue."""
    number = int(issue.get("number"))
    labels = _labels(issue)
    assignees = _assignees(issue)
    claim_value = (
        claim_statuses.get(number)
        if claim_statuses is not None
        else status_issue(number, remote=remote)
    )
    claim = (
        claim_value
        if isinstance(claim_value, dict)
        else _claim_status_payload(
            number,
            remote=remote,
            ok=False,
            claimed=None,
            sha=None,
            error="claim status unavailable",
        )
    )
    title = "" if issue.get("title") is None else issue.get("title", "")
    url = "" if issue.get("url") is None else issue.get("url", "")
    classification, reason = _portfolio_classification(
        labels=labels,
        title=str(title),
        assignees=assignees,
        claim=claim,
    )
    return {
        "number": number,
        "title": title,
        "url": url,
        "labels": labels,
        "classification": classification,
        "reason": reason,
        "owner_type": _portfolio_owner_type(labels, classification),
        "next_action": _portfolio_next_action(classification),
        "label_recommendation": _portfolio_label_recommendation(labels, classification),
    }


def _active_portfolio_markdown(rows: list[dict[str, Any]]) -> str:
    """Return a compact Markdown portfolio table for maintainer review."""
    lines = [
        "# Active Issue Portfolio",
        "",
        "| Issue | Classification | Owner | Next action | Label recommendation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        issue = f"#{row['number']} {row['title']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(issue),
                    _markdown_cell(row["classification"]),
                    _markdown_cell(row["owner_type"]),
                    _markdown_cell(row["next_action"]),
                    _markdown_cell(row["label_recommendation"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def snapshot_active_issue_portfolio(
    *, repo: str, remote: str, report_path: str = "", limit: int
) -> dict[str, Any]:
    """Return a compact active issue portfolio for routing and demotion review."""
    result = _gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            "number,title,state,labels,url,assignees",
        ]
    )
    if result.returncode != 0:
        rows: list[dict[str, Any]] = []
        errors = [
            {
                "status": "error",
                "error": result.stderr.strip() or f"gh returned exit code {result.returncode}",
            }
        ]
    else:
        errors = []
        try:
            listed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            listed = []
            errors = [{"status": "error", "error": f"invalid gh JSON: {exc}"}]
        if not errors and not isinstance(listed, list):
            listed = []
            errors = [
                {
                    "status": "error",
                    "error": "expected gh issue list JSON array",
                }
            ]
        issue_numbers = [
            int(issue["number"])
            for issue in listed
            if isinstance(issue, dict) and str(issue.get("number", "")).isdigit()
        ]
        claim_statuses = _batch_claim_statuses(issue_numbers, remote=remote)
        rows = [
            _active_portfolio_row(issue, remote=remote, claim_statuses=claim_statuses)
            for issue in listed
            if isinstance(issue, dict) and issue.get("number") is not None
        ]
    counts = dict.fromkeys(sorted(PORTFOLIO_CLASSIFICATIONS), 0)
    for row in rows:
        classification = str(row.get("classification", ""))
        if classification in counts:
            counts[classification] += 1
    markdown = _active_portfolio_markdown(rows)
    if report_path:
        path = pathlib.Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {
        "schema": "active_issue_portfolio.v1",
        "repo": repo,
        "mode": "active_portfolio",
        "rows": rows,
        "row_count": len(rows),
        "classification_counts": counts,
        "report_path": report_path,
        "markdown": markdown,
        "errors": errors,
    }


def _context_capsule(issue: dict[str, Any]) -> dict[str, Any]:
    """Return a compact worker-seeding capsule for one issue snapshot."""
    return {
        "schema": "issue_context_capsule.v1",
        "issue": {
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "url": issue.get("url", ""),
            "labels": issue.get("labels", []),
            "body_excerpt": issue.get("body_excerpt", ""),
        },
        "claim": issue.get("claim", {}),
        "files_to_read": [issue.get("recommended_context_pack", "docs/context/INDEX.md")],
        "tests_to_run": [],
        "docs_to_update": [],
        "known_risks": [],
        "worker_prompt_seed": (
            "Use this capsule as the first context source. Avoid broad repository search until "
            "these files have been inspected and summarized."
        ),
    }


def _write_capsules(issues: list[dict[str, Any]], capsule_dir: pathlib.Path) -> None:
    """Write one optional context capsule per successfully fetched issue."""
    capsule_dir.mkdir(parents=True, exist_ok=True)
    for issue in issues:
        if issue.get("status") != "ok":
            continue
        path = capsule_dir / f"issue_{issue['number']}_context_capsule.json"
        path.write_text(
            json.dumps(_context_capsule(issue), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        issue["context_capsule_path"] = str(path)


def snapshot_issues(
    numbers: list[int], *, repo: str, body_limit: int, remote: str, capsule_dir: str = ""
) -> dict[str, Any]:
    """Return a compact issue-batch snapshot."""
    issues = [
        fetch_issue(number, repo=repo, body_limit=body_limit, remote=remote) for number in numbers
    ]
    if capsule_dir:
        _write_capsules(issues, pathlib.Path(capsule_dir))
    return {
        "schema": "issue_batch_snapshot.v1",
        "repo": repo,
        "body_excerpt_chars": body_limit,
        "issues": issues,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "issues", nargs="*", type=int, help="Issue numbers; two values form a range."
    )
    parser.add_argument(
        "--claimable",
        action="store_true",
        help="Discover bounded open claimable issues without explicit issue numbers.",
    )
    parser.add_argument(
        "--include-blocked-external",
        action="store_true",
        help="Include blocked external-input issues in --claimable output.",
    )
    parser.add_argument(
        "--blocked-external-report",
        action="store_true",
        help="Generate a compact blocked external-assets report instead of claim routing.",
    )
    parser.add_argument(
        "--active-portfolio",
        action="store_true",
        help="Generate a compact active issue portfolio with label recommendations.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional Markdown path for --blocked-external-report.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--body-chars", type=int, default=BODY_EXCERPT_CHARS)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CLAIMABLE_LIMIT,
        help="Limit for --claimable discovery mode.",
    )
    parser.add_argument(
        "--capsule-dir",
        default="",
        help="Optional directory for per-issue context capsule JSON files.",
    )
    parser.add_argument(
        "--no-expand-range",
        action="store_true",
        help="Treat two issue numbers as exactly two issues instead of an inclusive range.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> int:
    """Return nonzero after printing a CLI contract error."""
    if args.claimable and args.issues:
        print(
            "--claimable cannot be combined with explicit issue numbers",
            file=sys.stderr,
        )
        return 1
    if args.include_blocked_external and not args.claimable:
        print(
            "--include-blocked-external requires --claimable",
            file=sys.stderr,
        )
        return 1
    if args.blocked_external_report and (args.claimable or args.issues):
        print(
            "--blocked-external-report cannot be combined with --claimable or issue numbers",
            file=sys.stderr,
        )
        return 1
    if args.active_portfolio and (args.claimable or args.blocked_external_report or args.issues):
        print(
            "--active-portfolio cannot be combined with --claimable, "
            "--blocked-external-report, or issue numbers",
            file=sys.stderr,
        )
        return 1
    return 0


def _build_payload(args: argparse.Namespace, numbers: list[int]) -> dict[str, Any]:
    """Build the requested CLI payload after argument validation."""
    if args.active_portfolio:
        return snapshot_active_issue_portfolio(
            repo=args.repo,
            remote=args.remote,
            report_path=args.report_path,
            limit=max(args.limit, 1),
        )
    if args.blocked_external_report:
        return snapshot_blocked_external_issues(
            repo=args.repo,
            report_path=args.report_path,
            limit=max(args.limit, 1),
        )
    if args.claimable:
        return snapshot_claimable_issues(
            repo=args.repo,
            remote=args.remote,
            body_limit=max(args.body_chars, 0),
            limit=max(args.limit, 1),
            include_blocked_external=args.include_blocked_external,
        )
    return snapshot_issues(
        numbers,
        repo=args.repo,
        body_limit=max(args.body_chars, 0),
        remote=args.remote,
        capsule_dir=args.capsule_dir,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    validation_error = _validate_args(args)
    if validation_error:
        return validation_error
    numbers = expand_issue_numbers(args.issues, expand_range=not args.no_expand_range)
    try:
        if not (
            args.active_portfolio or args.blocked_external_report or args.claimable or args.issues
        ):
            print(
                "at least one issue number is required unless --claimable, "
                "--blocked-external-report, or --active-portfolio is used",
                file=sys.stderr,
            )
            return 1
        payload = _build_payload(args, numbers)
    except FileNotFoundError:
        print("gh or git command not found", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as exc:
        print(f"snapshot command timed out: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload))
    if "issues" in payload:
        return 1 if any(issue.get("status") == "error" for issue in payload["issues"]) else 0
    return 1 if payload.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
