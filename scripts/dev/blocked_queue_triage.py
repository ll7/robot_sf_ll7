#!/usr/bin/env python3
"""Produce and optionally publish a fail-closed triage packet for blocked issues.

The tool keeps ``state:blocked`` issues open and makes the next unblock event
explicit.  It is report-only by default.  ``--apply-comments`` updates one
idempotent, marker-bound issue comment per blocked issue; it never changes issue
labels, state, project fields, or closure status.

The classification is intentionally conservative.  Strong labels and explicit
issue wording produce a higher-confidence class; ambiguous rows remain
``external_fact`` with a human-review note instead of being silently promoted
to an implementation queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_LABEL = "state:blocked"
SCHEMA = "blocked_queue_triage.v1"
COMMENT_MARKER = "blocked-queue-triage.v1"
COMMENTS_PAGE_SIZE = 100
MAX_BODY_EXCERPT = 180
MAX_EVIDENCE = 3

BLOCKER_CLASSES = (
    "external_fact",
    "compute",
    "licence",
    "maintainer",
    "dependency",
    "upstream_issue",
)
MACHINE_TESTABLE_CLASSES = frozenset({"dependency", "upstream_issue"})
BOT_LOGINS = frozenset(
    {
        "github-actions",
        "github-actions[bot]",
        "coderabbitai",
        "dependabot[bot]",
        "renovate[bot]",
    }
)
ISSUE_REF_RE = re.compile(r"(?<![\w-])#(?P<number>[1-9]\d*)\b")
BLOCKED_BY_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:blocked\s+by|depends\s+on|prerequisite|upstream)\b"
)

Runner = Callable[[list[str], str | None], subprocess.CompletedProcess[str]]


class TriageError(RuntimeError):
    """Raised when the source inventory or a write/readback is incomplete."""


def _run_gh(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` with bounded output capture."""

    try:
        return subprocess.run(
            ["gh", *args],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TriageError("gh CLI is not installed or unavailable on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise TriageError(f"gh command timed out after {exc.timeout}s") from exc


def _json_result(result: subprocess.CompletedProcess[str], *, operation: str) -> Any:
    """Parse one successful ``gh api`` response or fail closed."""

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise TriageError(f"{operation} failed with exit code {result.returncode}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TriageError(f"{operation} returned invalid JSON: {exc}") from exc


def _flatten_pages(payload: Any, *, operation: str) -> list[dict[str, Any]]:
    """Flatten ``gh api --paginate --slurp`` output and validate object rows."""

    if not isinstance(payload, list):
        raise TriageError(f"{operation} returned a non-list pagination payload")
    rows: list[Any]
    if payload and all(isinstance(page, list) for page in payload):
        rows = [item for page in payload for item in page]
    else:
        rows = payload
    if any(not isinstance(row, dict) for row in rows):
        raise TriageError(f"{operation} returned a malformed row")
    return [row for row in rows if isinstance(row, dict)]


def _api_json(
    path: str,
    *,
    runner: Runner,
    operation: str,
    method: str | None = None,
    payload: Mapping[str, Any] | None = None,
    paginate: bool = False,
    slurp: bool = False,
) -> Any:
    """Call a GitHub REST endpoint through ``gh api``."""

    args = ["api"]
    if paginate:
        args.append("--paginate")
    if slurp:
        args.append("--slurp")
    if method:
        args.extend(["--method", method])
    args.append(path)
    input_text = None
    if payload is not None:
        args.extend(["--input", "-"])
        input_text = json.dumps(dict(payload), sort_keys=True)
    return _json_result(runner(args, input_text), operation=operation)


def _labels(row: Mapping[str, Any]) -> set[str]:
    """Return normalized label names from a REST issue row."""

    raw = row.get("labels", [])
    if not isinstance(raw, list):
        raise TriageError(f"issue #{row.get('number', '?')} has malformed labels")
    labels: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TriageError(f"issue #{row.get('number', '?')} has malformed label entry")
        labels.add(item["name"])
    return labels


def _is_triage_comment(comment: Mapping[str, Any]) -> bool:
    """Return whether a comment was generated by this triage tool."""

    return COMMENT_MARKER in str(comment.get("body") or "")


def _flatten_text(row: Mapping[str, Any], comments: Iterable[Mapping[str, Any]]) -> str:
    """Join title, body, and comment text for conservative signal matching."""

    parts = [str(row.get("title") or ""), str(row.get("body") or "")]
    for comment in comments:
        if _is_triage_comment(comment):
            continue
        parts.append(str(comment.get("body") or ""))
    return "\n".join(parts)


def _snippet(text: str, terms: Iterable[str]) -> str | None:
    """Return a bounded line containing one of ``terms``."""

    for line in text.splitlines():
        compact = " ".join(line.split())
        lowered = compact.casefold()
        if compact and any(term.casefold() in lowered for term in terms):
            return compact[:MAX_BODY_EXCERPT]
    return None


def _referenced_blockers(text: str) -> tuple[int, ...]:
    """Return issue references from explicit dependency/upstream sections."""

    if not BLOCKED_BY_RE.search(text):
        return ()
    seen: set[int] = set()
    references: list[int] = []
    for match in ISSUE_REF_RE.finditer(text):
        number = int(match.group("number"))
        if number not in seen:
            references.append(number)
            seen.add(number)
    return tuple(references)


def _classify(  # noqa: C901
    row: Mapping[str, Any], comments: list[Mapping[str, Any]]
) -> tuple[str, str, list[str], tuple[int, ...]]:
    """Classify a blocked issue from explicit labels and current text."""

    labels = _labels(row)
    text = _flatten_text(row, comments)
    lowered = text.casefold()
    evidence: list[str] = []

    if labels & {"resource:slurm", "resource:gpu", "resource:compute"} or any(
        term in lowered for term in ("slurm", "sbatch", "compute authorization", "gpu allocation")
    ):
        matched = sorted(labels & {"resource:slurm", "resource:gpu", "resource:compute"})
        evidence.extend(f"label: {label}" for label in matched)
        if not matched:
            evidence.append("text: compute or scheduler gate")
        return "compute", "high", evidence[:MAX_EVIDENCE], ()

    if labels & {"resource:license", "resource:licence", "rights", "licence"} or any(
        term in lowered for term in ("license", "licence", "licensing", "copyright", "rights")
    ):
        matched = sorted(labels & {"resource:license", "resource:licence", "rights", "licence"})
        evidence.extend(f"label: {label}" for label in matched)
        if not matched:
            evidence.append("text: license or rights gate")
        return "licence", "high", evidence[:MAX_EVIDENCE], ()

    if "decision-required" in labels or any(
        term in lowered
        for term in (
            "maintainer approval",
            "maintainer decision",
            "owner decision",
            "requires approval",
            "decision is required",
        )
    ):
        if "decision-required" in labels:
            evidence.append("label: decision-required")
        else:
            evidence.append("text: maintainer decision or approval")
        return "maintainer", "high", evidence[:MAX_EVIDENCE], ()

    references = _referenced_blockers(text)
    if references and any(term in lowered for term in ("upstream", "blocked by")):
        evidence.append(f"text: explicit upstream blocker references {list(references)}")
        return "upstream_issue", "high", evidence[:MAX_EVIDENCE], references

    if references or any(term in lowered for term in ("depends on", "prerequisite", "source_pr")):
        evidence.append("text: dependency or prerequisite gate")
        return "dependency", "medium", evidence[:MAX_EVIDENCE], references

    if labels & {"resource:external-data", "state:blocked-external-input"} or any(
        term in lowered
        for term in (
            "external data",
            "external asset",
            "dataset",
            "checkpoint",
            "model weights",
            "missing artifact",
        )
    ):
        matched = sorted(labels & {"resource:external-data", "state:blocked-external-input"})
        evidence.extend(f"label: {label}" for label in matched)
        if not matched:
            evidence.append("text: external input or artifact gate")
        return "external_fact", "high", evidence[:MAX_EVIDENCE], ()

    snippet = _snippet(text, ("blocked", "pending", "unavailable", "missing"))
    evidence.append(snippet or "no specific blocker signal; human confirmation required")
    return "external_fact", "low", evidence[:MAX_EVIDENCE], ()


def _condition_for(
    blocker_class: str, references: tuple[int, ...], *, repo: str
) -> tuple[str, str, str]:
    """Return condition text, watcher, and machine/human mode."""

    watcher = f"scripts/dev/blocked_queue_triage.py --repo {repo} --report-only"
    if blocker_class == "compute":
        return (
            "A fresh canonical queue/readiness record and explicit compute authorization are both present.",
            watcher,
            "human_observable",
        )
    if blocker_class == "licence":
        return (
            "The required licence, rights, or permission is recorded in the provenance path and approved.",
            watcher,
            "human_observable",
        )
    if blocker_class == "maintainer":
        return (
            "A maintainer or domain owner records the required decision or approval in the issue thread.",
            watcher,
            "human_observable",
        )
    if blocker_class in {"dependency", "upstream_issue"}:
        suffix = (
            f" (referenced issues: {', '.join(f'#{n}' for n in references)})" if references else ""
        )
        return (
            "The named prerequisite reaches the required terminal state and a fresh issue/PR readback verifies it"
            + suffix
            + ".",
            "scripts/tools/check_blocker_staleness.py plus this triage report",
            "machine_testable",
        )
    return (
        "The named external fact, dataset, model, or artifact is available with a durable path, checksum, and provenance record.",
        watcher,
        "human_observable",
    )


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a GitHub ISO timestamp into UTC."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_bot_comment(comment: Mapping[str, Any]) -> bool:
    """Return whether a comment is an automated status/comment source."""

    user = comment.get("user")
    login = user.get("login") if isinstance(user, dict) else user
    normalized = str(login or "").casefold()
    return normalized in BOT_LOGINS or normalized.endswith("[bot]")


def _last_progress(row: Mapping[str, Any], comments: list[Mapping[str, Any]]) -> tuple[str, str]:
    """Select the latest human-authored comment, falling back to issue activity."""

    human_comments = [
        comment
        for comment in comments
        if not _is_bot_comment(comment) and not _is_triage_comment(comment)
    ]
    dated = [
        (timestamp, comment)
        for comment in human_comments
        if (timestamp := _parse_timestamp(comment.get("created_at"))) is not None
    ]
    if dated:
        timestamp, _ = max(dated, key=lambda item: item[0])
        return timestamp.isoformat().replace("+00:00", "Z"), "latest_human_comment"
    if comments:
        issue_timestamp = _parse_timestamp(row.get("created_at"))
        fallback_source = "issue_creation_fallback"
    else:
        issue_timestamp = _parse_timestamp(row.get("updated_at")) or _parse_timestamp(
            row.get("created_at")
        )
        fallback_source = "issue_activity_fallback"
    if issue_timestamp is not None:
        return issue_timestamp.isoformat().replace("+00:00", "Z"), fallback_source
    return "", "unavailable"


def _triage_row(
    row: Mapping[str, Any],
    comments: list[Mapping[str, Any]],
    *,
    repo: str,
    next_check_at: str,
) -> dict[str, Any]:
    """Build one normalized issue triage row."""

    raw_number = row.get("number")
    if type(raw_number) is not int or raw_number < 1:
        raise TriageError("blocked issue has an invalid number")
    title = row.get("title")
    if not isinstance(title, str) or not title:
        raise TriageError(f"issue #{raw_number} has an invalid title")
    blocker_class, confidence, evidence, references = _classify(row, comments)
    condition, watcher, condition_mode = _condition_for(blocker_class, references, repo=repo)
    last_progress, progress_source = _last_progress(row, comments)
    return {
        "issue": raw_number,
        "title": title,
        "url": str(row.get("html_url") or row.get("url") or ""),
        "blocker_class": blocker_class,
        "classification_confidence": confidence,
        "unblock_condition": condition,
        "condition_mode": condition_mode,
        "watcher": watcher,
        "next_check_at": next_check_at,
        "last_meaningful_progress_at": last_progress,
        "last_progress_source": progress_source,
        "evidence": evidence,
        "referenced_issues": list(references),
        "closure_recommendation": "keep_open",
    }


def _age_bucket(timestamp: str, *, as_of: datetime) -> str:
    """Classify progress age into stable report buckets."""

    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return "unknown"
    age = max(as_of - parsed, timedelta(0))
    if age < timedelta(days=7):
        return "under_7_days"
    if age < timedelta(days=30):
        return "7_to_29_days"
    if age < timedelta(days=90):
        return "30_to_89_days"
    return "90_days_or_more"


def _digest(row: Mapping[str, Any]) -> str:
    """Return a stable digest for one comment payload."""

    encoded = json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_comment(row: Mapping[str, Any], *, generated_at: str) -> str:
    """Render the idempotent issue comment for one triage row."""

    issue = int(row["issue"])
    digest = _digest(row)
    evidence = "\n".join(f"  - {item}" for item in row.get("evidence", [])) or "  - unavailable"
    references = ", ".join(f"#{number}" for number in row.get("referenced_issues", [])) or "none"
    return (
        "\n".join(
            [
                f"<!-- {COMMENT_MARKER} issue={issue} digest={digest} -->",
                "## Blocked queue triage",
                "",
                f"Generated at: `{generated_at}`  ",
                f"Blocker class: `{row['blocker_class']}`  ",
                f"Classification confidence: `{row['classification_confidence']}`  ",
                f"Condition mode: `{row['condition_mode']}`",
                "",
                f"- Unblock condition: {row['unblock_condition']}",
                f"- Watcher: `{row['watcher']}`",
                f"- Next check: `{row['next_check_at']}`",
                f"- Last meaningful progress: `{row['last_meaningful_progress_at'] or 'unavailable'}` ({row['last_progress_source']})",
                f"- Referenced issues: {references}",
                "- Closure recommendation: **keep open** under the maintainer TRIAGE-WITH-UNBLOCK-CONDITIONS ruling.",
                "",
                "Evidence used:",
                evidence,
                "",
                "This bookkeeping comment does not authorize implementation, compute, evidence admission, publication, or closure.",
            ]
        )
        + "\n"
    )


def _existing_comment(comments: list[Mapping[str, Any]], issue: int) -> Mapping[str, Any] | None:
    """Return the newest existing triage comment for ``issue``."""

    marker = f"<!-- {COMMENT_MARKER} issue={issue} "
    matches = [comment for comment in comments if marker in str(comment.get("body") or "")]
    if not matches:
        return None
    return max(matches, key=lambda comment: str(comment.get("created_at") or ""))


def apply_comments(
    rows: list[dict[str, Any]],
    comments_by_issue: Mapping[int, list[Mapping[str, Any]]],
    *,
    repo: str,
    generated_at: str,
    max_mutations: int,
    runner: Runner,
) -> list[dict[str, Any]]:
    """Create or update triage comments with verified idempotent writes."""

    operations: list[dict[str, Any]] = []
    for row in rows:
        issue = int(row["issue"])
        body = render_comment(row, generated_at=generated_at)
        existing = _existing_comment(list(comments_by_issue.get(issue, [])), issue)
        existing_body = str(existing.get("body") or "") if existing else ""
        if existing_body == body:
            operations.append({"issue": issue, "action": "unchanged"})
            continue
        if (
            len([item for item in operations if item["action"] in {"created", "updated"}])
            >= max_mutations
        ):
            raise TriageError(f"mutation budget exhausted at {max_mutations} writes")
        if existing is None:
            path = f"repos/{repo}/issues/{issue}/comments"
            result = _api_json(
                path,
                runner=runner,
                operation=f"create triage comment for issue #{issue}",
                method="POST",
                payload={"body": body},
            )
            action = "created"
        else:
            comment_id = existing.get("id")
            if type(comment_id) is not int or comment_id < 1:
                raise TriageError(f"existing triage comment for issue #{issue} has invalid id")
            path = f"repos/{repo}/issues/comments/{comment_id}"
            result = _api_json(
                path,
                runner=runner,
                operation=f"update triage comment for issue #{issue}",
                method="PATCH",
                payload={"body": body},
            )
            action = "updated"
        if not isinstance(result, dict) or result.get("body") != body:
            raise TriageError(f"triage comment readback failed for issue #{issue}")
        operations.append({"issue": issue, "action": action, "comment_id": result.get("id")})
    return operations


def _fetch_blocked_issues(
    *, repo: str, label: str, limit: int, runner: Runner
) -> list[dict[str, Any]]:
    """Fetch the complete open blocked issue inventory, excluding pull requests."""

    path = f"repos/{repo}/issues?state=open&labels={label.replace(':', '%3A')}&per_page={COMMENTS_PAGE_SIZE}"
    payload = _api_json(
        path,
        runner=runner,
        operation=f"list open issues labeled {label}",
        paginate=True,
        slurp=True,
    )
    rows = _flatten_pages(payload, operation="open blocked issue inventory")
    issues = [row for row in rows if "pull_request" not in row]
    if len(issues) > limit:
        raise TriageError(
            f"blocked issue inventory contains {len(issues)} rows, above limit {limit}; increase --limit"
        )
    return issues


def _fetch_comments(issue: int, *, repo: str, runner: Runner) -> list[dict[str, Any]]:
    """Fetch all comments for one issue without silently truncating pages."""

    path = f"repos/{repo}/issues/{issue}/comments?per_page={COMMENTS_PAGE_SIZE}"
    payload = _api_json(
        path,
        runner=runner,
        operation=f"comments for issue #{issue}",
        paginate=True,
        slurp=True,
    )
    return _flatten_pages(payload, operation=f"comments for issue #{issue}")


def build_report(
    issues: list[dict[str, Any]],
    comments_by_issue: Mapping[int, list[Mapping[str, Any]]],
    *,
    repo: str,
    label: str,
    generated_at: str,
    next_check_at: str,
) -> dict[str, Any]:
    """Build the complete triage report from a live issue inventory."""

    rows = [
        _triage_row(
            issue,
            list(comments_by_issue.get(int(issue["number"]), [])),
            repo=repo,
            next_check_at=next_check_at,
        )
        for issue in issues
    ]
    as_of = _parse_timestamp(generated_at) or datetime.now(UTC)
    class_counts = Counter(row["blocker_class"] for row in rows)
    confidence_counts = Counter(row["classification_confidence"] for row in rows)
    condition_counts = Counter(row["condition_mode"] for row in rows)
    age_counts = Counter(
        _age_bucket(row["last_meaningful_progress_at"], as_of=as_of) for row in rows
    )
    closure_candidates: list[dict[str, Any]] = []
    return {
        "schema": SCHEMA,
        "generated_at_utc": generated_at,
        "repo": repo,
        "source": {
            "label": label,
            "open_issue_count": len(rows),
            "pagination_complete": True,
            "comments_pagination_complete": True,
        },
        "counts": {
            "by_blocker_class": {name: class_counts.get(name, 0) for name in BLOCKER_CLASSES},
            "by_confidence": dict(sorted(confidence_counts.items())),
            "by_condition_mode": {
                "machine_testable": condition_counts.get("machine_testable", 0),
                "human_observable": condition_counts.get("human_observable", 0),
            },
            "by_progress_age": {
                name: age_counts.get(name, 0)
                for name in (
                    "under_7_days",
                    "7_to_29_days",
                    "30_to_89_days",
                    "90_days_or_more",
                    "unknown",
                )
            },
        },
        "closure_candidates": closure_candidates,
        "closure_policy": "never close unilaterally; keep open unless dead or duplicate is independently verified",
        "watcher_recommendation": {
            "command": f"scripts/dev/blocked_queue_triage.py --repo {repo} --label {label} --report-only",
            "machine_testable_conditions": [
                "dependency or upstream issue references can be rechecked against current issue state",
                "progress-age and triage-comment coverage can be checked on every scheduled run",
            ],
            "human_observable_conditions": [
                "compute authorization",
                "licence or rights approval",
                "maintainer/domain decision",
                "external fact or artifact provenance",
            ],
            "schedule": "weekly blocker-staleness workflow or an explicit workflow_dispatch run",
        },
        "issues": rows,
    }


def _generated_at(value: str | None) -> str:
    """Return a normalized UTC timestamp for reports and comment digests."""

    if value:
        parsed = _parse_timestamp(value)
        if parsed is None:
            raise TriageError(f"invalid --generated-at timestamp: {value}")
        return parsed.isoformat().replace("+00:00", "Z")
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    """Build the blocked-queue triage CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--limit", type=int, default=500)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply-comments",
        action="store_true",
        help="Create/update marker-bound triage comments; never changes issue state or labels.",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Only emit the inventory report (the default); perform no GitHub writes.",
    )
    parser.add_argument("--max-mutations", type=int, default=100)
    parser.add_argument(
        "--generated-at", help="Override report time with an ISO-8601 UTC timestamp."
    )
    parser.add_argument(
        "--next-check-at",
        default="next scheduled blocker-staleness workflow run",
        help="Timestamp or event recorded in every row.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the JSON report to this path as well as stdout."
    )
    return parser


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    """Run the report-only or bounded comment-application workflow."""

    args = _build_parser().parse_args(argv)
    try:
        if args.limit < 1 or args.max_mutations < 0:
            raise TriageError("--limit must be >= 1 and --max-mutations must be >= 0")
        gh_runner = runner or _run_gh
        generated_at = _generated_at(args.generated_at)
        issues = _fetch_blocked_issues(
            repo=args.repo,
            label=args.label,
            limit=args.limit,
            runner=gh_runner,
        )
        comments_by_issue = {
            int(issue["number"]): _fetch_comments(
                int(issue["number"]), repo=args.repo, runner=gh_runner
            )
            for issue in issues
        }
        report = build_report(
            issues,
            comments_by_issue,
            repo=args.repo,
            label=args.label,
            generated_at=generated_at,
            next_check_at=args.next_check_at,
        )
        if args.apply_comments:
            operations = apply_comments(
                report["issues"],
                comments_by_issue,
                repo=args.repo,
                generated_at=generated_at,
                max_mutations=args.max_mutations,
                runner=gh_runner,
            )
            report["mutations"] = {
                "applied": True,
                "operations": operations,
                "created": sum(item["action"] == "created" for item in operations),
                "updated": sum(item["action"] == "updated" for item in operations),
                "unchanged": sum(item["action"] == "unchanged" for item in operations),
            }
        else:
            report["mutations"] = {"applied": False, "operations": []}
        encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
        return 0
    except (OSError, TriageError, TypeError, ValueError) as exc:
        print(f"blocked queue triage failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
