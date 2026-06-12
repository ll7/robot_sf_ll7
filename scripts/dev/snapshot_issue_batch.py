#!/usr/bin/env python3
"""Emit a compact issue-batch snapshot for token-efficient goal orchestration."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

from scripts.dev.issue_claim import status_issue

BODY_EXCERPT_CHARS = 300
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"


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
    excerpt, truncated = _body_excerpt(issue.get("body"), limit=body_limit)
    claim = status_issue(number, remote=remote)
    return {
        "number": issue.get("number", number),
        "status": "ok",
        "title": issue.get("title", ""),
        "state": issue.get("state", ""),
        "url": issue.get("url", ""),
        "labels": labels,
        "assignees": _assignees(issue),
        "body_excerpt": excerpt,
        "body_truncated": truncated,
        "claim": {
            "ok": claim.get("ok"),
            "claimed": claim.get("claimed"),
            "claim_ref": claim.get("claim_ref"),
            "sha": claim.get("sha"),
        },
        "linked_prs": [],
        "recommended_context_pack": _recommended_context_pack(
            int(issue.get("number", number)), labels, str(issue.get("title", ""))
        ),
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
        path.write_text(json.dumps(_context_capsule(issue), indent=2, sort_keys=True) + "\n")
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
        "issues", nargs="+", type=int, help="Issue numbers; two values form a range."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--body-chars", type=int, default=BODY_EXCERPT_CHARS)
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    numbers = expand_issue_numbers(args.issues, expand_range=not args.no_expand_range)
    try:
        payload = snapshot_issues(
            numbers,
            repo=args.repo,
            body_limit=max(args.body_chars, 0),
            remote=args.remote,
            capsule_dir=args.capsule_dir,
        )
    except FileNotFoundError:
        print("gh or git command not found", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as exc:
        print(f"snapshot command timed out: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload))
    return 1 if any(issue.get("status") == "error" for issue in payload["issues"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
