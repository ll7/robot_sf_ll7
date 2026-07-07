#!/usr/bin/env python3
"""CI script to validate PR body and changes against repository contract rules.

This script implements issue #4735: PR contract checks for closes-discipline,
closure declaration, state-refresh-only, evidence tree hygiene, successor discipline,
and worker-lane labeling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Match keywords followed by #N or a GitHub issue URL
CLOSING_PATTERN = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+`?(?:#(\d+)|https?://github\.com/[^/\s]+/[^/\s]+/issues/(\d+))\b",
    re.IGNORECASE,
)


def is_negated(text: str, match_start: int) -> bool:
    """Check if the matched word is negated in the preceding context."""
    prefix = text[max(0, match_start - 30) : match_start].lower()
    negations = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bprevent\b",
        r"\bavoid\b",
        r"\bdoesnt\b",
        r"\bdoesn't\b",
        r"\bdont\b",
        r"\bdon't\b",
    ]
    for neg in negations:
        if re.search(neg, prefix):
            return True
    return False


def find_closed_issues(body: str) -> list[str]:
    """Extract issue numbers that this PR claims to close."""
    issues = []
    for match in CLOSING_PATTERN.finditer(body):
        if is_negated(body, match.start()):
            continue
        num1, num2 = match.groups()
        issues.append(num1 or num2)
    return sorted({i for i in issues if i}, key=int)


def find_title_issues(title: str) -> list[str]:
    """Extract issue numbers from the PR title."""
    pattern = re.compile(r"(?:#|issue\s+|issues\s+)(\d+)", re.IGNORECASE)
    return sorted(set(pattern.findall(title)), key=int)


def has_declaration_for_issue(issue: str, body: str) -> bool:
    """Check if there is a closes or refs declaration for the given issue."""
    pattern = re.compile(
        rf"\b(?:closes?|fixes?|resolves?|refs?|references?)\s+`?(?:#|https?://github\.com/[^/\s]+/[^/\s]+/issues/)?{issue}\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(body))


def get_issue_labels(issue: str, repo: str) -> list[str]:
    """Query GitHub API to get labels for a specific issue."""
    try:
        res = subprocess.run(
            ["gh", "issue", "view", issue, "--json", "labels", "--repo", repo],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            return [lbl["name"].lower() for lbl in data.get("labels", [])]
    except Exception:  # noqa: BLE001
        pass
    return []


def get_new_files(base_ref: str) -> set[str]:
    """Get the set of files added (created) in this branch relative to base_ref."""
    new_files = set()
    try:
        res = subprocess.run(
            ["git", "diff", "--name-status", base_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    status, path = parts
                    if status.startswith("A"):
                        new_files.add(path.replace("\\", "/").strip())
    except Exception:  # noqa: BLE001
        pass
    return new_files


def is_file_new(path: str, base_ref: str = "origin/main") -> bool:
    """Check if the file is new (does not exist on base_ref)."""
    if not os.path.exists(path):
        return False
    res = subprocess.run(["git", "show", f"{base_ref}:{path}"], capture_output=True, check=False)
    return res.returncode != 0


def check_closes_discipline(body: str, repo: str) -> list[str]:
    """Rule 1: Demand Refs #N instead of Closes #N if N is an epic issue."""
    blockers = []
    closed_issues = find_closed_issues(body)
    for issue in closed_issues:
        labels = get_issue_labels(issue, repo)
        if "epic" in labels:
            blockers.append(
                f"BLOCKER: PR body attempts to close epic issue #{issue}. "
                f"Epic issues cannot be closed by a single PR. Please use 'Refs #{issue}' instead."
            )
    return blockers


def check_closure_declaration(title: str, body: str) -> list[str]:
    """Rule 2: Warn if title has an issue number but body has no declaration."""
    warnings = []
    issues = find_title_issues(title)
    for issue in issues:
        if not has_declaration_for_issue(issue, body):
            warnings.append(
                f"WARN: PR title references issue #{issue}, but the PR body lacks a closure "
                f"declaration (e.g., 'Closes #{issue}' or 'Refs #{issue}')."
            )
    return warnings


def check_state_refresh_only(changed_files: list[str], title: str, body: str) -> list[str]:
    """Rule 3: Reject state-refresh-only PRs."""
    if not changed_files:
        return []

    only_state = True
    for f in changed_files:
        f = f.replace("\\", "/").strip()
        if not f:
            continue
        is_state_file = f.startswith("docs/context/") and not f.startswith("docs/context/evidence/")
        if not is_state_file:
            only_state = False
            break

    if not only_state:
        return []

    state_refresh_pattern = re.compile(
        r"\b(?:closure[-_ ]audit|state[-_ ]refresh|refresh[-_ ]state|state[-_ ]update|update[-_ ]state|status[-_ ]refresh|refresh[-_ ]status|state\.yaml|closure\.md)\b",
        re.IGNORECASE,
    )
    if state_refresh_pattern.search(title) or state_refresh_pattern.search(body):
        return [
            "BLOCKER: This PR touches ONLY docs/context/** state/closure files, and the body/title "
            "matches closure-audit/state patterns. State-refresh-only PRs are rejected; please "
            "post these updates as comments on the issue instead."
        ]
    return []


def check_evidence_tree_hygiene(changed_files: list[str], base_ref: str) -> list[str]:  # noqa: C901
    """Rule 4: Evidence tree marker and provenance checks."""
    blockers = []
    new_files = get_new_files(base_ref)

    for f in changed_files:
        f_norm = f.replace("\\", "/").strip()
        if not f_norm:
            continue

        # Determine relative path for validation checks
        f_rel = f_norm
        for pattern in ("docs/context/evidence/", "docs/context/"):
            if pattern in f_norm:
                f_rel = f_norm[f_norm.index(pattern) :]
                break

        if not f_rel.startswith("docs/context/evidence/"):
            continue
        if not os.path.exists(f):
            continue

        try:
            with open(f, encoding="utf-8") as file_obj:
                content = file_obj.read()
        except Exception:  # noqa: BLE001
            continue

        # 1. Marker check for new evidence files
        is_new = (f in new_files) or is_file_new(f, base_ref)
        if is_new:
            if "AI-GENERATED" not in content or "NEEDS-REVIEW" not in content:
                blockers.append(
                    f"BLOCKER: New evidence file '{f}' is missing the required AI-GENERATED "
                    f"and NEEDS-REVIEW marker convention."
                )

        # 2. Evidence README claims check
        if "README" in os.path.basename(f).upper():
            content_lower = content.lower()
            has_proves = "proves" in content_lower
            has_dem_stability = "demonstrates stability" in content_lower
            if has_proves or has_dem_stability:
                has_seeds = "seeds" in content_lower
                has_config = "config" in content_lower
                has_hash = "hash" in content_lower

                missing = []
                if not has_seeds:
                    missing.append("seeds")
                if not has_config:
                    missing.append("config")
                if not has_hash:
                    missing.append("hash")

                if missing:
                    claim = "proves" if has_proves else "demonstrates stability"
                    blockers.append(
                        f"BLOCKER: Evidence README '{f}' contains claim '{claim}', "
                        f"but is missing required provenance fields: {', '.join(missing)}."
                    )
    return blockers


def check_successor_discipline(title: str, body: str, repo: str) -> list[str]:
    """Rule 5: Successor statement warning for issues with merged PRs."""
    issues = find_title_issues(title)
    warnings = []
    body_lower = body.lower()
    has_successor_stmt = "successor slice" in body_lower or "does not duplicate" in body_lower

    if not has_successor_stmt:
        for issue in issues:
            try:
                res = subprocess.run(
                    [
                        "gh",
                        "pr",
                        "list",
                        "--search",
                        f"is:merged {issue}",
                        "--json",
                        "number",
                        "--repo",
                        repo,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if res.returncode == 0:
                    prs = json.loads(res.stdout)
                    if len(prs) >= 1:
                        warnings.append(
                            f"WARN: Issue #{issue} has already been referenced in {len(prs)} merged PR(s), "
                            f"but the PR body does not contain a successor statement ('successor slice' "
                            f"or 'does not duplicate')."
                        )
            except Exception:  # noqa: BLE001
                pass
    return warnings


def check_worker_lane_provenance(body: str, pr_number: str | None, repo: str) -> tuple[str, bool]:
    """Rule 6: Worker-lane provenance detection and labeling."""
    if "cheap implementation lane" in body.lower():
        if pr_number:
            try:
                res = subprocess.run(
                    [
                        "gh",
                        "pr",
                        "edit",
                        pr_number,
                        "--add-label",
                        "cheap-lane",
                        "--repo",
                        repo,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    return "INFO: Automatically added 'cheap-lane' label to the PR.", True
                else:
                    return (
                        f"INFO: Detected cheap-lane provenance, but failed to add label: {res.stderr.strip()}",
                        True,
                    )
            except Exception as e:  # noqa: BLE001
                return (
                    f"INFO: Detected cheap-lane provenance, but failed to run gh to add label: {e}",
                    True,
                )
        else:
            return (
                "INFO: Detected cheap-lane provenance. (Not in PR context; label not added).",
                True,
            )
    return "No worker-lane provenance detected.", False


def post_or_update_comment(pr_number: str, repo: str, comment_body: str) -> None:
    """Post or edit in-place the summary comment on the pull request."""
    signature = "<!-- pr-contract-check-signature -->"
    comment_id = None
    try:
        res = subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/{pr_number}/comments"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if res.returncode == 0:
            comments = json.loads(res.stdout)
            for c in comments:
                if signature in c.get("body", ""):
                    comment_id = c.get("id")
                    break
    except Exception as e:  # noqa: BLE001
        print(f"Error searching for comment: {e}", file=sys.stderr)

    if comment_id:
        try:
            subprocess.run(
                [
                    "gh",
                    "api",
                    "-X",
                    "PATCH",
                    f"repos/{repo}/issues/comments/{comment_id}",
                    "-f",
                    f"body={comment_body}",
                ],
                check=True,
            )
            print("Successfully updated existing comment.")
        except Exception as e:  # noqa: BLE001
            print(f"Error updating comment: {e}", file=sys.stderr)
    else:
        try:
            subprocess.run(
                [
                    "gh",
                    "pr",
                    "comment",
                    pr_number,
                    "--body",
                    comment_body,
                    "--repo",
                    repo,
                ],
                check=True,
            )
            print("Successfully created new comment.")
        except Exception as e:  # noqa: BLE001
            print(f"Error creating comment: {e}", file=sys.stderr)


def run_all_checks(
    title: str,
    body: str,
    changed_files: list[str],
    repo: str,
    base_ref: str,
    pr_number: str | None,
) -> tuple[list[str], list[str], list[str]]:
    """Run all 6 contract checks."""
    blockers = []
    warnings = []
    infos = []

    # 1. Closes-discipline
    closes_blockers = check_closes_discipline(body, repo)
    blockers.extend(closes_blockers)

    # 2. Closure declaration
    closure_warnings = check_closure_declaration(title, body)
    warnings.extend(closure_warnings)

    # 3. State-refresh-only
    state_blockers = check_state_refresh_only(changed_files, title, body)
    blockers.extend(state_blockers)

    # 4. Evidence tree hygiene
    evidence_blockers = check_evidence_tree_hygiene(changed_files, base_ref)
    blockers.extend(evidence_blockers)

    # 5. Successor discipline
    successor_warnings = check_successor_discipline(title, body, repo)
    warnings.extend(successor_warnings)

    # 6. Worker-lane provenance
    lane_info, _ = check_worker_lane_provenance(body, pr_number, repo)
    infos.append(lane_info)

    return blockers, warnings, infos


def build_comment_body(
    blockers: list[str], warnings: list[str], infos: list[str], status: str
) -> str:
    """Build the Markdown comment body to post on the PR."""
    rows = []

    def get_status_str(has_failures: bool, is_blocker: bool = True) -> str:
        if has_failures:
            return "❌ FAILED" if is_blocker else "⚠️ WARNING"
        return "✅ PASSED"

    rows.append(
        f"| 1. Closes-discipline | {get_status_str(any('closes epic' in b.lower() for b in blockers))} | Demand Refs #N for epic issues |"
    )
    rows.append(
        f"| 2. Closure declaration | {get_status_str(bool(warnings), is_blocker=False)} | Require Closes/Refs for title issues |"
    )
    rows.append(
        f"| 3. State-refresh-only | {get_status_str(any('state-refresh-only' in b.lower() for b in blockers))} | Reject docs/context state updates |"
    )
    rows.append(
        f"| 4. Evidence hygiene | {get_status_str(any('evidence' in b.lower() for b in blockers))} | Checks markers and provenance fields |"
    )
    rows.append(
        f"| 5. Successor discipline | {get_status_str(any('successor' in w.lower() for w in warnings), is_blocker=False)} | Require successor statement on multi-PR issues |"
    )

    lane_detected = "Added 'cheap-lane' label" in "".join(infos)
    rows.append(
        f"| 6. Worker-lane label | {'🏷️ cheap-lane' if lane_detected else '⚪ None'} | Label PRs from cheap worker lane |"
    )

    comment = [
        "<!-- pr-contract-check-signature -->",
        "## 🔍 PR Contract Check Summary",
        "",
        f"**Overall Status**: {status}",
        "",
        "| Check | Status | Description |",
        "| --- | --- | --- |",
        "\n".join(rows),
        "",
    ]

    if blockers:
        comment.append("### ❌ Blockers")
        for b in blockers:
            comment.append(f"- {b}")
        comment.append("")

    if warnings:
        comment.append("### ⚠️ Warnings")
        for w in warnings:
            comment.append(f"- {w}")
        comment.append("")

    if infos:
        comment.append("### ℹ️ Info")
        for i in infos:
            comment.append(f"- {i}")
        comment.append("")

    comment.append("---")
    comment.append("*This check is mechanized. Please resolve any blockers to pass CI.*")

    return "\n".join(comment)


def get_changed_files(changed_files_file: Path | None, base_ref: str) -> list[str]:
    """Get list of changed files from a file or fallback to git diff."""
    if changed_files_file and os.path.exists(changed_files_file):
        try:
            with open(changed_files_file, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:  # noqa: BLE001
            pass

    try:
        res = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        pass
    return []


def main() -> int:  # noqa: C901
    """Run PR contract validation checks."""
    parser = argparse.ArgumentParser(description="PR contract check.")
    parser.add_argument("--github-event-path", type=Path, help="Path to github event path JSON.")
    parser.add_argument("--changed-files-file", type=Path, help="Path to changed files list file.")
    parser.add_argument("--pr-body-file", type=Path, help="PR body file for local run/test.")
    parser.add_argument("--pr-title", type=str, help="PR title for local run/test.")
    parser.add_argument("--pr-number", type=str, help="PR number for local run/test.")
    parser.add_argument("--repo", type=str, default="ll7/robot_sf_ll7", help="Repo name.")
    parser.add_argument("--base-ref", type=str, default="origin/main", help="Git base branch ref.")
    parser.add_argument(
        "--post-comment", action="store_true", help="Post summary comment on the PR."
    )

    args = parser.parse_args()

    pr_body = ""
    pr_title = ""
    pr_number = args.pr_number
    repo = args.repo

    # Load from event path if available
    event_path = args.github_event_path or os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, encoding="utf-8") as f:
                event = json.load(f)
            pull_request = event.get("pull_request", {})
            pr_body = pull_request.get("body") or ""
            pr_title = pull_request.get("title") or ""
            pr_number = str(pull_request.get("number", ""))
            repo = event.get("repository", {}).get("full_name") or repo
        except Exception as e:  # noqa: BLE001
            print(f"Error reading event path: {e}", file=sys.stderr)

    # CLI overrides
    if args.pr_body_file and os.path.exists(args.pr_body_file):
        with open(args.pr_body_file, encoding="utf-8") as f:
            pr_body = f.read()
    if args.pr_title:
        pr_title = args.pr_title

    changed_files = get_changed_files(args.changed_files_file, args.base_ref)

    # Run checks
    blockers, warnings, infos = run_all_checks(
        pr_title, pr_body, changed_files, repo, args.base_ref, pr_number
    )

    # Build status
    status = "🔴 FAILED" if blockers else "🟢 PASSED"

    # Report results
    print(f"=== PR CONTRACT CHECK RESULTS: {status} ===")
    if blockers:
        print("\nBlockers:")
        for b in blockers:
            print(f"  [BLOCKER] {b}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  [WARN] {w}")
    if infos:
        print("\nInfo:")
        for i in infos:
            print(f"  [INFO] {i}")

    # Post comment if requested
    if args.post_comment and pr_number:
        comment_body = build_comment_body(blockers, warnings, infos, status)
        post_or_update_comment(pr_number, repo, comment_body)

    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
