#!/usr/bin/env python3
"""Read, add, or remove issue/PR labels through GitHub's REST API.

Why this exists
---------------
``gh pr edit --add-label`` and ``gh issue edit --label`` can fail on GitHub CLI
versions that query the retired Projects Classic GraphQL field. This helper uses
the paginated ``GET /repos/{owner}/{repo}/issues/{number}/labels`` route for
inventory reads, ``POST /repos/{owner}/{repo}/issues/{number}/labels`` for adds,
and ``DELETE /repos/{owner}/{repo}/issues/{number}/labels/{label}`` for removals;
write operations verify that GitHub actually applied or removed the requested
label. It is deliberately REST-only: authentication, authorization, malformed
responses, and verification mismatches fail closed.

The REST issues-labels endpoint works for both issues and PRs because GitHub
treats PRs as issues for labeling. One helper covers ``gh pr edit --add-label``
and ``gh issue edit --label``.

Usage
-----
::

    uv run python scripts/dev/gh_pr_label_rest.py list 5220 \\
        --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py add 5220 \\
        --label cheap-lane --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py add 5220 \\
        --label merge-ready --expected-head-sha <head_sha> \\
        --expected-base-sha <base_sha> --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py remove 5220 \\
        --label cheap-lane --repo ll7/robot_sf_ll7
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from scripts.dev._gh_rest import gh_api_delete as _gh_api_delete
from scripts.dev._gh_rest import gh_api_label_get as _gh_api_get
from scripts.dev._gh_rest import gh_api_post as _gh_api_post
from scripts.dev._gh_rest import subprocess
from scripts.dev.github_transport_policy import get_transport_contract
from scripts.dev.pr_carrier_gate import check_merge_ready_carriers
from scripts.dev.pr_write_guard import guard_pr_write, pr_write_lock

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REPO = "ll7/robot_sf_ll7"
LABEL_PAGE_SIZE = 100
LABEL_PAGE_CEILING = 10
TRANSPORT_CONTRACT = get_transport_contract("gh_pr_label_rest.py")


def _is_absent_label_delete(result: subprocess.CompletedProcess[str]) -> bool:
    """Recognize only GitHub's idempotent missing-label DELETE response."""
    if result.returncode == 0:
        return False
    detail = (result.stderr or result.stdout).strip().lower()
    return "http 404" in detail and "label does not exist" in detail


def _get_label_names(number: int, *, repo: str = DEFAULT_REPO, timeout: int = 30) -> dict[str, Any]:
    """Return a complete, strictly validated label inventory, or an error dict."""
    names: list[str] = []
    for page in range(1, LABEL_PAGE_CEILING + 1):
        path = f"repos/{repo}/issues/{number}/labels?per_page={LABEL_PAGE_SIZE}&page={page}"
        result = _gh_api_get(path, timeout=timeout)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
            return {"status": "error", "error": f"could not read labels page {page}: {detail}"}
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            snippet = result.stdout.strip()[:200]
            return {
                "status": "error",
                "error": f"label page {page} was not valid JSON: {exc}; "
                f"stdout snippet: {snippet!r}",
            }
        if not isinstance(data, list):
            return {
                "status": "error",
                "error": f"expected a list from labels page {page}, got {type(data).__name__}",
            }
        for row in data:
            if not isinstance(row, dict):
                return {
                    "status": "error",
                    "error": f"malformed label row on page {page}: expected an object",
                }
            name = row.get("name")
            if not isinstance(name, str) or not name.strip():
                return {
                    "status": "error",
                    "error": f"malformed label row on page {page}: name must be non-empty text",
                }
            names.append(name)
        if len(data) < LABEL_PAGE_SIZE:
            return {"status": "ok", "labels": names}
    return {
        "status": "error",
        "error": f"label pagination exceeded the page ceiling of {LABEL_PAGE_CEILING}",
    }


def get_label_names(number: int, *, repo: str = DEFAULT_REPO, timeout: int = 30) -> dict[str, Any]:
    """Read the complete current label inventory for an issue or pull request."""
    if type(number) is not int or number < 1:
        return {"status": "error", "error": f"issue/PR number must be positive, got {number}"}
    return _get_label_names(number, repo=repo, timeout=timeout)


def list_labels(number: int, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Return a compact CLI payload containing the verified current labels."""
    result = get_label_names(number, repo=repo)
    if result["status"] != "ok":
        return result
    return {
        "status": "ok",
        "number": number,
        "action": "list",
        "repo": repo,
        "labels": result["labels"],
    }


def validate_result_envelope(
    result: object,
    *,
    action: str,
    number: int,
    repo: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Validate a successful CLI result before an orchestrator trusts it.

    Shell callers and direct library callers use the same contract: a success
    result must identify the operation, exact issue/PR number, repository, and
    label where applicable. List results additionally require a distinct,
    non-empty string inventory.
    """
    if action not in {"list", "add", "remove"}:
        raise ValueError(f"unsupported label helper action {action!r}")
    _validate_result_identity(result, action=action, number=number, repo=repo)
    if action == "list":
        _validate_list_result(result)
    else:
        _validate_write_result(result, label=label)
    return result


def _validate_result_identity(result: object, *, action: str, number: int, repo: str) -> None:
    """Validate the common identity fields in a successful label result."""
    if not isinstance(result, dict):
        raise ValueError("label helper result must be a JSON object")
    if result.get("status") != "ok":
        raise ValueError(f"label helper result status must be ok, got {result.get('status')!r}")
    if type(number) is not int or number < 1:
        raise ValueError(f"expected issue/PR number must be a positive integer, got {number!r}")
    if result.get("action") != action:
        raise ValueError(
            f"label helper result action does not match request "
            f"({result.get('action')!r} != {action!r})"
        )
    if type(result.get("number")) is not int or result.get("number") != number:
        raise ValueError(
            f"label helper result number does not match request "
            f"({result.get('number')!r} != {number})"
        )
    if result.get("repo") != repo:
        raise ValueError(
            f"label helper result repository does not match request "
            f"({result.get('repo')!r} != {repo!r})"
        )


def _validate_list_result(result: dict[str, Any]) -> None:
    """Validate the label inventory in a successful list result."""
    labels = result.get("labels")
    if not isinstance(labels, list):
        raise ValueError("label helper list result labels must be a list")
    if any(type(name) is not str or not name.strip() for name in labels):
        raise ValueError("label helper list result labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("label helper list result labels must be distinct")


def _validate_write_result(result: dict[str, Any], *, label: str | None) -> None:
    """Validate the requested label in a successful add/remove result."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("expected label must be a non-empty string for add/remove")
    if result.get("label") != label:
        raise ValueError(
            f"label helper result label does not match request "
            f"({result.get('label')!r} != {label!r})"
        )


def _guarded_merge_ready_write(
    number: int,
    *,
    repo: str,
    expected_head_sha: str | None,
    expected_base_sha: str | None,
    write: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run a merge-ready label write only after the exact-head/base preflight.

    The write additionally requires the PR body and its exact-head review
    comments to be bound to the live head/base: a stale body or a
    stale-narrative review comment (including any pending domain-review
    disposition) withholds ``merge-ready`` fail-closed (issue #7610).
    """
    try:
        with pr_write_lock(repo, number):
            guard = guard_pr_write(
                number,
                repo=repo,
                expected_head_sha=expected_head_sha,
                expected_base_sha=expected_base_sha,
                operation="merge_ready_label",
            )
            if guard["status"] != "ok":
                return guard
            observed_base = guard.get("observed_base_sha")
            if not observed_base:
                return {
                    "status": "error",
                    "error": "live PR base SHA unavailable for the carrier gate",
                }
            carriers = check_merge_ready_carriers(
                number,
                repo=repo,
                live_head=guard["observed_head_sha"],
                live_base=observed_base,
            )
            if carriers["status"] != "ok":
                return carriers
            return write()
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def add_label(
    number: int,
    label: str,
    *,
    repo: str = DEFAULT_REPO,
    expected_head_sha: str | None = None,
    expected_base_sha: str | None = None,
) -> dict[str, Any]:
    """Add *label* to issue/PR *number* and verify it was applied.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if type(number) is not int or number < 1:
        return {"status": "error", "error": f"issue/PR number must be positive, got {number}"}
    if not isinstance(label, str) or not label.strip():
        return {"status": "error", "error": "label must be a non-empty string"}

    def _write() -> dict[str, Any]:
        """Apply and verify one label after any required PR preflight."""
        path = f"repos/{repo}/issues/{number}/labels"
        result = _gh_api_post(path, {"labels": [label]})
        if result.returncode != 0:
            detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
            return {"status": "error", "error": f"label add failed: {detail}"}
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            snippet = result.stdout.strip()[:200]
            return {
                "status": "error",
                "error": f"label add returned invalid JSON: {exc}; stdout snippet: {snippet!r}",
            }

        current = get_label_names(number, repo=repo)
        if current["status"] == "error":
            return current
        if label not in current["labels"]:
            return {
                "status": "error",
                "error": f"label '{label}' was not found in labels after add; "
                "the write may not have taken effect",
            }
        return {
            "status": "ok",
            "number": number,
            "label": label,
            "action": "add",
            "repo": repo,
        }

    if label != "merge-ready":
        return _write()
    return _guarded_merge_ready_write(
        number,
        repo=repo,
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
        write=_write,
    )


def remove_label(number: int, label: str, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Remove *label* from issue/PR *number* and verify it was removed.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if type(number) is not int or number < 1:
        return {"status": "error", "error": f"issue/PR number must be positive, got {number}"}
    if not isinstance(label, str) or not label.strip():
        return {"status": "error", "error": "label must be a non-empty string"}

    path = f"repos/{repo}/issues/{number}/labels/{quote(label, safe='')}"
    result = _gh_api_delete(path)
    idempotent = _is_absent_label_delete(result)
    if result.returncode != 0 and not idempotent:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return {"status": "error", "error": f"label remove failed: {detail}"}

    # Verify the label was actually removed by re-reading labels.
    current = get_label_names(number, repo=repo)
    if current["status"] == "error":
        return current
    if label in current["labels"]:
        return {
            "status": "error",
            "error": f"label '{label}' was still found in labels after remove; "
            "the delete may not have taken effect",
        }
    response = {
        "status": "ok",
        "number": number,
        "label": label,
        "action": "remove",
        "repo": repo,
    }
    if idempotent:
        response["idempotent"] = True
    return response


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("list", "add", "remove"),
        help="Whether to list, add, or remove labels.",
    )
    parser.add_argument("number", type=int, help="Issue or PR number to inspect or update.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to update (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--label",
        help="Label name to add or remove (required for add/remove).",
    )
    parser.add_argument(
        "--expected-head-sha",
        help="Full PR head SHA required when adding merge-ready.",
    )
    parser.add_argument(
        "--expected-base-sha",
        help="Full PR base SHA required when adding merge-ready.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the label helper and emit one compact JSON result."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "list":
        result = list_labels(args.number, repo=args.repo)
    elif not args.label:
        parser.error("--label is required for add/remove")
    elif args.action == "add":
        result = add_label(
            args.number,
            args.label,
            repo=args.repo,
            expected_head_sha=args.expected_head_sha,
            expected_base_sha=args.expected_base_sha,
        )
    else:
        result = remove_label(args.number, args.label, repo=args.repo)

    if result.get("status") == "ok":
        try:
            validate_result_envelope(
                result,
                action=args.action,
                number=args.number,
                repo=args.repo,
                label=args.label,
            )
        except ValueError as exc:
            result = {"status": "error", "error": f"invalid label helper result: {exc}"}

    stream = sys.stdout if result["status"] == "ok" else sys.stderr
    print(json.dumps(result, sort_keys=True), file=stream)
    if result["status"] == "ok":
        return 0
    if result["status"] == "review_skipped_stale_state":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
