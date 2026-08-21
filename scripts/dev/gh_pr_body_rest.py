#!/usr/bin/env python3
"""Update or reconcile a pull-request title/body through GitHub's REST API.

Why this exists
---------------
``gh pr edit --body-file`` can fail on GitHub CLI versions that query the retired
Projects Classic GraphQL field. This helper uses only ``PATCH /repos/{owner}/{repo}/pulls/{number}``
and verifies that GitHub returned the requested body. It is deliberately REST-only:
authentication, authorization, malformed responses, and body mismatches fail closed.
The ``--reconcile`` mode reads the current title/body first, performs one atomic
title-and-body PATCH only when either field differs, and verifies both fields. All
helper writers serialize per-PR through a host-local advisory lock held from the
read through the post-update verification. A final read detects an external writer
that does not use the lock and fails closed instead of claiming reconciliation.

Usage
-----
::

    source .venv/bin/activate

    uv run python scripts/dev/gh_pr_body_rest.py 5220 \
        --repo ll7/robot_sf_ll7 --body-file /tmp/pr-body.md

    uv run python scripts/dev/gh_pr_body_rest.py 5220 --reconcile \
        --title "fix: final title" --repo ll7/robot_sf_ll7 \
        --body-file /tmp/pr-body.md
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.dev._gh_rest import gh_api_metadata_get as _gh_api_get
from scripts.dev._gh_rest import gh_api_patch as _gh_api_patch
from scripts.dev._gh_rest import subprocess
from scripts.dev.pr_loop_policy import (
    extract_sha_carriers,
    invalid_sha_carriers,
    metadata_conflict_handoff,
)
from scripts.dev.pr_metadata import metadata_digest, validate_pr_title

if TYPE_CHECKING:
    from collections.abc import Iterator


DEFAULT_REPO = "ll7/robot_sf_ll7"
_LOCK_DIR_ENV = "ROBOT_SF_PR_METADATA_LOCK_DIR"


@contextmanager
def _metadata_write_lock(repo: str, number: int) -> Iterator[None]:
    """Serialize helper writers for one repository/PR on the current host."""
    lock_root = Path(os.environ.get(_LOCK_DIR_ENV, "") or tempfile.gettempdir())
    lock_root = lock_root / "robot_sf_ll7_pr_metadata"
    lock_key = hashlib.sha256(f"{repo}:{number}".encode()).hexdigest()
    lock_path = lock_root / f"{lock_key}.lock"
    try:
        lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"could not acquire PR metadata writer lock {lock_path}: {exc}") from exc


def _read_body_file(body_file: Path) -> tuple[str | None, str | None]:
    """Read a UTF-8 PR body file, returning ``(body, error)``."""
    try:
        body = body_file.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read body file {body_file}: {exc}"
    if not body.strip():
        return None, f"refusing to write an empty PR body from {body_file}"
    return body, None


def _decode_object(
    result: subprocess.CompletedProcess[str], *, operation: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Decode a successful ``gh api`` result as a JSON object."""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return None, f"{operation} failed: {detail}"
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return None, f"{operation} returned invalid JSON: {exc}; stdout snippet: {snippet!r}"
    if not isinstance(response, dict):
        return None, f"{operation} response was not an object"
    return response, None


def _head_sha(payload: dict[str, Any]) -> str | None:
    """Return a live PR head SHA when the REST payload provides one."""
    head = payload.get("head")
    if isinstance(head, dict) and isinstance(head.get("sha"), str) and head["sha"]:
        return head["sha"]
    value = payload.get("head_sha")
    return value if isinstance(value, str) and value else None


def _resolve_live_head(repo: str, number: int) -> tuple[str | None, str | None]:
    """Resolve the live PR head SHA (REST ``head.sha``) or an error string."""
    result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    response, error = _decode_object(result, operation="PR live-head read")
    if error:
        return None, error
    assert response is not None
    head = response.get("head")
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str) or not head["sha"]:
        return None, "PR live-head read returned a malformed head"
    return head["sha"], None


def _git_object_diagnostics(shas: list[str]) -> dict[str, str]:
    """Return per-SHA local object types via ``git cat-file -t``, best-effort.

    Diagnostics only: a missing local object never blocks the write itself;
    equality to the live remote head remains the admission rule.
    """
    result: dict[str, str] = {}
    for sha in shas:
        try:
            proc = subprocess.run(
                ["git", "cat-file", "-t", sha],
                capture_output=True,
                text=True,
                timeout=10,
            )
            result[sha] = (proc.stdout or "").strip() or "missing"
        except (OSError, subprocess.TimeoutExpired):
            result[sha] = "unavailable"
    return result


def _validate_sha_carriers(body: str, live_head_sha: str) -> str | None:
    """Return a fail-closed error when *body* carries a non-live exact-head SHA.

    Admission rule: every ``gate-verdict``/``base-policy``/``Exact head`` SHA
    carrier must be a full 40-hex SHA equal to the live PR head. Abbreviated
    SHAs and any SHA naming a different commit are invalid and block the write
    before any PATCH is issued.
    """
    carriers = extract_sha_carriers(body)
    if not carriers:
        return None
    invalid = invalid_sha_carriers(carriers, live_head_sha)
    if not invalid:
        return None
    details = "; ".join(
        f"{carrier.kind} {carrier.sha}" + ("" if carrier.full else " (abbreviated)")
        for carrier in invalid
    )
    diagnostic = ""
    full_shas = [carrier.sha for carrier in invalid if carrier.full]
    if full_shas:
        objects = _git_object_diagnostics(full_shas)
        diagnostic = " local object type: " + ", ".join(
            f"{sha}={obj}" for sha, obj in objects.items()
        )
    return (
        f"PR body carries exact-head SHA carrier(s) that do not match the "
        f"live head {live_head_sha}: {details}.{diagnostic}"
    )


def _guard_update_body(body: str, repo: str, number: int) -> str | None:
    """Resolve the live head and validate the body's exact-head carriers.

    Returns a fail-closed error before any PATCH when the body carries a
    non-live exact-head SHA, or when a carrier-bearing body cannot be checked
    against a resolvable live head.
    """
    if not extract_sha_carriers(body):
        return None
    live_head, head_error = _resolve_live_head(repo, number)
    if head_error:
        return head_error
    assert live_head is not None
    return _validate_sha_carriers(body, live_head)


def _guard_reconcile_body(body: str, current: dict[str, Any]) -> str | None:
    """Validate a desired body's exact-head carriers against the current PR head.

    The current GET response supplies the live ``head.sha``; it is read inside
    the writer lock immediately before mutation, so it is the admission
    reference for this reconcile attempt.
    """
    if not extract_sha_carriers(body):
        return None
    head = current.get("head")
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str) or not head["sha"]:
        return "PR metadata read returned a malformed head"
    return _validate_sha_carriers(body, head["sha"])


def update_pr_body(number: int, body_file: Path, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Update PR *number* from *body_file* and verify the REST response.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    body, body_error = _read_body_file(body_file)
    if body_error:
        return {"status": "error", "error": body_error}
    assert body is not None

    try:
        with _metadata_write_lock(repo, number):
            guard_error = _guard_update_body(body, repo, number)
            if guard_error:
                return {"status": "error", "error": guard_error}
            result = _gh_api_patch(f"repos/{repo}/pulls/{number}", {"body": body})
            if result.returncode != 0:
                detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
                return {"status": "error", "error": f"PR body update failed: {detail}"}
            try:
                response = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                snippet = result.stdout.strip()[:200]
                return {
                    "status": "error",
                    "error": f"PR body update returned invalid JSON: {exc}; stdout snippet: {snippet!r}",
                }
            if not isinstance(response, dict):
                return {"status": "error", "error": "PR body update response was not an object"}
            if response.get("body") != body:
                return {
                    "status": "error",
                    "error": "PR body update response did not preserve the requested body",
                }
            return {
                "status": "ok",
                "number": number,
                "repo": repo,
                "url": str(response.get("html_url", "")),
            }
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def reconcile_pr_metadata(  # noqa: C901
    number: int,
    title: str,
    body_file: Path,
    *,
    repo: str = DEFAULT_REPO,
) -> dict[str, Any]:
    """Reconcile a PR's final title/body with one verified atomic REST update.

    The current PR is read before mutation. A matching title/body returns an
    explicit no-op; otherwise one PATCH carries both fields and the response
    must preserve both exact requested strings.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    title_error = validate_pr_title(title)
    if title_error:
        return {"status": "error", "error": title_error}
    body, body_error = _read_body_file(body_file)
    if body_error:
        return {"status": "error", "error": body_error}
    assert body is not None

    try:
        with _metadata_write_lock(repo, number):
            current_result = _gh_api_get(f"repos/{repo}/pulls/{number}")
            current, current_error = _decode_object(current_result, operation="PR metadata read")
            if current_error:
                return {"status": "error", "error": current_error}
            assert current is not None
            current_title = current.get("title")
            current_body = current.get("body")
            if not isinstance(current_title, str):
                return {"status": "error", "error": "PR metadata read returned a malformed title"}
            if current_body is None:
                current_body = ""
            if not isinstance(current_body, str):
                return {"status": "error", "error": "PR metadata read returned a malformed body"}
            current_head_sha = _head_sha(current)

            guard_error = _guard_reconcile_body(body, current)
            if guard_error:
                return {"status": "error", "error": guard_error}

            desired_digest = metadata_digest(title, body)
            current_digest = metadata_digest(current_title, current_body)
            changed_fields = [
                field
                for field, current_value, desired_value in (
                    ("title", current_title, title),
                    ("body", current_body, body),
                )
                if current_value != desired_value
            ]
            base_result = {
                "number": number,
                "repo": repo,
                "url": str(current.get("html_url", "")),
                "metadata_digest": desired_digest,
                "previous_metadata_digest": current_digest,
                "changed_fields": changed_fields,
            }
            if not changed_fields:
                return {"status": "unchanged", **base_result, "changed": False}

            patch_result = _gh_api_patch(
                f"repos/{repo}/pulls/{number}",
                {"title": title, "body": body},
            )
            response, patch_error = _decode_object(patch_result, operation="PR metadata update")
            if patch_error:
                return {"status": "error", "error": patch_error}
            assert response is not None
            if response.get("title") != title or response.get("body") != body:
                return {
                    "status": "error",
                    "error": (
                        "PR metadata update response did not preserve the requested title and body"
                    ),
                }

            verify_result = _gh_api_get(f"repos/{repo}/pulls/{number}")
            verified, verify_error = _decode_object(
                verify_result, operation="PR metadata post-update verification"
            )
            if verify_error:
                return {"status": "error", "error": verify_error}
            assert verified is not None
            if verified.get("title") != title or verified.get("body") != body:
                observed_title = verified.get("title")
                observed_body = verified.get("body")
                observed_digest = (
                    metadata_digest(observed_title, observed_body)
                    if isinstance(observed_title, str) and isinstance(observed_body, str)
                    else None
                )
                conflict = {
                    "status": "conflict",
                    "error": (
                        "PR metadata changed during reconciliation; refusing to claim success"
                    ),
                    **base_result,
                    "changed": True,
                    "observed_metadata_digest": observed_digest,
                    "previous_head_sha": current_head_sha,
                    "observed_head_sha": _head_sha(verified),
                }
                conflict.update(metadata_conflict_handoff(conflict))
                return conflict
            return {
                "status": "ok",
                **base_result,
                "url": str(response.get("html_url", current.get("html_url", ""))),
                "changed": True,
            }
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Pull-request number to update.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to update (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--body-file", type=Path, required=True, help="Markdown body file to apply."
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Read current metadata and atomically reconcile title and body when needed.",
    )
    parser.add_argument(
        "--title",
        help="Final PR title; required with --reconcile.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PR-body updater and emit one compact JSON result."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.reconcile and args.title is None:
        parser.error("--reconcile requires --title")
    if not args.reconcile and args.title is not None:
        parser.error("--title requires --reconcile")
    if args.reconcile:
        result = reconcile_pr_metadata(
            args.number,
            args.title,
            args.body_file,
            repo=args.repo,
        )
    else:
        result = update_pr_body(args.number, args.body_file, repo=args.repo)
    success = result["status"] in {"ok", "unchanged"}
    stream = sys.stdout if success else sys.stderr
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
