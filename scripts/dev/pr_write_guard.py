#!/usr/bin/env python3
"""Fail-closed, exact-head preflight for writes targeting an open pull request."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.dev._gh_rest import gh_api_get as _gh_api_get
from scripts.dev._gh_rest import parse_json as _parse_json

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_REPO = "ll7/robot_sf_ll7"
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
STALE_WRITE_STATUS = "review_skipped_stale_state"
_LOCK_DIR_ENV = "ROBOT_SF_PR_WRITE_LOCK_DIR"


@contextmanager
def pr_write_lock(repo: str, number: int) -> Iterator[None]:
    """Serialize same-host PR review/label writers for one repository and PR."""
    lock_root = Path(os.environ.get(_LOCK_DIR_ENV, "") or tempfile.gettempdir())
    lock_root = lock_root / "robot_sf_pr_write"
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
        raise RuntimeError(f"could not acquire PR write lock {lock_path}: {exc}") from exc


def _base_result(
    number: int,
    *,
    repo: str,
    operation: str,
    expected_head_sha: str | None,
    observed_state: str,
    observed_head_sha: str,
    merged_at: Any,
) -> dict[str, Any]:
    """Build the shared machine-readable state fields."""
    return {
        "number": number,
        "repo": repo,
        "operation": operation,
        "expected_head_sha": expected_head_sha or "",
        "observed_state": observed_state,
        "observed_head_sha": observed_head_sha,
        "merged_at": merged_at,
    }


def guard_pr_write(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    expected_head_sha: str | None,
    operation: str,
) -> dict[str, Any]:
    """Read PR state/head immediately before a review or merge-ready write.

    A non-open PR or a head mismatch returns ``review_skipped_stale_state`` and
    must never be followed by a write. Transport or malformed-payload failures
    return ``error`` so callers fail closed rather than treating uncertainty as
    a safe skip.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    if not isinstance(expected_head_sha, str) or not FULL_SHA_RE.fullmatch(expected_head_sha):
        return {
            "status": "error",
            "error": "expected_head_sha must be a full 40-character SHA",
            "number": number,
            "repo": repo,
            "operation": operation,
        }

    result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    payload, error = _parse_json(result, what=f"PR {number} write-state read")
    if error:
        return {"status": "error", "error": error}
    if not isinstance(payload, dict):
        return {"status": "error", "error": "PR write-state payload was not an object"}

    raw_state = payload.get("state")
    raw_head = payload.get("head")
    merged_at = payload.get("merged_at")
    if not isinstance(raw_state, str) or not raw_state:
        return {"status": "error", "error": "PR write-state payload has no state"}
    if not isinstance(raw_head, dict) or not isinstance(raw_head.get("sha"), str):
        return {"status": "error", "error": "PR write-state payload has no head SHA"}
    if merged_at is not None and not isinstance(merged_at, str):
        return {"status": "error", "error": "PR write-state payload has malformed merged_at"}

    observed_state = raw_state.upper()
    observed_head_sha = raw_head["sha"]
    base = _base_result(
        number,
        repo=repo,
        operation=operation,
        expected_head_sha=expected_head_sha,
        observed_state=observed_state,
        observed_head_sha=observed_head_sha,
        merged_at=merged_at,
    )
    if observed_state != "OPEN" or merged_at:
        return {
            "status": STALE_WRITE_STATUS,
            "reason": "pr_not_open",
            **base,
        }
    if observed_head_sha.lower() != expected_head_sha.lower():
        return {
            "status": STALE_WRITE_STATUS,
            "reason": "head_sha_changed",
            **base,
        }
    return {"status": "ok", **base}
