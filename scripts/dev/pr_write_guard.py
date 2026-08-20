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
    expected_shas: tuple[str | None, str | None],
    observed_state: str,
    observed_head_sha: str,
    observed_base_sha: str | None,
    merged_at: Any,
) -> dict[str, Any]:
    """Build the shared machine-readable state fields."""
    expected_head_sha, expected_base_sha = expected_shas
    return {
        "number": number,
        "repo": repo,
        "operation": operation,
        "expected_head_sha": expected_head_sha or "",
        "expected_base_sha": expected_base_sha or "",
        "observed_state": observed_state,
        "observed_head_sha": observed_head_sha,
        "observed_base_sha": observed_base_sha or "",
        "merged_at": merged_at,
    }


def _write_verdict(
    *,
    observed_state: str,
    merged_at: Any,
    observed_head_sha: str,
    expected_head_sha: str,
    observed_base_sha: str | None,
    expected_base_sha: str | None,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Return the fail-closed stale verdict or the admission verdict."""
    if observed_state != "OPEN" or merged_at:
        return {"status": STALE_WRITE_STATUS, "reason": "pr_not_open", **base}
    if observed_head_sha.lower() != expected_head_sha.lower():
        return {"status": STALE_WRITE_STATUS, "reason": "head_sha_changed", **base}
    if (
        expected_base_sha is not None
        and observed_base_sha is not None
        and observed_base_sha.lower() != expected_base_sha.lower()
    ):
        return {"status": STALE_WRITE_STATUS, "reason": "base_sha_changed", **base}
    return {"status": "ok", **base}


def _validate_expected_shas(
    *,
    number: int,
    repo: str,
    operation: str,
    expected_head_sha: str | None,
    expected_base_sha: str | None,
) -> dict[str, Any] | None:
    """Return a fail-closed error dict for invalid expected SHAs, or None."""
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
    if expected_base_sha is not None and (
        not isinstance(expected_base_sha, str) or not FULL_SHA_RE.fullmatch(expected_base_sha)
    ):
        return {
            "status": "error",
            "error": "expected_base_sha must be a full 40-character SHA when provided",
            "number": number,
            "repo": repo,
            "operation": operation,
        }
    return None


def guard_pr_write(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    expected_head_sha: str | None,
    expected_base_sha: str | None = None,
    operation: str,
) -> dict[str, Any]:
    """Read PR state/head/base immediately before a review or merge-ready write.

    A non-open PR, a head mismatch, or a base mismatch when
    ``expected_base_sha`` is supplied returns ``review_skipped_stale_state`` and
    must never be followed by a write. Transport or malformed-payload failures
    return ``error`` so callers fail closed rather than treating uncertainty as
    a safe skip.
    """
    validation_error = _validate_expected_shas(
        number=number,
        repo=repo,
        operation=operation,
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
    )
    if validation_error is not None:
        return validation_error
    assert isinstance(expected_head_sha, str)

    result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    payload, error = _parse_json(result, what=f"PR {number} write-state read")
    if error:
        return {"status": "error", "error": error}
    if not isinstance(payload, dict):
        return {"status": "error", "error": "PR write-state payload was not an object"}

    raw_state = payload.get("state")
    raw_head = payload.get("head")
    raw_base = payload.get("base")
    merged_at = payload.get("merged_at")
    if not isinstance(raw_state, str) or not raw_state:
        return {"status": "error", "error": "PR write-state payload has no state"}
    if not isinstance(raw_head, dict) or not isinstance(raw_head.get("sha"), str):
        return {"status": "error", "error": "PR write-state payload has no head SHA"}
    if merged_at is not None and not isinstance(merged_at, str):
        return {"status": "error", "error": "PR write-state payload has malformed merged_at"}
    if expected_base_sha is not None and (
        not isinstance(raw_base, dict) or not isinstance(raw_base.get("sha"), str)
    ):
        return {"status": "error", "error": "PR write-state payload has no base SHA"}

    observed_state = raw_state.upper()
    observed_head_sha = raw_head["sha"]
    observed_base_sha: str | None = None
    if isinstance(raw_base, dict) and isinstance(raw_base.get("sha"), str):
        observed_base_sha = raw_base["sha"]
    base = _base_result(
        number,
        repo=repo,
        operation=operation,
        expected_shas=(expected_head_sha, expected_base_sha),
        observed_state=observed_state,
        observed_head_sha=observed_head_sha,
        observed_base_sha=observed_base_sha,
        merged_at=merged_at,
    )
    return _write_verdict(
        observed_state=observed_state,
        merged_at=merged_at,
        observed_head_sha=observed_head_sha,
        expected_head_sha=expected_head_sha,
        observed_base_sha=observed_base_sha,
        expected_base_sha=expected_base_sha,
        base=base,
    )
