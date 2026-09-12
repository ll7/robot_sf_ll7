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
responses, and verification mismatches fail closed. A rate-limited mutation
retries only within a small bounded policy and otherwise returns an explicit
``blocked`` receipt carrying the rate-limit reset evidence (issue #9146).

The REST issues-labels endpoint works for both issues and PRs because GitHub
treats PRs as issues for labeling. One helper covers ``gh pr edit --add-label``
and ``gh issue edit --label``.

Usage
-----
::

    uv run python scripts/dev/gh_pr_label_rest.py list 5220 \\
        --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py add 5220 \\
        --target issue --label cheap-lane --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py add 5220 \\
        --target pr --label merge-ready --expected-head-sha <head_sha> \\
        --expected-base-sha <base_sha> --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py remove 5220 \\
        --target issue --label cheap-lane --repo ll7/robot_sf_ll7
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

if __package__ in {None, ""}:
    # Direct execution must resolve this checkout's transport and write guards,
    # ahead of any competing checkout or editable installation on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev._gh_rest import gh_api_delete as _gh_api_delete
from scripts.dev._gh_rest import gh_api_label_get as _gh_api_get
from scripts.dev._gh_rest import gh_api_post as _gh_api_post
from scripts.dev._gh_rest import subprocess
from scripts.dev.github_transport_policy import get_transport_contract
from scripts.dev.pr_write_guard import guard_pr_write, pr_write_lock

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REPO = "ll7/robot_sf_ll7"
LABEL_PAGE_SIZE = 100
LABEL_PAGE_CEILING = 10
RATE_LIMIT_STATUS = "rate_limited"
BLOCKED_STATUS = "blocked"
RATE_LIMIT_MAX_ATTEMPTS = 3
RATE_LIMIT_MAX_WAIT_SECONDS = 15.0
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "too many requests")
_SECONDARY_RATE_LIMIT_MARKERS = ("secondary rate", "secondary-rate", "abuse detection")
MAX_RETRY_AFTER_SECONDS = 7 * 24 * 60 * 60
MAX_RATE_LIMIT_EPOCH = 2**63 - 1
_RATE_LIMIT_RESET_RE = re.compile(r"(?im)(?:^|[\s,;(])x-ratelimit-reset\s*[:=]\s*([^\s,;)]+)")
_RATE_LIMIT_RETRY_AFTER_RE = re.compile(r"(?im)(?:^|[\s,;(])retry[- ]after\s*[:=]\s*([^\s,;)]+)")
_RATE_LIMIT_RESET_HEADER_RE = re.compile(r"(?im)(?:^|[\s,;(])x-ratelimit-reset\s*[:=]")
_RATE_LIMIT_RETRY_AFTER_HEADER_RE = re.compile(r"(?im)(?:^|[\s,;(])retry[- ]after\s*[:=]")
TRANSPORT_CONTRACT = get_transport_contract("gh_pr_label_rest.py")


def check_merge_ready_carriers(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    live_head: str,
    live_base: str,
) -> dict[str, Any]:
    """Load the canonical carrier guard only when needed, failing closed if unavailable."""
    try:
        from scripts.dev.pr_carrier_gate import check_merge_ready_carriers as check_carriers
    except ImportError as exc:
        return {"status": "error", "error": f"could not load merge-ready carrier checker: {exc}"}
    return check_carriers(number, repo=repo, live_head=live_head, live_base=live_base)


def _label_name_error(raw_name: object, *, context: str) -> str | None:
    """Return a validation error for a label name, or ``None`` when it is safe to transport."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        return f"{context} must be non-empty text"
    if not raw_name.isprintable():
        return f"{context} must be printable text"
    return None


def _is_absent_label_delete(result: subprocess.CompletedProcess[str]) -> bool:
    """Recognize only GitHub's idempotent missing-label DELETE response."""
    if result.returncode == 0:
        return False
    detail = (result.stderr or result.stdout).strip().lower()
    return "http 404" in detail and "label does not exist" in detail


def _is_rate_limit_failure(result: subprocess.CompletedProcess[str]) -> bool:
    """Return whether a failed gh result reports a REST rate or secondary limit."""
    if result.returncode == 0:
        return False
    text = _response_detail(result).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _response_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Combine both process streams without dropping a rate-limit diagnostic."""
    streams = [stream.strip() for stream in (result.stderr, result.stdout) if stream.strip()]
    return "\n".join(dict.fromkeys(streams))


def _parse_bounded_decimal(raw_value: str, *, maximum: int) -> int | None:
    """Parse an ASCII decimal value only when it fits the supplied safety bound."""
    if not raw_value or not raw_value.isascii() or not raw_value.isdigit():
        return None
    maximum_text = str(maximum)
    if len(raw_value) > len(maximum_text):
        return None
    significant = raw_value.lstrip("0") or "0"
    if len(significant) > len(maximum_text) or (
        len(significant) == len(maximum_text) and significant > maximum_text
    ):
        return None
    try:
        return int(significant)
    except (OverflowError, ValueError):
        return None


def _retry_after_seconds(result: subprocess.CompletedProcess[str]) -> int | None:
    """Extract a bounded numeric ``Retry-After`` header value, if present."""
    match = _RATE_LIMIT_RETRY_AFTER_RE.search(_response_detail(result))
    if match is None:
        return None
    return _parse_bounded_decimal(match.group(1), maximum=MAX_RETRY_AFTER_SECONDS)


def _is_secondary_rate_limit(result: subprocess.CompletedProcess[str]) -> bool:
    """Return whether a response has explicit or structurally valid secondary evidence."""
    detail = _response_detail(result).lower()
    return any(marker in detail for marker in _SECONDARY_RATE_LIMIT_MARKERS) or (
        _retry_after_seconds(result) is not None
    )


def _retry_after_utc(seconds: int) -> str | None:
    """Render a bounded retry delay without allowing timestamp conversion errors to escape."""
    if type(seconds) is not int or seconds < 0 or seconds > MAX_RETRY_AFTER_SECONDS:
        return None
    try:
        now = time.time()
        if not math.isfinite(now):
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + seconds))
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _parse_rate_limit_reset_epoch(text: str, *, now: float) -> int | None:
    """Parse an absolute reset epoch or relative Retry-After delay, else None."""
    reset_match = _RATE_LIMIT_RESET_RE.search(text or "")
    if reset_match is not None:
        return _parse_bounded_decimal(reset_match.group(1), maximum=MAX_RATE_LIMIT_EPOCH)
    retry_match = _RATE_LIMIT_RETRY_AFTER_RE.search(text or "")
    if retry_match is not None:
        seconds = _parse_bounded_decimal(retry_match.group(1), maximum=MAX_RETRY_AFTER_SECONDS)
        if seconds is None:
            return None
        try:
            if not math.isfinite(now):
                return None
            now_epoch = math.floor(now)
            reset_at = now_epoch + seconds
        except (OverflowError, TypeError, ValueError):
            return None
        return reset_at if 0 <= reset_at <= MAX_RATE_LIMIT_EPOCH else None
    return None


def _rate_limit_evidence(
    result: subprocess.CompletedProcess[str], *, now: float
) -> dict[str, Any] | None:
    """Return reset evidence for a rate-limited mutation failure, else None."""
    if not _is_rate_limit_failure(result):
        return None
    text = _response_detail(result)
    reset_header = _RATE_LIMIT_RESET_HEADER_RE.search(text)
    retry_after_header = _RATE_LIMIT_RETRY_AFTER_HEADER_RE.search(text)
    retry_after_seconds = _retry_after_seconds(result)
    retry_after_utc = (
        _retry_after_utc(retry_after_seconds) if retry_after_seconds is not None else None
    )
    reset_invalid = (
        (reset_header is not None and _parse_rate_limit_reset_epoch(text, now=now) is None)
        or (retry_after_header is not None and retry_after_seconds is None)
        or (retry_after_seconds is not None and retry_after_utc is None)
    )
    secondary = _is_secondary_rate_limit(result)
    if secondary and retry_after_header is None:
        reset_invalid = True
    return {
        "reset_at": _parse_rate_limit_reset_epoch(text, now=now),
        "_rate_limit_kind": "secondary" if secondary else "core",
        "_rate_limit_reset_invalid": reset_invalid,
        "_retry_after_seconds": retry_after_seconds,
        "_retry_after_utc": retry_after_utc,
    }


def _fetch_core_rate_limit_reset_at() -> int | None:
    """Return the canonical REST core-quota reset epoch, or None when unavailable."""
    try:
        from scripts.dev.github_quota import fetch_core_reset_at
    except ImportError:
        return None
    return fetch_core_reset_at()


def _safe_reset_epoch(value: object) -> int | None:
    """Return a reset epoch safe for arithmetic and JSON receipts, or ``None``."""
    if type(value) is not int or value < 0 or value > MAX_RATE_LIMIT_EPOCH:
        return None
    return value


def _bounded_wait_seconds(reset_at: int, now: float) -> float | None:
    """Return a finite wait only when the reset is inside the retry bound."""
    try:
        if not math.isfinite(now):
            return None
        now_floor = math.floor(now)
        if reset_at > now_floor + int(RATE_LIMIT_MAX_WAIT_SECONDS):
            return None
        wait_seconds = max(0.0, reset_at - now)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(wait_seconds) or wait_seconds > RATE_LIMIT_MAX_WAIT_SECONDS:
        return None
    return wait_seconds


def _blocked_label_mutation_result(
    result: dict[str, Any],
    *,
    action: str,
    number: int,
    repo: str,
    label: str,
    attempts: int,
    reset_at: int | None,
) -> dict[str, Any]:
    """Build the stable blocked receipt without leaking unsafe parser values."""
    blocked = {
        "status": BLOCKED_STATUS,
        "reason": "rate_limited",
        "action": action,
        "number": number,
        "repo": repo,
        "label": label,
        "attempts": attempts,
        "reset_at": reset_at,
        "error": result.get("error", "GitHub rate limit blocked the label mutation"),
    }
    rate_limit_kind = result.get("_rate_limit_kind")
    if rate_limit_kind in {"core", "secondary"}:
        blocked["rate_limit_kind"] = rate_limit_kind
    if rate_limit_kind == "secondary":
        blocked["retry_after_seconds"] = result.get("_retry_after_seconds")
        blocked["retry_after_utc"] = result.get("_retry_after_utc")
    return blocked


def _run_bounded_label_mutation(
    attempt: Callable[[], dict[str, Any]],
    *,
    action: str,
    number: int,
    repo: str,
    label: str,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    reset_fetcher: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """Run one label mutation with a bounded rate-limit retry policy.

    Only a rate-limit failure is retryable, and only when a reset time is known
    and falls inside ``RATE_LIMIT_MAX_WAIT_SECONDS`` under an attempt cap of
    ``RATE_LIMIT_MAX_ATTEMPTS``. Unknown reset evidence blocks fail-closed with
    an explicit receipt. Guarded callers pass an ``attempt`` that re-runs their
    live exact-head/base CAS preflight, so a retry can never mutate a moved PR.
    """
    now_fn = now or time.time
    sleep_fn = sleep or time.sleep
    reset_fn = reset_fetcher or _fetch_core_rate_limit_reset_at
    attempts = 0
    while True:
        attempts += 1
        result = attempt()
        if result.get("status") != RATE_LIMIT_STATUS:
            if result.get("status") == "ok" and attempts > 1:
                result = {**result, "attempts": attempts}
            return result
        reset_at = _safe_reset_epoch(result.get("reset_at"))
        if result.get("_rate_limit_reset_invalid"):
            return _blocked_label_mutation_result(
                result,
                action=action,
                number=number,
                repo=repo,
                label=label,
                attempts=attempts,
                reset_at=None,
            )
        rate_limit_kind = result.get("_rate_limit_kind", "core")
        if reset_at is None and rate_limit_kind != "secondary":
            try:
                reset_at = _safe_reset_epoch(reset_fn())
            except (OverflowError, OSError, TypeError, ValueError):
                reset_at = None
        if reset_at is not None and attempts < RATE_LIMIT_MAX_ATTEMPTS:
            try:
                wait_seconds = _bounded_wait_seconds(reset_at, now_fn())
            except (OverflowError, OSError, TypeError, ValueError):
                wait_seconds = None
            if wait_seconds is not None:
                sleep_fn(min(max(1.0, wait_seconds + 1.0), RATE_LIMIT_MAX_WAIT_SECONDS))
                continue
        return _blocked_label_mutation_result(
            result,
            action=action,
            number=number,
            repo=repo,
            label=label,
            attempts=attempts,
            reset_at=reset_at,
        )


def _get_label_names(number: int, *, repo: str = DEFAULT_REPO, timeout: int = 30) -> dict[str, Any]:
    """Return a complete, strictly validated label inventory, or an error dict."""
    names: list[str] = []
    seen_names: set[str] = set()
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
            raw_name = row.get("name")
            if (
                name_error := _label_name_error(
                    raw_name, context=f"malformed label row on page {page}: name"
                )
            ) is not None:
                return {
                    "status": "error",
                    "error": name_error,
                }
            assert isinstance(raw_name, str)
            name = raw_name
            if name in seen_names:
                return {
                    "status": "error",
                    "error": f"duplicate label row on page {page}: {name!r}",
                }
            seen_names.add(name)
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
    for name in labels:
        if (
            name_error := _label_name_error(name, context="label helper list result labels")
        ) is not None:
            raise ValueError(name_error)
    if len(set(labels)) != len(labels):
        raise ValueError("label helper list result labels must be distinct")


def _validate_write_result(result: dict[str, Any], *, label: str | None) -> None:
    """Validate the requested label in a successful add/remove result."""
    label_error = _label_name_error(label, context="expected label")
    if label_error is not None:
        raise ValueError(label_error)
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
    disposition) withholds ``merge-ready`` fail-closed (issue #7610). The
    preflight is re-run before every mutation attempt, including each bounded
    rate-limit retry, so a moved head or base can never receive the write.
    """
    if expected_head_sha is None or expected_base_sha is None:
        return {
            "status": "error",
            "error": "PR label writes require both expected_head_sha and expected_base_sha",
        }
    try:
        with pr_write_lock(repo, number):

            def _attempt() -> dict[str, Any]:
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

            return _run_bounded_label_mutation(
                _attempt,
                action="add",
                number=number,
                repo=repo,
                label="merge-ready",
            )
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def _guarded_pr_label_write(
    number: int,
    *,
    repo: str,
    label: str,
    action: str,
    expected_head_sha: str | None,
    expected_base_sha: str | None,
    write: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run a non-``merge-ready`` PR label write under exact live head/base CAS."""
    if expected_head_sha is None or expected_base_sha is None:
        return {
            "status": "error",
            "error": "PR label writes require both expected_head_sha and expected_base_sha",
        }
    try:
        with pr_write_lock(repo, number):

            def _attempt() -> dict[str, Any]:
                guard = guard_pr_write(
                    number,
                    repo=repo,
                    expected_head_sha=expected_head_sha,
                    expected_base_sha=expected_base_sha,
                    operation=f"label_{action}",
                )
                if guard["status"] != "ok":
                    return guard
                return write()

            return _run_bounded_label_mutation(
                _attempt,
                action=action,
                number=number,
                repo=repo,
                label=label,
            )
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def _label_target_error(
    label: str,
    *,
    target: object,
    expected_head_sha: str | None,
    expected_base_sha: str | None,
) -> str | None:
    """Validate the issue/PR target and its required compare-and-swap inputs."""
    if target not in ("issue", "pr"):
        return f"unsupported label target: {target!r}"
    if target == "issue":
        if expected_head_sha is not None or expected_base_sha is not None:
            return "expected PR SHAs require target=pr"
        if label == "merge-ready":
            return "merge-ready labels require target=pr"
        return None
    if expected_head_sha is None or expected_base_sha is None:
        return "PR label writes require both expected_head_sha and expected_base_sha"
    return None


def _label_request_error(
    number: int,
    label: str,
    *,
    target: object,
    expected_head_sha: str | None,
    expected_base_sha: str | None,
) -> str | None:
    """Validate the common label mutation inputs before any transport call."""
    if type(number) is not int or number < 1:
        return f"issue/PR number must be positive, got {number}"
    if (label_error := _label_name_error(label, context="label")) is not None:
        return label_error
    return _label_target_error(
        label,
        target=target,
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
    )


def add_label(
    number: int,
    label: str,
    *,
    repo: str = DEFAULT_REPO,
    target: Literal["issue", "pr"] = "issue",
    expected_head_sha: str | None = None,
    expected_base_sha: str | None = None,
) -> dict[str, Any]:
    """Add *label* to issue/PR *number* and verify it was applied.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if (
        request_error := _label_request_error(
            number,
            label,
            target=target,
            expected_head_sha=expected_head_sha,
            expected_base_sha=expected_base_sha,
        )
    ) is not None:
        return {"status": "error", "error": request_error}

    def _write() -> dict[str, Any]:
        """Apply and verify one label after any required PR preflight."""
        path = f"repos/{repo}/issues/{number}/labels"
        result = _gh_api_post(path, {"labels": [label]})
        if result.returncode != 0:
            detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
            if (rate := _rate_limit_evidence(result, now=time.time())) is not None:
                return {
                    "status": RATE_LIMIT_STATUS,
                    "error": f"label add failed: {detail}",
                    **rate,
                }
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

    if target == "issue":
        return _run_bounded_label_mutation(
            _write,
            action="add",
            number=number,
            repo=repo,
            label=label,
        )
    if label == "merge-ready":
        return _guarded_merge_ready_write(
            number,
            repo=repo,
            expected_head_sha=expected_head_sha,
            expected_base_sha=expected_base_sha,
            write=_write,
        )
    return _guarded_pr_label_write(
        number,
        repo=repo,
        label=label,
        action="add",
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
        write=_write,
    )


def remove_label(
    number: int,
    label: str,
    *,
    repo: str = DEFAULT_REPO,
    target: Literal["issue", "pr"] = "issue",
    expected_head_sha: str | None = None,
    expected_base_sha: str | None = None,
) -> dict[str, Any]:
    """Remove *label* from issue/PR *number* and verify it was removed.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if (
        request_error := _label_request_error(
            number,
            label,
            target=target,
            expected_head_sha=expected_head_sha,
            expected_base_sha=expected_base_sha,
        )
    ) is not None:
        return {"status": "error", "error": request_error}

    def _remove() -> dict[str, Any]:
        """Delete and verify one label, classifying rate-limit failures."""
        path = f"repos/{repo}/issues/{number}/labels/{quote(label, safe='')}"
        result = _gh_api_delete(path)
        idempotent = _is_absent_label_delete(result)
        if result.returncode != 0 and not idempotent:
            detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
            if (rate := _rate_limit_evidence(result, now=time.time())) is not None:
                return {
                    "status": RATE_LIMIT_STATUS,
                    "error": f"label remove failed: {detail}",
                    **rate,
                }
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

    if target == "issue":
        return _run_bounded_label_mutation(
            _remove,
            action="remove",
            number=number,
            repo=repo,
            label=label,
        )
    return _guarded_pr_label_write(
        number,
        repo=repo,
        label=label,
        action="remove",
        expected_head_sha=expected_head_sha,
        expected_base_sha=expected_base_sha,
        write=_remove,
    )


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
        "--target",
        choices=("issue", "pr"),
        default="issue",
        help="Target kind; PR writes require both expected head and base SHAs.",
    )
    parser.add_argument(
        "--expected-head-sha",
        help="Full PR head SHA required for an exact-head PR label write.",
    )
    parser.add_argument(
        "--expected-base-sha",
        help="Full PR base SHA required for an exact-head PR label write.",
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
            target=args.target,
            expected_head_sha=args.expected_head_sha,
            expected_base_sha=args.expected_base_sha,
        )
    else:
        result = remove_label(
            args.number,
            args.label,
            repo=args.repo,
            target=args.target,
            expected_head_sha=args.expected_head_sha,
            expected_base_sha=args.expected_base_sha,
        )

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
