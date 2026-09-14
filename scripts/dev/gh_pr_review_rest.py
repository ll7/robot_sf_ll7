#!/usr/bin/env python3
"""Publish an exact-head pull-request review through the REST API.

When a caller has already computed the exact title/body metadata digest, pass
``--expected-metadata-digest`` so the live PR metadata is re-read and compared
before the review POST. A mismatched ``pr-metadata`` carrier is an error
(CLI exit 1), while a mismatch against the live PR title/body metadata is a
safe stale-state skip (CLI exit 2).

A rejected review POST returns structured evidence instead of an opaque error:
the endpoint, HTTP status, bounded stderr/response excerpts, and an explicit
``fallback`` handoff describing the idempotent top-level comment route. That
route runs only with ``--fallback-comment`` (never automatically), marks the
comment with ``REVIEW_FALLBACK_MARKER`` plus the expected head so a repeat run
cannot duplicate it, and reports ``authoritative_for_review: false`` because a
comment is not a formal review event. An unavailable endpoint exits 2 once the
fallback is recorded; a real review publication exits 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Direct execution must resolve this checkout's transport and write guards,
    # ahead of any competing checkout or editable installation on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev._gh_rest import gh_api_comments_get as _gh_api_comments_get
from scripts.dev._gh_rest import gh_api_get as _gh_api_get
from scripts.dev._gh_rest import gh_api_review_post as _gh_api_post
from scripts.dev._gh_rest import parse_json as _parse_json
from scripts.dev.github_transport_policy import get_transport_contract
from scripts.dev.pr_carrier_gate import _declared_base_sha, extract_full_shas
from scripts.dev.pr_metadata import extract_metadata_digests, metadata_digest
from scripts.dev.pr_write_guard import DEFAULT_REPO, guard_pr_write, pr_write_lock

REVIEW_EVENTS = ("COMMENT", "APPROVE", "REQUEST_CHANGES")
SELF_AUTHORED_REVIEW_STATUS = "review_skipped_self_authored"
METADATA_DIGEST_STATUS = "review_skipped_stale_state"
REVIEW_FALLBACK_STATUS = "review_fallback_comment_recorded"
FULL_METADATA_DIGEST_LENGTH = 64
REVIEW_FALLBACK_MARKER = "<!-- exact-head-review-fallback:v1 -->"
REVIEW_ENDPOINT_FAILURE_STATUSES = frozenset({422, 500, 502, 503, 504})
REVIEW_AUTHORIZATION_STATUSES = frozenset({401, 403})
_EXCERPT_LIMIT = 400
_COMMENTS_PAGE_SIZE = 100
_HTTP_STATUS_PATTERN = re.compile(r"\(HTTP (\d{3})\)|\bHTTP (\d{3})\b")
TRANSPORT_CONTRACT = get_transport_contract("gh_pr_review_rest.py")


def _read_body_file(body_file: Path) -> tuple[str | None, str | None]:
    """Read a non-empty UTF-8 review body file."""
    try:
        body = body_file.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read review body file {body_file}: {exc}"
    if not body.strip():
        return None, "review body file must not be empty"
    return body, None


def _validate_review_body_shas(
    body: str,
    *,
    expected_head_sha: str,
    observed_base_sha: str | None,
) -> dict[str, Any] | None:
    """Require the live head and reject an unverifiable or mismatched declared base."""
    cited_shas = {sha.lower() for sha in extract_full_shas(body)}
    if expected_head_sha.lower() not in cited_shas:
        return {
            "status": "error",
            "error": f"review body does not cite the expected head SHA {expected_head_sha}",
        }

    declared_base = _declared_base_sha(body)
    if declared_base is None:
        return None
    if not isinstance(observed_base_sha, str) or not observed_base_sha:
        return {
            "status": "error",
            "error": (
                f"review body declares base SHA {declared_base}, but the live base SHA "
                "is unavailable"
            ),
        }
    if declared_base.lower() != observed_base_sha.lower():
        return {
            "status": "error",
            "error": (
                f"review body declares base SHA {declared_base} which does not match "
                f"live base SHA {observed_base_sha}"
            ),
        }
    return None


def _read_authenticated_actor() -> tuple[str | None, dict[str, Any] | None]:
    """Read the authenticated GitHub login, failing closed on uncertainty."""
    result = _gh_api_get("user")
    payload, error = _parse_json(result, what="authenticated GitHub actor read")
    if error:
        return None, {"status": "error", "error": error}
    if not isinstance(payload, dict):
        return None, {"status": "error", "error": "authenticated actor payload was not an object"}
    login = payload.get("login")
    if not isinstance(login, str) or not login.strip():
        return None, {"status": "error", "error": "authenticated actor payload has no login"}
    return login.strip(), None


def _read_live_metadata_digest(
    number: int, *, repo: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Read the live PR title/body and return its exact canonical metadata digest."""
    result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    payload, error = _parse_json(result, what=f"PR {number} metadata read")
    if error:
        return None, {"status": "error", "error": error}
    if not isinstance(payload, dict):
        return None, {"status": "error", "error": "PR metadata payload was not an object"}
    title = payload.get("title")
    body = payload.get("body")
    if not isinstance(title, str):
        return None, {"status": "error", "error": "PR metadata payload has no title"}
    if body is None:
        body = ""
    if not isinstance(body, str):
        return None, {"status": "error", "error": "PR metadata payload has malformed body"}
    return metadata_digest(title, body), None


def _validate_expected_metadata_digest(expected_digest: str | None) -> str | None:
    """Validate an optional exact title/body metadata digest supplied by the caller."""
    if expected_digest is None:
        return None
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != FULL_METADATA_DIGEST_LENGTH
        or any(character not in "0123456789abcdefABCDEF" for character in expected_digest)
    ):
        return "expected_metadata_digest must be a full 64-character hexadecimal digest"
    return None


def _guard_review(
    number: int,
    *,
    repo: str,
    expected_head_sha: str,
    event: str,
) -> dict[str, Any]:
    """Run the event-specific exact-head guard."""
    guard_kwargs: dict[str, Any] = {
        "repo": repo,
        "expected_head_sha": expected_head_sha,
        "operation": "commented_review" if event == "COMMENT" else "review",
    }
    if event == "REQUEST_CHANGES":
        guard_kwargs["include_author"] = True
    return guard_pr_write(number, **guard_kwargs)


def _self_authored_guidance(
    guard: dict[str, Any],
    *,
    authenticated_actor_login: str,
) -> dict[str, Any] | None:
    """Return explicit comment guidance for a self-authored request, if needed."""
    observed_author_login = guard.get("observed_author_login")
    if not isinstance(observed_author_login, str) or not observed_author_login:
        return {
            "status": "error",
            "error": "PR write-state guard did not return an author login",
        }
    if authenticated_actor_login.casefold() != observed_author_login.casefold():
        return None
    return {
        **guard,
        "status": SELF_AUTHORED_REVIEW_STATUS,
        "reason": "self_authored_request_changes_forbidden",
        "fallback_event": "COMMENT",
        "automatic_fallback": False,
        "body_preserved": True,
        "authenticated_actor_login": authenticated_actor_login,
    }


def _review_preflight(
    number: int,
    body: str,
    *,
    repo: str,
    expected_head_sha: str,
    event: str,
) -> dict[str, Any]:
    """Run actor, exact-head, body-carrier, and self-authored checks before a write."""
    authenticated_actor_login: str | None = None
    if event == "REQUEST_CHANGES":
        authenticated_actor_login, actor_error = _read_authenticated_actor()
        if actor_error is not None:
            return actor_error
        assert authenticated_actor_login is not None

    guard = _guard_review(
        number,
        repo=repo,
        expected_head_sha=expected_head_sha,
        event=event,
    )
    if guard["status"] != "ok":
        return guard

    body_sha_error = _validate_review_body_shas(
        body,
        expected_head_sha=expected_head_sha,
        observed_base_sha=guard.get("observed_base_sha"),
    )
    if body_sha_error is not None:
        return body_sha_error

    if authenticated_actor_login is not None:
        guidance = _self_authored_guidance(
            guard,
            authenticated_actor_login=authenticated_actor_login,
        )
        if guidance is not None:
            return guidance
    return guard


def _metadata_digest_preflight(
    number: int,
    *,
    repo: str,
    expected_digest: str | None,
    review_body: str,
    preflight: dict[str, Any],
) -> dict[str, Any] | None:
    """Enforce carrier and live-metadata digests before publishing a review.

    A mismatched review-body carrier returns ``status="error"`` (CLI exit 1).
    A changed live PR title/body returns ``METADATA_DIGEST_STATUS`` (CLI exit 2).
    """
    if expected_digest is None:
        return None
    body_digests = extract_metadata_digests(review_body)
    mismatched_body_digests = [
        digest for digest in body_digests if digest.casefold() != expected_digest.casefold()
    ]
    if mismatched_body_digests:
        return {
            **preflight,
            "status": "error",
            "error": (
                "review body metadata digest does not match expected metadata digest: "
                + ", ".join(mismatched_body_digests)
            ),
            "expected_metadata_digest": expected_digest.lower(),
            "observed_review_body_metadata_digests": mismatched_body_digests,
        }
    observed_digest, metadata_error = _read_live_metadata_digest(number, repo=repo)
    if metadata_error is not None:
        return metadata_error
    assert observed_digest is not None
    if observed_digest.casefold() == expected_digest.casefold():
        return None
    return {
        **preflight,
        "status": METADATA_DIGEST_STATUS,
        "reason": "metadata_digest_changed",
        "expected_metadata_digest": expected_digest.lower(),
        "observed_metadata_digest": observed_digest,
    }


def _bounded_excerpt(value: str, limit: int = _EXCERPT_LIMIT) -> str:
    """Return a single-line, length-bounded excerpt for structured evidence."""
    text = " ".join(str(value or "").split())
    return text[:limit]


def _http_status(texts: tuple[str, ...]) -> int | None:
    """Return the first HTTP status found in the given texts, if any."""
    for text in texts:
        match = _HTTP_STATUS_PATTERN.search(text or "")
        if match:
            return int(match.group(1) or match.group(2))
    return None


def _fallback_handoff(number: int, *, repo: str, expected_head_sha: str) -> dict[str, Any]:
    """Describe the explicit, idempotent top-level-comment fallback route."""
    return {
        "kind": "top_level_comment",
        "endpoint": f"repos/{repo}/issues/{number}/comments",
        "event": "COMMENT",
        "automatic": False,
        "authoritative_for_review": False,
        "idempotency_marker": REVIEW_FALLBACK_MARKER,
        "expected_head_sha": expected_head_sha,
        "cli_flag": "--fallback-comment",
    }


def _review_publication_failure(
    result: Any,
    *,
    repo: str,
    number: int,
    expected_head_sha: str,
    event: str,
) -> dict[str, Any]:
    """Return structured, actionable evidence for a failed review POST."""
    stderr = str(getattr(result, "stderr", "") or "")
    stdout = str(getattr(result, "stdout", "") or "")
    status = _http_status((stderr, stdout))
    validation_error = False
    if status == 422:
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError:
            response = None
        validation_error = isinstance(response, dict) and bool(response.get("errors"))
    authorization_error = status in REVIEW_AUTHORIZATION_STATUSES
    unavailable = (
        not authorization_error
        and not validation_error
        and (status is None or status in REVIEW_ENDPOINT_FAILURE_STATUSES)
    )
    if authorization_error:
        error_class = "review_authorization_failed"
    elif validation_error:
        error_class = "review_validation_failed"
    else:
        error_class = "review_publication_failed"
    return {
        "status": "error",
        "error": f"PR {number} review publication failed: {_bounded_excerpt(stderr or stdout)}",
        "error_class": error_class,
        "endpoint": f"repos/{repo}/pulls/{number}/reviews",
        "http_status": status,
        "returncode": getattr(result, "returncode", None),
        "stderr_excerpt": _bounded_excerpt(stderr),
        "response_body_excerpt": _bounded_excerpt(stdout),
        "expected_head_sha": expected_head_sha,
        "event": event,
        "fallback_recommended": unavailable,
        "fallback": _fallback_handoff(number, repo=repo, expected_head_sha=expected_head_sha),
    }


def _fallback_marker(head_sha: str) -> str:
    """Return the idempotency marker binding one fallback comment to one head."""
    return f"{REVIEW_FALLBACK_MARKER} head: {head_sha.lower()}"


def _existing_fallback_comment(
    number: int, *, repo: str, expected_head_sha: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Return an existing fallback comment for this head, or a read error."""
    marker = _fallback_marker(expected_head_sha)
    page = 1
    while True:
        result = _gh_api_comments_get(
            f"repos/{repo}/issues/{number}/comments?per_page={_COMMENTS_PAGE_SIZE}&page={page}"
        )
        payload, error = _parse_json(result, what=f"PR {number} fallback comment read")
        if error:
            return None, error
        if not isinstance(payload, list):
            return None, f"PR {number} fallback comment read was not a list"
        for comment in payload:
            body = str((comment or {}).get("body") or "") if isinstance(comment, dict) else ""
            if marker in body:
                return {
                    "comment_id": comment.get("id"),
                    "url": str((comment or {}).get("html_url", "")),
                }, None
        if len(payload) < _COMMENTS_PAGE_SIZE:
            break
        page += 1
    return None, None


def _record_fallback_comment(
    number: int,
    body: str,
    *,
    repo: str,
    expected_head_sha: str,
) -> dict[str, Any]:
    """Post (or find) the idempotent top-level comment fallback for one head."""
    existing, read_error = _existing_fallback_comment(
        number, repo=repo, expected_head_sha=expected_head_sha
    )
    if read_error is not None:
        return {"status": "error", "error": read_error, "error_class": "fallback_read_failed"}
    if existing is not None:
        return {
            "status": REVIEW_FALLBACK_STATUS,
            "number": number,
            "repo": repo,
            "expected_head_sha": expected_head_sha,
            "authoritative_for_review": False,
            "duplicate_prevented": True,
            **existing,
        }
    marked_body = f"{_fallback_marker(expected_head_sha)}\n\n{body}"
    result = _gh_api_post(f"repos/{repo}/issues/{number}/comments", {"body": marked_body})
    payload, error = _parse_json(result, what=f"PR {number} fallback comment publication")
    if error:
        return {"status": "error", "error": error, "error_class": "fallback_publication_failed"}
    if not isinstance(payload, dict):
        return {
            "status": "error",
            "error": "fallback comment response was not an object",
            "error_class": "fallback_publication_failed",
        }
    comment_id = payload.get("id")
    if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id < 1:
        return {
            "status": "error",
            "error": "fallback comment response had no numeric id",
            "error_class": "fallback_response_unrecognized",
        }
    return {
        "status": REVIEW_FALLBACK_STATUS,
        "number": number,
        "repo": repo,
        "expected_head_sha": expected_head_sha,
        "authoritative_for_review": False,
        "duplicate_prevented": False,
        "comment_id": comment_id,
        "url": str(payload.get("html_url", "")),
    }


def _publish_review(
    number: int,
    body: str,
    *,
    repo: str,
    expected_head_sha: str,
    event: str,
) -> dict[str, Any]:
    """POST a review and validate the response's exact commit binding."""
    result = _gh_api_post(
        f"repos/{repo}/pulls/{number}/reviews",
        {"body": body, "event": event, "commit_id": expected_head_sha},
    )
    payload, error = _parse_json(result, what=f"PR {number} review publication")
    if error:
        return _review_publication_failure(
            result,
            repo=repo,
            number=number,
            expected_head_sha=expected_head_sha,
            event=event,
        )
    if not isinstance(payload, dict):
        return {"status": "error", "error": "review response was not an object"}
    response_commit = payload.get("commit_id")
    if response_commit and str(response_commit).lower() != expected_head_sha.lower():
        return {
            "status": "error",
            "error": "review response was bound to a different commit",
            "expected_head_sha": expected_head_sha,
            "observed_review_commit_id": response_commit,
        }
    review_id = payload.get("id")
    if isinstance(review_id, bool) or not isinstance(review_id, int) or review_id < 1:
        return {"status": "error", "error": "review response had no numeric id"}
    return {
        "status": "ok",
        "number": number,
        "repo": repo,
        "event": event,
        "head_sha": expected_head_sha,
        "review_id": review_id,
        "url": str(payload.get("html_url", "")),
    }


def post_review(
    number: int,
    body_file: Path,
    *,
    expected_head_sha: str,
    event: str = "COMMENT",
    repo: str = DEFAULT_REPO,
    expected_metadata_digest: str | None = None,
    fallback_comment: bool = False,
) -> dict[str, Any]:
    """Post one review only when the PR state and optional metadata digest are current.

    With ``fallback_comment``, an unavailable review endpoint records the review
    body as an idempotent top-level comment instead; the result then reports
    ``authoritative_for_review: false`` and never substitutes for a review.
    """
    body, body_error = _read_body_file(body_file)
    if body_error:
        return {"status": "error", "error": body_error}
    assert body is not None
    event = event.upper()
    if event not in REVIEW_EVENTS:
        return {
            "status": "error",
            "error": f"event must be one of {', '.join(REVIEW_EVENTS)}",
        }
    digest_error = _validate_expected_metadata_digest(expected_metadata_digest)
    if digest_error is not None:
        return {"status": "error", "error": digest_error}

    try:
        with pr_write_lock(repo, number):
            preflight = _review_preflight(
                number,
                body,
                repo=repo,
                expected_head_sha=expected_head_sha,
                event=event,
            )
            if preflight["status"] != "ok":
                return preflight

            metadata_preflight = _metadata_digest_preflight(
                number,
                repo=repo,
                expected_digest=expected_metadata_digest,
                review_body=body,
                preflight=preflight,
            )
            if metadata_preflight is not None:
                return metadata_preflight

            published = _publish_review(
                number,
                body,
                repo=repo,
                expected_head_sha=expected_head_sha,
                event=event,
            )
            if (
                fallback_comment
                and published.get("status") == "error"
                and published.get("fallback_recommended")
            ):
                fallback = _record_fallback_comment(
                    number, body, repo=repo, expected_head_sha=expected_head_sha
                )
                return {**published, "fallback_result": fallback}
            return published
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Pull-request number to review.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/repo to update.")
    parser.add_argument("--body-file", type=Path, required=True, help="Review body file.")
    parser.add_argument(
        "--expected-head-sha",
        required=True,
        help="Full 40-character PR head SHA captured by the review lane.",
    )
    parser.add_argument(
        "--expected-metadata-digest",
        help=(
            "Optional full 64-character SHA-256 digest of the exact live PR title/body; "
            "a carrier mismatch errors (exit 1), while a live metadata mismatch "
            "skips publication (exit 2)."
        ),
    )
    parser.add_argument("--event", choices=REVIEW_EVENTS, default="COMMENT")
    parser.add_argument(
        "--fallback-comment",
        action="store_true",
        help=(
            "When the review endpoint is unavailable, record the review body as an "
            "idempotent top-level PR comment (never a review substitute; exit 2)."
        ),
    )
    return parser


def _exit_code(result: dict[str, Any]) -> int:
    """Map a result payload to the CLI exit code."""
    status = result.get("status")
    if status == "ok":
        return 0
    if status in {"review_skipped_stale_state", SELF_AUTHORED_REVIEW_STATUS}:
        return 2
    fallback = result.get("fallback_result")
    if isinstance(fallback, dict) and fallback.get("status") == REVIEW_FALLBACK_STATUS:
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the guarded REST review writer and emit one JSON result."""
    args = _build_parser().parse_args(argv)
    result = post_review(
        args.number,
        args.body_file,
        expected_head_sha=args.expected_head_sha,
        event=args.event,
        repo=args.repo,
        expected_metadata_digest=args.expected_metadata_digest,
        fallback_comment=args.fallback_comment,
    )
    print(
        json.dumps(result, sort_keys=True),
        file=sys.stdout if result.get("status") == "ok" else sys.stderr,
    )
    return _exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
