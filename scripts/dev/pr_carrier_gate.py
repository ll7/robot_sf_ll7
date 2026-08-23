#!/usr/bin/env python3
"""Fail-closed carrier binding checks for the merge-ready label gate.

Why this exists
---------------
``pr_write_guard`` guarantees that a merge-ready write targets the exact live
PR head, but it cannot tell whether the review *evidence* (the PR body and the
exact-head self-review comments) is bound to that same live state. Issue #7610
exposed a merged PR whose body and review carriers cited an older head/base
while the live PR object had moved on, and which carried ``merge-ready``
despite a "not current-base merge evidence" refresh note.

This module closes that class of drift:

- a top-level review carrier is valid only when it names the exact live head
  and, when it declares a base, the exact live base;
- a human ``COMMENTED`` review from the PR reviews endpoint is valid only when
  its body names that live head/base and its ``commit_id`` is the live head;
- a body or comment carrying stale-narrative sentinels (including
  "not current-base merge evidence" and pending domain-review dispositions)
  invalidates the merge-ready disposition;
- a body carrying ``gate-verdict``/``base-policy``/``Exact head`` SHA carriers
  must name the live head, so a stale body cannot back a current-head label
  write;
- any failure returns a structured ``error`` and the caller must not write.

All reads go through the shared REST helpers and fail closed on transport or
payload uncertainty. A valid top-level carrier remains a compatibility path
when the review endpoint is unavailable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from scripts.dev._gh_rest import gh_api_get as _gh_api_get
from scripts.dev._gh_rest import parse_json as _parse_json
from scripts.dev.pr_loop_policy import extract_sha_carriers, invalid_sha_carriers
from scripts.dev.pr_metadata import find_not_ready_body_sentinels

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_REPO = "ll7/robot_sf_ll7"
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

REVIEW_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"exact[- ]head\s+self[- ]review", re.IGNORECASE),
    re.compile(r"exact[- ]head\s+implementation[- ]review", re.IGNORECASE),
    re.compile(r"self[- ]review[^\n]{0,40}exact\s+head", re.IGNORECASE),
)

STALE_CARRIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"not\s+current[- ]base\s+merge\s+evidence", re.IGNORECASE),
    re.compile(r"pending[- ]domain[- ]review", re.IGNORECASE),
    re.compile(r"pending\s+domain(?:\s+aware)?\s+(?:approval|review)", re.IGNORECASE),
    re.compile(r"domain[- ]?aware\s+approval\s+remain\w*\s+pending", re.IGNORECASE),
)

_BOT_AUTHORS = frozenset(
    {"github-actions[bot]", "coderabbitai[bot]", "dependabot[bot]", "renovate[bot]"}
)


def extract_full_shas(text: str) -> list[str]:
    """Return every full 40-hex SHA contained in *text*, in order."""
    if not isinstance(text, str):
        return []
    return [match.group(0) for match in re.finditer(r"[0-9a-fA-F]{40}", text)]


def is_review_carrier_comment(comment: str) -> bool:
    """Return True when *comment* is an exact-head self-review carrier."""
    if not isinstance(comment, str):
        return False
    return any(pattern.search(comment) for pattern in REVIEW_HEADER_PATTERNS)


def _declared_base_sha(comment: str) -> str | None:
    """Return the base SHA declared next to a base label in *comment*, if any."""
    match = re.search(
        r"\b(?:exact\s+)?base(?:\s+(?:reviewed|sha|commit))?\s*"
        r"(?:[:=]\s*`?|\s+`?)([0-9a-fA-F]{40})",
        comment,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).lower()


def review_comment_covers(comment: str, *, live_head: str, live_base: str) -> bool:
    """Return True when *comment* is a review carrier bound to the live state.

    The comment must be an exact-head review marker and name the live head.
    When the comment declares a base SHA, that base must equal the live base so
    a review performed against an older base cannot count as current evidence.
    """
    if not is_review_carrier_comment(comment):
        return False
    if live_head.lower() not in {sha.lower() for sha in extract_full_shas(comment)}:
        return False
    declared_base = _declared_base_sha(comment)
    if declared_base is not None and declared_base != live_base.lower():
        return False
    return True


def stale_carrier_sentinels(text: str) -> list[str]:
    """Return stale-narrative sentinels found in *text*.

    These markers declare that the evidence carrier is no longer current-base
    merge evidence or that a domain-aware review/approval remains pending; any
    such narrative invalidates a merge-ready disposition.
    """
    if not isinstance(text, str):
        return []
    matches: list[str] = []
    for pattern in STALE_CARRIER_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(match.group(0))
    return list(dict.fromkeys(matches))


def _body_carrier_error(body: str, *, live_head: str) -> str | None:
    """Return an error when the body carries exact-head SHA carriers for a different head.

    A body that declares a ``gate-verdict``, ``base-policy``, or ``Exact head``
    carrier must name the live head (the same admission rule used by
    ``gh_pr_body_rest``). Abbreviated or mismatched carriers mean the body is a
    stale carrier and the merge-ready disposition must be withheld.
    """
    carriers = extract_sha_carriers(body)
    if not carriers:
        return None
    invalid = invalid_sha_carriers(carriers, live_head)
    if not invalid:
        return None
    details = "; ".join(f"{carrier.kind} {carrier.sha}" for carrier in invalid)
    return (
        f"PR body carries exact-head SHA carrier(s) that do not match the "
        f"live head {live_head}: {details}"
    )


def _live_pr_identity_error(
    payload: dict[str, Any], *, live_head: str, live_base: str
) -> str | None:
    """Return an error if the carrier read observes a different PR head or base."""
    observed_head = payload.get("head")
    observed_base = payload.get("base")
    if not isinstance(observed_head, dict) or not isinstance(observed_head.get("sha"), str):
        return "PR carrier payload has no head SHA"
    if not isinstance(observed_base, dict) or not isinstance(observed_base.get("sha"), str):
        return "PR carrier payload has no base SHA"
    observed_head_sha = observed_head["sha"]
    observed_base_sha = observed_base["sha"]
    if not FULL_SHA_RE.fullmatch(observed_head_sha):
        return "PR carrier payload has malformed head SHA"
    if not FULL_SHA_RE.fullmatch(observed_base_sha):
        return "PR carrier payload has malformed base SHA"
    if observed_head_sha.lower() != live_head.lower():
        return "PR head changed during carrier read; merge-ready must be withheld"
    if observed_base_sha.lower() != live_base.lower():
        return "PR base changed during carrier read; merge-ready must be withheld"
    return None


def _review_carrier_error(
    comments: Sequence[dict[str, Any]],
    *,
    live_head: str,
    live_base: str,
    require_review_commit: bool = False,
) -> str | None:
    """Return an error when no human review comment is bound to the live state."""
    for entry in comments:
        if not isinstance(entry, dict):
            continue
        author = entry.get("user")
        if isinstance(author, dict) and author.get("login") in _BOT_AUTHORS:
            continue
        body = entry.get("body")
        if not isinstance(body, str):
            continue
        if require_review_commit:
            if str(entry.get("state", "")).upper() != "COMMENTED":
                continue
            review_commit = entry.get("commit_id")
            if not isinstance(review_commit, str) or not FULL_SHA_RE.fullmatch(review_commit):
                continue
            if review_commit.lower() != live_head.lower():
                continue
        if review_comment_covers(body, live_head=live_head, live_base=live_base):
            return None
    return (
        f"no exact-head review carrier comment covers the live head {live_head} (base {live_base})"
    )


def _reviews_verdict(
    number: int,
    *,
    repo: str,
    live_head: str,
    live_base: str,
) -> dict[str, Any]:
    """Read PR reviews and validate a canonical review-endpoint carrier.

    A review must be a live-head ``COMMENTED`` review with its commit id bound to
    the live head. The caller checks the top-level issue-comment compatibility
    carrier first, so older GitHub CLI/API setups do not need this endpoint when
    that legacy carrier is already present.
    """
    reviews_result = _gh_api_get(f"repos/{repo}/pulls/{number}/reviews?per_page=100")
    reviews, reviews_error = _parse_json(reviews_result, what=f"PR {number} review carriers read")
    if reviews_error:
        return {"status": "error", "error": reviews_error}
    if not isinstance(reviews, list):
        return {"status": "error", "error": "PR review carriers payload was not a list"}

    narrative_error = _stale_narrative_error(reviews, body="")
    if narrative_error:
        return {"status": "error", "error": narrative_error}

    review_error = _review_carrier_error(
        reviews,
        live_head=live_head,
        live_base=live_base,
        require_review_commit=True,
    )
    if review_error:
        return {"status": "error", "error": review_error}
    return {"status": "ok", "carrier_source": "pull_request_review"}


def _stale_narrative_error(comments: Sequence[dict[str, Any]], *, body: str) -> str | None:
    """Return an error when the body or any comment carries stale narratives."""
    body_sentinels = find_not_ready_body_sentinels(body)
    if body_sentinels:
        return "PR body carries not-ready sentinels: " + "; ".join(body_sentinels)
    for entry in comments:
        if not isinstance(entry, dict):
            continue
        text = entry.get("body")
        if not isinstance(text, str):
            continue
        sentinels = stale_carrier_sentinels(text)
        if sentinels:
            author = entry.get("user")
            who = author.get("login") if isinstance(author, dict) else "unknown"
            return (
                "review comment carries stale-carrier sentinel(s) naming "
                + "; ".join(sentinels)
                + f" (author {who}); merge-ready must be withheld"
            )
    return None


def _comments_verdict(
    number: int,
    *,
    repo: str,
    body: str,
    live_head: str,
    live_base: str,
) -> dict[str, Any]:
    """Read PR comments and return the carrier verdict for them."""
    comments_result = _gh_api_get(f"repos/{repo}/issues/{number}/comments")
    comments, comments_error = _parse_json(
        comments_result, what=f"PR {number} carrier comments read"
    )
    if comments_error:
        return {"status": "error", "error": comments_error}
    if not isinstance(comments, list):
        return {"status": "error", "error": "PR carrier comments payload was not a list"}

    narrative_error = _stale_narrative_error(comments, body=body)
    if narrative_error:
        return {"status": "error", "error": narrative_error}

    review_error = _review_carrier_error(comments, live_head=live_head, live_base=live_base)
    if review_error is None:
        return {"status": "ok", "carrier_source": "issue_comment"}
    reviews_verdict = _reviews_verdict(
        number,
        repo=repo,
        live_head=live_head,
        live_base=live_base,
    )
    if reviews_verdict["status"] != "ok":
        return reviews_verdict

    return reviews_verdict


def check_merge_ready_carriers(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    live_head: str,
    live_base: str,
) -> dict[str, Any]:
    """Validate that the PR's body and review carriers bind the live state.

    Reads the live PR object and its issue comments immediately, then checks
    body-digest freshness, body not-ready sentinels, stale-carrier narratives,
    and the existence of a human exact-head review comment bound to
    ``live_head``/``live_base``. Any failure returns ``status: error`` and the
    merge-ready write must be withheld.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    if not FULL_SHA_RE.fullmatch(live_head) or not FULL_SHA_RE.fullmatch(live_base):
        return {
            "status": "error",
            "error": "live_head and live_base must be full 40-character SHAs",
        }

    pr_result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    pr_payload, pr_error = _parse_json(pr_result, what=f"PR {number} carrier read")
    if pr_error:
        return {"status": "error", "error": pr_error}
    if not isinstance(pr_payload, dict):
        return {"status": "error", "error": "PR carrier payload was not an object"}

    body = pr_payload.get("body")
    if not isinstance(body, str):
        return {"status": "error", "error": "PR carrier payload has no body"}
    identity_error = _live_pr_identity_error(pr_payload, live_head=live_head, live_base=live_base)
    if identity_error:
        return {"status": "error", "error": identity_error}

    metadata_error = _body_carrier_error(body, live_head=live_head)
    if metadata_error:
        return {"status": "error", "error": metadata_error}

    comments_verdict = _comments_verdict(
        number,
        repo=repo,
        body=body,
        live_head=live_head,
        live_base=live_base,
    )
    if comments_verdict["status"] != "ok":
        return comments_verdict

    return {
        "status": "ok",
        "number": number,
        "repo": repo,
        "live_head_sha": live_head,
        "live_base_sha": live_base,
        "carrier_source": comments_verdict.get("carrier_source", "unknown"),
        "operation": "merge_ready_carrier_gate",
    }
