#!/usr/bin/env python3
"""Canonical lane-coordination marker parser and formatter (issue #9254).

Lanes coordinate by emitting and parsing each other's comment prose:
``review-claim:``, ``gate-verdict:``, ``base-policy:``, ``pr-metadata:
reconciled``, and the ``merge-ready`` label. Producer/consumer skew has bitten
twice (#9243: human prefaces mixed into JSON artifacts; #9246: secondary
reason codes overwriting primary ones), so every marker has exactly one
implementation here that both sides share.

``pr_loop_policy.py`` and ``pr_metadata.py`` re-export these names for
backward compatibility; new code imports from this module. Marker semantics
are unchanged: this module only unifies the spelling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

MERGE_READY_LABEL = "merge-ready"

# Minimum overlap (hex chars) required to treat an abbreviated trailer SHA as a
# match for a longer head SHA. Seven mirrors git's default short SHA width.
GATE_VERDICT_MIN_SHA_OVERLAP = 7
GATE_VERDICT_PROJECTION_SOURCE = "trusted-review-projection"
_GATE_VERDICT_STATUSES = frozenset({"accepted", "hold", "missing", "malformed", "ambiguous"})

# Matches accepted and blocking HOLD trailers embedded in comment or review
# body excerpts, capturing the verdict and hex SHA. The verdict word is
# matched case-insensitively; surrounding markdown/code fences are tolerated.
_GATE_VERDICT_RE = re.compile(
    r"(?=gate-verdict\s*:\s*(?:accepted|hold)\s*@\s*"
    r"(?P<sha>[0-9a-fA-F]{7,40})\b)"
    r"gate-verdict\s*:\s*(?P<verdict>accepted|hold)\s*@\s*"
    r"[0-9a-fA-F]{7,40}\b",
    re.IGNORECASE,
)
GATE_VERDICT_RE = _GATE_VERDICT_RE
# Gate-verdict events are control trailers, so require the marker to begin a
# Markdown-style line (after optional list/quote/fence decoration). This keeps
# prose such as ``keep `gate-verdict: hold`;`` from becoming a malformed event
# while leaving malformed dedicated trailers fail-closed below.
_GATE_VERDICT_MARKER_RE = re.compile(
    r"^[ \t]*(?:[-*+>]\s*)*(?:`{1,3}\s*)?"
    r"(?P<marker>gate-verdict\s*:\s*(?P<verdict>accepted|hold)\b)",
    re.IGNORECASE | re.MULTILINE,
)
_BASE_POLICY_RE = re.compile(
    r"base-policy\s*:\s*(ordinary-cas|current-base)\s*@\s*([0-9a-fA-F]{7,40})\b",
    re.IGNORECASE,
)
BASE_POLICY_RE = _BASE_POLICY_RE
_EXACT_HEAD_RE = re.compile(
    r"exact\s+head\s*:\s*([0-9a-fA-F]{7,40})\b",
    re.IGNORECASE,
)
EXACT_HEAD_RE = _EXACT_HEAD_RE


def gate_verdict_matches(text: str) -> list[re.Match[str]]:
    """Return complete gate-verdict carriers from dedicated Markdown lines.

    The broad SHA carrier parser remains available for provenance inspection, but
    gate-event consumers must not promote inline prose into control state. A
    dedicated marker with an invalid or missing SHA is intentionally omitted here;
    event consumers that need fail-closed malformed detection inspect the marker
    stream directly.
    """
    if not isinstance(text, str) or not text:
        return []
    matches: list[re.Match[str]] = []
    for marker in _GATE_VERDICT_MARKER_RE.finditer(text):
        match = _GATE_VERDICT_RE.match(text, marker.start("marker"))
        if match is not None:
            matches.append(match)
    return matches


# A ``review-claim`` comment announces a lane's mutable-write window; admission
# gates treat an unexpired trusted claim as a hold. The same comment thread is
# released with ``review-claim: released @ <head-sha>``. The ``until`` timestamp
# is the ISO-8601 UTC expiry (default claim window 90 min).
_REVIEW_CLAIM_RE = re.compile(
    r"review-claim\s*:\s*(?P<lane>[^\s@]+)\s*@\s*(?P<sha>[0-9a-fA-F]{7,40})\b"
    r"\s+until\s+(?P<until>\S+)",
    re.IGNORECASE,
)
REVIEW_CLAIM_RE = _REVIEW_CLAIM_RE
_REVIEW_CLAIM_RELEASED_RE = re.compile(
    r"review-claim\s*:\s*released\s*@\s*(?P<sha>[0-9a-fA-F]{7,40})\b",
    re.IGNORECASE,
)
REVIEW_CLAIM_RELEASED_RE = _REVIEW_CLAIM_RELEASED_RE
_PR_METADATA_RE = re.compile(
    r"pr-metadata\s*:\s*reconciled\s*@\s*([0-9a-fA-F]{64})(?![0-9a-fA-F])",
    re.IGNORECASE,
)
PR_METADATA_RE = _PR_METADATA_RE


@dataclass(frozen=True, slots=True)
class ShaCarrier:
    """One exact-head SHA carrier parsed from PR metadata text.

    ``kind`` is one of ``gate-verdict``, ``base-policy``, or ``exact-head``;
    ``sha`` is the hex carrier as written, lowercased; ``full`` is True only
    for the 40-hex form that can be checked against a live head.
    """

    kind: str
    sha: str
    full: bool


@dataclass(frozen=True, slots=True)
class ReviewClaim:
    """One parsed ``review-claim`` marker from a trusted comment (issue #7508).

    ``lane`` is the claiming lane id, ``sha`` the claimed head SHA as written
    (lowercased), and ``expires_at`` the parsed ``until`` timestamp (UTC) or
    ``None`` when the timestamp is missing or unparseable.
    """

    lane: str
    sha: str
    expires_at: datetime | None


def _valid_head_sha(sha: str) -> bool:
    """Return whether a SHA is a plausible 7-40 hex commit prefix."""
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", sha))


def format_review_claim(lane: str, head_sha: str, until: str) -> str:
    """Format the canonical ``review-claim`` marker producers emit.

    Raises:
        ValueError: when the lane is blank or the SHA is not hex.
    """
    lane = lane.strip()
    head_sha = head_sha.strip().lower()
    if not lane or "@" in lane or any(part.isspace() for part in lane):
        raise ValueError(f"invalid review-claim lane: {lane!r}")
    if not _valid_head_sha(head_sha):
        raise ValueError(f"invalid review-claim head SHA: {head_sha!r}")
    if not until.strip():
        raise ValueError("review-claim expiry must not be blank")
    return f"review-claim: {lane} @ {head_sha} until {until.strip()}"


def format_review_claim_release(head_sha: str) -> str:
    """Format the canonical ``review-claim: released`` marker producers emit."""
    head_sha = head_sha.strip().lower()
    if not _valid_head_sha(head_sha):
        raise ValueError(f"invalid review-claim release SHA: {head_sha!r}")
    return f"review-claim: released @ {head_sha}"


def format_gate_verdict(verdict: str, head_sha: str) -> str:
    """Format the canonical ``gate-verdict`` trailer reviewers emit."""
    verdict = verdict.strip().lower()
    head_sha = head_sha.strip().lower()
    if verdict not in {"accepted", "hold"}:
        raise ValueError(f"invalid gate verdict: {verdict!r}")
    if not _valid_head_sha(head_sha):
        raise ValueError(f"invalid gate-verdict head SHA: {head_sha!r}")
    return f"gate-verdict: {verdict} @ {head_sha}"


def format_base_policy(policy: str, head_sha: str) -> str:
    """Format the canonical ``base-policy`` trailer reviewers emit."""
    policy = policy.strip().lower()
    head_sha = head_sha.strip().lower()
    if policy not in {"ordinary-cas", "current-base"}:
        raise ValueError(f"invalid base policy: {policy!r}")
    if not _valid_head_sha(head_sha):
        raise ValueError(f"invalid base-policy head SHA: {head_sha!r}")
    return f"base-policy: {policy} @ {head_sha}"


def format_pr_metadata(digest: str) -> str:
    """Format the canonical ``pr-metadata: reconciled`` trailer reviewers emit."""
    digest = digest.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("pr-metadata digest must be 64 hex chars")
    return f"pr-metadata: reconciled @ {digest}"


def metadata_trailer(digest: str) -> str:
    """Return the canonical review-evidence trailer for *digest*."""
    return format_pr_metadata(digest)


def has_merge_ready_label(labels: list[str] | Any) -> bool:
    """Return whether a PR label list carries the canonical merge-ready label."""
    if not isinstance(labels, list):
        return False
    return MERGE_READY_LABEL in {str(label) for label in labels}


def extract_sha_carriers(text: str) -> list[ShaCarrier]:
    """Extract exact-head SHA carriers from a PR metadata text blob.

    Covers the three canonical trailer forms: ``gate-verdict: accepted @
    <sha>``, ``base-policy: (ordinary-cas|current-base) @ <sha>``, and
    ``Exact head: <sha>``, plus a validated v2 ``exact_head`` field. Surrounding
    markdown/code fences are tolerated so quoted historical evidence is still
    surfaced for fail-closed validation.
    Carriers are returned in document order with the SHA as written.
    """
    if not isinstance(text, str) or not text:
        return []
    from scripts.dev.pr_contract_v2 import parse_pr_contract_v2

    carriers: list[ShaCarrier] = []
    for match in _GATE_VERDICT_RE.finditer(text):
        raw = match.group("sha")
        carriers.append(ShaCarrier(kind="gate-verdict", sha=raw.lower(), full=len(raw) == 40))
    for match in _BASE_POLICY_RE.finditer(text):
        raw = match.group(2)
        carriers.append(ShaCarrier(kind="base-policy", sha=raw.lower(), full=len(raw) == 40))
    for match in _EXACT_HEAD_RE.finditer(text):
        raw = match.group(1)
        carriers.append(ShaCarrier(kind="exact-head", sha=raw.lower(), full=len(raw) == 40))
    v2_result = parse_pr_contract_v2(text, source="sha-carrier")
    if v2_result.contract is not None and v2_result.contract.exact_head:
        raw = v2_result.contract.exact_head
        carriers.append(ShaCarrier(kind="exact-head", sha=raw, full=len(raw) == 40))
    return carriers


def invalid_sha_carriers(carriers: list[ShaCarrier], live_head_sha: str) -> list[ShaCarrier]:
    """Return the carrier subset that fails the live-head admission rule.

    Admission rule (issue #7448): a carrier admits only when it carries the
    full 40-hex SHA equal to the live head, case-insensitively. Abbreviated
    carriers, carriers naming a different commit, and carriers with no
    comparable live head all fail closed.
    """
    live_head = live_head_sha.lower()
    return [carrier for carrier in carriers if not carrier.full or carrier.sha != live_head]


def _parse_review_claim_marker(text: str) -> ReviewClaim | None:
    """Parse one ``review-claim: <lane> @ <sha> until <UTC>`` marker, or None.

    Returns ``None`` for non-string/empty blobs and for blobs whose timestamp
    cannot be parsed as an ISO-8601 UTC datetime (fail closed: an unparseable
    claim is treated as expired rather than parking the PR forever).
    """
    if not isinstance(text, str) or not text:
        return None
    match = _REVIEW_CLAIM_RE.search(text)
    if not match:
        return None
    raw_until = match.group("until")
    try:
        expires_at = datetime.fromisoformat(raw_until.replace("Z", "+00:00"))
    except ValueError:
        return None
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return ReviewClaim(
        lane=match.group("lane").lower(),
        sha=match.group("sha").lower(),
        expires_at=expires_at.astimezone(UTC),
    )


def parse_review_claim(text: str) -> ReviewClaim | None:
    """Parse one canonical ``review-claim`` marker, or ``None`` when absent."""
    return _parse_review_claim_marker(text)


def _review_claim_released_shas(text: str) -> set[str]:
    """Return lowercased SHAs released by ``review-claim: released @ <sha>`` markers."""
    if not isinstance(text, str) or not text:
        return set()
    return {match.group("sha").lower() for match in _REVIEW_CLAIM_RELEASED_RE.finditer(text)}


def review_claim_released_shas(text: str) -> set[str]:
    """Return lowercased SHAs released by canonical release markers."""
    return _review_claim_released_shas(text)


def extract_metadata_digests(text: str) -> list[str]:
    """Extract canonical metadata digests from a review/comment body."""
    if not isinstance(text, str):
        return []
    return list(dict.fromkeys(match.group(1).lower() for match in _PR_METADATA_RE.finditer(text)))
