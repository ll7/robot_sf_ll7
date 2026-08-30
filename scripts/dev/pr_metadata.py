#!/usr/bin/env python3
"""Pure helpers for final pull-request title/body reconciliation evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

PR_TITLE_MAX_LENGTH = 256
_PR_METADATA_RE = re.compile(
    r"pr-metadata\s*:\s*reconciled\s*@\s*([0-9a-fA-F]{64})(?![0-9a-fA-F])",
    re.IGNORECASE,
)
PR_METADATA_RE = _PR_METADATA_RE


def validate_pr_title(title: str) -> str | None:
    """Return an error for an invalid GitHub PR title, or ``None`` when valid."""
    if not isinstance(title, str):
        return "PR title must be a string"
    if not title.strip():
        return "PR title must not be empty"
    if "\n" in title or "\r" in title:
        return "PR title must be a single line"
    if len(title) > PR_TITLE_MAX_LENGTH:
        return f"PR title must be at most {PR_TITLE_MAX_LENGTH} characters"
    return None


def metadata_digest(title: str, body: str) -> str:
    """Return the exact SHA-256 digest for a PR title/body pair.

    The digest input is canonical JSON containing the two exact strings in
    title/body order. JSON keeps the pair unambiguous even when either field
    contains newlines or separator-like text, while ``ensure_ascii=False``
    preserves Unicode characters before UTF-8 hashing.
    """
    encoded = json.dumps(
        [title, body],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_trailer(digest: str) -> str:
    """Return the canonical review-evidence trailer for *digest*."""
    return f"pr-metadata: reconciled @ {digest.lower()}"


def extract_metadata_digests(text: str) -> list[str]:
    """Extract canonical metadata digests from a review/comment body."""
    if not isinstance(text, str):
        return []
    return list(dict.fromkeys(match.group(1).lower() for match in _PR_METADATA_RE.finditer(text)))


_NOT_READY_SENTINEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnot\s+merge[-\s]?ready\b", re.IGNORECASE),
    re.compile(r"\bremains\s+unapproved\b", re.IGNORECASE),
    re.compile(
        r"\bpending\s+(?:independent|hosted|external)(?:\s+[\w-]+)*\s+review\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdo\s+not\s+merge\b", re.IGNORECASE),
    re.compile(r"\bunapproved\s+and\s+not\s+merge[-\s]?ready\b", re.IGNORECASE),
)


def find_not_ready_body_sentinels(body: str) -> list[str]:
    """Return matches for stale unapproved/not-ready narrative phrases in *body*."""
    if not isinstance(body, str):
        return []
    matches: list[str] = []
    for pattern in _NOT_READY_SENTINEL_PATTERNS:
        for match in pattern.finditer(body):
            matches.append(match.group(0))
    return list(dict.fromkeys(matches))


def has_not_ready_body_narrative(body: str) -> bool:
    """Return True if *body* contains any unapproved or not-ready narrative sentinels."""
    return bool(find_not_ready_body_sentinels(body))


EPOCH_SCHEMA = "pr_metadata_epoch.v1"


@dataclass(frozen=True)
class PrMetadataEpochInputs:
    """Bound inputs for a deterministic PR metadata epoch record (issue #7649)."""

    pr_number: int
    repository: str
    head_sha: str
    base_sha: str
    title: str
    body: str
    linked_issues: list[int] = field(default_factory=list)
    closing_references: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    requested_reviewers: list[str] = field(default_factory=list)
    review_decision: str = ""
    domain_approval_required: bool = False


def build_pr_metadata_epoch(
    inputs: PrMetadataEpochInputs,
    producer: str = "scripts/dev/pr_metadata_epoch.py",
) -> dict:
    """Build a deterministic ``pr_metadata_epoch.v1`` record from *inputs*.

    The record binds the dimensions that make a PR's preparation-to-integration
    transition stable: exact head/base identity, normalized title and body
    digest, linked-issue and closing-reference sets, admission-relevant labels,
    reviewer and approval state. ``producer`` and ``observed_at`` are provenance
    fields and are intentionally excluded from the digest.
    """
    record: dict = {
        "schema": EPOCH_SCHEMA,
        "pr_number": int(inputs.pr_number),
        "repository": inputs.repository,
        "head_sha": str(inputs.head_sha),
        "base_sha": str(inputs.base_sha),
        "title_normalized": normalize_pr_title(inputs.title),
        "body_digest": body_digest(inputs.body),
        "linked_issues": sorted({int(n) for n in inputs.linked_issues}),
        "closing_references": sorted({int(n) for n in inputs.closing_references}),
        "labels": sorted(set(inputs.labels)),
        "requested_reviewers": sorted(set(inputs.requested_reviewers)),
        "review_decision": str(inputs.review_decision),
        "domain_approval_required": bool(inputs.domain_approval_required),
        "producer": producer,
    }
    record["digest"] = epoch_digest(record)
    return record


# Dimensions bound into the epoch digest. Comment text is deliberately excluded:
# an automated comment-only review must not invalidate the metadata epoch.
_EPOCH_BOUND_FIELDS: tuple[str, ...] = (
    "pr_number",
    "repository",
    "head_sha",
    "base_sha",
    "title_normalized",
    "body_digest",
    "linked_issues",
    "closing_references",
    "labels",
    "requested_reviewers",
    "review_decision",
    "domain_approval_required",
)


def normalize_pr_text(text: str) -> str:
    """Normalize cosmetic whitespace in PR text (documented deterministic rule).

    The only cosmetic differences that are normalized are line-ending style
    (``\\r\\n`` becomes ``\\n``) and trailing whitespace on each line. Any other
    byte difference remains material and changes the epoch digest.
    """
    if not isinstance(text, str):
        return ""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalize_pr_title(title: str) -> str:
    """Normalize a PR title: strip and collapse internal whitespace runs."""
    if not isinstance(title, str):
        return ""
    return " ".join(title.split())


def body_digest(body: str) -> str:
    """Return the SHA-256 digest of the normalized PR body."""
    return hashlib.sha256(normalize_pr_text(body).encode("utf-8")).hexdigest()


def epoch_digest(record: dict) -> str:
    """Return the digest for the metadata-bound fields of an epoch record."""
    bound = {field: record[field] for field in _EPOCH_BOUND_FIELDS if field in record}
    encoded = json.dumps(
        bound,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def diff_epochs(previous: dict, current: dict) -> list[dict]:
    """Compare two epoch records and report every changed bound dimension.

    Each entry carries ``dimension``, ``material`` (False only when both sides
    normalize to the same value), and bounded ``before``/``after`` summaries.
    Changes to unbound provenance fields (``producer``, ``observed_at``) are
    reported as non-material observations.
    """
    changes: list[dict] = []
    for dimension in _EPOCH_BOUND_FIELDS:
        before = previous.get(dimension)
        after = current.get(dimension)
        if before != after:
            changes.append(
                {
                    "dimension": dimension,
                    "material": True,
                    "before": _summarize(before),
                    "after": _summarize(after),
                }
            )
    for dimension in ("producer", "observed_at"):
        before = previous.get(dimension)
        after = current.get(dimension)
        if before is not None or after is not None:
            if before != after:
                changes.append(
                    {
                        "dimension": dimension,
                        "material": False,
                        "before": _summarize(before),
                        "after": _summarize(after),
                    }
                )
    return changes


def _summarize(value: object, limit: int = 120) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…({len(text)} chars)"
