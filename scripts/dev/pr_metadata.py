#!/usr/bin/env python3
"""Pure helpers for final pull-request title/body reconciliation evidence."""

from __future__ import annotations

import hashlib
import json
import re

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
