"""Shared release tag/source-SHA identity semantics (issue #7938).

Future benchmark-release tags may carry a SHA-like identity component.  When
they do, that component must be derived from the final immutable source SHA —
never from a preliminary planning/base SHA.  This module owns the extraction,
derivation, and consistency-check helpers so the release doctor, the release
CLI, and tests share one versioned contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Matches a trailing 40-hex SHA-like tag suffix (e.g. ``-cd831d7582c1`` is a
#: 12-hex abbreviation; a full 40-hex component is the canonical identity).
_SHA_SUFFIX_RE = re.compile(r"[_-](?P<sha>[0-9a-f]{40})$")
#: Full 40-hex SHA anywhere in the tag (strict identity carrier).
_SHA_ANYWHERE_RE = re.compile(r"\b[0-9a-f]{40}\b")

# This release predates the source-SHA tag contract.  It is retained only as
# an explicit read-only compatibility exception; no future release may use
# this exception or mutate the published tag/assets.
HISTORICAL_RELEASE_TAG = "paper-matrix-v2-h600-s30-2026-08-cd831d7582c1"
HISTORICAL_RELEASE_SOURCE_SHA = "b1d5ab6de708385c0828c99501a9d1c29727ec11"


@dataclass(frozen=True)
class TagShaIdentity:
    """Result of extracting a SHA-like component from a release tag."""

    tag: str
    sha_component: str | None
    full_sha_present: bool
    scheme: str

    def as_dict(self) -> dict[str, str | bool | None]:
        """Return a JSON-ready dictionary."""
        return {
            "tag": self.tag,
            "sha_component": self.sha_component,
            "full_sha_present": self.full_sha_present,
            "scheme": self.scheme,
        }


def is_historical_release_tag(tag: str) -> bool:
    """Return whether ``tag`` is the immutable pre-contract release tag."""
    return tag == HISTORICAL_RELEASE_TAG


def extract_tag_sha_component(tag: str) -> TagShaIdentity:
    """Extract the SHA-like component of a release tag.

    A tag may use a full 40-hex SHA suffix (canonical), a shorter hex
    abbreviation, or no SHA at all (semantic identifier scheme).  A full
    40-hex SHA anywhere in the tag is the authoritative identity carrier.

    Returns:
        A :class:`TagShaIdentity` with the extracted component.
    """
    suffix_match = _SHA_SUFFIX_RE.search(tag)
    anywhere_match = _SHA_ANYWHERE_RE.search(tag)
    if suffix_match:
        sha = suffix_match.group("sha")
        return TagShaIdentity(
            tag=tag,
            sha_component=sha,
            full_sha_present=True,
            scheme="sha_suffix",
        )
    if anywhere_match:
        return TagShaIdentity(
            tag=tag,
            sha_component=anywhere_match.group(0),
            full_sha_present=True,
            scheme="sha_embedded",
        )
    # No full 40-hex SHA: check for a shorter hex abbreviation (>=8 hex chars
    # bounded by separators), which is treated as an ambiguous identity.
    short = re.search(r"[_-](?P<sha>[0-9a-f]{8,39})(?:$|[_-])", tag)
    if short:
        return TagShaIdentity(
            tag=tag,
            sha_component=short.group("sha"),
            full_sha_present=False,
            scheme="sha_abbreviated",
        )
    return TagShaIdentity(
        tag=tag,
        sha_component=None,
        full_sha_present=False,
        scheme="semantic",
    )


def derive_sha_tag(prefix: str, source_sha: str, *, separator: str = "-") -> str:
    """Deterministically derive a SHA-bearing tag from the final source SHA.

    Args:
        prefix: Human-readable release prefix (e.g. ``paper-matrix-v2-h600-s30``).
        source_sha: Full 40-hex final immutable source SHA.
        separator: Separator before the SHA component.

    Returns:
        The derived tag string.

    Raises:
        ValueError: When ``source_sha`` is not a full 40-hex SHA.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be a full 40-character lowercase hexadecimal SHA")
    return f"{prefix}{separator}{source_sha}"


def check_tag_source_consistency(
    tag: str, source_sha: str, *, allow_semantic: bool = True
) -> list[str]:
    """Return fail-closed problems when a SHA-bearing tag disagrees with source_sha.

    A tag with a full 40-hex SHA component must match ``source_sha`` exactly.
    A semantic (non-SHA) tag passes when ``allow_semantic`` is true.  A short
    hex abbreviation that is neither the full SHA nor a recognized prefix fails
    closed because it cannot be verified.

    Returns:
        Problem strings; empty when the tag is consistent.
    """
    identity = extract_tag_sha_component(tag)
    problems: list[str] = []
    if identity.full_sha_present:
        if identity.sha_component != source_sha:
            problems.append(
                f"tag SHA component {identity.sha_component!r} disagrees with "
                f"source_sha {source_sha!r}; a SHA-bearing tag must be derived from "
                "the final immutable source SHA (planning/base SHAs are separate fields)"
            )
        return problems
    if identity.scheme == "sha_abbreviated":
        if not source_sha.startswith(identity.sha_component or ""):
            problems.append(
                f"tag abbreviation {identity.sha_component!r} is not a prefix of "
                f"source_sha {source_sha!r}; ambiguous SHA-like identity fails closed"
            )
        return problems
    if not allow_semantic:
        problems.append(
            f"tag {tag!r} carries no SHA identity but allow_semantic=false; "
            "use derive_sha_tag() or an explicit semantic scheme"
        )
    return problems


def check_canonical_source_tag(tag: str, source_sha: str) -> list[str]:
    """Require one unambiguous ``<prefix>-<full source SHA>`` identity.

    Semantic and abbreviated tags remain supported by the older prospective
    compatibility checker.  Generated benchmark-data identities use this
    stricter boundary so two SHA tokens, an embedded token, or a stale source
    cannot acquire a second interpretation.

    Returns:
        Problem strings; empty only for one exact full-SHA suffix.
    """
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        return ["source_sha must be a full 40-character lowercase hexadecimal SHA"]
    normalized_tag = str(tag).strip()
    if not isinstance(tag, str) or tag != normalized_tag:
        return [
            "release tag must have one canonical full-SHA suffix without surrounding whitespace"
        ]
    suffix = f"-{source_sha}"
    prefix = normalized_tag[: -len(suffix)] if normalized_tag.endswith(suffix) else ""
    sha_tokens = re.findall(r"[0-9A-Fa-f]{40,}", normalized_tag)
    if not prefix or sha_tokens != [source_sha]:
        return ["release tag must have one canonical full-SHA suffix derived from source_sha"]
    if derive_sha_tag(prefix, source_sha) != normalized_tag:
        return ["release tag must have one canonical full-SHA suffix derived from source_sha"]
    return []
