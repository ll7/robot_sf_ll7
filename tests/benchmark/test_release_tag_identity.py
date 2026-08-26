"""Tests for release tag/source-SHA identity semantics (issue #7938)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.release_tag_identity import (
    check_tag_source_consistency,
    derive_sha_tag,
    extract_tag_sha_component,
)

SOURCE_SHA = "b1d5ab6de708385c0828c99501a9d1c29727ec11"
PLANNING_SHA = "cd831d7582c117ac9529065e7d1c60386933c92d"


def test_extract_full_sha_suffix() -> None:
    identity = extract_tag_sha_component(f"paper-matrix-v2-h600-s30-{SOURCE_SHA}")
    assert identity.full_sha_present is True
    assert identity.sha_component == SOURCE_SHA
    assert identity.scheme == "sha_suffix"


def test_extract_embedded_sha() -> None:
    identity = extract_tag_sha_component(f"release-{SOURCE_SHA}-final")
    assert identity.full_sha_present is True
    assert identity.sha_component == SOURCE_SHA
    assert identity.scheme == "sha_embedded"


def test_extract_short_abbreviation() -> None:
    identity = extract_tag_sha_component("paper-matrix-v2-h600-s30-cd831d7582c1")
    assert identity.full_sha_present is False
    assert identity.sha_component == "cd831d7582c1"
    assert identity.scheme == "sha_abbreviated"


def test_extract_semantic_tag() -> None:
    identity = extract_tag_sha_component("paper-matrix-v2-h600-s30-2026-08")
    assert identity.sha_component is None
    assert identity.full_sha_present is False
    assert identity.scheme == "semantic"


def test_derive_sha_tag_is_deterministic() -> None:
    derived = derive_sha_tag("paper-matrix-v2-h600-s30", SOURCE_SHA)
    assert derived == f"paper-matrix-v2-h600-s30-{SOURCE_SHA}"
    assert derive_sha_tag("paper-matrix-v2-h600-s30", SOURCE_SHA) == derived


def test_derive_sha_tag_rejects_bad_sha() -> None:
    with pytest.raises(ValueError):
        derive_sha_tag("prefix", "not-a-sha")
    with pytest.raises(ValueError):
        derive_sha_tag("prefix", "abc123")


def test_consistency_full_sha_match_passes() -> None:
    tag = derive_sha_tag("paper-matrix-v2-h600-s30", SOURCE_SHA)
    assert check_tag_source_consistency(tag, SOURCE_SHA) == []


def test_consistency_full_sha_mismatch_fails() -> None:
    tag = f"paper-matrix-v2-h600-s30-{PLANNING_SHA}"
    problems = check_tag_source_consistency(tag, SOURCE_SHA)
    assert len(problems) == 1
    assert "disagrees with" in problems[0]
    assert "planning/base SHAs are separate fields" in problems[0]


def test_consistency_abbreviation_prefix_passes() -> None:
    tag = f"paper-matrix-v2-h600-s30-{SOURCE_SHA[:12]}"
    assert check_tag_source_consistency(tag, SOURCE_SHA) == []


def test_consistency_abbreviation_not_prefix_fails() -> None:
    tag = f"paper-matrix-v2-h600-s30-{PLANNING_SHA[:12]}"
    problems = check_tag_source_consistency(tag, SOURCE_SHA)
    assert len(problems) == 1
    assert "not a prefix" in problems[0]


def test_consistency_semantic_allowed_by_default() -> None:
    assert check_tag_source_consistency("paper-matrix-v2-h600-s30-2026-08", SOURCE_SHA) == []


def test_consistency_semantic_forbidden_when_disallowed() -> None:
    problems = check_tag_source_consistency(
        "paper-matrix-v2-h600-s30-2026-08", SOURCE_SHA, allow_semantic=False
    )
    assert len(problems) == 1
    assert "no SHA identity" in problems[0]


def test_planning_sha_never_satisfies_source_check() -> None:
    """A planning/base SHA in a SHA-bearing tag must never pass the source check."""
    tag = f"paper-matrix-v2-h600-s30-{PLANNING_SHA}"
    problems = check_tag_source_consistency(tag, SOURCE_SHA)
    assert problems  # fails closed; the planning SHA is not the source SHA
