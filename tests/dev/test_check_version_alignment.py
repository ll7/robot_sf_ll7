"""Tests for the git-tag single-source version alignment checker."""

from __future__ import annotations

import pytest

from scripts.dev.check_version_alignment import (
    base_version,
    evaluate,
    git_all_tags,
    latest_release_tag,
    load_citation_version,
    numeric_version_from_tag,
)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("0.0.2", "0.0.2"),
        ("rc0.0.3", "0.0.3"),
        ("v1.2.3", "1.2.3"),
        ("12.4.9", "12.4.9"),
        ("artifact/models-2026-05-registry-v1", None),
        ("camera-ready-v0.0.1a", None),
        ("main", None),
        ("", None),
    ],
)
def test_numeric_version_from_tag(tag: str, expected: str | None) -> None:
    """Only release-line tags (X.Y.Z / vX.Y.Z / rcX.Y.Z) map to a number."""
    assert numeric_version_from_tag(tag) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.0.2", "0.0.2"),
        ("0.0.3.dev4+g1234abc", "0.0.3"),
        ("1.2.3rc1", "1.2.3"),
        ("0.0.0+unknown", "0.0.0"),
    ],
)
def test_base_version(version: str, expected: str) -> None:
    """The base version strips dev/local/pre-release suffixes."""
    assert base_version(version) == expected


def test_latest_release_tag_excludes_candidates() -> None:
    """rc tags are pre-releases and must not win the latest-release election."""
    tags = ["0.0.1", "0.0.2", "rc0.0.3", "v0.0.1", "artifact/foo"]
    assert latest_release_tag(tags) == "0.0.2"


def test_latest_release_tag_orders_numerically() -> None:
    """Numeric ordering, not lexicographic, decides the latest release."""
    assert latest_release_tag(["0.0.9", "0.0.10", "0.0.2"]) == "0.0.10"


def test_latest_release_tag_none_when_no_full_release() -> None:
    """Without a full release tag there is nothing to align CITATION against."""
    assert latest_release_tag(["rc0.0.3", "artifact/foo"]) is None


def test_evaluate_aligned_on_release_candidate_tag() -> None:
    """HEAD on rc0.0.3 with package 0.0.3 and CITATION 0.0.2 is aligned."""
    problems = evaluate(
        head_tags=["rc0.0.3"],
        all_tags=["0.0.2", "rc0.0.3"],
        package_version="0.0.3",
        citation_version="0.0.2",
    )
    assert problems == []


def test_evaluate_aligned_on_full_release_tag() -> None:
    """HEAD on 0.0.2 with a dev-suffixed build base still counts as derived."""
    problems = evaluate(
        head_tags=["0.0.2"],
        all_tags=["0.0.2"],
        package_version="0.0.2",
        citation_version="0.0.2",
    )
    assert problems == []


def test_evaluate_aligned_when_head_untagged() -> None:
    """Untagged HEAD skips the tag-derivation check and only guards CITATION."""
    problems = evaluate(
        head_tags=[],
        all_tags=["0.0.2", "rc0.0.3"],
        package_version="0.0.4.dev2+gdeadbee",
        citation_version="0.0.2",
    )
    assert problems == []


def test_evaluate_flags_package_not_derived_from_tag() -> None:
    """A build whose version ignores the HEAD tag is drift."""
    problems = evaluate(
        head_tags=["0.0.2"],
        all_tags=["0.0.2"],
        package_version="9.9.9",
        citation_version="0.0.2",
    )
    assert any("does not derive" in problem for problem in problems)


def test_evaluate_flags_missing_install_on_tagged_head() -> None:
    """On a tagged HEAD, an uninstalled package cannot be verified."""
    problems = evaluate(
        head_tags=["rc0.0.3"],
        all_tags=["0.0.2", "rc0.0.3"],
        package_version=None,
        citation_version="0.0.2",
    )
    assert any("not installed" in problem for problem in problems)


def test_evaluate_flags_citation_drift() -> None:
    """The historical benchmark-protocol label is flagged as drift."""
    problems = evaluate(
        head_tags=[],
        all_tags=["0.0.2"],
        package_version="0.0.2",
        citation_version="benchmark-protocol-0.1.0",
    )
    assert any("CITATION.cff" in problem for problem in problems)


def test_evaluate_flags_missing_release_tag() -> None:
    """Without any full release tag the CITATION cannot be aligned."""
    problems = evaluate(
        head_tags=[],
        all_tags=["rc0.0.3"],
        package_version="0.0.3",
        citation_version="0.0.2",
    )
    assert any("no full release tag" in problem for problem in problems)


def test_repo_citation_matches_latest_release_tag() -> None:
    """The committed CITATION.cff must match the repo's latest release tag.

    This is the concrete guard that the current repository state is aligned;
    it uses the real git tags and the real CITATION.cff, independent of any
    installed-package state.
    """
    from scripts.dev.check_version_alignment import DEFAULT_CITATION

    all_tags = git_all_tags()
    latest = latest_release_tag(all_tags)
    assert latest is not None, "expected at least one full release tag in the repo"

    citation_version = load_citation_version(DEFAULT_CITATION)
    # head_tags=[] scopes this guard to CITATION alignment only, independent of
    # whether HEAD currently sits on a release tag or of the installed package.
    problems = evaluate(
        head_tags=[],
        all_tags=all_tags,
        package_version=numeric_version_from_tag(latest),
        citation_version=citation_version,
    )
    assert problems == [], problems
