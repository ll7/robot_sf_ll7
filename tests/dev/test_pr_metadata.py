"""Unit tests for pr_metadata module."""

from __future__ import annotations

import pytest

from scripts.dev.pr_metadata import (
    find_not_ready_body_sentinels,
    has_not_ready_body_narrative,
    metadata_digest,
    metadata_trailer,
    validate_pr_title,
)


@pytest.mark.parametrize(
    "body, expected_matches",
    [
        (
            "The PR remains unapproved and not merge-ready pending independent exact-head review and current hosted checks.",
            ["not merge-ready", "remains unapproved", "pending independent exact-head review"],
        ),
        (
            "This change remains unapproved pending hosted review.",
            ["remains unapproved", "pending hosted review"],
        ),
        (
            "WIP: do not merge yet!",
            ["do not merge"],
        ),
        (
            "Status: unapproved and not merge-ready.",
            ["not merge-ready", "unapproved and not merge-ready"],
        ),
        (
            "This is a clean, reconciled PR ready for review.",
            [],
        ),
        (
            "",
            [],
        ),
    ],
)
def test_find_not_ready_body_sentinels(body: str, expected_matches: list[str]) -> None:
    """Verify exact match of sentinel patterns against PR body text."""
    matches = find_not_ready_body_sentinels(body)
    if expected_matches:
        assert len(matches) >= len(expected_matches)
        for expected in expected_matches:
            assert any(expected.lower() in m.lower() for m in matches)
    else:
        assert matches == []
    assert has_not_ready_body_narrative(body) == bool(expected_matches)


def test_validate_pr_title() -> None:
    """Validate PR title constraints."""
    assert validate_pr_title("valid title") is None
    assert validate_pr_title("") == "PR title must not be empty"
    assert validate_pr_title("   ") == "PR title must not be empty"
    assert validate_pr_title("multi\nline") == "PR title must be a single line"
    assert validate_pr_title("a" * 300) is not None


def test_metadata_digest_and_trailer() -> None:
    """Verify metadata digest computation and trailer formatting."""
    digest = metadata_digest("title", "body")
    assert isinstance(digest, str)
    assert len(digest) == 64
    trailer = metadata_trailer(digest)
    assert trailer == f"pr-metadata: reconciled @ {digest}"
