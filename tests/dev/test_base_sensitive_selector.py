"""Tests for the explicit base-sensitive changed-file selector."""

from __future__ import annotations

from scripts.dev.base_sensitive_selector import (
    BASE_SENSITIVE,
    ORDINARY,
    SELECTOR_VERSION,
    UNKNOWN,
    classify_changed_files,
)

SENSITIVE = ["tests/test_snapshot.py", "tests/test_tuple_contract.py"]


def test_selector_classifies_marker_file_intersection() -> None:
    result = classify_changed_files(
        ["robot_sf/runtime.py", "tests/test_snapshot.py"],
        sensitive_files=SENSITIVE,
    )

    assert result["status"] == BASE_SENSITIVE
    assert result["selector"] == SELECTOR_VERSION
    assert result["changed_sensitive_files"] == ["tests/test_snapshot.py"]


def test_selector_classifies_unrelated_change_as_ordinary() -> None:
    result = classify_changed_files(["scripts/dev/helper.py"], sensitive_files=SENSITIVE)

    assert result["status"] == ORDINARY
    assert result["changed_sensitive_files"] == []


def test_selector_fails_closed_when_inventory_is_missing() -> None:
    result = classify_changed_files(None, sensitive_files=SENSITIVE)

    assert result["status"] == UNKNOWN
    assert result["reason"] == "changed_file_inventory_unavailable"


def test_path_normalization_preserves_hidden_directory_names() -> None:
    """Normalization removes only an explicit relative prefix."""
    result = classify_changed_files(["./.agents/skills/example.md"], sensitive_files=SENSITIVE)

    assert result["status"] == ORDINARY
    assert result["changed_files"] == [".agents/skills/example.md"]
