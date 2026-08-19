"""Characterization tests for shared dependency-neutral string validation."""

from __future__ import annotations

from collections import UserDict

import pytest

from robot_sf.common.string_validation import require_non_empty_string


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 1.5, float("nan"), float("inf"), "", " \t"],
)
def test_require_non_empty_string_rejects_non_text_and_blank_values(value: object) -> None:
    """The migrated helpers reject every historical invalid-value category."""

    errors: list[str] = []
    require_non_empty_string({"field": value}, "field", errors)

    assert errors == ["field must be a non-empty string"]


@pytest.mark.parametrize("value", ["value", " value ", "0", "NaN"])
def test_require_non_empty_string_accepts_non_empty_text(value: str) -> None:
    """Non-empty strings retain their original value and produce no error."""

    errors: list[str] = []
    require_non_empty_string(UserDict(field=value), "field", errors)

    assert errors == []
