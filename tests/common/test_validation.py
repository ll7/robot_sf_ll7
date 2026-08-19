"""Tests for shared common validation helpers."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from robot_sf.common.validation import append_non_empty_string_error


@pytest.mark.parametrize(
    "value",
    [None, True, 3, 3.5, "", " \t"],
)
def test_append_non_empty_string_error_preserves_append_contract(value: object) -> None:
    """Invalid values append the historical error after existing errors."""
    errors = ["prior error"]

    append_non_empty_string_error({"field": value}, "field", errors)

    assert errors == ["prior error", "field must be a non-empty string"]


def test_append_non_empty_string_error_reports_missing_key() -> None:
    """Missing keys append the same error as other invalid values."""
    errors: list[str] = []

    append_non_empty_string_error({}, "field", errors)

    assert errors == ["field must be a non-empty string"]


def test_append_non_empty_string_error_accepts_mapping_like_inputs() -> None:
    """The shared helper accepts read-only mapping implementations."""
    errors: list[str] = []

    append_non_empty_string_error(MappingProxyType({"field": "valid text"}), "field", errors)

    assert errors == []


@pytest.mark.parametrize("value", ["valid", " text "])
def test_append_non_empty_string_error_accepts_nonblank_strings(value: str) -> None:
    """Nonblank strings do not mutate the existing error list."""
    errors = ["prior error"]

    append_non_empty_string_error({"field": value}, "field", errors)

    assert errors == ["prior error"]
