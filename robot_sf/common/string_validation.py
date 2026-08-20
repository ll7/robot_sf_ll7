"""Dependency-neutral validation primitives for structured string fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def require_non_empty_string(mapping: Mapping[str, Any], key: str, errors: list[str]) -> None:
    """Record the historical validation error for a missing or blank string field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")


__all__ = ["require_non_empty_string"]
