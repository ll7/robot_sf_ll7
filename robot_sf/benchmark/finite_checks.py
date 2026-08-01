"""Shared fail-closed finite-value checks for diagnostic producers.

Re-exports from :mod:`robot_sf.common.validation` for backward compatibility.
"""

from robot_sf.common.validation import (
    require_finite_array,
    require_finite_fields,
    require_finite_scalar,
)

__all__ = ["require_finite_array", "require_finite_fields", "require_finite_scalar"]
