"""Shared fail-closed finite-value checks for diagnostic producers."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def require_finite_scalar(name: str, value: Real) -> float:
    """Return a real numeric scalar as float or raise when it is non-finite."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} is not finite: {numeric}")
    return numeric


def require_finite_array(name: str, values: Any) -> np.ndarray:
    """Return ``values`` as a float64 array or raise when any entry is NaN/Inf."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def require_finite_fields(label: str, obj: Any, fields: Iterable[str]) -> None:
    """Raise when any named numeric field on ``obj`` is NaN/Inf."""
    for field in fields:
        require_finite_scalar(f"{label}.{field}", getattr(obj, field))


def _require_finite(name: str, value: float) -> None:
    """Raise a clear error when a numeric grid parameter is not finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _require_finite_coerce(value: float, *, key: str) -> float:
    """Raise ``ValueError`` if ``value`` is non-finite (NaN/inf).

    Non-finite limits would silently corrupt threshold comparisons and could cause an
    infeasible maneuver to be reported feasible. Reject them at the input boundary.

    Returns:
        The validated finite float value.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite; got {value}")
    return numeric


def _require_finite_ndarray(name: str, arr: np.ndarray) -> None:
    """Raise ``ValueError`` if ``arr`` holds any non-finite (NaN/inf) value.

    Non-finite inputs would silently defeat every threshold comparison — e.g.
    ``nan < min_clearance`` evaluates to ``False`` — and could cause an unsafe
    trajectory to be accepted. Reject them at the input boundary (fail closed).
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")


def _require_finite_non_negative(name: str, value: float) -> None:
    """Raise ValueError unless value is finite and >= 0."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def _require_finite_non_negative_coerce(value: Any, *, key: str) -> float:
    """Coerce ``value`` to a float, raising ``ValueError`` unless it is finite and non-negative.

    Returns:
        The coerced finite, non-negative float value.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return numeric


def _require_finite_number(
    value: float, field_name: str, *, positive: bool = False, non_negative: bool = False
) -> None:
    """Validate ``value`` is finite (and optionally positive/non-negative), else raise."""
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    if positive and float(value) <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    if non_negative and float(value) < 0.0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_finite_bounded_values(
    *,
    values: tuple[float, ...],
    field_name: str,
    max_abs_value: float,
    max_field_name: str,
) -> None:
    """Validate one lattice axis against its configured absolute bound."""
    if not values:
        raise ValueError(f"{field_name} must contain at least one command value")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        if abs(value) > max_abs_value:
            raise ValueError(f"{field_name} exceed {max_field_name}")


def _require_finite_real(value: Any, field_name: str) -> float:
    """Return a finite real value while rejecting YAML booleans explicitly."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite real number, not a boolean")
    return float(value)


def _require_finite_position(position: list[object]) -> None:
    """Validate that every component of a trace position is a finite number."""
    if not all(isinstance(value, int | float) and math.isfinite(value) for value in position):
        raise ValueError("trace positions must be finite numbers")


__all__ = [
    "require_finite_array",
    "require_finite_fields",
    "require_finite_scalar",
    "_require_finite",
    "_require_finite_bounded_values",
    "_require_finite_coerce",
    "_require_finite_ndarray",
    "_require_finite_non_negative",
    "_require_finite_non_negative_coerce",
    "_require_finite_number",
    "_require_finite_position",
    "_require_finite_real",
]
