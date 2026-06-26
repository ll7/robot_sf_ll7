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


__all__ = ["require_finite_array", "require_finite_fields", "require_finite_scalar"]
