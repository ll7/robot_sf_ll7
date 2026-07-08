"""Shared validation helpers for SNQI weight mappings.

This module centralizes validation logic used by optimization and
recompute scripts so that future changes to required keys or numeric
constraints happen in one place.

The validator intentionally stays lightweight (no external deps) and
performs the following checks:

* All required weight keys present (defined in WEIGHT_NAMES)
* Values convertible to float, finite and >= 0 (a weight of 0 disables the corresponding metric)
* Warn (not error) on extraneous keys to preserve forward compatibility
* Warn on unusually large weights (>10) which may indicate scale errors

Raises ValueError on structural / numeric failures. Returns a new dict
with all values cast to float for downstream use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .compute import WEIGHT_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

# Backward-compatible aliasing for weight names.
#
# We keep 'w_near' as the canonical key for now (to avoid breaking changes), but
# accept 'w_near_misses' as an alias to match the metric name 'near_misses'.
# If both are present, 'w_near' wins and the alias is ignored with a warning.
_WEIGHT_ALIASES: dict[str, str] = {
    "w_near_misses": "w_near",
}


def _apply_weight_aliases(working: dict[str, object]) -> None:
    """Normalize aliases in-place on a working mapping.

    Rules:
    - If canonical key missing but alias present, copy alias value to canonical and log at INFO.
    - If both present and values differ (after float coercion), keep canonical and warn.
    """
    for alias, canonical in _WEIGHT_ALIASES.items():
        has_alias = alias in working
        has_canonical = canonical in working
        if not has_alias:
            continue
        if not has_canonical:
            working[canonical] = working[alias]
            logger.warning(
                f"Weight '{alias}' is deprecated; use '{canonical}'. Mapping alias automatically."
            )
            continue
        # Both present: check if numerically equal; if not, warn and prefer canonical.
        try:
            alias_val = float(working[alias])  # type: ignore[arg-type]
            canon_val = float(working[canonical])  # type: ignore[arg-type]
        except (ValueError, TypeError):
            logger.warning(
                f"Alias '{alias}' provided alongside '{canonical}'; "
                "ignoring alias due to non-numeric values"
            )
            continue
        if alias_val != canon_val:
            logger.warning(
                f"Alias '{alias}' provided alongside '{canonical}'; ignoring alias (values differ)"
            )


def validate_weights_mapping(raw: Mapping[str, object]) -> dict[str, float]:
    """Validate and normalize an external weights mapping.

    Args:
        raw: Mapping of weight name -> numeric value (any convertible to float)

    Returns:
        A new dict[str, float] with validated values.

    Raises:
        ValueError: If required keys missing or values invalid.
    """
    working: dict[str, object] = dict(raw)
    _apply_weight_aliases(working)

    missing = [k for k in WEIGHT_NAMES if k not in working]
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    extraneous = [k for k in working if k not in WEIGHT_NAMES and k not in _WEIGHT_ALIASES]
    if extraneous:
        logger.warning(f"Extraneous weight keys will be ignored: {extraneous}")
    validated: dict[str, float] = {}
    for k in WEIGHT_NAMES:
        v = working[k]
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Non-numeric weight for {k}: {v}") from e
        # Accept zero as a valid weight to disable a term; reject negatives and non-finite.
        if not np.isfinite(fv) or fv < 0:
            raise ValueError(f"Invalid weight value for {k}: {fv}")
        if fv > 10:
            logger.warning(f"Weight {k} unusually large ({fv:.3f}) > 10")
        validated[k] = fv
    return validated


__all__ = ["validate_weights_mapping"]
