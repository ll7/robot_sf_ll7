"""Shared validation helpers for SNQI weight mappings.

This module centralizes validation logic used by optimization and
recompute scripts so that future changes to required keys or numeric
constraints happen in one place.

The validator intentionally stays lightweight (no external deps) and
performs the following checks:

* All required weight keys present (defined in WEIGHT_NAMES)
* Values convertible to float, finite and > 0
* Warn (not error) on extraneous keys to preserve forward compatibility
* Warn on unusually large weights (>10) which may indicate scale errors

Raises ValueError on structural / numeric failures. Returns a new dict
with all values cast to float for downstream use.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping

import numpy as np

from .compute import WEIGHT_NAMES

logger = logging.getLogger(__name__)


def validate_weights_mapping(raw: Mapping[str, object]) -> Dict[str, float]:
    """Validate and normalize an external weights mapping.

    Args:
        raw: Mapping of weight name -> numeric value (any convertible to float)

    Returns:
        A new dict[str, float] with validated values.

    Raises:
        ValueError: If required keys missing or values invalid.
    """
    missing = [k for k in WEIGHT_NAMES if k not in raw]
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    extraneous = [k for k in raw.keys() if k not in WEIGHT_NAMES]
    if extraneous:
        logger.warning("Extraneous weight keys will be ignored: %s", extraneous)
    validated: Dict[str, float] = {}
    for k in WEIGHT_NAMES:
        v = raw[k]
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Non-numeric weight for {k}: {v}") from e
        if not np.isfinite(fv) or fv <= 0:
            raise ValueError(f"Invalid weight value for {k}: {fv}")
        if fv > 10:
            logger.warning("Weight %s unusually large (%.3f) > 10", k, fv)
        validated[k] = fv
    return validated


__all__ = ["validate_weights_mapping"]
