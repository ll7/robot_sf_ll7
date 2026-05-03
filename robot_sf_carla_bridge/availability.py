"""Optional CARLA runtime availability checks."""

from __future__ import annotations

import importlib.util
from typing import Any


def check_carla_availability() -> dict[str, Any]:
    """Return explicit availability metadata without importing CARLA.

    The CARLA Python API can be installed through simulator-specific paths and should not be a
    normal Robot-SF dependency. This helper intentionally uses ``find_spec`` so normal imports of
    :mod:`robot_sf_carla_bridge` never require CARLA.
    """

    if importlib.util.find_spec("carla") is None:
        return {
            "status": "not-available",
            "reason": "CARLA Python API package 'carla' is not importable",
            "dependency": "carla",
        }
    return {
        "status": "available",
        "reason": "CARLA Python API package is importable",
        "dependency": "carla",
    }
