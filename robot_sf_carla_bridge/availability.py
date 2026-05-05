"""Optional CARLA runtime availability checks."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


class CarlaUnavailableError(RuntimeError):
    """Raised when a CARLA-dependent bridge path is used without CARLA installed."""


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


def require_carla() -> Any:
    """Import and return the optional CARLA Python API or raise a clear bridge error.

    Returns:
        Imported ``carla`` module for CARLA-dependent replay entry points.

    Raises:
        CarlaUnavailableError: When the CARLA Python API cannot be imported.
    """

    try:
        return importlib.import_module("carla")
    except ModuleNotFoundError as exc:
        if exc.name != "carla":
            raise
        raise CarlaUnavailableError(
            "CARLA Python API package 'carla' is not importable. Install CARLA and ensure its "
            "Python API is on PYTHONPATH before using CARLA replay entry points."
        ) from exc
