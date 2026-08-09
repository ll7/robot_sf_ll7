"""Simulator backend modules for the Robot SF facade.

This package collects the backend modules that the simulator registry
(``robot_sf.sim.registry``) loads on demand. Each backend module exposes a
factory matching the ``robot_sf.sim.facade.SimulatorFactory`` contract:

- ``dummy_backend``: deterministic smoke-test backend providing
  ``DummySimulator`` and ``dummy_factory`` (always available).
- ``fast_pysf_backend``: fast-pysf physics backend providing
  ``fast_pysf_factory`` (requires the optional fast-pysf dependency).

Backend modules resolve lazily so that importing this package never pulls the
optional fast-pysf dependency chain: a missing optional dependency still lets
``import robot_sf.sim.backends`` succeed, and the registry keeps its usual
import-time skip behavior for the fast-pysf backend.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static type information only
    from . import dummy_backend, fast_pysf_backend

__all__ = ["dummy_backend", "fast_pysf_backend"]

_BACKEND_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Resolve backend submodules only when a caller requests them.

    Keeping package import light preserves the optional fast-pysf dependency
    policy: importing ``robot_sf.sim.backends`` must not eagerly import
    ``fast_pysf_backend`` or its optional dependency chain.

    Returns:
        The requested backend submodule.
    """
    if name in _BACKEND_EXPORTS:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exported backend submodule names in discovery.

    Returns:
        Available package attribute names.
    """
    return sorted(set(globals()) | set(__all__))
