"""Backend registry for simulator creation.

Provides register/get/list helpers and registers the default "fast-pysf" backend.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # avoid runtime imports
    from robot_sf.sim.facade import SimulatorFactory

_REGISTRY: dict[str, SimulatorFactory] = {}
_BACKEND_PERFORMANCE_ORDER = {
    "fast-pysf": 0,
    "dummy": 100,
}
_DEFAULT_PERFORMANCE_SCORE = 1000


def register_backend(key: str, factory: SimulatorFactory, *, override: bool = False) -> None:
    """TODO docstring. Document this function.

    Args:
        key: TODO docstring.
        factory: TODO docstring.
        override: TODO docstring.
    """
    k = key.strip()
    if not k:
        raise ValueError("Backend key must be a non-empty string")
    if k in _REGISTRY and not override:
        raise KeyError(f"Backend '{k}' already registered; pass override=True to replace")
    _REGISTRY[k] = factory
    logger.info("Registered simulator backend: {}", k)


def get_backend(key: str) -> SimulatorFactory:
    """TODO docstring. Document this function.

    Args:
        key: TODO docstring.

    Returns:
        TODO docstring.
    """
    try:
        return _REGISTRY[key]
    except KeyError as e:  # provide suggestions
        known = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown backend '{key}'. Known: {known}") from e


def list_backends() -> list[str]:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    return sorted(_REGISTRY.keys())


def select_best_backend(preferred: str | None = None) -> str:
    """Return the best available backend, optionally honoring a preferred choice.

    Args:
        preferred: Explicit backend name to try first. If ``None`` the selector
            also checks the ``ROBOT_SF_BACKEND`` environment variable before
            evaluating the registry.

    Returns:
        The chosen backend key.

    Raises:
        RuntimeError: If no backends are registered.
    """

    available = list_backends()
    if not available:
        raise RuntimeError("No simulator backends are registered")

    explicit_choice = preferred or os.environ.get("ROBOT_SF_BACKEND")
    if explicit_choice:
        explicit_choice = explicit_choice.strip()
        if explicit_choice in available:
            logger.info("Using explicitly requested backend: {}", explicit_choice)
            return explicit_choice
        logger.warning(
            "Requested backend '{}' not available (known: {})",
            explicit_choice,
            ", ".join(available),
        )

    def _score(name: str) -> tuple[int, str]:
        """TODO docstring. Document this function.

        Args:
            name: TODO docstring.

        Returns:
            TODO docstring.
        """
        return (_BACKEND_PERFORMANCE_ORDER.get(name, _DEFAULT_PERFORMANCE_SCORE), name)

    best = min(available, key=_score)
    logger.info("Selected backend '{}' based on performance preference", best)
    return best


# Default backend registration (fast-pysf)
# Register default backends on import
try:
    from robot_sf.sim.backends.fast_pysf_backend import fast_pysf_factory

    register_backend("fast-pysf", fast_pysf_factory, override=True)
except (ValueError, KeyError) as _e:  # pragma: no cover - defensive
    logger.warning("Failed to register default fast-pysf backend: {}", _e)

try:
    from robot_sf.sim.backends.dummy_backend import dummy_factory

    register_backend("dummy", dummy_factory, override=True)
except (ValueError, KeyError) as _e:  # pragma: no cover - defensive
    logger.warning("Failed to register dummy backend: {}", _e)
