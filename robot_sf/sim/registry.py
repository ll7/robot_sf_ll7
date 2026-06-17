"""Backend registry for simulator creation.

Provides register/get/list helpers and registers the default "fast-pysf" backend.
"""

from __future__ import annotations

import importlib
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
_FAST_PYSF_IMPORT_PATHS = {
    "robot_sf.sim.backends.fast_pysf_backend",
    "robot_sf.sim.fast_pysf_backend",
    "pysocialforce",
}


def _is_fast_pysf_dependency_error(error: ImportError) -> bool:
    """Return whether an import error belongs to optional fast-pysf requirements."""
    missing = getattr(error, "name", None)
    if not isinstance(missing, str):
        return False
    return any(missing == path or missing.startswith(f"{path}.") for path in _FAST_PYSF_IMPORT_PATHS)


def register_backend(key: str, factory: SimulatorFactory, *, override: bool = False) -> None:
    """Register a simulator backend factory under a string key.

    Args:
        key: Backend identifier used by factory and configuration code.
        factory: Callable that creates a simulator facade for this backend.
        override: Replace an existing registration when ``True``.
    """
    k = key.strip()
    if not k:
        raise ValueError("Backend key must be a non-empty string")
    already_registered = k in _REGISTRY
    if already_registered and not override:
        raise KeyError(f"Backend '{k}' already registered; pass override=True to replace")
    _REGISTRY[k] = factory
    if override and already_registered:
        # Subprocess workers re-import this module; keep these re-registration logs quiet.
        logger.debug("Re-registered simulator backend (override): {}", k)
    else:
        logger.info("Registered simulator backend: {}", k)


def get_backend(key: str) -> SimulatorFactory:
    """Return the simulator backend factory registered for ``key``.

    Args:
        key: Backend identifier to resolve.

    Returns:
        Registered simulator factory.
    """
    try:
        return _REGISTRY[key]
    except KeyError as e:  # provide suggestions
        known = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown backend '{key}'. Known: {known}") from e


def list_backends() -> list[str]:
    """List registered simulator backend keys.


    Returns:
        Backend keys sorted lexicographically.
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
        """Return the backend preference score used for automatic selection.

        Args:
            name: Registered backend key.

        Returns:
            Tuple ordered by performance preference, then name for stable ties.
        """
        return (_BACKEND_PERFORMANCE_ORDER.get(name, _DEFAULT_PERFORMANCE_SCORE), name)

    best = min(available, key=_score)
    logger.info("Selected backend '{}' based on performance preference", best)
    return best


def _register_optional_fast_pysf_backend() -> None:
    """Register the default fast-pysf backend when optional dependencies exist."""
    try:
        fast_pysf_backend = importlib.import_module(
            "robot_sf.sim.backends.fast_pysf_backend",
        )
        fast_pysf_factory = fast_pysf_backend.fast_pysf_factory
        register_backend("fast-pysf", fast_pysf_factory, override=True)
    except (ValueError, KeyError) as error:
        logger.warning("Failed to register default fast-pysf backend: {}", error)
    except ImportError as error:
        if _is_fast_pysf_dependency_error(error):
            logger.warning(
                "fast-pysf backend is unavailable/skipped: dependency '{}' is missing.",
                getattr(error, "name", "unknown"),
            )
            return
        raise


# Default backend registration (fast-pysf)
# Register default backends on import
_register_optional_fast_pysf_backend()

try:
    from robot_sf.sim.backends.dummy_backend import dummy_factory

    register_backend("dummy", dummy_factory, override=True)
except (ValueError, KeyError) as _e:  # pragma: no cover - defensive
    logger.warning("Failed to register dummy backend: {}", _e)
