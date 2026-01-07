"""Matplotlib utility functions for backend management and plotting helpers."""

from __future__ import annotations

import os
import platform

import matplotlib
from loguru import logger


def is_headless_environment() -> bool:
    """Check if we're running in a headless environment.

    Returns:
        True if the environment is headless (no display available).

    Examples:
        >>> from robot_sf.common.matplotlib_utils import is_headless_environment
        >>> if is_headless_environment():
        ...     print("Running headless")
    """
    # Check for explicit headless environment variable
    if os.environ.get("MPLBACKEND") == "Agg":
        return True

    # Check for DISPLAY on Unix systems
    if platform.system() != "Windows" and not os.environ.get("DISPLAY"):
        return True

    # Check if current backend is Agg
    backend = matplotlib.get_backend().lower()
    return backend == "agg"


def _get_backend_candidates() -> list[str]:
    """Get list of backend candidates based on platform.

    Returns:
        List of backend names to try, in order of preference.
    """
    if platform.system() == "Darwin":  # macOS
        return ["MacOSX", "QtAgg", "Qt5Agg"]
    return ["QtAgg", "Qt5Agg", "TkAgg", "WXAgg"]


def _try_set_backend(backend: str, verbose: bool) -> bool:
    """Attempt to set a specific matplotlib backend.

    Args:
        backend: Backend name to try.
        verbose: If True, log attempt details.

    Returns:
        True if backend was successfully set, False otherwise.
    """
    try:
        matplotlib.use(backend, force=True)
        if verbose:
            logger.debug(f"Successfully set matplotlib backend to {backend}")
        return True
    except ImportError:
        if verbose:
            logger.debug(f"Backend {backend} not available")
        return False
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.debug(f"Failed to set backend {backend}: {e}")
        return False


def ensure_interactive_backend(verbose: bool = False) -> bool:
    """Switch away from headless Agg backend when possible to show plots interactively.

    This function attempts to switch to an interactive matplotlib backend
    suitable for the current platform. It tries multiple backends in order
    of preference until one succeeds.

    Platform-specific backend preferences:
    - macOS: MacOSX (native), QtAgg, Qt5Agg
    - Other platforms: QtAgg, Qt5Agg, TkAgg, WXAgg

    If no interactive backend is available, the current backend is left unchanged.

    Args:
        verbose: If True, log backend selection details.

    Returns:
        True if an interactive backend was successfully set, False otherwise.

    Examples:
        >>> from robot_sf.common.matplotlib_utils import ensure_interactive_backend
        >>> if ensure_interactive_backend():
        ...     # Interactive mode available
        ...     plt.show()
        ... else:
        ...     # Fall back to headless mode
        ...     plt.savefig("output.png")
    """
    # Check if already in headless mode explicitly
    if os.environ.get("MPLBACKEND") == "Agg":
        if verbose:
            logger.debug("MPLBACKEND=Agg set, staying in headless mode")
        return False

    # Check current backend
    current_backend = matplotlib.get_backend()
    if verbose:
        logger.debug(f"Current matplotlib backend: {current_backend}")

    # If already interactive, keep it
    if current_backend.lower() not in ("agg", "pdf", "ps", "svg"):
        if verbose:
            logger.debug(f"Already using interactive backend: {current_backend}")
        return True

    # Try each backend candidate
    for backend in _get_backend_candidates():
        if _try_set_backend(backend, verbose):
            return True

    # No interactive backend available
    if verbose:
        logger.warning("No interactive matplotlib backend available")
    return False


__all__ = ["ensure_interactive_backend", "is_headless_environment"]
