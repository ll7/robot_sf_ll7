"""Matplotlib utility functions for backend management and plotting helpers."""

from __future__ import annotations

import matplotlib


def ensure_interactive_backend() -> None:
    """Switch away from headless Agg backend when possible to show plots interactively.

    This function attempts to switch to an interactive matplotlib backend
    (MacOSX or TkAgg) if the current backend is Agg (non-interactive).
    It's useful when running scripts that may be in headless mode but
    should display plots when possible.

    If no interactive backend is available, the current backend is left unchanged.

    Examples:
        >>> from robot_sf.common.matplotlib_utils import ensure_interactive_backend
        >>> ensure_interactive_backend()
        >>> # Now matplotlib will use an interactive backend if available
    """
    backend = matplotlib.get_backend().lower()
    if backend != "agg":
        return
    for candidate in ("MacOSX", "TkAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:  # noqa: BLE001 - Fallback logic for unavailable backends
            continue


__all__ = ["ensure_interactive_backend"]
