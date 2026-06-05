"""Shared helpers for oldest-to-newest temporal sensor stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def fill_history_stack(stacked_state: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    """Fill ``stacked_state`` in place with ``current_state`` in every temporal row.

    Returns
    -------
    np.ndarray
        The same ``stacked_state`` array, modified in place.
    """
    stacked_state[:] = current_state
    return stacked_state


def append_history_row(stacked_state: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    """Append ``current_state`` as the newest row in an oldest-to-newest stack.

    Returns
    -------
    np.ndarray
        Updated stack with previous rows shifted toward index ``0`` and the current state at
        index ``-1``.
    """
    stacked_state[:-1] = stacked_state[1:]
    stacked_state[-1] = current_state
    return stacked_state


def reset_history_stack(stacked_state: np.ndarray) -> np.ndarray:
    """Zero-fill ``stacked_state`` in place.

    Returns
    -------
    np.ndarray
        The same ``stacked_state`` array, modified in place.
    """
    stacked_state[:] = 0
    return stacked_state
