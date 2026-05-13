"""Shared helpers for oldest-to-newest temporal sensor stacks."""

from __future__ import annotations

import numpy as np


def fill_history_stack(stacked_state: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    """Return a stack prefilled with ``current_state`` in every temporal row."""
    return np.repeat(current_state[np.newaxis, :], stacked_state.shape[0], axis=0).astype(
        stacked_state.dtype, copy=False
    )


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
    """Return a zero-filled stack preserving the existing shape and dtype."""
    return np.zeros_like(stacked_state)
