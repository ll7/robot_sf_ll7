"""NumPy-free numerical thread environment bootstrap.

This module must stay dependency-free so command entry points can force thread
caps before importing NumPy, SciPy, or any benchmark module that reaches them.
"""

from __future__ import annotations

import os

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_thread_env_for_determinism() -> dict[str, str]:
    """Force numerical thread pools to one before scientific-library imports.

    Returns:
        The effective one-thread environment mapping.
    """
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"
    return {name: os.environ[name] for name in THREAD_ENV_VARS}
