"""Seed utilities for deterministic experiments.

Provides helpers to set and inspect global RNG state across Python's random,
NumPy, and optionally PyTorch (if installed). Designed for use in benchmark
runners and scripts to harden reproducibility guarantees.
"""

from __future__ import annotations

import importlib
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class SeedReport:
    """Summary of applied seed settings."""

    seed: int
    deterministic: bool
    has_torch: bool
    torch_deterministic: bool | None = None
    torch_benchmark: bool | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the dataclass as a regular dictionary for logging/JSON.

        Returns:
            dict[str, Any]: Dictionary representation suitable for serialization.
        """
        return asdict(self)


def _import_torch():
    """Import torch lazily and return the module when available.

    Returns:
        Any | None: Imported torch module when available, otherwise None.
    """
    try:
        return importlib.import_module("torch")  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None


def _set_torch_deterministic_algorithms(torch_module: Any, deterministic: bool) -> bool:
    """Set torch's deterministic-algorithm flag without the Torch 2.13.0 CI crash path.

    Torch 2.13.0 on Python 3.12 imports ``torch._inductor.config`` from the public
    ``use_deterministic_algorithms`` wrapper.  That import reaches Dynamo/Triton and can
    segfault in the repository's test and smoke workers.  The C-level setter is already
    loaded with torch and preserves the operation-level determinism flag without loading
    Inductor.  Other interpreter/version combinations retain the public API path.

    Args:
        torch_module: Imported torch module or a compatible test double.
        deterministic: Whether deterministic algorithms should be enabled.

    Returns:
        bool: Whether the requested flag was applied.
    """
    public_setter = getattr(torch_module, "use_deterministic_algorithms", None)
    if public_setter is None:
        return False

    version = str(getattr(torch_module, "__version__", "")).split("+", 1)[0]
    if sys.version_info[:2] == (3, 12) and version == "2.13.0":
        c_module = getattr(torch_module, "_C", None)
        c_setter = getattr(c_module, "_set_deterministic_algorithms", None)
        if c_setter is None:
            return False
        c_setter(bool(deterministic), warn_only=False)
        return True

    public_setter(bool(deterministic))
    return True


def set_global_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """Set global seeds for ``random``, ``numpy``, and torch (if available).

    Args:
        seed: Global seed applied to all supported generators.
        deterministic: When ``True``, request deterministic torch behavior
            (CUDNN deterministic kernels, ``use_deterministic_algorithms``) and disable
            the CUDNN autotuner. Ignored when torch is unavailable.

    Returns:
        SeedReport: Summary of the applied settings for logging/debugging.
    """
    # Core RNGs
    random.seed(seed)
    np.random.seed(seed)

    # Torch (optional)
    torch = _import_torch()
    report = SeedReport(seed=seed, deterministic=deterministic, has_torch=torch is not None)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Determinism flags
            determinism_applied = _set_torch_deterministic_algorithms(torch, deterministic)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = bool(deterministic)
                # Disable CUDNN autotuner for determinism (perf tradeoff).
                torch.backends.cudnn.benchmark = bool(not deterministic)
                report.torch_deterministic = bool(deterministic) if determinism_applied else None
                report.torch_benchmark = bool(torch.backends.cudnn.benchmark)
            if not determinism_applied:
                report.notes = "torch deterministic algorithm flag could not be applied"
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - rare platforms
            report.notes = f"torch seed partially applied: {exc}"

    # Matplotlib headless safety for plotting in tests/CI
    os.environ.setdefault("MPLBACKEND", "Agg")

    return report


def get_seed_state_sample(n: int = 5) -> dict[str, Any]:
    """Return small sample sequences for quick sanity checks.

    Returns:
        dict[str, Any]: Example sequences for Python's `random` and NumPy RNGs.
    """
    # Python random
    rand_seq = [random.random() for _ in range(n)]
    # NumPy
    np_seq = np.random.random(n).tolist()
    return {"random": rand_seq, "numpy": np_seq}
