"""Seed utilities for deterministic experiments.

Provides helpers to set and inspect global RNG state across Python's random,
NumPy, and optionally PyTorch (if installed). Designed for use in benchmark
runners and scripts to harden reproducibility guarantees.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SeedReport:
    """Summary of applied seed settings."""

    seed: int
    deterministic: bool
    has_torch: bool
    torch_deterministic: Optional[bool] = None
    torch_benchmark: Optional[bool] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:  # pragma: no cover - optional dependency
        return None


def set_global_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """Set global seeds for random, numpy, and torch (if available).

    Parameters
    - seed: Global seed to apply.
    - deterministic: If True, set torch.backends.cudnn.deterministic True and
      disable CUDNN benchmarking for deterministic convs (if torch available).

    Returns
    - SeedReport with applied settings.
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
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(deterministic)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = bool(deterministic)
                # Disable CUDNN autotuner for determinism (perf tradeoff).
                torch.backends.cudnn.benchmark = bool(not deterministic)
                report.torch_deterministic = bool(deterministic)
                report.torch_benchmark = bool(torch.backends.cudnn.benchmark)
        except Exception as exc:  # pragma: no cover - rare platforms
            report.notes = f"torch seed partially applied: {exc}"

    # Matplotlib headless safety for plotting in tests/CI
    os.environ.setdefault("MPLBACKEND", "Agg")

    return report


def get_seed_state_sample(n: int = 5) -> Dict[str, Any]:
    """Return small sample sequences for quick sanity checks."""
    # Python random
    rand_seq = [random.random() for _ in range(n)]
    # NumPy
    np_seq = np.random.random(n).tolist()
    return {"random": rand_seq, "numpy": np_seq}
