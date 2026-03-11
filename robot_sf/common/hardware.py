"""Hardware-capacity helpers for training/runtime resource planning.

The helpers in this module are intentionally lightweight and avoid shelling out to
cluster-specific tools. They rely on environment variables exposed by schedulers
such as Slurm, CUDA visibility settings, and optional torch runtime checks.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from typing import Final

_AUTO_DISABLED_GPU_TOKEN: Final[str] = "-1"
_CPU_ENV_KEYS: Final[tuple[str, ...]] = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")
_GPU_ENV_KEYS: Final[tuple[str, ...]] = ("SLURM_GPUS_ON_NODE", "SLURM_GPUS", "SLURM_JOB_GPUS")


@dataclass(slots=True, frozen=True)
class HardwareCapacity:
    """Detected CPU/GPU capacity visible to the current process.

    Attributes:
        logical_cpus: Total logical CPUs reported by the host OS.
        allocated_cpus: Scheduler-allocated CPUs, if detectable.
        usable_cpus: CPU budget after applying reserve/minimum constraints.
        allocated_gpus: Scheduler-allocated GPU count, if detectable.
        visible_gpus: GPU count visible to the current process.
    """

    logical_cpus: int
    allocated_cpus: int | None
    usable_cpus: int
    allocated_gpus: int | None
    visible_gpus: int


def _parse_positive_int(raw: str | None) -> int | None:
    """Extract the first positive integer from text values like ``24`` or ``24(x2)``.

    Returns:
        int | None: Parsed non-negative integer, or ``None`` when parsing fails.
    """
    if raw is None:
        return None
    match = re.match(r"^\s*(\d+)", raw)
    if match is None:
        return None
    value = int(match.group(1))
    return value if value >= 0 else None


def _count_gpu_token(token: str, *, treat_numeric_as_device_id: bool = False) -> int | None:
    """Parse one GPU allocation token into a concrete device count.

    Returns:
        int | None: Concrete count for one token, or ``None`` if token is unknown.
    """
    item = token.strip()
    if not item:
        return None
    if item == _AUTO_DISABLED_GPU_TOKEN:
        return 0

    numeric = _parse_positive_int(item)
    if numeric is not None and item.isdigit():
        return 1 if treat_numeric_as_device_id else numeric

    # Slurm formats like "gpu:a30:1".
    if ":" in item:
        suffix = item.rsplit(":", maxsplit=1)[-1]
        numeric_suffix = _parse_positive_int(suffix)
        if numeric_suffix is not None:
            return numeric_suffix

    # Device ranges like "0-3".
    range_match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", item)
    if range_match is not None:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if end >= start:
            return (end - start) + 1

    # Device IDs like "0" were handled above; unresolved tokens are unknown.
    return None


def _parse_gpu_count(raw: str | None, *, treat_numeric_as_device_id: bool = False) -> int | None:
    """Parse GPU count from Slurm/CUDA environment variable formats.

    Returns:
        int | None: Total parsed GPU count, or ``None`` when format is unsupported.
    """
    if raw is None:
        return None
    tokens = [token for token in raw.split(",") if token.strip()]
    if not tokens:
        return None

    counts = [
        _count_gpu_token(token, treat_numeric_as_device_id=treat_numeric_as_device_id)
        for token in tokens
    ]
    if any(count is None for count in counts):
        return None

    return int(sum(int(count) for count in counts if count is not None))


def _env_first_count(keys: tuple[str, ...], parser) -> int | None:
    """Return the first parseable count from the provided environment keys."""
    for key in keys:
        parsed = parser(os.environ.get(key))
        if parsed is not None:
            return parsed
    return None


def _torch_visible_gpu_count() -> int | None:
    """Best-effort torch CUDA device count probe.

    Returns:
        int | None: Visible CUDA device count (or zero), ``None`` if probing fails.
    """
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return None
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except (AttributeError, OSError, RuntimeError, TypeError):
        return None


def detect_hardware_capacity(
    *, reserve_cpu_cores: int = 0, minimum_cpus: int = 1
) -> HardwareCapacity:
    """Detect CPU/GPU capacity for the current process context.

    Args:
        reserve_cpu_cores: Number of CPU cores to keep free for system/runtime overhead.
        minimum_cpus: Minimum usable CPU count to report.

    Returns:
        HardwareCapacity: Structured CPU/GPU capacity snapshot.
    """
    if minimum_cpus < 1:
        raise ValueError("minimum_cpus must be >= 1")
    if reserve_cpu_cores < 0:
        raise ValueError("reserve_cpu_cores must be >= 0")

    logical_cpus = max(1, os.cpu_count() or 1)
    allocated_cpus = _env_first_count(_CPU_ENV_KEYS, _parse_positive_int)
    cpu_budget = allocated_cpus if allocated_cpus is not None else logical_cpus
    usable_cpus = max(minimum_cpus, cpu_budget - reserve_cpu_cores)

    allocated_gpus = _env_first_count(_GPU_ENV_KEYS, _parse_gpu_count)
    cuda_visible = _parse_gpu_count(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        treat_numeric_as_device_id=True,
    )
    torch_visible = _torch_visible_gpu_count()
    if cuda_visible is not None:
        visible_gpus = max(0, cuda_visible)
    elif allocated_gpus is not None:
        visible_gpus = max(0, allocated_gpus)
    elif torch_visible is not None:
        visible_gpus = max(0, torch_visible)
    else:
        visible_gpus = 0

    return HardwareCapacity(
        logical_cpus=logical_cpus,
        allocated_cpus=allocated_cpus,
        usable_cpus=usable_cpus,
        allocated_gpus=allocated_gpus,
        visible_gpus=visible_gpus,
    )


def recommend_env_runners(capacity: HardwareCapacity, *, cpu_headroom: int = 4) -> int:
    """Recommend RL env-runner count from detected CPU capacity.

    Returns:
        int: Suggested env-runner count after reserving CPU headroom.
    """
    if cpu_headroom < 0:
        raise ValueError("cpu_headroom must be >= 0")
    return max(1, capacity.usable_cpus - cpu_headroom)


__all__ = ["HardwareCapacity", "detect_hardware_capacity", "recommend_env_runners"]
