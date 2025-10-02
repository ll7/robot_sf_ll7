"""Runtime hardware inspection for multi-extractor training runs."""

from __future__ import annotations

import platform
from typing import Optional

from loguru import logger

from robot_sf.training.multi_extractor_models import HardwareProfile


def _collect_gpu_metadata() -> tuple[Optional[str], Optional[str]]:
    """Return (gpu_model, cuda_version) when CUDA is available."""

    try:
        import torch
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:  # pragma: no cover
        logger.debug("torch import failed during hardware probe: {}", exc)
        return None, None

    if not torch.cuda.is_available():
        logger.info("CUDA not available; recording CPU-only hardware profile")
        return None, None

    try:
        device_index = torch.cuda.current_device()
        gpu_model = torch.cuda.get_device_name(device_index)
        cuda_version = getattr(torch.version, "cuda", None)
        return gpu_model, cuda_version
    except (OSError, RuntimeError) as exc:  # pragma: no cover - relies on runtime GPU state
        logger.warning("Failed to capture CUDA metadata: {}", exc)
        return None, None


def collect_hardware_profile(*, worker_count: int, skip_gpu: bool = False) -> HardwareProfile:
    """Capture hardware traits for logging and summary reproduction."""

    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    platform_label = platform.platform()
    arch = platform.machine()
    python_version = platform.python_version()

    if skip_gpu:
        gpu_model, cuda_version = None, None
    else:
        gpu_model, cuda_version = _collect_gpu_metadata()

    profile = HardwareProfile(
        platform=platform_label,
        arch=arch,
        python_version=python_version,
        workers=worker_count,
        gpu_model=gpu_model,
        cuda_version=cuda_version,
    )

    logger.info(
        "Hardware profile captured",
        platform=platform_label,
        arch=arch,
        python_version=python_version,
        workers=worker_count,
        gpu_model=gpu_model,
        cuda_version=cuda_version,
    )

    return profile
