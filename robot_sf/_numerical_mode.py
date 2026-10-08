"""Opt-in, versioned numerical policy for prospective learned-policy campaigns."""

from __future__ import annotations

import os
import sys

PINNED_MODE = "pinned_float64_v1"
PINNED_ENV = {
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_CBWR": "COMPATIBLE",
    "OPENBLAS_CORETYPE": "Haswell",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def bootstrap_numerical_mode(mode: str | None) -> None:
    """Pin dispatch before libraries initialize; reject late initialization."""
    if mode is None:
        return
    if mode != PINNED_MODE:
        raise ValueError(f"Unsupported numerical_mode: {mode}")
    if any(name in sys.modules for name in ("numpy", "torch")):
        raise RuntimeError("Pinned numerical mode must be initialized before NumPy/Torch import")
    os.environ.update(PINNED_ENV)
    os.environ["ROBOT_SF_NUMERICAL_MODE"] = mode


def effective_numerical_mode() -> dict:
    """Observe the live kernel context.

    Returns:
        Effective dispatch flags and kernel policy, independent of requested config.
    """
    import torch  # noqa: PLC0415
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    blas = [
        {key: item.get(key) for key in ("internal_api", "version", "architecture", "num_threads")}
        for item in threadpool_info()
        if item.get("user_api") == "blas"
    ]
    return {
        "mode": os.environ.get("ROBOT_SF_NUMERICAL_MODE", "default"),
        "blas": blas,
        "kernel_env": {key: os.environ.get(key) for key in PINNED_ENV},
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def initialize_pinned_torch() -> None:
    """Apply Torch flags in every process that inherited the early bootstrap."""
    if os.environ.get("ROBOT_SF_NUMERICAL_MODE") == PINNED_MODE:
        import torch  # noqa: PLC0415

        torch.backends.mkldnn.enabled = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


def validate_numerical_mode(claim: dict, observed: dict) -> None:
    """Reject pinned claims without effective dispatch and actor dtype evidence."""
    if not isinstance(claim, dict) or not isinstance(observed, dict):
        raise ValueError("Pinned numerical mode requires mapping evidence")
    if claim.get("mode") != PINNED_MODE:
        raise ValueError("Unsupported numerical mode claim")
    expected = {
        "mode": PINNED_MODE,
        "kernel_env": PINNED_ENV,
        "torch_cpu_capability": "DEFAULT",
        "mkldnn_enabled": False,
        "deterministic_algorithms": True,
    }
    capability = observed.get("torch_cpu_capability")
    if capability not in {"DEFAULT", "NO AVX"}:
        raise ValueError("Pinned numerical mode requires DEFAULT Torch CPU dispatch")
    expected["torch_cpu_capability"] = capability
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned numerical mode does not match effective kernel context")
    blas = observed.get("blas", [])
    if not blas or any(
        item.get("num_threads") != 1
        or item.get("internal_api") != "openblas"
        or item.get("architecture") != "Haswell"
        for item in blas
    ):
        raise ValueError("Pinned numerical mode requires observed single-threaded Haswell OpenBLAS")
    if claim.get("inference_dtype") != "float64" or observed.get("inference_dtype") != "float64":
        raise ValueError("Pinned numerical mode requires observed float64 inference")
