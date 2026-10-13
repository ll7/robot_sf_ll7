"""Study-only interpreter hook; activates only in explicitly marked children.

The kernel-only mode retains the normal float32 actor. The supplemental float64
mode is selected separately by the canonical campaign configuration.
"""

import atexit
import json
import os
from pathlib import Path


def initialize():
    """Apply flags before policy imports and record each process's live context."""
    mode = os.environ.get("XCPU_MODE")
    # Float64 uses main's early bootstrap, which refuses prior library imports.
    if mode not in {"default", "pinned"}:
        return
    try:
        # Use the repository guard before touching Torch/SB3's lazy import graph.
        from robot_sf.common.seed import _set_torch_deterministic_algorithms  # noqa: I001

        import torch
        from threadpoolctl import threadpool_info

        if mode != "default":
            torch.backends.mkldnn.enabled = False
            if not _set_torch_deterministic_algorithms(torch, True):
                raise RuntimeError("deterministic algorithms setter unavailable")
            torch.set_num_threads(1)
            if torch.backends.cpu.get_cpu_capability() not in {"DEFAULT", "NO AVX"}:
                raise RuntimeError("kernel pin did not take effect")

        def record(phase):
            payload = {
                "mode": mode,
                "pid": os.getpid(),
                "phase": phase,
                "torch_version": torch.__version__,
                "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "torch_threads": torch.get_num_threads(),
                "blas": threadpool_info(),
            }
            root = Path(os.environ["XCPU_CONTEXT_ROOT"])
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{os.getpid()}-{phase}.json").write_text(json.dumps(payload, indent=2))

        record("initial")
        atexit.register(record, "final")
        os.register_at_fork(after_in_child=lambda: record("fork"))
    except Exception:  # noqa: BLE001 - sitecustomize must fail closed on every startup error.
        # Python otherwise ignores sitecustomize errors and continues unpinned.
        import traceback

        traceback.print_exc()
        os._exit(70)


initialize()
