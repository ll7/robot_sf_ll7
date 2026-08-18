"""Dependency-free execution-context provenance primitives.

The exact-repeat and result-provenance paths need the same numerical context
vocabulary without importing NumPy, Numba, or the benchmark package.  Host
identity is deliberately not part of the canonical context: two distinct
hosts can be numerically comparable when their execution contexts match.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf._numerical_thread_env import THREAD_ENV_VARS

if TYPE_CHECKING:
    from collections.abc import Mapping

EXECUTION_CONTEXT_SCHEMA_VERSION = "benchmark_execution_context.v1"

EXECUTION_CONTEXT_FIELDS = (
    "schema_version",
    "cpu_model",
    "platform",
    "python_version",
    "thread_env",
    "numpy_version",
    "numba_version",
    "cpu_only",
    "workers",
)


def cpu_model() -> str:
    """Return a best-effort CPU model string without importing numeric stacks."""
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.lower().startswith("model name") and ":" in line:
                    model = line.split(":", 1)[1].strip()
                    if model:
                        return model
    except OSError:
        pass
    return platform.processor() or "Unknown CPU"


def build_execution_context(
    *,
    numpy_version: str | None = None,
    numba_version: str | None = None,
    cpu_only: bool | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Build the canonical numerical execution context.

    Every field is optional and omitted when the caller cannot observe it, so
    the context never asserts an execution mode it did not verify.  The NumPy
    and Numba values are supplied by callers after those libraries are
    imported.  ``cpu_only`` and ``workers`` are supplied only by callers that
    enforce or observe the execution mode (for example the exact-repeat path,
    whose contract is CPU-only single-worker execution); general result
    provenance records the real worker count in its own run metadata instead of
    restating it here.

    Returns:
        A JSON-serialisable canonical execution-context mapping.
    """
    context: dict[str, Any] = {
        "schema_version": EXECUTION_CONTEXT_SCHEMA_VERSION,
        "cpu_model": cpu_model(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }
    if numpy_version is not None:
        context["numpy_version"] = str(numpy_version)
    if numba_version is not None:
        context["numba_version"] = str(numba_version)
    if cpu_only is not None:
        context["cpu_only"] = bool(cpu_only)
    if workers is not None:
        context["workers"] = int(workers)
    return context


def execution_context_digest(context: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 digest of a canonical context."""
    normalized = dict(context)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def public_machine_id(machine_id: str) -> str:
    """Return a stable, non-reversible host label for public comparison output."""
    digest = hashlib.sha256(machine_id.encode("utf-8")).hexdigest()
    return f"host-{digest[:16]}"


__all__ = [
    "EXECUTION_CONTEXT_FIELDS",
    "EXECUTION_CONTEXT_SCHEMA_VERSION",
    "build_execution_context",
    "cpu_model",
    "execution_context_digest",
    "public_machine_id",
]
