"""Regression tests for lazy benchmark package imports (issue #5090).

These tests verify that importing a lightweight ``robot_sf.benchmark``
sub-module does not eagerly initialise TensorFlow, simulator registries,
or other heavy stacks — and that the package public API surface remains
accessible via lazy attribute lookup.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

# --- subprocess isolation helpers -----------------------------------------


def _run_import(code: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# --- startup-noise tests --------------------------------------------------


_LIGHTWEIGHT_MODULES = [
    # These modules have no transitive heavy deps (TF / simulator-registry).
    # scenario_failure_cause is intentionally excluded: it imports from
    # robot_sf.scenario_certification.failure_cause which itself triggers the
    # sensor registry.  Fixing that chain is out of scope for #5090.
    "robot_sf.benchmark.errors",
    "robot_sf.benchmark.benchmark_protocol",
    "robot_sf.benchmark.helper_registry",
]

_HEAVY_INIT_PATTERNS = [
    "oneDNN",
    "cpu_feature_guard",
    "TensorFlow",
    "Registered sensor",
    "Registered simulator backend",
]


@pytest.mark.parametrize("module", _LIGHTWEIGHT_MODULES)
def test_lightweight_module_no_heavy_stderr(module: str) -> None:
    """Importing a schema/contract module must not emit TF or simulator noise."""
    result = _run_import(f"import {module}; print('OK')")
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert result.stdout.strip() == "OK"
    for pattern in _HEAVY_INIT_PATTERNS:
        assert pattern not in result.stderr, (
            f"Importing {module!r} emitted heavy-stack noise (pattern {pattern!r}).\n"
            f"stderr snippet: {result.stderr[:500]}"
        )


@pytest.mark.parametrize("module", _LIGHTWEIGHT_MODULES)
def test_lightweight_module_fast_startup(module: str) -> None:
    """Importing a lightweight module must complete within a loose budget.

    Budget: 5 seconds. The budget is intentionally generous to accommodate
    cold-start filesystem overhead while still catching accidental TF eager
    initialisation (~8–10 s on this machine before the fix).
    """
    t0 = time.monotonic()
    result = _run_import(f"import {module}; print('OK')")
    elapsed = time.monotonic() - t0
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert elapsed < 5.0, (
        f"Importing {module!r} took {elapsed:.1f}s — exceeds 5 s budget. "
        "A heavy dependency was likely pulled in eagerly."
    )


# --- public API surface ---------------------------------------------------


def test_benchmark_all_names_resolvable() -> None:
    """Every name in robot_sf.benchmark.__all__ must be resolvable."""
    import robot_sf.benchmark as b

    missing = []
    for name in b.__all__:
        try:
            getattr(b, name)
        except AttributeError:
            missing.append(name)
    assert not missing, f"Names in __all__ not resolvable: {missing}"


def test_benchmark_dir_includes_all_exports() -> None:
    """__dir__ must surface every public export for interactive discovery."""
    import robot_sf.benchmark as b

    visible = set(dir(b))
    missing = [n for n in b.__all__ if n not in visible]
    assert not missing, f"Names missing from dir(robot_sf.benchmark): {missing}"


def test_aggregation_metadata_error_importable() -> None:
    """AggregationMetadataError must remain importable from the package root."""
    from robot_sf.benchmark import AggregationMetadataError

    assert issubclass(AggregationMetadataError, (ValueError, Exception))


def test_lazy_export_caches_after_first_access() -> None:
    """A lazy export must be placed in the module globals after first access."""
    import robot_sf.benchmark as b

    # Access a name that was not imported eagerly.
    _ = b.SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION

    # After first access, the name is cached and __getattr__ is not re-called.
    assert "SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION" in vars(b)
