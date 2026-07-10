"""Regression tests for lazy scenario-certification package imports (issue #5121).

These tests verify that importing a lightweight sub-module such as
``robot_sf.scenario_certification.failure_cause`` does not eagerly
initialise the sensor registry, shapely, or other heavy stacks — and
that the package public API surface remains accessible via lazy attribute
lookup.
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
    # failure_cause.py is pure stdlib; the package __init__ must not pull in
    # v1.py (shapely / gym_env / sensor-registry) when this submodule is imported.
    # This is the direct fix for issue #5121.
    "robot_sf.scenario_certification.failure_cause",
    # robot_sf.benchmark.scenario_failure_cause is intentionally excluded here:
    # it still triggers robot_sf.benchmark.__init__ which is heavy until
    # PR #5122 (lazy benchmark imports) merges.  Once that PR lands, add this
    # module to tests/benchmark/test_benchmark_package_lazy_imports.py instead
    # of here.
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
    """Importing a schema/constant module must not emit sensor-registry noise."""
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

    Budget: 5 seconds — generous for cold-start filesystem overhead while
    catching accidental sensor-registry eager init (~3.8 s before the fix).
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


_LIGHTWEIGHT_EXPORTS = [
    # From criticality_summary — no pysf/gym_env chain.
    "CRITICALITY_SUMMARY_SCHEMA_VERSION",
    "CriticalitySummaryV1",
    "build_criticality_summary_from_compact_evidence",
    "build_criticality_summary_from_pilot",
    "criticality_summary_to_dict",
    "validate_criticality_summary",
    # From perturbation_family_registry — pure dataclasses/stdlib only.
    "PerturbationFamily",
    "perturbation_families",
    "perturbation_family",
    "supported_perturbation_families",
    "validate_perturbation_family_parameters",
]


def test_scenario_certification_lightweight_names_resolvable() -> None:
    """Lazy exports from deps-free sub-modules must resolve without heavy chain."""
    import robot_sf.scenario_certification as sc

    missing = []
    for name in _LIGHTWEIGHT_EXPORTS:
        try:
            getattr(sc, name)
        except AttributeError:
            missing.append(name)
    assert not missing, f"Names not resolvable via lazy __getattr__: {missing}"


def test_scenario_certification_dir_includes_all_exports() -> None:
    """__dir__ must surface every public export for interactive discovery."""
    import robot_sf.scenario_certification as sc

    visible = set(dir(sc))
    missing = [n for n in sc.__all__ if n not in visible]
    assert not missing, f"Names missing from dir(robot_sf.scenario_certification): {missing}"


def test_lazy_export_caches_after_first_access() -> None:
    """A lazy export must be placed in module globals after first access."""
    import robot_sf.scenario_certification as sc

    # Use an export from criticality_summary (no heavy pysf/gym_env chain).
    _ = sc.CRITICALITY_SUMMARY_SCHEMA_VERSION
    assert "CRITICALITY_SUMMARY_SCHEMA_VERSION" in vars(sc)


def test_unknown_attribute_raises_attribute_error() -> None:
    """Accessing a name not in __all__ must raise AttributeError."""
    import robot_sf.scenario_certification as sc

    with pytest.raises(AttributeError, match="definitely_missing"):
        _ = sc.definitely_missing  # type: ignore[attr-defined]
