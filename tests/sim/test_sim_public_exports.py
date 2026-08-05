"""Focused export contract for the robot_sf.sim facade and backends package.

Guards the reviewed ``__all__`` surface for issue #6486: every declared export
resolves to the pre-change object on its pre-change import path, no declared
name is missing, and missing, misspelled, or stale names never leak into the
public surface.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import robot_sf.sim as sim_facade
import robot_sf.sim.backends as backends_facade

SIM_FACADE_ALL = ["_assert_fast_pysf_initialized"]
BACKENDS_FACADE_ALL = ["dummy_backend", "fast_pysf_backend"]
DUMMY_BACKEND_ALL = ["DummySimulator", "dummy_factory"]
FAST_PYSF_BACKEND_ALL = ["fast_pysf_factory"]


def test_sim_facade_declares_the_reviewed_guard_surface() -> None:
    """The facade exports exactly the reviewed fast-pysf initialization guard."""
    assert sim_facade.__all__ == SIM_FACADE_ALL
    assert set(sim_facade.__all__) <= set(dir(sim_facade))


def test_backends_facade_declares_the_reviewed_backend_surface() -> None:
    """The backends package exports exactly the two backend submodules."""
    assert backends_facade.__all__ == BACKENDS_FACADE_ALL
    assert set(backends_facade.__all__) <= set(dir(backends_facade))


@pytest.mark.parametrize("name", SIM_FACADE_ALL)
def test_sim_facade_exports_resolve_on_pre_change_paths(name: str) -> None:
    """Every declared facade export resolves with its pre-change identity."""
    export = getattr(sim_facade, name)
    assert export.__module__ == "robot_sf.sim"
    assert export.__qualname__ == name


@pytest.mark.parametrize("name", BACKENDS_FACADE_ALL)
def test_backend_exports_resolve_on_pre_change_paths(name: str) -> None:
    """Every declared backend export resolves to the pre-change submodule."""
    module = importlib.import_module(f"robot_sf.sim.backends.{name}")
    assert getattr(backends_facade, name) is module
    assert module.__name__ == f"robot_sf.sim.backends.{name}"


@pytest.mark.parametrize("name", BACKENDS_FACADE_ALL)
def test_backends_lazy_resolver_returns_the_submodule(name: str) -> None:
    """The lazy resolver returns the backend submodule on demand.

    Calls the resolver directly so the branch stays covered even when other
    test modules already imported the submodule and bound the attribute.
    """
    resolved = backends_facade.__getattr__(name)
    assert resolved is importlib.import_module(f"robot_sf.sim.backends.{name}")


def test_dummy_backend_declares_its_factory_surface() -> None:
    """dummy_backend exports DummySimulator and dummy_factory unchanged."""
    dummy_backend = importlib.import_module("robot_sf.sim.backends.dummy_backend")
    assert dummy_backend.__all__ == DUMMY_BACKEND_ALL
    for name in DUMMY_BACKEND_ALL:
        export = getattr(dummy_backend, name)
        assert export.__module__ == "robot_sf.sim.backends.dummy_backend"
        assert export.__qualname__ == name


def test_fast_pysf_backend_declares_its_factory_surface() -> None:
    """fast_pysf_backend exports fast_pysf_factory unchanged."""
    fast_pysf_backend = importlib.import_module("robot_sf.sim.backends.fast_pysf_backend")
    assert fast_pysf_backend.__all__ == FAST_PYSF_BACKEND_ALL
    for name in FAST_PYSF_BACKEND_ALL:
        export = getattr(fast_pysf_backend, name)
        assert export.__module__ == "robot_sf.sim.backends.fast_pysf_backend"
        assert export.__qualname__ == name


@pytest.mark.parametrize(
    "name",
    [
        "fast_pysf_backend",  # stale pre-package robot_sf.sim.fast_pysf_backend path
        "Simulator",  # belongs to robot_sf.sim.simulator, not the facade
        "init_simulators",  # belongs to robot_sf.sim.simulator, not the facade
        "_has_installed_pysocialforce",  # intentionally private guard helper
    ],
)
def test_sim_facade_keeps_foreign_stale_and_private_names_unexported(name: str) -> None:
    """Foreign, stale, and private symbols stay out of the facade export list."""
    assert name not in sim_facade.__all__


@pytest.mark.parametrize(
    "name",
    [
        "dumy_backend",  # misspelled submodule
        "fast_pysf_factry",  # misspelled factory
        "dummy_factory",  # factory stays on the dummy_backend submodule
        "fast_pysf_factory",  # factory stays on the fast_pysf_backend submodule
    ],
)
def test_backends_facade_rejects_missing_misspelled_and_stale_names(name: str) -> None:
    """Misspelled and stale names are neither declared nor resolvable."""
    assert name not in backends_facade.__all__
    with pytest.raises(AttributeError):
        getattr(backends_facade, name)


def test_backends_package_import_stays_lazy() -> None:
    """Importing the package must not import any backend module eagerly.

    Protects the optional fast-pysf dependency policy: ``import
    robot_sf.sim.backends`` must succeed and stay cheap even when the optional
    dependency is missing.
    """
    script = (
        "import sys\n"
        "import robot_sf.sim.backends as backends\n"
        "assert backends.__all__ == ['dummy_backend', 'fast_pysf_backend']\n"
        "assert 'robot_sf.sim.backends.dummy_backend' not in sys.modules\n"
        "assert 'robot_sf.sim.backends.fast_pysf_backend' not in sys.modules\n"
        "print('lazy-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy-ok" in result.stdout
