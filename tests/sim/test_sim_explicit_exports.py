"""Contract tests for the explicit public exports of ``robot_sf.sim``.

Issue #6486 asks for explicit, reviewed ``__all__`` declarations on the
``robot_sf.sim`` facade and its backend package so that downstream consumers
have a stable, documented import surface. These tests lock the public surface
and prove every declared name actually resolves, so a missing, misspelled, or
stale entry fails CI instead of silently widening the API.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

_EXPECTED_ALL: dict[str, list[str]] = {
    "robot_sf.sim": ["_assert_fast_pysf_initialized"],
    "robot_sf.sim.backends": ["dummy_backend", "fast_pysf_backend"],
    "robot_sf.sim.backends.dummy_backend": ["DummySimulator", "dummy_factory"],
    "robot_sf.sim.backends.fast_pysf_backend": ["fast_pysf_factory"],
}

_SUBMODULE_EXPORTS = {"robot_sf.sim.backends"}


@pytest.mark.parametrize("module_name", sorted(_EXPECTED_ALL))
def test_public_all_matches_expected_surface(module_name: str) -> None:
    """Verify each module's ``__all__`` matches the reviewed public surface."""
    module = importlib.import_module(module_name)
    declared = sorted(module.__all__)
    expected = sorted(_EXPECTED_ALL[module_name])
    assert declared == expected
    assert len(module.__all__) == len(set(module.__all__)), "duplicate __all__ entries"


@pytest.mark.parametrize(
    ("module_name", "entry"),
    [(module_name, entry) for module_name, entries in _EXPECTED_ALL.items() for entry in entries],
)
def test_every_all_entry_resolves(module_name: str, entry: str) -> None:
    """Verify every declared export resolves to a real object or submodule."""
    module = importlib.import_module(module_name)

    if module_name in _SUBMODULE_EXPORTS:
        submodule = importlib.import_module(f"{module_name}.{entry}")
        assert submodule is not None
    else:
        value = getattr(module, entry)
        assert value is not None


def test_exports_keep_import_identities() -> None:
    """Declared exports must be the same objects consumers already import."""
    from robot_sf.sim import _assert_fast_pysf_initialized
    from robot_sf.sim.backends.dummy_backend import DummySimulator, dummy_factory
    from robot_sf.sim.backends.fast_pysf_backend import fast_pysf_factory

    sim_module = importlib.import_module("robot_sf.sim")
    dummy_module = importlib.import_module("robot_sf.sim.backends.dummy_backend")
    fast_module = importlib.import_module("robot_sf.sim.backends.fast_pysf_backend")

    assert sim_module._assert_fast_pysf_initialized is _assert_fast_pysf_initialized
    assert dummy_module.DummySimulator is DummySimulator
    assert dummy_module.dummy_factory is dummy_factory
    assert fast_module.fast_pysf_factory is fast_pysf_factory
    assert inspect.isfunction(_assert_fast_pysf_initialized)
    assert inspect.isfunction(fast_pysf_factory)
