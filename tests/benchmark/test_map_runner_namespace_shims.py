"""Contract tests for the map-runner namespace migration (issue #6905 / #7536)."""

from __future__ import annotations

import importlib
import sys

import pytest

import robot_sf.benchmark.map_runner as map_runner_package
import robot_sf.benchmark.map_runner_env as flat_env_alias
from robot_sf.benchmark.map_runner import map_runner_env as canonical_env
from robot_sf.benchmark.map_runner.map_runner import run_map_batch

# The flat shims are imported at module level so the full CI suite measures
# their execution during test collection in every shard, not only inside the
# parametrized test below.
FLAT_SHIM_MODULES: tuple[str, ...] = (
    "map_runner_batch_plan",
    "map_runner_batch_runner",
    "map_runner_batch_summary",
    "map_runner_episode",
    "map_runner_identity",
    "map_runner_jsonl",
    "map_runner_metrics",
    "map_runner_native_command",
    "map_runner_observations",
    "map_runner_provenance",
    "map_runner_static_deadlock",
    "map_runner_trace",
    "map_runner_view_integrity",
    "map_runner_worker",
)
for _module_name in FLAT_SHIM_MODULES:
    importlib.import_module(f"robot_sf.benchmark.{_module_name}")


def test_flat_shim_is_identity_preserving() -> None:
    """The legacy top-level path must resolve to the canonical package module."""
    import robot_sf.benchmark.map_runner_env as flat_env

    assert flat_env is canonical_env
    assert flat_env.__name__ == "robot_sf.benchmark.map_runner.map_runner_env"


def test_package_delegates_bare_module_surface() -> None:
    """``robot_sf.benchmark.map_runner`` is the package; public names delegate to the core module."""

    assert map_runner_package.__name__ == "robot_sf.benchmark.map_runner"
    assert callable(map_runner_package.run_map_batch)
    assert map_runner_package.run_map_batch is run_map_batch


def test_canonical_submodule_imports_resolve() -> None:
    """Canonical dotted paths from the migration contract must import."""

    assert flat_env_alias is canonical_env
    assert "robot_sf.benchmark.map_runner.map_runner_env" in sys.modules
    assert run_map_batch.__module__ == "robot_sf.benchmark.map_runner.map_runner"


@pytest.mark.parametrize("module_name", list(FLAT_SHIM_MODULES))
def test_all_moved_submodules_import(module_name: str) -> None:
    """Every moved module is reachable through the canonical package path."""

    canonical = importlib.import_module(f"robot_sf.benchmark.map_runner.{module_name}")
    flat = importlib.import_module(f"robot_sf.benchmark.{module_name}")
    assert flat is canonical
