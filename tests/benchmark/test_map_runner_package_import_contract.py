"""Import-contract checks for the canonical map-runner helper package."""

from __future__ import annotations

import importlib
import sys

import pytest

_MOVED_HELPERS = (
    "map_runner_batch_plan",
    "map_runner_batch_runner",
    "map_runner_batch_summary",
    "map_runner_env",
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


@pytest.mark.parametrize("module_name", _MOVED_HELPERS)
def test_legacy_map_runner_helper_is_identity_alias(module_name: str) -> None:
    """Historical flat helper imports resolve to the canonical module object."""
    legacy_name = f"robot_sf.benchmark.{module_name}"
    canonical_name = f"robot_sf.benchmark.map_runner.{module_name}"

    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)

    assert legacy is canonical
    assert sys.modules[legacy_name] is canonical
    assert canonical.__name__ == canonical_name


def test_map_runner_package_keeps_public_batch_entrypoint() -> None:
    """The former map_runner.py module remains the package-level public API."""
    module = importlib.import_module("robot_sf.benchmark.map_runner")

    assert module.__name__ == "robot_sf.benchmark.map_runner"
    assert callable(module.run_map_batch)
    assert callable(module._build_policy)
