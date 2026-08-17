"""Import-contract checks for the map-runner policy namespace migration."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "map_runner_policy_actions",
        "map_runner_policy_common",
        "map_runner_policy_metadata",
        "map_runner_policy_resolution",
    ),
)
def test_legacy_policy_module_is_identity_alias(module_name: str) -> None:
    """Historical flat imports resolve to the canonical package module object."""
    legacy_name = f"robot_sf.benchmark.{module_name}"
    canonical_name = f"robot_sf.benchmark.map_runner_policies.{module_name}"

    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)

    assert legacy is canonical
    assert sys.modules[legacy_name] is canonical
    assert canonical.__name__ == canonical_name
