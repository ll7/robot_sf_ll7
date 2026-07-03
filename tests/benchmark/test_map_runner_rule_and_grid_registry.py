"""Regression tests for #3384 rule/grid and gap policy-builder registry migrations."""

from __future__ import annotations

import pytest

from robot_sf.benchmark import map_runner
from robot_sf.benchmark.map_runner_policies import gap_reference, rule_and_grid


@pytest.mark.parametrize(
    "algo_key",
    [
        "policy_stack_v1",
        "hybrid_rule_local_planner",
        "hybrid_rule_v0_minimal",
        "actuation_aware_hybrid_rule_v0",
        "topology_guided_hybrid_rule_v0",
        "lidar_grid_route",
        "lidar_occupancy_grid_route",
    ],
)
def test_rule_and_grid_keys_resolve_through_registry_bridge(algo_key: str) -> None:
    """Migrated rule/grid keys are owned by the registry builder."""
    assert map_runner._POLICY_BUILDERS[algo_key] is rule_and_grid.build


@pytest.mark.parametrize(
    "algo_key",
    ["stream_gap", "gap_prediction", "trivial_reference", "reference_adapter"],
)
def test_gap_reference_keys_resolve_through_registry_bridge(algo_key: str) -> None:
    """Migrated gap/reference keys are owned by the registry builder."""
    assert map_runner._POLICY_BUILDERS[algo_key] is gap_reference.build
