"""Regression tests for #3384 rule/grid policy-builder registry migration."""

from __future__ import annotations

import pytest

from robot_sf.benchmark import map_runner
from robot_sf.benchmark.map_runner_policies import rule_and_grid


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
    """Migrated rule/grid keys should be owned by the registry builder."""
    assert map_runner._POLICY_BUILDERS[algo_key] is rule_and_grid.build


@pytest.mark.parametrize("algo_key", ["stream_gap", "gap_prediction", "trivial_reference"])
def test_deferred_rule_gap_keys_remain_legacy_dispatch(algo_key: str) -> None:
    """Later #3384 slices still own helper-dependent rule/gap branches."""
    assert algo_key not in map_runner._POLICY_BUILDERS
