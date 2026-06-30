"""Determinism checks for issue #3813 sustained-flow variant generation."""

from __future__ import annotations

from robot_sf.scenario_certification.sustained_flow import (
    EXPECTED_SUSTAINED_FLOW_TIERS,
    REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE,
    generate_expected_sustained_flow_scenarios,
)


def test_issue_3813_sustained_flow_variants_are_stable_and_ordered() -> None:
    """Generator output has deterministic variant naming ordered density tiers."""
    generated = generate_expected_sustained_flow_scenarios()
    expected_tiers = tuple(tier for tier, *_ in EXPECTED_SUSTAINED_FLOW_TIERS)
    expected_spawn_rates = tuple(
        spawn_rate_per_min for _, _, spawn_rate_per_min, _ in EXPECTED_SUSTAINED_FLOW_TIERS
    )

    assert tuple(scenario["metadata"]["density"] for scenario in generated) == expected_tiers
    assert tuple(scenario["name"] for scenario in generated) == tuple(
        f"issue_3813_sustained_flow_t_intersection_{tier}" for tier in expected_tiers
    )
    assert (
        tuple(
            scenario["metadata"]["continuous_spawn"]["spawn_rate_per_min"] for scenario in generated
        )
        == expected_spawn_rates
    )
    assert tuple(
        tuple(scenario["metadata"]["requires_before_benchmark_use"]) for scenario in generated
    ) == tuple(REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE for _ in expected_tiers)
