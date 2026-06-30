"""Enumeration coverage issue #3813 sustained-flow scenario variants."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from robot_sf.scenario_certification.sustained_flow import (
    DEFAULT_SUSTAINED_FLOW_SCENARIO_SET,
    EXPECTED_SUSTAINED_FLOW_TIERS,
    REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE,
    generate_expected_sustained_flow_scenarios,
    preflight_sustained_flow_scenario_set,
    sustained_flow_preflight_to_dict,
)
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_issue_3813_sustained_flow_scenarios_enumerate_expected_tiers() -> None:
    """Scenario matrix enumerates light, medium, heavy sustained-flow tiers."""
    scenarios = load_scenarios(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET)
    assert len(scenarios) == 3
    expected = {
        tier: {
            "ped_density": ped_density,
            "spawn_rate_per_min": spawn_rate_per_min,
            "seeds": list(seeds),
        }
        for tier, ped_density, spawn_rate_per_min, seeds in EXPECTED_SUSTAINED_FLOW_TIERS
    }
    observed_tiers = [scenario["metadata"]["density"] for scenario in scenarios]
    assert observed_tiers == ["light", "medium", "heavy"]
    for scenario in scenarios:
        metadata = scenario["metadata"]
        tier = metadata["density"]
        assert scenario["name"] == f"issue_3813_sustained_flow_t_intersection_{tier}"
        assert scenario["simulation_config"]["max_episode_steps"] == 600
        assert scenario["simulation_config"]["ped_density"] == expected[tier]["ped_density"]
        assert (
            scenario["metadata"]["continuous_spawn"]["spawn_rate_per_min"]
            == expected[tier]["spawn_rate_per_min"]
        )
        assert scenario["seeds"] == expected[tier]["seeds"]


def test_issue_3813_sustained_flow_preflight_report_conforms() -> None:
    """The no-submit preflight reports scenario scaffold as enumerable only."""
    report = preflight_sustained_flow_scenario_set(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET)
    payload = sustained_flow_preflight_to_dict(report)

    assert report.conforms
    assert report.errors == ()
    assert payload["benchmark_evidence"] is False
    assert payload["runtime_support"] == "metadata_only"
    assert payload["variant_count"] == 3
    assert [variant["density_tier"] for variant in payload["variants"]] == [
        "light",
        "medium",
        "heavy",
    ]
    assert payload["errors"] == []


def test_issue_3813_sustained_flow_preflight_cli_json() -> None:
    """The validation CLI exposes the same preflight contract used by PR checks."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validation/preflight_sustained_flow_scenarios_issue_3813.py",
            "--scenario-set",
            str(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET),
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = yaml.safe_load(result.stdout)

    assert payload["conforms"] is True
    assert payload["variant_count"] == 3
    assert payload["runtime_support"] == "metadata_only"
    assert [variant["density_tier"] for variant in payload["variants"]] == [
        "light",
        "medium",
        "heavy",
    ]


def test_issue_3813_sustained_flow_checked_in_matrix_matches_generated_variant_specs() -> None:
    """Generated preflight specs match checked-in sustained-flow scaffold."""
    scenarios = load_scenarios(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET)
    generated = generate_expected_sustained_flow_scenarios()
    assert scenarios == generated


def test_issue_3813_sustained_flow_variants_are_stable_and_ordered() -> None:
    """Generator output has deterministic variant naming and ordered density tiers."""
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
