"""Enumeration coverage for issue #3813 sustained-flow scenario variants."""

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
    """Scenario matrix enumerates the light, medium, and heavy sustained-flow tiers."""

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
        assert scenario["seeds"] == expected[tier]["seeds"]
        assert (
            metadata["continuous_spawn"]["spawn_rate_per_min"]
            == expected[tier]["spawn_rate_per_min"]
        )
        assert metadata["continuous_spawn"]["target_density_tier"] == tier


def test_issue_3813_sustained_flow_scenarios_fail_closed_for_benchmark_use() -> None:
    """The scaffold remains metadata-only and cannot be mistaken for benchmark evidence."""

    raw = yaml.safe_load(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "robot_sf.scenario_matrix.v1"

    for scenario in raw["scenarios"]:
        metadata = scenario["metadata"]
        assert metadata["enabled_by_default"] is False
        assert metadata["benchmark_evidence"] is False
        assert metadata["status"] == "pre_benchmark_scaffold"
        assert metadata["requires_before_benchmark_use"] == list(
            REQUIRED_BLOCKERS_BEFORE_BENCHMARK_USE
        )
        assert metadata["continuous_spawn"] == {
            "required_before_benchmark_use": True,
            "intended_process": "poisson_respawn",
            "spawn_rate_per_min": metadata["continuous_spawn"]["spawn_rate_per_min"],
            "target_density_tier": metadata["density"],
            "current_runtime_support": "metadata_only",
        }
        assert metadata["termination"]["mode"] == "time_bounded"
        assert metadata["termination"]["goal_reach_is_not_primary_success"] is True
        assert metadata["success_metric"]["id"] == "sustained_progress_rate_m_per_s"


def test_issue_3813_sustained_flow_preflight_report_conforms() -> None:
    """The no-submit preflight reports the scenario scaffold as enumerable only."""

    report = preflight_sustained_flow_scenario_set(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET)
    payload = sustained_flow_preflight_to_dict(report)

    assert report.conforms is True
    assert report.errors == ()
    assert payload["benchmark_evidence"] is False
    assert payload["runtime_support"] == "metadata_only"
    assert payload["variant_count"] == 3
    assert [variant["density_tier"] for variant in payload["variants"]] == [
        "light",
        "medium",
        "heavy",
    ]


def test_issue_3813_sustained_flow_preflight_cli_json() -> None:
    """The validation CLI exposes the same preflight contract for PR checks."""

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
    assert payload["benchmark_evidence"] is False
    assert payload["runtime_support"] == "metadata_only"


def test_issue_3813_checked_in_matrix_matches_generated_variant_specs() -> None:
    """Generated preflight specs match the checked-in sustained-flow scaffold."""

    scenarios = load_scenarios(DEFAULT_SUSTAINED_FLOW_SCENARIO_SET)
    generated = generate_expected_sustained_flow_scenarios()

    assert scenarios == generated
