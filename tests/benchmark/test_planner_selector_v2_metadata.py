"""Benchmark catalog tests for planner-selector v2 diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    enrich_algorithm_metadata,
)
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)
from scripts.validation.run_policy_search_candidate import load_candidate_definition

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_planner_selector_v2_is_experimental_and_requires_opt_in() -> None:
    """The diagnostic selector must stay behind the experimental testing gate."""
    spec = get_algorithm_readiness("planner_selector_v2_diagnostic")

    assert spec is not None
    assert spec.canonical_name == "planner_selector_v2_diagnostic"
    assert spec.tier == "experimental"
    assert spec.requires_explicit_opt_in is True

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="planner_selector_v2_diagnostic",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    assert (
        require_algorithm_allowed(
            algo="planner_selector_v2_diagnostic",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
        == spec
    )


def test_planner_selector_v2_metadata_is_diagnostic_only() -> None:
    """Algorithm metadata should prevent benchmark-strength overclaiming."""
    assert canonical_algorithm_name("planner_selector_v2_diagnostic") == (
        "planner_selector_v2_diagnostic"
    )

    meta = enrich_algorithm_metadata(
        algo="planner_selector_v2_diagnostic",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    planner = meta["planner_kinematics"]
    assert meta["baseline_category"] == "diagnostic"
    assert meta["policy_semantics"] == "deterministic_diagnostic_planner_selector"
    assert planner["adapter_name"] == "PlannerSelectorV2DiagnosticAdapter"
    assert planner["diagnostic_only"] is True
    assert planner["benchmark_strength"] is False


def test_planner_selector_v2_registry_config_declares_no_leakage_sources() -> None:
    """The policy-search row should expose diagnostic-only selector inputs."""
    registry_path = REPO_ROOT / "docs/context/policy_search/candidate_registry.yaml"
    entry, payload, merged, config_path = load_candidate_definition(
        registry_path,
        "planner_selector_v2_diagnostic",
    )
    expected_path = (
        REPO_ROOT / "configs/policy_search/candidates/planner_selector_v2_diagnostic.yaml"
    )

    assert entry["status"] == "experimental_spike"
    assert entry["claim_scope"] == "diagnostic_only"
    assert payload["algo"] == "planner_selector_v2_diagnostic"
    assert config_path == expected_path
    selector = merged["selector"]
    assert "classic_realworld_double_bottleneck_high" in selector["topology_scenarios"]
    assert "francis2023_join_group" in selector["seed_sensitive_scenarios"]
    assert selector["hard_seed_values"][0] == 116
    assert selector["evidence_sources"] == [
        "docs/context/issue_1608_seed_sensitivity_analysis.md",
        "docs/context/issue_1692_topology_hypothesis_probe.md",
    ]
    assert sorted(merged["candidate_config_paths"]) == [
        "baseline",
        "fast_progress_static_escape",
        "proxemic_conservative",
        "topology_route",
    ]
