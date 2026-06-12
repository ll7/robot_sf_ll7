"""Contract tests for issue #2452 mechanism-aware local-navigation suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_2452_mechanism_aware_local_nav_suites_v0.yaml"
ISSUE_2544_SMOKE_MATRIX = ROOT / "configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml"
ISSUE_2653_MATRIX = (
    ROOT / "configs/scenarios/sets/issue_2653_static_deadlock_activation_capable_h500.yaml"
)
ISSUE_2653_FUNNEL = (
    ROOT / "configs/policy_search/transfer/issue_2653_static_deadlock_activation_capable_h500.yaml"
)
SCENARIO_ROOT = ROOT / "configs/scenarios"

REQUIRED_SUITES: set[str] = {
    "static_deadlock_recovery",
    "topology_hypothesis_selection",
    "dynamic_phase_sensitivity",
    "proxemic_tradeoff",
    "actuation_feasibility",
    "guard_domination",
    "learned_low_progress",
}

REQUIRED_FIELDS: set[str] = {
    "suite_id",
    "target_mechanism",
    "scenario_ids",
    "seed_set",
    "baseline_candidates",
    "intervention_candidates",
    "required_trace_fields",
    "required_metrics",
    "minimum_evidence_tier",
    "claim_boundary",
}

RELATED_ISSUES: set[int] = {2220, 2232, 2389, 2447}


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scenario_inventory() -> set[str]:
    inventory: set[str] = set()
    for path in SCENARIO_ROOT.rglob("*.yaml"):
        data = _load_yaml(path)
        scenarios = data.get("scenarios", []) if isinstance(data, dict) else data
        if not isinstance(scenarios, list):
            continue
        for item in scenarios:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                inventory.add(item["name"])
    return inventory


def test_manifest_declares_proposal_only_evidence_boundary() -> None:
    """Verify the manifest is explicit launch infrastructure, not benchmark evidence."""
    payload = _load_yaml(CONFIG_PATH)
    assert payload["contract_kind"] == "benchmark-suite-proposal"
    assert payload["evidence_tier"] == "proposal"
    assert payload["benchmark_evidence"] is False
    assert payload["status"] == "proposed"
    claim_boundary = payload["claim_boundary"].lower()
    assert "does not establish" in claim_boundary
    assert "planner rankings" in claim_boundary
    assert "paper-facing claims" in claim_boundary


def test_manifest_has_versioned_registry_id() -> None:
    """Verify the suite registry exposes stable identifiers and schema versioning."""
    payload = _load_yaml(CONFIG_PATH)
    assert payload["registry_id"] == "mechanism_aware_local_nav_suites_v0"
    assert payload["schema_version"] == "mechanism_aware_local_nav_suites.v0"
    assert payload["suite_version"] == "0"


def test_manifest_links_required_research_and_guard_issues() -> None:
    """Verify the issue links requested by the contract are present."""
    payload = _load_yaml(CONFIG_PATH)
    issues = {entry["issue"] for entry in payload["related_issues"]}
    assert RELATED_ISSUES <= issues
    for entry in payload["related_issues"]:
        assert "claim" in entry, f"Issue #{entry['issue']} missing claim boundary"
        assert entry["status"] not in {"resolved_by_this_manifest", "benchmark_complete"}


def test_manifest_declares_all_required_suites() -> None:
    """Verify the registry covers all seven mechanism-aware suite IDs."""
    payload = _load_yaml(CONFIG_PATH)
    declared = {suite["suite_id"] for suite in payload["suites"]}
    assert declared == REQUIRED_SUITES


def test_schema_names_required_fields() -> None:
    """Verify the manifest schema records the required suite fields from issue #2452."""
    payload = _load_yaml(CONFIG_PATH)
    assert set(payload["suite_schema"]["required_fields"]) == REQUIRED_FIELDS


def test_each_suite_has_required_contract_fields() -> None:
    """Verify every suite declares the scenario, evidence, metric, and claim surfaces."""
    payload = _load_yaml(CONFIG_PATH)
    for suite in payload["suites"]:
        missing = REQUIRED_FIELDS - set(suite)
        assert not missing, f"{suite.get('suite_id')} missing fields: {sorted(missing)}"
        for field in (
            "scenario_ids",
            "seed_set",
            "baseline_candidates",
            "intervention_candidates",
            "required_trace_fields",
            "required_metrics",
        ):
            assert suite[field], f"{suite['suite_id']} has empty {field}"
        assert suite["minimum_evidence_tier"] in payload["suite_schema"]["evidence_tiers"]
        assert "does not" in suite["claim_boundary"].lower()
        assert "row_status" in suite["required_trace_fields"]


def test_suite_scenario_ids_exist_in_checked_in_scenarios() -> None:
    """Verify suite rows use existing checked-in scenario IDs."""
    payload = _load_yaml(CONFIG_PATH)
    inventory = _scenario_inventory()
    declared = {scenario_id for suite in payload["suites"] for scenario_id in suite["scenario_ids"]}
    missing = declared - inventory
    assert not missing, f"Scenario IDs missing from checked-in configs: {sorted(missing)}"


def test_fallback_policy_is_fail_closed() -> None:
    """Verify fallback and unavailable modes are never counted as success evidence."""
    payload = _load_yaml(CONFIG_PATH)
    fallback_policy = payload["fallback_policy"]
    assert fallback_policy["fallback_is_success"] is False
    assert fallback_policy["degraded_is_success"] is False
    assert fallback_policy["not_available_is_success"] is False
    assert fallback_policy["failed_is_success"] is False
    assert "excluded from suite-strengthening evidence" in fallback_policy["rule"]


def test_issue_2544_static_deadlock_smoke_matrix_binds_first_suite() -> None:
    """Verify issue #2544 binds one proposal suite to an executable smoke matrix."""
    from robot_sf.training.scenario_loader import load_scenarios

    scenarios = [dict(scenario) for scenario in load_scenarios(ISSUE_2544_SMOKE_MATRIX)]
    assert [scenario["name"] for scenario in scenarios] == [
        "classic_bottleneck_low",
        "classic_head_on_corridor_low",
        "narrow_passage",
    ]
    for scenario in scenarios:
        assert scenario["seeds"] == [111]
        metadata = scenario["metadata"]
        assert metadata["mechanism_aware_suite_id"] == "static_deadlock_recovery"
        assert metadata["issue"] == 2544
        assert metadata["evidence_tier"] == "diagnostic_smoke"
        assert "not planner ranking" in metadata["claim_boundary"]


def test_issue_2653_static_deadlock_activation_capable_contract() -> None:
    """Verify #2653 predeclares the denominator, stop rule, and fail-closed boundary."""
    from robot_sf.training.scenario_loader import load_scenarios

    scenarios = [dict(scenario) for scenario in load_scenarios(ISSUE_2653_MATRIX)]
    assert [scenario["name"] for scenario in scenarios] == [
        "classic_bottleneck_low",
        "classic_head_on_corridor_low",
        "narrow_passage",
    ]
    for scenario in scenarios:
        assert scenario["seeds"] == [111, 112, 113]
        metadata = scenario["metadata"]
        assert metadata["mechanism_aware_suite_id"] == "static_deadlock_recovery"
        assert metadata["issue"] == 2653
        assert metadata["evidence_tier"] == "predeclared_activation_capable_h500"
        claim_boundary = metadata["claim_boundary"].lower()
        assert "recenter_activation_count > 0" in metadata["active_row_denominator_definition"]
        assert "does not establish planner ranking" in claim_boundary
        assert "fallback" in claim_boundary
        assert "degraded" in claim_boundary

    funnel = _load_yaml(ISSUE_2653_FUNNEL)
    stage = funnel["stages"]["full_matrix"]
    assert stage["scenario_matrix"] == (
        "configs/scenarios/sets/issue_2653_static_deadlock_activation_capable_h500.yaml"
    )
    assert stage["horizon"] == 500
    transfer = stage["mechanism_transfer"]
    assert transfer["issue"] == 2653
    gate = transfer["activation_capable_gate"]
    assert gate["min_active_row_denominator"] == 2
    assert gate["min_terminal_outcome_delta_count"] == 1
    assert "no planner-promotion" in gate["stop_rule"].lower()
    assert "fallback" in gate["claim_boundary"].lower()
