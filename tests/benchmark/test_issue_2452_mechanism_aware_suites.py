"""Contract tests for issue #2452 mechanism-aware local-navigation suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_2452_mechanism_aware_local_nav_suites_v0.yaml"
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
