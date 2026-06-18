"""Contract tests for issue #3059 research-engine scenario suite proposal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_3059_research_engine_suite_v0.yaml"
SCENARIO_ROOT = ROOT / "configs/scenarios"

REQUIRED_FAMILIES: set[str] = {
    "frame_consistency_sanity",
    "static_obstacle_detour",
    "topology_and_local_minima",
    "paired_pedestrian_interactions",
    "crowd_flow_and_density",
    "social_protocol_francis",
}

REQUIRED_FAMILY_FIELDS: set[str] = {
    "family_id",
    "research_question",
    "scenario_ids",
    "scenario_contract",
    "planner_compatibility",
    "required_metrics",
    "row_status_policy",
    "claim_boundary",
}


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scenario_inventory() -> set[str]:
    inventory: set[str] = set()
    for path in SCENARIO_ROOT.rglob("*.yaml"):
        data = _load_yaml(path)
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            inventory.add(data["name"])
        scenarios = data.get("scenarios", []) if isinstance(data, dict) else data
        if not isinstance(scenarios, list):
            continue
        for item in scenarios:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                inventory.add(item["name"])
    return inventory


def test_manifest_declares_proposal_only_boundary() -> None:
    """The suite proposal must not claim benchmark evidence or planner rankings."""
    payload = _load_yaml(CONFIG_PATH)

    assert payload["contract_kind"] == "benchmark-suite-proposal"
    assert payload["evidence_tier"] == "proposal"
    assert payload["benchmark_evidence"] is False
    assert payload["status"] == "proposed"
    claim_boundary = payload["claim_boundary"].lower()
    assert "does not establish benchmark results" in claim_boundary
    assert "planner rankings" in claim_boundary
    assert "paper-facing claims" in claim_boundary


def test_manifest_has_versioned_registry_id() -> None:
    """Stable identifiers let downstream schedulers pin the v0 suite contract."""
    payload = _load_yaml(CONFIG_PATH)

    assert payload["registry_id"] == "research_engine_scenario_suite_v0"
    assert payload["schema_version"] == "research_engine_scenario_suite.v0"
    assert payload["suite_version"] == "0"


def test_manifest_defines_six_research_families() -> None:
    """The v0 suite should stay compact at 5-7 scenario families."""
    payload = _load_yaml(CONFIG_PATH)
    families = payload["scenario_families"]

    assert 5 <= len(families) <= 7
    assert {family["family_id"] for family in families} == REQUIRED_FAMILIES


def test_schema_and_each_family_name_required_fields() -> None:
    """Every family carries the scenario, seed, row-status, and claim surfaces."""
    payload = _load_yaml(CONFIG_PATH)

    assert set(payload["suite_schema"]["required_family_fields"]) == REQUIRED_FAMILY_FIELDS
    for family in payload["scenario_families"]:
        missing = REQUIRED_FAMILY_FIELDS - set(family)
        assert not missing, f"{family.get('family_id')} missing fields: {sorted(missing)}"
        assert len(family["scenario_ids"]) >= 3
        assert family["required_metrics"]
        assert family["scenario_contract"]["status"] == "contract_available"
        assert "fallback" in family["row_status_policy"].lower()
        assert "does not" in family["claim_boundary"].lower()


def test_scenario_ids_and_source_paths_exist() -> None:
    """Referenced scenarios and contract source paths should be checked in."""
    payload = _load_yaml(CONFIG_PATH)
    inventory = _scenario_inventory()

    declared = {
        scenario_id
        for family in payload["scenario_families"]
        for scenario_id in family["scenario_ids"]
    }
    missing_scenarios = declared - inventory
    assert not missing_scenarios, f"Missing scenario IDs: {sorted(missing_scenarios)}"

    source_paths = {
        path
        for family in payload["scenario_families"]
        for path in family["scenario_contract"]["source_paths"]
    }
    missing_paths = [path for path in source_paths if not (ROOT / path).exists()]
    assert not missing_paths, f"Missing source paths: {missing_paths}"


def test_seed_policy_is_frozen_and_has_s5_s10_s20_sets() -> None:
    """Seed escalation inputs must be explicit before scheduler issues consume them."""
    payload = _load_yaml(CONFIG_PATH)
    seed_policy = payload["seed_policy"]

    assert seed_policy["mode"] == "fixed-escalation-sets"
    assert seed_policy["frozen_before_execution"] is True
    assert seed_policy["pilot_set"]["name"] == "S5"
    assert len(seed_policy["pilot_set"]["seeds"]) == 5
    escalation = {entry["name"]: entry["seeds"] for entry in seed_policy["escalation_sets"]}
    assert set(escalation) == {"S10", "S20"}
    assert len(escalation["S10"]) == 10
    assert len(escalation["S20"]) == 20
    assert set(seed_policy["stop_rules"]) == {
        "stop",
        "diagnostic_only",
        "escalate_to_s10",
        "escalate_to_s20",
    }


def test_fallback_policy_and_publication_boundary_are_fail_closed() -> None:
    """Fallback/degraded rows must stay caveats, never suite success evidence."""
    payload = _load_yaml(CONFIG_PATH)
    fallback_policy = payload["fallback_policy"]

    assert fallback_policy["fallback_is_success"] is False
    assert fallback_policy["degraded_is_success"] is False
    assert fallback_policy["unavailable_is_success"] is False
    assert fallback_policy["failed_is_success"] is False
    assert fallback_policy["diagnostic_only_is_benchmark_success"] is False
    assert "excluded from benchmark-strength" in fallback_policy["rule"]

    publication = payload["publication_boundary"]
    assert "diagnostic infrastructure" in publication["diagnostic_use"]
    assert "no fallback/degraded rows counted as success" in publication["benchmark_candidate_use"]
    assert "must not be cited as planner ranking" in publication["non_claim_use"]
