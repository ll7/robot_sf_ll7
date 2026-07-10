"""Contract tests for the committed benchmark v0.1 four-suite freeze."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.release_suite_contract import (
    evaluate_release_suite_contract,
    load_release_suite_contract,
)
from robot_sf.benchmark.release_suite_reference_validation import (
    evaluate_release_suite_references,
)
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs/benchmarks/releases/benchmark_v0_1_release_suites.yaml"
EXPECTED_SUITE_KINDS = {
    "nominal": "nominal",
    "stress": "stress",
    "adversarial": "adversarial",
    "amv_specific": "autonomous_mobile_vehicle_specific",
}
EXCLUDED_SCENARIOS = {
    "francis2023_exiting_elevator",
    "francis2023_narrow_doorway",
}


def _payload() -> dict[str, Any]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_committed_release_suite_manifest_freezes_expected_roster() -> None:
    """The release roster should name exactly one suite of every required kind."""

    payload = _payload()
    suites = {suite["suite_id"]: suite for suite in payload["suites"]}

    assert payload["freeze_status"] == "frozen_definition_pending_semantic_validation"
    assert payload["publication_status"] == "blocked_pending_release_rebase"
    assert {suite_id: suite["suite_kind"] for suite_id, suite in suites.items()} == (
        EXPECTED_SUITE_KINDS
    )
    assert len(suites["nominal"]["scenario_ids"]) == 37
    assert len(suites["stress"]["scenario_ids"]) == 9
    assert len(suites["adversarial"]["scenario_family_ids"]) == 2
    assert len(suites["amv_specific"]["scenario_ids"]) == 4

    nominal = set(suites["nominal"]["scenario_ids"])
    stress = set(suites["stress"]["scenario_ids"])
    assert nominal.isdisjoint(stress)
    assert not EXCLUDED_SCENARIOS.intersection(nominal | stress)


def test_committed_release_suite_definitions_are_real_parseable_configs() -> None:
    """Every frozen suite should point to a tracked, non-empty mapping config."""

    for suite in _payload()["suites"]:
        definition_path = ROOT / suite["suite_definition"]
        assert definition_path.is_file(), suite["suite_id"]
        definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
        assert isinstance(definition, dict) and definition, suite["suite_id"]


def test_frozen_membership_matches_accepted_policy_and_owned_configs() -> None:
    """Membership should be derived from the named policy/config owners, not copied loosely."""

    payload = _payload()
    suites = {suite["suite_id"]: suite for suite in payload["suites"]}
    policy = yaml.safe_load((ROOT / suites["nominal"]["selection_source"]).read_text())
    nominal_policy = policy["nominal_release_suite"]
    excluded = {item["scenario_id"] for item in nominal_policy["excluded_scenarios"]}
    stress = {item["scenario_id"] for item in nominal_policy["routed_stress_only_scenarios"]}
    release_scenarios = {
        scenario["name"]
        for scenario in load_scenarios(
            ROOT / "configs/scenarios/classic_interactions_francis2023.yaml"
        )
    }

    assert set(suites["nominal"]["scenario_ids"]) == release_scenarios - stress - excluded
    assert set(suites["stress"]["scenario_ids"]) == stress

    adversarial_definition = yaml.safe_load(
        (ROOT / suites["adversarial"]["suite_definition"]).read_text()
    )
    assert set(suites["adversarial"]["scenario_family_ids"]) == {
        family["family_id"] for family in adversarial_definition["scenario_families"]
    }

    amv_definition = yaml.safe_load((ROOT / suites["amv_specific"]["suite_definition"]).read_text())
    assert set(suites["amv_specific"]["scenario_ids"]) == {
        scenario["name"] for scenario in load_scenarios(ROOT / amv_definition["scenario_matrix"])
    }


def test_committed_release_suite_metadata_is_complete_and_dereferenceable() -> None:
    """All six metadata owners for all four suites should resolve fail-closed."""

    manifest = load_release_suite_contract(MANIFEST_PATH)
    structural_report = evaluate_release_suite_contract(manifest)
    reference_report = evaluate_release_suite_references(manifest, ROOT)

    assert structural_report["status"] == "pass"
    assert structural_report["suite_count"] == 4
    assert reference_report["status"] == "pass"
    assert reference_report["resolved_suite_count"] == 4
    assert reference_report["resolved_reference_count"] == 24
