"""Tests for the issue #2477 ScenarioBelief contract manifest."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from robot_sf.representation import (
    EntityBelief,
    Estimate2D,
    ScenarioBelief,
    scenario_belief_from_simulator_oracle,
    scenario_belief_from_visibility_limited_simulator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs/representation/scenario_belief_contract_issue_2477.yaml"


def _manifest() -> dict[str, object]:
    """Load the ScenarioBelief contract manifest."""
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_manifest_references_existing_contract_surfaces() -> None:
    """Every tracked contract surface should exist in the repository."""
    manifest = _manifest()
    surfaces = manifest["contract_surfaces"]
    assert isinstance(surfaces, dict)

    for rel_path in surfaces.values():
        assert (REPO_ROOT / str(rel_path)).exists(), rel_path


def test_manifest_fields_match_dataclass_contracts() -> None:
    """Manifest field lists should stay synchronized with the implemented dataclasses."""
    manifest = _manifest()
    scenario_fields = {field.name for field in fields(ScenarioBelief)}
    entity_fields = {field.name for field in fields(EntityBelief)}
    estimate_fields = {field.name for field in fields(Estimate2D)}

    assert set(manifest["required_top_level_fields"]).issubset(scenario_fields)
    assert set(manifest["required_entity_fields"]).issubset(entity_fields)
    assert set(manifest["required_estimate_fields"]).issubset(estimate_fields)


def test_manifest_producer_and_consumer_boundaries_resolve() -> None:
    """Producer functions and consumer methods named in the manifest should exist."""
    manifest = _manifest()
    producers = manifest["producer_boundaries"]
    consumers = manifest["consumer_boundaries"]
    assert isinstance(producers, list)
    assert isinstance(consumers, list)

    producer_functions = {
        "scenario_belief_from_simulator_oracle": scenario_belief_from_simulator_oracle,
        "scenario_belief_from_visibility_limited_simulator": (
            scenario_belief_from_visibility_limited_simulator
        ),
    }
    for producer in producers:
        assert isinstance(producer, dict)
        assert producer_functions[str(producer["function"])]

    for consumer in consumers:
        assert isinstance(consumer, dict)
        method_name = str(consumer["method"]).split(".")[-1]
        assert hasattr(ScenarioBelief, method_name), method_name


def test_manifest_preserves_claim_boundary_and_first_fixture() -> None:
    """The manifest should stay explicit that this is not benchmark evidence."""
    manifest = _manifest()
    claim_boundary = str(manifest["claim_boundary"]).lower()
    fixture = manifest["first_fixture_or_smoke_path"]
    assert isinstance(fixture, dict)

    assert manifest["benchmark_evidence"] is False
    assert "diagnostic" in claim_boundary
    assert "does not prove" in claim_boundary
    assert (REPO_ROOT / "tests/representation/test_scenario_belief.py").is_file()
    assert (
        "oracle and visibility-limited producers share schema and projection keys"
        in fixture["validates"]
    )
