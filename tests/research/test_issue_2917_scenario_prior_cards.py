"""Tests for the issue #2917 ScenarioPrior provenance-card registry."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "configs/research/scenario_prior_cards_issue_2917.yaml"


def _registry() -> dict[str, object]:
    """Load the ScenarioPrior card registry."""
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_registry_preserves_proposal_claim_boundary() -> None:
    """The registry should not present prior cards as benchmark or realism evidence."""
    registry = _registry()
    claim_boundary = str(registry["claim_boundary"]).lower()
    card_contract = registry["card_contract"]
    assert isinstance(card_contract, dict)

    assert registry["schema_version"] == "scenario-prior-card-registry.v1"
    assert registry["benchmark_evidence"] is False
    assert "does not prove" in claim_boundary
    assert card_contract["claim_boundary_fields"] == {
        "benchmark_evidence": False,
        "realism_claim": False,
        "training_input": False,
    }


def test_cards_cover_known_initial_prior_families() -> None:
    """Known authored, proxy, and staged-data-candidate prior families should have cards."""
    registry = _registry()
    cards = {card["card_id"]: card for card in registry["cards"]}

    assert {
        "authored_scenario_contract_priors",
        "issue_2523_proxy_scenario_prior_fixture",
        "sdd_scenario_distribution_candidate",
    }.issubset(cards)
    assert cards["authored_scenario_contract_priors"]["classification"] == "authored"
    assert cards["issue_2523_proxy_scenario_prior_fixture"]["classification"] == "proxy_only"
    assert cards["sdd_scenario_distribution_candidate"]["classification"] == (
        "external_dataset_candidate"
    )


def test_every_card_has_required_fields_and_existing_source_traces() -> None:
    """Each card should be auditable through repo-local source traces."""
    registry = _registry()
    required_fields = set(registry["card_contract"]["required_fields"])
    cards = registry["cards"]
    assert isinstance(cards, list)

    for card in cards:
        assert required_fields.issubset(card), card["card_id"]
        traces = card["source_traces"]
        assert isinstance(traces, list)
        assert traces, card["card_id"]
        for rel_path in traces:
            assert (REPO_ROOT / rel_path).exists(), (card["card_id"], rel_path)


def test_cards_reject_unsupported_prior_claims() -> None:
    """All initial cards should explicitly reject realism and benchmark promotion claims."""
    registry = _registry()
    required_rejections = {
        "learned_prior_realism",
        "benchmark_usefulness",
        "cross_dataset_generalization",
        "planner_performance_improvement",
    }

    for card in registry["cards"]:
        unsupported = set(card["unsupported_claims"])
        assert required_rejections.issubset(unsupported), card["card_id"]
        assert card["excluded_populations"], card["card_id"]
        assert card["odd_conditions"], card["card_id"]
