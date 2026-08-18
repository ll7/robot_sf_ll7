"""Tests for the deterministic benchmark namespace inventory in issue #7331."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev import audit_benchmark_namespace

REPO_ROOT = Path(__file__).resolve().parents[2]
# Current ``main`` includes the fixture-only figure-interpretation evaluator added by #7062
# and the trace-dossier package delivered by #7114; the result-interpretation
# packet module was delivered by #7029.
# Keep this explicit so a new direct child fails the audit until it is
# deliberately classified, rather than silently changing the inventory size.
EXPECTED_DIRECT_CHILD_COUNT = 296


@pytest.fixture(scope="module")
def inventory() -> dict[str, object]:
    """Build one current-source inventory for the module's assertions."""
    return audit_benchmark_namespace.build_inventory(REPO_ROOT)


def test_current_namespace_is_complete_and_routes_fail_closed(
    inventory: dict[str, object],
) -> None:
    """Every direct child is classified and no low-risk move is selected."""
    payload = inventory

    assert payload["schema"] == "benchmark-namespace-residual-inventory.v1"
    assert payload["direct_child_count"] == EXPECTED_DIRECT_CHILD_COUNT
    assert len(payload["direct_children"]) == payload["direct_child_count"]
    assert len({row["name"] for row in payload["direct_children"]}) == EXPECTED_DIRECT_CHILD_COUNT
    assert payload["recommendation"]["code"] == "pause_no_low_risk_cluster"
    assert payload["import_cycle_ledger"]
    assert all(cycle == sorted(cycle) for cycle in payload["import_cycle_ledger"])
    assert all(row["ownership"]["status"] for row in payload["direct_children"])
    assert payload["ownership_reconciliation"]["duplicate_ownership_check"] == "passed"
    assert all(row["compatibility_action"] for row in payload["direct_children"])
    assert payload["scope_boundary"]["production_moves"] is False


def test_known_facades_and_clusters_are_classified(inventory: dict[str, object]) -> None:
    """Known aliases, map-runner, scenario, and campaign surfaces remain distinct."""
    payload = inventory
    rows = {row["name"]: row for row in payload["direct_children"]}

    assert rows["campaign_atlas.py"]["compatibility_shim"] is True
    assert rows["campaign_atlas.py"]["classification"] == (
        "already_migrated_implementation_with_compatibility_shim"
    )
    assert rows["map_runner_policies"]["classification"] == "unresolved_map_runner_cluster"
    assert rows["scenario"]["classification"] == (
        "unresolved_scenario_generation_certification_cluster"
    )
    assert rows["camera_ready"]["classification"] == (
        "unresolved_camera_ready_campaign_facade_cluster"
    )
    assert rows["result_interpretation_packet.py"]["classification"] == (
        "cross_cutting_schema_evidence_readiness_artifact_metric_utility_surface"
    )
    assert rows["result_interpretation_packet.py"]["compatibility_action"] == (
        "no_compatibility_action"
    )
    assert rows["__init__.py"]["classification"] == "canonical_top_level_facade_api"


def test_serialized_outputs_are_deterministic(inventory: dict[str, object]) -> None:
    """Repeated inventory and rendering on one commit are byte-identical."""
    first = inventory
    second = audit_benchmark_namespace.build_inventory(REPO_ROOT)

    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second, indent=2, sort_keys=True
    )
    assert audit_benchmark_namespace.render_markdown(
        first
    ) == audit_benchmark_namespace.render_markdown(second)


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """The issue-scoped CLI emits both required report forms."""
    json_path = tmp_path / "namespace_inventory.json"
    markdown_path = tmp_path / "namespace_inventory.md"

    result = audit_benchmark_namespace.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert result == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["direct_child_count"] == EXPECTED_DIRECT_CHILD_COUNT
    assert "# Benchmark namespace residual inventory (issue #7331)" in markdown_path.read_text(
        encoding="utf-8"
    )
