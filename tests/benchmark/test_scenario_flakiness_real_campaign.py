"""Real-campaign integration test for the scenario flakiness audit (issue #4978).

The audit capability landed in PR #5069 (standalone CLI + ``scenario_flakiness.v1``
schema) and PR #5115 (aggregate embedding). Both explicitly deferred the
*real-campaign application* as compute-only follow-up. This test exercises that
deferred path end-to-end: it runs ``compute_flakiness_audit`` on a tracked,
compact subset of a real multi-planner benchmark campaign (shipped under
``tests/fixtures/``) and asserts the *real* per-cell stability structure —
including concrete knife-edge cells — so the advisory report is regression-
protected and reproducible from committed data alone.

This is diagnostic-grade evidence: it proves the audit produces sensible,
deterministic output on real benchmark episodes. It makes no planner-quality
or benchmark-ranking claim.

The fixture and the committed report live next to this test::

    tests/fixtures/benchmark/scenario_flakiness_issue_4978/
        real_campaign_episodes.jsonl            # 400 real episodes (5 scenarios x 4 planners x 20 seeds)
        real_campaign_flakiness_report.json     # committed scenario_flakiness.v1 report
        README.md                               # provenance + regeneration instructions
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.scenario_flakiness import (
    SCHEMA_VERSION,
    compute_flakiness_audit,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "benchmark"
    / "scenario_flakiness_issue_4978"
)
EPISODES = FIXTURE_DIR / "real_campaign_episodes.jsonl"
COMMITTED_REPORT = FIXTURE_DIR / "real_campaign_flakiness_report.json"


def _load_episodes() -> list[dict]:
    records = []
    with EPISODES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def real_report() -> dict:
    """Run the flakiness audit on the committed real-campaign fixture."""
    return compute_flakiness_audit(_load_episodes(), group_by="algo", fallback_group_by="algo")


def test_fixture_is_present_and_real():
    """The tracked fixture exists and carries real benchmark episodes with seed structure."""
    assert EPISODES.exists(), f"missing real-campaign fixture: {EPISODES}"
    records = _load_episodes()
    # 5 scenarios x 4 planners x 20 seeds == 400 real episodes.
    assert len(records) == 400
    scenarios = {r["scenario_id"] for r in records}
    planners = {r["algo"] for r in records}
    seeds = {r["seed"] for r in records}
    assert len(scenarios) == 5
    assert planners == {"goal", "orca", "ppo", "social_force"}
    assert len(seeds) == 20
    # Every episode must carry a provenance git_hash (faithful, reproducible subset).
    assert all(r.get("git_hash") for r in records)


def test_audit_scores_all_cells_with_seed_support(real_report):
    """Every (scenario, planner) cell is assessable with full seed support."""
    assert real_report["schema_version"] == SCHEMA_VERSION
    summary = real_report["summary"]
    assert summary["n_records"] == 400
    assert summary["n_cells"] == 5 * 4
    # No records dropped: every episode had a scenario_id and a binary outcome.
    assert summary["n_records_missing_outcome"] == 0
    assert summary["n_records_missing_scenario"] == 0
    assert summary["n_assessable_cells"] == 20
    # Each cell was scored from all 20 seeds (>= min_seeds default of 2).
    assert all(c["assessable"] for c in real_report["cells"])
    assert all(c["n_seeds"] == 20 for c in real_report["cells"])


def test_audit_flags_real_knife_edge_cells(real_report):
    """Issue #4978: the audit surfaces real per-cell outcome instability on real data.

    The classic_doorway_medium / ppo cell is a perfect coin-flip across 20
    independent seeds (stability_score == 0.5) — exactly the hidden variance the
    issue says per-seed confidence intervals understate. Asserting it here locks
    the detector against regression on a real, non-synthetic campaign subset.
    """
    by_cell = {(c["scenario_id"], c["planner"]): c for c in real_report["cells"]}

    # A concrete perfect knife-edge cell: 50/50 across 20 seeds.
    ppo_doorway = by_cell[("classic_doorway_medium", "ppo")]
    assert ppo_doorway["knife_edge"] is True
    assert ppo_doorway["stability_score"] == pytest.approx(0.5)
    assert ppo_doorway["success_rate"] == pytest.approx(0.5)

    # Cross-trap high is knife-edge for both learned/reactive planners.
    assert by_cell[("classic_cross_trap_high", "orca")]["knife_edge"] is True
    assert by_cell[("classic_cross_trap_high", "ppo")]["knife_edge"] is True

    # social_force is unanimous-fail across every selected scenario: never knife-edge.
    for scenario_id, planner in by_cell:
        if planner == "social_force":
            assert by_cell[(scenario_id, "social_force")]["knife_edge"] is False
            assert by_cell[(scenario_id, "social_force")]["stability_score"] == pytest.approx(1.0)

    # The headline aggregate: a substantial minority of cells are knife-edge.
    assert real_report["summary"]["n_knife_edge_cells"] == 7
    assert real_report["summary"]["knife_edge_fraction"] == pytest.approx(0.35)


def test_audit_reports_unknown_determinism_without_repeat_data(real_report):
    """Exact-repeat determinism fails closed: unknown (None) when no seed was repeated.

    Each (scenario, planner, seed) ran exactly once in the source campaign, so the
    audit must NOT assert determinism it cannot evidence — the detector stays honest.
    """
    exact = real_report["exact_repeat"]
    assert exact["checked_repeat_groups"] == 0
    assert exact["is_deterministic"] is None
    assert exact["examples"] == []


def test_committed_report_matches_fresh_computation(real_report):
    """The committed report is reproducible from the tracked fixture alone.

    Regenerating the audit from the fixture must reproduce the committed
    ``scenario_flakiness.v1`` report byte-for-byte at the contract level (summary,
    exact-repeat verdict, and per-cell stability/knife-edge structure).
    """
    assert COMMITTED_REPORT.exists(), f"missing committed report: {COMMITTED_REPORT}"
    committed = json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    assert committed["schema_version"] == SCHEMA_VERSION
    assert committed["summary"] == real_report["summary"]
    assert committed["exact_repeat"] == real_report["exact_repeat"]
    committed_cells = {(c["scenario_id"], c["planner"]): c for c in committed["cells"]}
    for cell in real_report["cells"]:
        key = (cell["scenario_id"], cell["planner"])
        committed_cell = committed_cells[key]
        assert committed_cell["stability_score"] == pytest.approx(cell["stability_score"])
        assert committed_cell["success_rate"] == pytest.approx(cell["success_rate"])
        assert committed_cell["knife_edge"] == cell["knife_edge"]
        assert committed_cell["n_seeds"] == cell["n_seeds"]
    # Provenance block is self-describing (not a benchmark claim).
    assert committed["_provenance"]["source_git_hash"]
    source_hashes = committed["_provenance"]["source_episode_sha256"]
    assert set(source_hashes) == {"goal", "orca", "ppo", "social_force"}
    assert all(len(value) == 64 for value in source_hashes.values())
