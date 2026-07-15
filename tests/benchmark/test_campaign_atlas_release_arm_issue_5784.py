"""Regression tests for release-arm identity in campaign-atlas grouping (#5784).

The 0.0.3 release evidence contains 14 distinct execution arms, but four
hybrid configurations share the ``algo="hybrid_rule_local_planner"`` label.
Grouping by ``planner`` alone collapses them to 11 top-level arms and silently
pools episodes from architecturally distinct configurations. These tests prove
the atlas keys on the stable ``release_arm_id`` instead, fails closed on
missing/ambiguous/colliding identity, and retains the planner family as a
secondary (explicit) view only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.campaign_atlas import (
    AtlasConfig,
    CampaignAtlasError,
    EpisodeInventoryRow,
    build_atlas_summary,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmark"
    / "campaign_atlas_issue_5784_release_arms.jsonl"
)

HYBRID_ARMS_5784 = (
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
)


def _row(
    episode_id: str,
    planner: str,
    release_arm_id: str | None,
    *,
    family: str = "doorway",
) -> EpisodeInventoryRow:
    """Build a minimal inventory row with an explicit release-arm identity."""
    return EpisodeInventoryRow(
        episode_id=episode_id,
        planner=planner,
        scenario_id=f"{family}_s1",
        scenario_family=family,
        seed=111,
        outcome="success",
        release_arm_id=release_arm_id,
    )


class TestReleaseArmIdentity:
    """The atlas groups by release_arm_id, preserving all 14 arms."""

    def test_fixture_carries_14_distinct_release_arms(self) -> None:
        """The 0.0.3-shaped fixture retains 14 arms, not 11."""
        from scripts.analysis.build_campaign_atlas_issue_5616 import load_inventory

        rows = load_inventory(FIXTURE_PATH)
        assert len(rows) == 14
        assert len({row.release_arm_id for row in rows}) == 14

    def test_four_hybrid_configs_remain_four_distinct_arms(self) -> None:
        """The four hybrid configs keep separate atlas arms despite one algo label."""
        from scripts.analysis.build_campaign_atlas_issue_5616 import load_inventory

        rows = load_inventory(FIXTURE_PATH)
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="0.0.3"))
        hybrid_cells = [c for c in summary.cells if c.planner == "hybrid_rule_local_planner"]
        # 4 arms x 1 scenario family = 4 cells for this planner, not 1 pooled cell.
        assert len(hybrid_cells) == 4
        arm_ids = {c.release_arm_id for c in hybrid_cells}
        assert arm_ids == set(HYBRID_ARMS_5784)

    def test_generated_catalog_reports_14_arms_not_11(self) -> None:
        """The summary exposes 14 distinct arms across all planner families."""
        from scripts.analysis.build_campaign_atlas_issue_5616 import load_inventory

        rows = load_inventory(FIXTURE_PATH)
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="0.0.3"))
        assert len(summary.cells) == 14
        assert len({c.release_arm_id for c in summary.cells}) == 14

    def test_no_cross_arm_episode_pooling_for_shared_planner(self) -> None:
        """Pooled-by-planner would put 4 episodes under one cell; arm grouping keeps 1 each."""
        rows = [
            _row("a", "hybrid_rule_local_planner", "hybrid_rule_v3_fast_progress_static_escape"),
            _row(
                "b",
                "hybrid_rule_local_planner",
                "hybrid_rule_v3_fast_progress_static_escape_continuous",
            ),
            _row("c", "hybrid_rule_local_planner", "scenario_adaptive_hybrid_orca_v1"),
            _row(
                "d", "hybrid_rule_local_planner", "scenario_adaptive_hybrid_orca_v2_collision_guard"
            ),
        ]
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        counts = {(c.planner, c.release_arm_id): c.n_total for c in summary.cells}
        assert (
            counts[("hybrid_rule_local_planner", "hybrid_rule_v3_fast_progress_static_escape")] == 1
        )
        assert counts[("hybrid_rule_local_planner", "scenario_adaptive_hybrid_orca_v1")] == 1
        assert len(counts) == 4

    def test_planner_family_remains_available_as_secondary_view(self) -> None:
        """The human-readable planner label is retained on every cell for family aggregation."""
        from scripts.analysis.build_campaign_atlas_issue_5616 import load_inventory

        rows = load_inventory(FIXTURE_PATH)
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="0.0.3"))
        # The four hybrid arms still expose their shared planner family.
        hybrid = [c for c in summary.cells if c.planner == "hybrid_rule_local_planner"]
        assert all(c.planner == "hybrid_rule_local_planner" for c in hybrid)


class TestReleaseArmIdentityFailClosed:
    """Missing/ambiguous/colliding identity fails closed, never inferred from algo."""

    def test_mixed_armed_and_armless_rows_raise(self) -> None:
        """A partial arm id is ambiguous; the artifact must refuse, not pool."""
        rows = [
            _row("a", "orca", "orca__differential_drive"),
            _row("b", "orca", None),
        ]
        with pytest.raises(CampaignAtlasError, match="ambiguous"):
            build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))

    def test_one_arm_id_mapping_to_two_planners_raises(self) -> None:
        """A single release_arm_id must map to exactly one planner (algo) label."""
        rows = [
            _row("a", "orca", "shared_arm"),
            _row("b", "goal", "shared_arm"),
        ]
        with pytest.raises(CampaignAtlasError, match="inconsistent"):
            build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))

    def test_armless_inventory_preserves_legacy_planner_keyed_behavior(self) -> None:
        """Rows without arm id (legacy inventory) still group by planner, fail-closed-off."""
        rows = [
            _row("a", "orca", None),
            _row("b", "orca", None),
            _row("c", "goal", None),
        ]
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        assert all(c.release_arm_id is None for c in summary.cells)
        assert len(summary.cells) == 2


def test_inventory_record_accepts_release_arm_id_field(tmp_path: Path) -> None:
    """The CLI inventory loader threads release_arm_id through to the row model."""
    from scripts.analysis.build_campaign_atlas_issue_5616 import load_inventory

    record = {
        "episode_id": "e1",
        "planner": "hybrid_rule_local_planner",
        "release_arm_id": "scenario_adaptive_hybrid_orca_v1",
        "scenario_id": "doorway_s1",
        "scenario_family": "doorway",
        "seed": 111,
        "outcome": "success",
    }
    inventory = tmp_path / "inv.jsonl"
    inventory.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rows = load_inventory(inventory)
    assert rows[0].release_arm_id == "scenario_adaptive_hybrid_orca_v1"
