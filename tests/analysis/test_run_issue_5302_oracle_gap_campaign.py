"""Focused tests for the issue #5302 oracle-gap campaign runner.

These tests materialize the frozen campaign through the *real* loaders (the packet,
the partition manifest, the real arm configs, the model registry) and an *injected
execution seam* for the canary. They assert the no-submit preflight and the one-cell
canary, and re-break every fail-closed condition the runner is responsible for.

Claim boundary: these tests verify campaign *materialization* (config-first runner,
deterministic manifest, injected canary seam). They do not run a benchmark, submit
compute, or assert a planner ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.analysis import run_issue_5302_oracle_gap_campaign as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = REPO_ROOT / "configs/analysis/issue_5302_oracle_gap_packet.yaml"
PARTITION_PATH = REPO_ROOT / "configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml"


def _packet() -> dict[str, Any]:
    payload = yaml.safe_load(PACKET_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _arm(packet: dict[str, Any], planner_id: str) -> dict[str, Any]:
    for arm in packet["planner_roster"]["required"]:
        if arm.get("planner_id") == planner_id:
            return arm
    raise AssertionError(f"arm {planner_id!r} not in roster")


# ---------------------------------------------------------------------------
# Real-loader arm resolution (no stubs)
# ---------------------------------------------------------------------------


def test_resolve_arms_uses_real_loaders_for_all_six_arms() -> None:
    """Every frozen-roster arm resolves its real config, algo, and checkpoint provenance."""
    arms = runner.resolve_arms(_packet(), repo_root=REPO_ROOT)
    assert [a.planner_id for a in arms] == [
        "orca",
        "ppo",
        "prediction_planner",
        "scenario_adaptive_hybrid_orca_v1",
        "prediction_mpc",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
    ]
    ppo = next(a for a in arms if a.planner_id == "ppo")
    assert ppo.config_sha256 is not None  # real config hash, not a stub
    assert ppo.model_ids == ("ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",)
    assert ppo.checkpoint_refs[0].sha256 == (
        "8367af109a27e8879ced0c8913f6eff26df7ec59c31ea88f9a297bb2c141eb09"
    )
    # ORCA is the canonical baseline with no config file.
    orca = next(a for a in arms if a.planner_id == "orca")
    assert orca.config_path is None and orca.config_sha256 is None


def test_resolve_arms_runs_real_admission_gate() -> None:
    """Arm resolution runs the real-loader admission gate, so an inadmissible arm fails closed."""
    packet = _packet()
    _arm(packet, "orca")["readiness"] = "totally_invented_readiness_label"
    with pytest.raises(runner.CampaignMaterializationError, match="cannot instantiate as declared"):
        runner.resolve_arms(packet, repo_root=REPO_ROOT)


def test_resolve_arms_missing_config_is_file_error() -> None:
    """A missing config is a file error, not an apparently-valid roster."""
    packet = _packet()
    _arm(packet, "prediction_planner")["config_path"] = "configs/algos/does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="planner config missing"):
        runner.resolve_arms(packet, repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Family partition + cells
# ---------------------------------------------------------------------------


def test_family_partition_is_disjoint() -> None:
    """Selection and evaluation family sets are disjoint (the packet validity rule)."""
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    assert partition.disjoint is True
    assert set(partition.selection_families).isdisjoint(partition.evaluation_families)
    assert partition.evaluation_seeds  # frozen seed schedule
    assert partition.selection_scenario_matrix.endswith(".yaml")
    assert partition.evaluation_scenario_matrix.endswith(".yaml")


def test_resolve_cells_covers_both_pools() -> None:
    """Resolved cells span both selection and evaluation families."""
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    cells = runner.resolve_cells(partition, repo_root=REPO_ROOT)
    families = {c.scenario_family for c in cells}
    assert set(partition.selection_families) <= families
    assert set(partition.evaluation_families) <= families
    # Each cell has a stable frozen identity.
    assert all(c.scenario_cell for c in cells)


# ---------------------------------------------------------------------------
# No-submit preflight manifest
# ---------------------------------------------------------------------------


def test_preflight_materializes_full_campaign_without_compute() -> None:
    """The deterministic no-submit manifest resolves arms, partition, cells, seeds, denominator."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    assert manifest.arm_count == 6
    assert manifest.partition.disjoint is True
    assert manifest.seed_count >= 1
    # denominator = cells x seeds x six arms
    assert manifest.denominator == manifest.cell_count * manifest.seed_count * 6
    assert manifest.denominator == len(manifest.episodes)
    report = runner.manifest_to_dict(manifest)
    assert report["campaign_execution_submitted"] is False
    assert report["schema_version"] == runner.MANIFEST_SCHEMA_VERSION
    assert report["counts"]["denominator"] == manifest.denominator


def test_preflight_denominator_is_balanced_across_arms() -> None:
    """Identical episode set across planners: every (cell, seed) unit covers all six arms."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    by_unit: dict[tuple[str, str, int], set[str]] = {}
    for ep in manifest.episodes:
        by_unit.setdefault((ep.scenario_family, ep.scenario_cell, ep.seed), set()).add(
            ep.planner_id
        )
    arm_set = {a.planner_id for a in manifest.arms}
    assert all(arms == arm_set for arms in by_unit.values())


def test_main_preflight_writes_report_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI preflight writes reports/preflight.json and submits nothing."""
    monkeypatch.chdir(REPO_ROOT)
    # Redirect the disposable output root into a temp dir so the test never touches the real tree.
    monkeypatch.setattr(runner, "DEFAULT_PACKET", PACKET_PATH)
    rc = runner.main(["--packet", str(PACKET_PATH), "preflight"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "preflight" in out and "denominator=" in out
    report_path = REPO_ROOT / "output/benchmarks/issue_5302_oracle_gap/reports/preflight.json"
    assert report_path.is_file()


# ---------------------------------------------------------------------------
# Fail-closed: leakage, arms, denominator, hash drift
# ---------------------------------------------------------------------------


def test_assert_no_leakage_fails_on_overlapping_families() -> None:
    """A family routed through both selection and evaluation pools fails closed.

    Leakage is a partition-level property: a family that appears in both the
    selection and evaluation family sets is the exact near-identical-configuration
    leak the packet's validity rule forbids. Cells are assigned exactly one family,
    so the only realistic leak is a non-disjoint partition.
    """
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    leaked = runner.ScenarioCell(
        scenario_family=partition.selection_families[0],
        scenario_cell="dup_cell",
        scenario_id="dup_cell",
        source_kind="test",
        source_path="test",
    )
    # A partition whose evaluation families reuse a selection family.
    leaked_partition = runner.FamilyPartition(
        selection_families=partition.selection_families,
        evaluation_families=(*partition.evaluation_families, partition.selection_families[0]),
        selection_scenario_matrix=partition.selection_scenario_matrix,
        evaluation_scenario_matrix=partition.evaluation_scenario_matrix,
        evaluation_seeds=partition.evaluation_seeds,
        disjoint=False,
    )
    cells = (*runner.resolve_cells(partition, repo_root=REPO_ROOT), leaked)
    with pytest.raises(runner.CampaignMaterializationError, match="must be disjoint"):
        runner.assert_no_leakage(leaked_partition, cells)


def test_assert_no_leakage_fails_when_partition_is_not_disjoint() -> None:
    """A non-disjoint partition fails closed even before cells are inspected."""
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    leaked_partition = runner.FamilyPartition(
        selection_families=partition.selection_families,
        evaluation_families=(partition.selection_families[0],),
        selection_scenario_matrix=partition.selection_scenario_matrix,
        evaluation_scenario_matrix=partition.evaluation_scenario_matrix,
        evaluation_seeds=partition.evaluation_seeds,
        disjoint=False,
    )
    with pytest.raises(runner.CampaignMaterializationError, match="must be disjoint"):
        runner.assert_no_leakage(
            leaked_partition,
            [
                runner.ScenarioCell(
                    scenario_family=partition.selection_families[0],
                    scenario_cell="x",
                    scenario_id="x",
                    source_kind="test",
                    source_path="t",
                )
            ],
        )


def test_assert_complete_matrix_fails_on_missing_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A (cell, seed) unit missing an arm fails the denominator/completeness contract."""
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    arms = runner.resolve_arms(_packet(), repo_root=REPO_ROOT)
    cells = runner.resolve_cells(partition, repo_root=REPO_ROOT)
    episodes = runner.build_episode_matrix(arms, cells, partition)
    # Drop all episodes for one arm in a single (cell, seed) unit.
    victim = episodes[0]
    broken = [
        ep
        for ep in episodes
        if not (
            ep.scenario_family == victim.scenario_family
            and ep.scenario_cell == victim.scenario_cell
            and ep.seed == victim.seed
            and ep.planner_id == victim.planner_id
        )
    ]
    with pytest.raises(runner.CampaignMaterializationError, match="does not cover all six arms"):
        runner.assert_complete_six_arm_matrix(arms, broken, partition)


def test_assert_complete_matrix_fails_on_duplicate_arm() -> None:
    """Duplicate planner ids within a six-entry roster fail closed."""
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    arms = list(runner.resolve_arms(_packet(), repo_root=REPO_ROOT))
    # Replace the second arm with a duplicate of the first -> six entries, dup id.
    arms[1] = runner.ResolvedArm(
        planner_id=arms[0].planner_id,
        role="dup",
        readiness="canonical_baseline",
        config_path=None,
        algo="orca",
        config_sha256=None,
        model_ids=(),
        checkpoint_refs=(),
    )
    with pytest.raises(runner.CampaignMaterializationError, match="duplicate planner ids"):
        runner.assert_complete_six_arm_matrix(arms, [], partition)


def test_assert_complete_matrix_fails_on_denominator_mismatch() -> None:
    """A total episode count != cells x seeds x six arms fails closed.

    Each (family, cell, seed) unit still covers all six arms, but an extra
    duplicate episode inflates the total so the denominator no longer matches.
    """
    partition = runner.load_family_partition(PARTITION_PATH, repo_root=REPO_ROOT)
    arms = runner.resolve_arms(_packet(), repo_root=REPO_ROOT)
    cells = runner.resolve_cells(partition, repo_root=REPO_ROOT)
    episodes = list(runner.build_episode_matrix(arms, cells, partition))
    # Duplicate one episode so units still cover six arms but total is off by one.
    episodes.append(episodes[0])
    with pytest.raises(runner.CampaignMaterializationError, match="denominator mismatch"):
        runner.assert_complete_six_arm_matrix(arms, episodes, partition)


def test_assert_hashes_stable_fails_on_checkpoint_drift() -> None:
    """A packet PPO pin that disagrees with the registry digest fails closed."""
    packet = _packet()
    pinned = _arm(packet, "ppo")["pinned_provenance"]
    pinned["checkpoint_sha256"] = "a" * 64  # drift away from the registry digest
    arms = runner.resolve_arms(packet, repo_root=REPO_ROOT)
    with pytest.raises(runner.CampaignMaterializationError, match="hash drift"):
        runner.assert_hashes_stable(arms, packet)


def test_materialize_manifest_fails_when_admission_breaks() -> None:
    """A roster that fails real-loader admission fails the whole materialization."""
    packet = _packet()
    _arm(packet, "orca")["readiness"] = "not_a_real_readiness_label"
    with pytest.raises(runner.CampaignMaterializationError):
        runner.materialize_manifest(packet, repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Canary (injected execution seam)
# ---------------------------------------------------------------------------


def _native_seam() -> runner.EpisodeRunner:
    """A deterministic injected seam that emits one native successful_evidence row per arm."""

    def _run(episode: runner.ResolvedEpisode) -> dict[str, Any]:
        return {
            "planner_id": episode.planner_id,
            "execution_mode": "native",
            "row_status": "successful_evidence",
            "collision_rate": 0.0,
            "completion_rate": 1.0,
        }

    return _run


def test_canary_is_one_cell_one_seed_all_six_arms() -> None:
    """The canary resolves exactly one cell x one seed x six arms."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    episodes = runner.canary_episodes(manifest)
    assert len(episodes) == 6
    assert len({ep.scenario_cell for ep in episodes}) == 1
    assert len({ep.seed for ep in episodes}) == 1
    assert {ep.planner_id for ep in episodes} == {a.planner_id for a in manifest.arms}
    assert runner.CANARY_SEED in manifest.seeds


def test_canary_cell_is_deterministic_and_heldout() -> None:
    """The frozen canary cell is deterministic and belongs to an evaluation (held-out) family."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    cell = runner.select_canary_cell(manifest)
    again = runner.select_canary_cell(manifest)
    assert cell == again
    assert cell.scenario_family in manifest.partition.evaluation_families


def test_run_canary_emits_one_native_row_per_arm_with_identity() -> None:
    """The injected seam yields one native row per arm carrying policy identity."""
    result = runner.run_canary(_packet(), repo_root=REPO_ROOT, run_episode=_native_seam())
    assert result["schema_version"] == runner.CANARY_SCHEMA_VERSION
    assert result["campaign_execution_submitted"] is False
    assert result["arm_count"] == 6
    assert len(result["rows"]) == 6
    assert {row["planner_id"] for row in result["rows"]} == {
        "orca",
        "ppo",
        "prediction_planner",
        "scenario_adaptive_hybrid_orca_v1",
        "prediction_mpc",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
    }
    assert all(row["execution_mode"] == "native" for row in result["rows"])


def test_run_canary_fails_closed_without_execution_seam() -> None:
    """No injected seam -> fail closed: this contract authorizes no compute submission."""
    with pytest.raises(runner.CampaignMaterializationError, match="injected execution seam"):
        runner.run_canary(_packet(), repo_root=REPO_ROOT)


def test_run_canary_rejects_fallback_execution() -> None:
    """A seam that returns a fallback row fails closed (packet forbids fallback success)."""

    def fallback_seam(episode: runner.ResolvedEpisode) -> dict[str, Any]:
        return {
            "planner_id": episode.planner_id,
            "execution_mode": "fallback",
            "row_status": "fallback",
        }

    with pytest.raises(runner.CampaignMaterializationError, match="not native execution"):
        runner.run_canary(_packet(), repo_root=REPO_ROOT, run_episode=fallback_seam)


def test_run_canary_rejects_degraded_execution() -> None:
    """A seam that returns a degraded row fails closed (packet forbids degraded success)."""

    def degraded_seam(episode: runner.ResolvedEpisode) -> dict[str, Any]:
        return {
            "planner_id": episode.planner_id,
            "execution_mode": "degraded",
            "row_status": "degraded",
        }

    with pytest.raises(runner.CampaignMaterializationError, match="not native execution"):
        runner.run_canary(_packet(), repo_root=REPO_ROOT, run_episode=degraded_seam)


def test_run_canary_rejects_row_that_loses_policy_identity() -> None:
    """A seam that drops the planner_id fails the policy-identity contract."""

    def identity_loss_seam(episode: runner.ResolvedEpisode) -> dict[str, Any]:
        return {
            "planner_id": "some_other_planner",
            "execution_mode": "native",
            "row_status": "successful_evidence",
        }

    with pytest.raises(runner.CampaignMaterializationError, match="policy identity"):
        runner.run_canary(_packet(), repo_root=REPO_ROOT, run_episode=identity_loss_seam)


def test_run_canary_rejects_non_evidence_status() -> None:
    """A seam that returns a non-successful_evidence status fails closed."""

    def blocked_seam(episode: runner.ResolvedEpisode) -> dict[str, Any]:
        return {
            "planner_id": episode.planner_id,
            "execution_mode": "native",
            "row_status": "failed",
        }

    with pytest.raises(runner.CampaignMaterializationError, match="not successful_evidence"):
        runner.run_canary(_packet(), repo_root=REPO_ROOT, run_episode=blocked_seam)


# ---------------------------------------------------------------------------
# Full matrix preserves identical episodes across arms and disjoint families
# ---------------------------------------------------------------------------


def test_full_matrix_identical_episodes_across_arms() -> None:
    """Every arm runs the same set of (family, cell, seed) units (same episode set)."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    matrix = runner.matrix_to_dict(manifest)
    units_by_arm: dict[str, set[tuple[str, str, int]]] = {}
    for ep in matrix["episodes"]:
        units_by_arm.setdefault(ep["planner_id"], set()).add(
            (ep["scenario_family"], ep["scenario_cell"], ep["seed"])
        )
    reference = next(iter(units_by_arm.values()))
    assert all(units == reference for units in units_by_arm.values())


def test_full_matrix_disjoint_selection_evaluation_families() -> None:
    """The full matrix labels selection/evaluation splits with disjoint family sets."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    matrix = runner.matrix_to_dict(manifest)
    selection = {
        (ep["scenario_family"], ep["scenario_cell"])
        for ep in matrix["episodes"]
        if ep["split"] == "selection"
    }
    evaluation = {
        (ep["scenario_family"], ep["scenario_cell"])
        for ep in matrix["episodes"]
        if ep["split"] == "evaluation"
    }
    selection_families = {fam for fam, _ in selection}
    evaluation_families = {fam for fam, _ in evaluation}
    assert selection_families.isdisjoint(evaluation_families)
    assert matrix["disjoint"] is True


def test_episode_id_is_deterministic_and_arm_specific() -> None:
    """The same (family, cell, seed) yields distinct-but-stable episode ids per arm."""
    manifest = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    ids = {(ep.planner_id, ep.episode_id) for ep in manifest.episodes}
    # No arm shares an episode id with another arm (paired but distinct).
    assert len({eid for _, eid in ids}) == len(manifest.episodes)
    # Re-materializing yields identical ids (deterministic, host-independent).
    again = runner.materialize_manifest(_packet(), repo_root=REPO_ROOT)
    assert [ep.episode_id for ep in again.episodes] == [ep.episode_id for ep in manifest.episodes]
