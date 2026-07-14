"""Tests for the issue #5446 release-bundle adapter.

Uses a tiny synthetic ``runs/<arm>/episodes.jsonl`` bundle (two rows across two
arms, one native and one adapter-mode) to check the adapter projects the
nested provenance fields the #5446 miner and #5616 atlas need without
inventing any value not literally present on the source row.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis import adapt_release_bundle_issue_5446_mining_run as adapter


def _episode_row(
    *,
    scenario_id: str,
    seed: int,
    algo: str,
    execution_mode: str,
    success: bool,
    collision_event: bool = False,
    timeout_event: bool = False,
    archetype: str | None = "doorway",
) -> dict:
    config_hash = f"cfg-{scenario_id}-{seed}"
    return {
        "episode_id": f"{scenario_id}--{seed}--{config_hash}",
        "scenario_id": scenario_id,
        "seed": seed,
        "config_hash": config_hash,
        "algo": algo,
        "git_hash": "top-level-git-hash",
        "result_provenance": {"repo_commit": "nested-repo-commit"},
        "algorithm_metadata": {"planner_kinematics": {"execution_mode": execution_mode}},
        "scenario_params": {"metadata": {"archetype": archetype}} if archetype else {},
        "outcome": {"collision_event": collision_event, "timeout_event": timeout_event},
        "metrics": {"success": success, "total_collision_count": int(collision_event)},
    }


def _write_bundle(tmp_path: Path) -> Path:
    payload = tmp_path / "payload"
    arm_native = payload / "runs" / "goal__differential_drive"
    arm_adapter = payload / "runs" / "orca__differential_drive"
    arm_native.mkdir(parents=True)
    arm_adapter.mkdir(parents=True)

    native_row = _episode_row(
        scenario_id="classic_doorway_low",
        seed=111,
        algo="goal",
        execution_mode="native",
        success=True,
    )
    adapter_row = _episode_row(
        scenario_id="classic_doorway_low",
        seed=111,
        algo="orca",
        execution_mode="adapter",
        success=False,
        collision_event=True,
    )
    (arm_native / "episodes.jsonl").write_text(json.dumps(native_row) + "\n", encoding="utf-8")
    (arm_adapter / "episodes.jsonl").write_text(json.dumps(adapter_row) + "\n", encoding="utf-8")
    return payload


def test_mining_row_projects_nested_provenance_to_top_level(tmp_path: Path) -> None:
    """repo_commit/execution_mode are lifted from nested paths without fabrication."""
    payload = _write_bundle(tmp_path)
    mining_out = tmp_path / "mining_rows.jsonl"
    summary = adapter.adapt(
        payload, mining_rows_out=mining_out, atlas_inventory_out=None, store_rows_out=None
    )

    assert summary["n_rows_total"] == 2
    assert summary["execution_mode_counts"] == {"native": 1, "adapter": 1}

    rows = [json.loads(line) for line in mining_out.read_text(encoding="utf-8").splitlines()]
    by_algo = {row["algo"]: row for row in rows}
    assert by_algo["goal"]["repo_commit"] == "nested-repo-commit"
    assert by_algo["goal"]["execution_mode"] == "native"
    assert by_algo["orca"]["execution_mode"] == "adapter"
    # config_hash/episode_id/scenario_id/seed pass through untouched.
    assert by_algo["goal"]["scenario_id"] == "classic_doorway_low"
    assert by_algo["goal"]["seed"] == 111


def test_mining_row_falls_back_to_top_level_git_hash(tmp_path: Path) -> None:
    """When result_provenance.repo_commit is absent, top-level git_hash is used."""
    row = _episode_row(
        scenario_id="classic_doorway_low",
        seed=111,
        algo="goal",
        execution_mode="native",
        success=True,
    )
    del row["result_provenance"]
    projected = adapter._mining_row(row)
    assert projected["repo_commit"] == "top-level-git-hash"


def test_atlas_row_derives_family_and_outcome_from_literal_fields(tmp_path: Path) -> None:
    """scenario_family comes from scenario_params.metadata.archetype; outcome folds real booleans."""
    payload = _write_bundle(tmp_path)
    atlas_out = tmp_path / "atlas_inventory.jsonl"
    adapter.adapt(payload, mining_rows_out=None, atlas_inventory_out=atlas_out, store_rows_out=None)

    rows = [json.loads(line) for line in atlas_out.read_text(encoding="utf-8").splitlines()]
    by_planner = {row["planner"]: row for row in rows}
    assert by_planner["goal"]["scenario_family"] == "doorway"
    assert by_planner["goal"]["outcome"] == "success"
    assert by_planner["orca"]["outcome"] == "collision"
    # No trajectory/event_anchors/predicate_timeline are fabricated.
    assert "trajectory" not in by_planner["goal"]
    assert "event_anchors" not in by_planner["goal"]


def test_scenario_family_falls_back_to_difficulty_stripped_id_without_archetype() -> None:
    """A row lacking scenario_params.metadata.archetype falls back, not fabrication."""
    row = _episode_row(
        scenario_id="classic_doorway_low",
        seed=111,
        algo="goal",
        execution_mode="native",
        success=True,
        archetype=None,
    )
    projected = adapter._atlas_row(row)
    assert projected["scenario_family"] == "classic_doorway"


def test_derive_outcome_label_precedence() -> None:
    """collision_event beats timeout_event beats metrics.success."""
    collision_row = {
        "outcome": {"collision_event": True, "timeout_event": True},
        "metrics": {"success": True},
    }
    timeout_row = {
        "outcome": {"collision_event": False, "timeout_event": True},
        "metrics": {"success": True},
    }
    other_row = {
        "outcome": {"collision_event": False, "timeout_event": False},
        "metrics": {"success": False},
    }
    assert adapter._derive_outcome_label(collision_row) == "collision"
    assert adapter._derive_outcome_label(timeout_row) == "timeout"
    assert adapter._derive_outcome_label(other_row) == "other"


def test_store_row_status_collapses_mixed_to_adapter() -> None:
    """campaign_result_store's ROW_STATUS_VALUES has no 'mixed' member; document the collapse."""
    row = _episode_row(
        scenario_id="classic_doorway_low",
        seed=111,
        algo="ppo",
        execution_mode="mixed",
        success=True,
    )
    store_row = adapter._store_row(
        row, run_id="run-1", line_no=0, source_path=Path("episodes.jsonl")
    )
    assert store_row["row_status"] == "adapter"


def test_adapter_is_deterministic_across_reruns(tmp_path: Path) -> None:
    """Re-running the adapter on the same bundle produces byte-identical output."""
    payload = _write_bundle(tmp_path)
    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    adapter.adapt(payload, mining_rows_out=out1, atlas_inventory_out=None, store_rows_out=None)
    adapter.adapt(payload, mining_rows_out=out2, atlas_inventory_out=None, store_rows_out=None)
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
