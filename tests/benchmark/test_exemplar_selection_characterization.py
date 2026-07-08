"""Characterization baseline tests for ``robot_sf/benchmark/exemplar_selection.py``.

These tests pin the *current observable behavior* of the exemplar selection
pipeline on small synthetic episode sets. They are table-driven and assert
exact golden values, including edge cases (empty input, single episode per
cell, ties, multi-cell ordering).

Purpose (issue #4874, Refs #4770): lock a behavioral baseline so the
post-submission refactor wave can prove behavior-preservation. If a test
reveals a genuine bug, do NOT fix it here — document it and file a separate
fix issue.

These tests are additive: they pin exact full ``SelectedEpisode`` records,
the grouping-key normalization table, the cell-key construction format, and
the manifest field set. They do not duplicate the selection-logic class
coverage in ``test_exemplar_selection.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.exemplar_selection import (
    METRIC_DIRECTIONS,
    SELECTION_MANIFEST_SCHEMA_VERSION,
    build_manifest,
    save_manifest,
    select_exemplars,
)

if TYPE_CHECKING:
    from pathlib import Path

_PLANNER_GROUP = ["planner_key"]


def _eps(values: list[float], *, planner: str = "A") -> list[dict[str, object]]:
    """Build episodes with a single ``collisions`` metric per episode."""
    return [
        {"episode_id": f"e{i}", "metrics": {"collisions": float(v)}, "algo": planner}
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# Exact SelectedEpisode golden records
# ---------------------------------------------------------------------------


def test_full_selected_episode_record_is_pinned() -> None:
    """Pin every field of a median selection including reason and seed default."""
    # values [3,1,4,2] -> sorted [1,2,3,4]; median idx = 4//2 = 2 -> value 3.0 (e0)
    episodes = _eps([3, 1, 4, 2])
    selected, skipped = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("median",)
    )
    assert skipped == []
    assert len(selected) == 1
    ep = selected[0]
    assert ep.episode_id == "e0"
    assert ep.planner_key == "A"
    assert ep.scenario_id == ""  # missing field defaults to empty string
    assert ep.seed == 0  # missing field defaults to 0
    assert ep.selection_mode == "median"
    assert ep.selection_rank == 2
    assert ep.metric_value == pytest.approx(3.0)
    assert ep.reason == "median within planner_key=A"


def test_best_and_worst_indices_respect_lower_is_better() -> None:
    """For ``collisions`` (lower better): best=min, worst=max after ascending sort."""
    episodes = _eps([3, 1, 4, 2])  # sorted [1(e1), 2(e3), 3(e0), 4(e2)]
    selected, _ = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("best", "worst")
    )
    by_mode = {s.selection_mode: s for s in selected}
    assert by_mode["best"].episode_id == "e1"  # value 1.0, rank 0
    assert by_mode["best"].metric_value == pytest.approx(1.0)
    assert by_mode["best"].selection_rank == 0
    assert by_mode["worst"].episode_id == "e2"  # value 4.0, rank 3
    assert by_mode["worst"].selection_rank == 3


def test_all_three_modes_produce_three_selections_per_cell() -> None:
    """All three modes produce one selection each, in the declared order."""
    episodes = _eps([1, 2, 3])
    selected, _ = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("median", "best", "worst")
    )
    modes = [s.selection_mode for s in selected]
    assert modes == ["median", "best", "worst"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_episodes_returns_empty_selection_and_no_skips() -> None:
    """An empty episode list yields no selections and no skipped cells."""
    selected, skipped = select_exemplars([], group_by=_PLANNER_GROUP, metric="collisions")
    assert selected == []
    assert skipped == []


def test_single_episode_per_cell_all_modes_select_same_episode() -> None:
    """A single-episode cell selects that episode under every mode at rank 0."""
    episodes = _eps([5.0])
    selected, skipped = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("median", "best", "worst")
    )
    assert skipped == []
    assert {s.episode_id for s in selected} == {"e0"}
    # Each mode lands on rank 0 of a length-1 cell.
    assert {s.selection_rank for s in selected} == {0}


def test_tie_keeps_first_in_input_order_after_stable_sort() -> None:
    """Equal metric values keep input order; best(lower) selects the first."""
    episodes = [
        {"episode_id": "first", "metrics": {"collisions": 2.0}, "algo": "A"},
        {"episode_id": "second", "metrics": {"collisions": 2.0}, "algo": "A"},
    ]
    selected, _ = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("best",)
    )
    assert selected[0].episode_id == "first"


def test_all_missing_metric_creates_skipped_cell_with_reason() -> None:
    """A cell whose episodes all lack the metric is skipped with a reason."""
    episodes = [
        {"episode_id": "e1", "metrics": {"other": 1.0}, "algo": "A"},
        {"episode_id": "e2", "metrics": {}, "algo": "A"},
    ]
    selected, skipped = select_exemplars(episodes, group_by=_PLANNER_GROUP, metric="collisions")
    assert selected == []
    assert len(skipped) == 1
    assert skipped[0].cell_key == "planner_key=A"
    assert "collisions" in skipped[0].reason


def test_cells_processed_in_sorted_cell_key_order() -> None:
    """Multi-cell selection emits results in sorted cell-key order (A before B)."""
    episodes = [
        {"episode_id": "b1", "metrics": {"collisions": 1.0}, "algo": "B"},
        {"episode_id": "a1", "metrics": {"collisions": 2.0}, "algo": "A"},
    ]
    selected, _ = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("best",)
    )
    assert [s.planner_key for s in selected] == ["A", "B"]


# ---------------------------------------------------------------------------
# Grouping-key normalization table
# ---------------------------------------------------------------------------


def test_planner_x_outcome_grouping_cell_key_format() -> None:
    """The cell key is ``key=value`` pairs sorted and pipe-joined."""
    episodes = [
        {
            "episode_id": "e1",
            "algo": "A",
            "outcome": {"route_complete": True},
            "metrics": {"collisions": 0.0},
        },
        {
            "episode_id": "e2",
            "algo": "A",
            "outcome": {"collision_event": True},
            "metrics": {"collisions": 1.0},
        },
    ]
    selected, _ = select_exemplars(
        episodes, group_by=["planner_key", "outcome"], metric="collisions", modes=("best",)
    )
    cell_keys = {s.reason.split(" within ", 1)[1] for s in selected}
    assert cell_keys == {"outcome=success|planner_key=A", "outcome=collision|planner_key=A"}


def test_metric_direction_auto_detection_mapping_is_locked() -> None:
    """Pin the documented metric-direction mapping used for best/worst selection."""
    assert METRIC_DIRECTIONS["collisions"] == "lower"
    assert METRIC_DIRECTIONS["path_efficiency"] == "higher"
    assert METRIC_DIRECTIONS["success"] == "higher"
    assert METRIC_DIRECTIONS["clearing_distance_min"] == "higher"
    assert METRIC_DIRECTIONS["near_misses"] == "lower"
    # Unknown metrics default to "lower" via ``.get(metric, "lower")``.
    assert METRIC_DIRECTIONS.get("totally_unknown_metric", "lower") == "lower"


# ---------------------------------------------------------------------------
# Manifest building and persistence
# ---------------------------------------------------------------------------


def test_build_manifest_pinned_fields_and_auto_direction(tmp_path: Path) -> None:
    """The manifest carries pinned provenance fields and the auto-detected direction."""
    episodes = _eps([3, 1, 4, 2])
    selected, skipped = select_exemplars(
        episodes, group_by=_PLANNER_GROUP, metric="collisions", modes=("median",)
    )
    source = tmp_path / "episodes.jsonl"
    source.write_text("\n".join(json.dumps(e) for e in episodes), encoding="utf-8")

    manifest = build_manifest(
        source_episodes_path=source,
        group_by=_PLANNER_GROUP,
        metric="collisions",
        selected=selected,
        skipped_cells=skipped,
    )
    assert manifest.schema_version == SELECTION_MANIFEST_SCHEMA_VERSION
    assert manifest.metric_direction == "lower"  # auto-detected from METRIC_DIRECTIONS
    assert manifest.group_by == ["planner_key"]
    assert manifest.metric == "collisions"
    assert manifest.source_episodes == str(source)
    assert len(manifest.source_sha256) == 64  # hex SHA-256 digest
    assert manifest.selected == selected
    assert manifest.skipped_cells == skipped


def test_save_manifest_roundtrips_to_json(tmp_path: Path) -> None:
    """Saved manifests round-trip through JSON with the locked field set."""
    episodes = _eps([1.0])
    selected, skipped = select_exemplars(episodes, group_by=_PLANNER_GROUP, metric="collisions")
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(episodes[0]), encoding="utf-8")
    manifest = build_manifest(
        source_episodes_path=source,
        group_by=_PLANNER_GROUP,
        metric="collisions",
        selected=selected,
        skipped_cells=skipped,
    )
    out = save_manifest(manifest, tmp_path / "nested" / "manifest.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "source_episodes",
        "source_sha256",
        "group_by",
        "metric",
        "metric_direction",
        "selected",
        "skipped_cells",
        "git_hash",
    }
    assert payload["selected"][0]["episode_id"] == "e0"
