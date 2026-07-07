"""Tests for exemplar episode selection (issue #4778).

Validates:
- Deterministic median selection for odd and even cell sizes
- Best/worst obey metric direction
- Missing metric creates skipped cell
- Tie-breaking is deterministic
- Manifest includes source hash and claim boundary
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.exemplar_selection import (
    ExemplarSelectionError,
    SelectionManifest,
    build_manifest,
    save_manifest,
    select_exemplars,
)


def _make_episode(
    episode_id: str,
    planner: str,
    scenario_id: str,
    seed: int,
    metric_value: float | None,
    metric_name: str = "path_efficiency",
    *,
    outcome: str | None = None,
) -> dict[str, any]:
    """Create a minimal episode record for testing."""
    metrics = {}
    if metric_value is not None:
        metrics[metric_name] = metric_value

    outcome_dict = {"collision_event": False, "route_complete": False, "timeout_event": True}
    if outcome == "success":
        outcome_dict = {"collision_event": False, "route_complete": True, "timeout_event": False}
    elif outcome == "collision":
        outcome_dict = {"collision_event": True, "route_complete": False, "timeout_event": False}

    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": planner,
        "metrics": metrics,
        "outcome": outcome_dict,
        "status": outcome if outcome else "timeout",
    }


class TestMedianSelection:
    """Test median selection determinism."""

    def test_odd_cell_size(self) -> None:
        """Median of 3 episodes selects the middle one (index 1)."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.5),
            _make_episode("ep2", "orca", "s1", 2, 0.8),
            _make_episode("ep3", "orca", "s1", 3, 0.3),
        ]
        selected, skipped = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["median"],
        )
        assert len(selected) == 1
        assert len(skipped) == 0
        # Sorted descending: [0.8, 0.5, 0.3], median index=1 -> 0.5 (ep1)
        assert selected[0].episode_id == "ep1"
        assert selected[0].selection_mode == "median"
        assert selected[0].metric_value == 0.5

    def test_even_cell_size(self) -> None:
        """Median of 4 episodes selects index 2 (floor(n/2))."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.1),
            _make_episode("ep2", "orca", "s1", 2, 0.5),
            _make_episode("ep3", "orca", "s1", 3, 0.8),
            _make_episode("ep4", "orca", "s1", 4, 0.9),
        ]
        selected, _skipped = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["median"],
        )
        assert len(selected) == 1
        # Sorted descending: [0.9, 0.8, 0.5, 0.1], median index=2 -> 0.5
        assert selected[0].episode_id == "ep2"
        assert selected[0].metric_value == 0.5


class TestBestWorstDirection:
    """Test that best/worst obey metric direction."""

    def test_higher_is_better(self) -> None:
        """With higher-is-better, best has highest value."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.3),
            _make_episode("ep2", "orca", "s1", 2, 0.9),
            _make_episode("ep3", "orca", "s1", 3, 0.6),
        ]
        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best", "worst"],
        )
        best = next(s for s in selected if s.selection_mode == "best")
        worst = next(s for s in selected if s.selection_mode == "worst")
        assert best.episode_id == "ep2"
        assert best.metric_value == 0.9
        assert worst.episode_id == "ep1"
        assert worst.metric_value == 0.3

    def test_lower_is_better(self) -> None:
        """With lower-is-better, best has lowest value."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 3, metric_name="collisions"),
            _make_episode("ep2", "orca", "s1", 2, 0, metric_name="collisions"),
            _make_episode("ep3", "orca", "s1", 3, 1, metric_name="collisions"),
        ]
        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="collisions",
            metric_direction="lower",
            modes=["best", "worst"],
        )
        best = next(s for s in selected if s.selection_mode == "best")
        worst = next(s for s in selected if s.selection_mode == "worst")
        assert best.episode_id == "ep2"
        assert best.metric_value == 0.0
        assert worst.episode_id == "ep1"
        assert worst.metric_value == 3.0


class TestMissingMetric:
    """Test handling of missing or non-finite metric values."""

    def test_all_missing_creates_skipped_cell(self) -> None:
        """When all episodes have missing metric, cell is skipped."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, None),
            _make_episode("ep2", "orca", "s1", 2, None),
        ]
        selected, skipped = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            modes=["median"],
        )
        assert len(selected) == 0
        assert len(skipped) == 1
        assert "missing" in skipped[0].reason.lower() or "non-finite" in skipped[0].reason.lower()

    def test_partial_missing_selects_valid(self) -> None:
        """Episodes with valid metric are selected; missing ones are ignored."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, None),
            _make_episode("ep2", "orca", "s1", 2, 0.7),
            _make_episode("ep3", "orca", "s1", 3, 0.9),
        ]
        selected, _skipped = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["median"],
        )
        assert len(selected) == 1
        # Only 2 valid episodes, median index=1 -> 0.7
        assert selected[0].metric_value == 0.7


class TestTieBreaking:
    """Test deterministic tie-breaking."""

    def test_same_metric_value_different_episode_id(self) -> None:
        """When metric values tie, sort is stable (by input order)."""
        episodes = [
            _make_episode("ep_a", "orca", "s1", 1, 0.5),
            _make_episode("ep_b", "orca", "s1", 2, 0.5),
            _make_episode("ep_c", "orca", "s1", 3, 0.5),
        ]
        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best", "median", "worst"],
        )
        # All have same value; best=first in sorted order, worst=last
        best = next(s for s in selected if s.selection_mode == "best")
        worst = next(s for s in selected if s.selection_mode == "worst")
        assert best.episode_id == "ep_a"
        assert worst.episode_id == "ep_c"


class TestGrouping:
    """Test multi-key grouping."""

    def test_planner_x_outcome_grouping(self) -> None:
        """Episodes are grouped by planner x outcome."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.9, outcome="success"),
            _make_episode("ep2", "orca", "s1", 2, 0.3, outcome="timeout"),
            _make_episode("ep3", "sf", "s1", 3, 0.8, outcome="success"),
            _make_episode("ep4", "sf", "s1", 4, 0.2, outcome="timeout"),
        ]
        selected, skipped = select_exemplars(
            episodes,
            group_by=["planner_key", "outcome"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best"],
        )
        # 4 groups x 1 mode = 4 selections
        assert len(selected) == 4
        assert len(skipped) == 0

        orca_success = next(
            s for s in selected if s.planner_key == "orca" and "success" in s.reason
        )
        assert orca_success.episode_id == "ep1"


class TestManifest:
    """Test manifest building and serialization."""

    def test_manifest_schema_version(self) -> None:
        """Manifest has correct schema version."""
        manifest = SelectionManifest()
        assert manifest.schema_version == "exemplar-selection.v1"

    def test_manifest_to_dict_roundtrip(self) -> None:
        """Manifest to_dict is JSON-serializable."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.9),
            _make_episode("ep2", "orca", "s1", 2, 0.5),
        ]
        selected, skipped = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            modes=["best"],
        )

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
            tmp_path = Path(f.name)

        try:
            manifest = build_manifest(
                source_episodes_path=tmp_path,
                group_by=["planner_key"],
                metric="path_efficiency",
                selected=selected,
                skipped_cells=skipped,
            )
            d = manifest.to_dict()
            json_str = json.dumps(d)
            assert "exemplar-selection.v1" in json_str
            assert len(d["selected"]) == 1
            assert d["source_sha256"] != "unknown"
        finally:
            tmp_path.unlink()

    def test_save_manifest_creates_file(self) -> None:
        """save_manifest writes a valid JSON file."""
        manifest = SelectionManifest(
            metric="path_efficiency",
            selected=[],
            skipped_cells=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_manifest(manifest, Path(tmpdir) / "sub" / "manifest.json")
            assert out.exists()
            data = json.loads(out.read_text())
            assert data["schema_version"] == "exemplar-selection.v1"


class TestInvalidMode:
    """Test invalid selection mode rejection."""

    def test_invalid_mode_raises(self) -> None:
        """Invalid selection mode raises ExemplarSelectionError."""
        episodes = [_make_episode("ep1", "orca", "s1", 1, 0.5)]
        with pytest.raises(ExemplarSelectionError, match="Invalid selection mode"):
            select_exemplars(
                episodes,
                group_by=["planner_key"],
                metric="path_efficiency",
                modes=["invalid_mode"],
            )


class TestMetricDirectionAutoDetection:
    """Test automatic metric direction detection."""

    def test_auto_detect_lower(self) -> None:
        """Known lower-is-better metrics are auto-detected."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 3, metric_name="collisions"),
            _make_episode("ep2", "orca", "s1", 2, 0, metric_name="collisions"),
        ]
        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="collisions",
            modes=["best"],
        )
        assert selected[0].episode_id == "ep2"
        assert selected[0].metric_value == 0.0

    def test_auto_detect_higher(self) -> None:
        """Known higher-is-better metrics are auto-detected."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.3),
            _make_episode("ep2", "orca", "s1", 2, 0.9),
        ]
        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            modes=["best"],
        )
        assert selected[0].episode_id == "ep2"
