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
        """Median of 4 episodes selects index 2 (floor(n/2)) in canonical ascending order."""
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
        # Canonical ascending sort: [0.1, 0.5, 0.8, 0.9], median index=2 -> 0.8
        assert selected[0].episode_id == "ep3"
        assert selected[0].metric_value == 0.8


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
        # Only 2 valid episodes, canonical ascending: [0.7, 0.9], median index=1 -> 0.9
        assert selected[0].metric_value == 0.9


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
        # All have same value; with canonical ascending sort and higher direction,
        # best = last (ep_c), worst = first (ep_a).
        best = next(s for s in selected if s.selection_mode == "best")
        worst = next(s for s in selected if s.selection_mode == "worst")
        assert best.episode_id == "ep_c"
        assert worst.episode_id == "ep_a"

    def test_tie_breaker_with_step_count(self) -> None:
        """Tie-breaker metric selects episode with higher step count when primary metric is tied."""
        episodes = [
            _make_episode("ep_short", "orca", "s1", 1, 1.0),
            _make_episode("ep_medium", "orca", "s1", 2, 1.0),
            _make_episode("ep_long", "orca", "s1", 3, 1.0),
        ]
        # Add step counts to metrics
        episodes[0]["metrics"]["step_count"] = 5
        episodes[1]["metrics"]["step_count"] = 50
        episodes[2]["metrics"]["step_count"] = 100

        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best"],
            tie_breaker="step_count",
        )
        # With tie_breaker=step_count (higher is better), best should be ep_long
        assert selected[0].episode_id == "ep_long"
        assert "tie_breaker" in selected[0].reason

    def test_tie_breaker_worst_selects_lowest_secondary(self) -> None:
        """With a higher-is-better primary, worst (first in ascending sort) gets the lowest tie-breaker value."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 1.0),
            _make_episode("ep2", "orca", "s1", 2, 1.0),
            _make_episode("ep3", "orca", "s1", 3, 1.0),
        ]
        # Add step counts
        episodes[0]["metrics"]["step_count"] = 100
        episodes[1]["metrics"]["step_count"] = 50
        episodes[2]["metrics"]["step_count"] = 5

        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["worst"],
            tie_breaker="step_count",
        )
        # worst (higher direction) = first in ascending sort
        # With tie_breaker=step_count ascending, worst should be ep3 (lowest step_count)
        assert selected[0].episode_id == "ep3"


class TestMinStepCount:
    """Test minimum step count filtering."""

    def test_filters_out_single_step_episodes(self) -> None:
        """Episodes with step_count < min_step_count are excluded."""
        episodes = [
            _make_episode("ep_single", "orca", "s1", 1, 0.9),
            _make_episode("ep_short", "orca", "s1", 2, 0.5),
            _make_episode("ep_long", "orca", "s1", 3, 0.1),
        ]
        # Add step counts
        episodes[0]["metrics"]["step_count"] = 1
        episodes[1]["metrics"]["step_count"] = 10
        episodes[2]["metrics"]["step_count"] = 100

        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best", "worst"],
            min_step_count=2,
        )
        # ep_single should be filtered out
        selected_ids = {s.episode_id for s in selected}
        assert "ep_single" not in selected_ids
        assert len(selected) == 2

    def test_skips_cell_when_all_below_min_step(self) -> None:
        """Cell is skipped when all episodes have step_count < min_step_count."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.9),
            _make_episode("ep2", "orca", "s1", 2, 0.5),
        ]
        episodes[0]["metrics"]["step_count"] = 1
        episodes[1]["metrics"]["step_count"] = 1

        selected, skipped = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best"],
            min_step_count=2,
        )
        assert len(selected) == 0
        assert len(skipped) == 1
        assert "step_count" in skipped[0].reason.lower()

    def test_min_step_count_with_tie_breaker(self) -> None:
        """min_step_count and tie_breaker work together."""
        episodes = [
            _make_episode("ep_single", "orca", "s1", 1, 1.0),
            _make_episode("ep_short", "orca", "s1", 2, 1.0),
            _make_episode("ep_long", "orca", "s1", 3, 1.0),
        ]
        episodes[0]["metrics"]["step_count"] = 1  # Will be filtered
        episodes[1]["metrics"]["step_count"] = 10
        episodes[2]["metrics"]["step_count"] = 100

        selected, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["best"],
            tie_breaker="step_count",
            min_step_count=2,
        )
        # ep_single filtered out; ep_long has highest step_count
        assert selected[0].episode_id == "ep_long"


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


class TestDirectionInvariantMedian:
    """Regression test for issue #4794: median selection must be invariant to metric_direction."""

    def test_same_median_under_both_directions(self) -> None:
        """Median episode is the same regardless of metric_direction flip."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.1),
            _make_episode("ep2", "orca", "s1", 2, 0.5),
            _make_episode("ep3", "orca", "s1", 3, 0.8),
            _make_episode("ep4", "orca", "s1", 4, 0.9),
        ]
        selected_higher, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["median"],
        )
        selected_lower, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="lower",
            modes=["median"],
        )
        assert len(selected_higher) == 1
        assert len(selected_lower) == 1
        assert selected_higher[0].episode_id == selected_lower[0].episode_id
        assert selected_higher[0].metric_value == selected_lower[0].metric_value

    def test_odd_size_same_median_both_directions(self) -> None:
        """Median of odd-sized group is also direction-invariant."""
        episodes = [
            _make_episode("ep1", "orca", "s1", 1, 0.2),
            _make_episode("ep2", "orca", "s1", 2, 0.6),
            _make_episode("ep3", "orca", "s1", 3, 0.8),
            _make_episode("ep4", "orca", "s1", 4, 0.9),
            _make_episode("ep5", "orca", "s1", 5, 1.0),
        ]
        selected_higher, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="higher",
            modes=["median"],
        )
        selected_lower, _ = select_exemplars(
            episodes,
            group_by=["planner_key"],
            metric="path_efficiency",
            metric_direction="lower",
            modes=["median"],
        )
        assert selected_higher[0].episode_id == selected_lower[0].episode_id
