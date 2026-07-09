"""Tests for the issue #4891 head-on corridor export script."""

from __future__ import annotations

# Import the module under test
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_sf.evidence.writers import sha256_file

# Load the export script as a module
_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "export_issue_4891_head_on_corridor_exemplars.py"
)
_spec = importlib.util.spec_from_file_location("export_issue_4891", _SCRIPT_PATH)
_export_module = importlib.util.module_from_spec(_spec)
sys.modules["export_issue_4891"] = _export_module
_spec.loader.exec_module(_export_module)


class TestMinDistance:
    """Test the _min_distance helper."""

    def test_empty_pedestrians(self) -> None:
        result, pid = _export_module._min_distance([0.0, 0.0], [])
        assert result is None
        assert pid is None

    def test_single_pedestrian(self) -> None:
        peds = [{"id": "1", "position": [3.0, 4.0]}]
        result, pid = _export_module._min_distance([0.0, 0.0], peds)
        assert result == pytest.approx(5.0)
        assert pid == "1"

    def test_nearest_selected(self) -> None:
        peds = [
            {"id": "far", "position": [10.0, 10.0]},
            {"id": "near", "position": [1.0, 0.0]},
        ]
        result, pid = _export_module._min_distance([0.0, 0.0], peds)
        assert result == pytest.approx(1.0)
        assert pid == "near"

    def test_invalid_position_skipped(self) -> None:
        peds = [
            {"id": "bad", "position": []},
            {"id": "good", "position": [2.0, 0.0]},
        ]
        result, pid = _export_module._min_distance([0.0, 0.0], peds)
        assert result == pytest.approx(2.0)
        assert pid == "good"


class TestSelectExemplars:
    """Test the select_exemplars_for_planner function."""

    def _make_episode(self, scenario_id: str, seed: int, path_eff: float) -> dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "seed": seed,
            "episode_id": f"{scenario_id}_s{seed}",
            "status": "success",
            "metrics": {"path_efficiency": path_eff},
        }

    def test_filters_head_on_corridor_only(self) -> None:
        episodes = [
            self._make_episode("classic_head_on_corridor_low", 1, 0.8),
            self._make_episode("classic_head_on_corridor_medium", 2, 0.9),
            self._make_episode("classic_doorway_low", 3, 0.7),  # should be excluded
        ]
        selected = _export_module.select_exemplars_for_planner(episodes, "goal")
        assert len(selected) == 3  # median, best, worst from 2 episodes
        for sel in selected:
            assert "head_on_corridor" in sel.scenario_id

    def test_selects_median_best_worst(self) -> None:
        episodes = [
            self._make_episode("classic_head_on_corridor_low", 1, 0.5),
            self._make_episode("classic_head_on_corridor_low", 2, 0.7),
            self._make_episode("classic_head_on_corridor_low", 3, 0.9),
        ]
        selected = _export_module.select_exemplars_for_planner(episodes, "orca")
        modes = {sel.selection_mode for sel in selected}
        assert modes == {"median", "best", "worst"}

        best = next(s for s in selected if s.selection_mode == "best")
        worst = next(s for s in selected if s.selection_mode == "worst")
        median = next(s for s in selected if s.selection_mode == "median")
        assert best.metric_value == 0.9
        assert worst.metric_value == 0.5
        assert median.metric_value == 0.7

    def test_empty_episodes(self) -> None:
        selected = _export_module.select_exemplars_for_planner([], "goal")
        assert selected == []

    def test_no_head_on_corridor(self) -> None:
        episodes = [self._make_episode("classic_doorway_low", 1, 0.8)]
        selected = _export_module.select_exemplars_for_planner(episodes, "goal")
        assert selected == []

    def test_tie_break_prefers_richer_trace_for_best(self) -> None:
        """When path_efficiency ties, best gets the MOST steps, worst the FEWEST.

        Regression guard for issue #4912: a descending step-count secondary sort
        inverts best/worst and would starve the best exemplar of trace content.
        """
        episodes = [
            self._make_episode("classic_head_on_corridor_low", 1, 1.0),
            self._make_episode("classic_head_on_corridor_low", 2, 1.0),
            self._make_episode("classic_head_on_corridor_low", 3, 1.0),
        ]
        episodes[0]["metrics"]["step_count"] = 10
        episodes[1]["metrics"]["step_count"] = 50
        episodes[2]["metrics"]["step_count"] = 100

        selected = _export_module.select_exemplars_for_planner(episodes, "orca")
        best = next(s for s in selected if s.selection_mode == "best")
        worst = next(s for s in selected if s.selection_mode == "worst")
        # episode_id is f"{scenario_id}_s{seed}"; seed 3 == 100 steps (richest)
        assert best.episode_id.endswith("_s3")
        assert worst.episode_id.endswith("_s1")

    def test_filters_single_step_episodes(self) -> None:
        """Single-step episodes (step_count < MIN_STEP_COUNT) are excluded."""
        episodes = [
            self._make_episode("classic_head_on_corridor_low", 1, 0.1),
            self._make_episode("classic_head_on_corridor_low", 2, 0.5),
            self._make_episode("classic_head_on_corridor_low", 3, 0.7),
            self._make_episode("classic_head_on_corridor_low", 4, 0.9),
        ]
        episodes[0]["metrics"]["step_count"] = 1  # single-step: filtered out
        episodes[1]["metrics"]["step_count"] = 20
        episodes[2]["metrics"]["step_count"] = 30
        episodes[3]["metrics"]["step_count"] = 40

        selected = _export_module.select_exemplars_for_planner(episodes, "orca")
        selected_ids = {s.episode_id for s in selected}
        assert "classic_head_on_corridor_low_s1" not in selected_ids
        assert len(selected) == 3


class TestDeriveTraceRows:
    """Test the derive_trace_rows function."""

    def _make_record(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "algorithm_metadata": {
                "simulation_step_trace": {"steps": steps},
                "algorithm": "goal",
            },
            "scenario_id": "classic_head_on_corridor_low",
            "seed": 1,
            "status": "success",
        }

    def test_valid_trace(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0], "velocity": [1.0, 0.0], "heading": 0.0},
                "planner": {"selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0}},
                "pedestrians": [{"id": "p1", "position": [5.0, 0.0]}],
            },
        ]
        record = self._make_record(steps)
        result = _export_module.derive_trace_rows(record)
        assert len(result.trace_rows) == 1
        assert len(result.min_distance_rows) == 1
        assert result.summary["step_count"] == 1
        assert result.summary["global_min_robot_ped_distance_m"] == pytest.approx(5.0)

    def test_missing_trace_raises(self) -> None:
        record = {"algorithm_metadata": {}}
        with pytest.raises(ValueError, match="lacks simulation_step_trace"):
            _export_module.derive_trace_rows(record)


class TestSha256File:
    """Test the sha256_file helper."""

    def test_known_content(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\n", encoding="utf-8")
        digest = sha256_file(test_file)
        assert len(digest) == 64  # SHA-256 hex length
        # Known SHA-256 of the literal bytes "hello\n"; pins the helper's correctness
        # rather than merely asserting the digest length.
        assert digest == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


class TestExtractMarkerDate:
    """Test the _extract_marker_date helper."""

    def test_extracts_date_from_iso_timestamp(self) -> None:
        metadata = {"generated_at_utc": "2026-07-08T23:13:18.867110+00:00"}
        assert _export_module._extract_marker_date(metadata) == "2026-07-08"

    def test_empty_generated_at_uses_today(self) -> None:
        metadata: dict[str, str] = {}
        result = _export_module._extract_marker_date(metadata)
        # Should be a valid YYYY-MM-DD string (today's date)
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"


class TestConstants:
    """Test module constants are set correctly."""

    def test_scenario_set(self) -> None:
        assert "classic_head_on_corridor_low" in _export_module.HEAD_ON_CORRIDOR_SCENARIOS
        assert "classic_head_on_corridor_medium" in _export_module.HEAD_ON_CORRIDOR_SCENARIOS
        assert len(_export_module.HEAD_ON_CORRIDOR_SCENARIOS) == 2

    def test_target_planners(self) -> None:
        assert _export_module.TARGET_PLANNERS == ["goal", "orca", "social_force"]

    def test_selection_modes(self) -> None:
        assert _export_module.SELECTION_MODES == ["median", "best", "worst"]
