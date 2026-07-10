"""Tests for the issue #4891 head-on corridor export script."""

from __future__ import annotations

# Import the module under test
import importlib.util
import json
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


class TestFailClosedInputs:
    """Regression coverage for issue #5003 evidence-input failure paths."""

    def test_missing_campaign_root_returns_nonzero(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """The CLI reports a missing campaign root and exits with a nonzero status."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export",
                "--campaign-root",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert _export_module.main() == 2
        assert "campaign root is not a directory" in capsys.readouterr().err

    def test_missing_planner_dir_fails_closed(self, tmp_path: Path) -> None:
        """A missing planner directory raises the typed exporter input error."""
        campaign_root = tmp_path / "runs"
        campaign_root.mkdir()
        with pytest.raises(_export_module.ExemplarExportInputError, match="planner directory"):
            _export_module._process_planner("goal", campaign_root, tmp_path / "output")

    def test_missing_episodes_jsonl_fails_closed(self, tmp_path: Path) -> None:
        """A planner directory without episodes.jsonl is rejected as a typed input error."""
        campaign_root = tmp_path / "runs"
        (campaign_root / "goal__differential_drive").mkdir(parents=True)
        with pytest.raises(_export_module.ExemplarExportInputError, match="not a file"):
            _export_module._process_planner("goal", campaign_root, tmp_path / "output")

    def test_directory_at_episodes_jsonl_path_fails_closed(self, tmp_path: Path) -> None:
        """A directory named episodes.jsonl cannot be accepted as an episode file."""
        campaign_root = tmp_path / "runs"
        (campaign_root / "goal__differential_drive" / "episodes.jsonl").mkdir(parents=True)
        with pytest.raises(_export_module.ExemplarExportInputError, match="not a file"):
            _export_module._process_planner("goal", campaign_root, tmp_path / "output")

    def test_empty_episodes_jsonl_fails_closed(self, tmp_path: Path) -> None:
        """An empty episode file cannot yield an empty-but-successful export."""
        episodes_path = tmp_path / "runs" / "goal__differential_drive" / "episodes.jsonl"
        episodes_path.parent.mkdir(parents=True)
        episodes_path.write_text("\n", encoding="utf-8")
        with pytest.raises(_export_module.ExemplarExportInputError, match="is empty"):
            _export_module._process_planner("goal", episodes_path.parents[1], tmp_path / "output")

    def test_all_nonfinite_selection_metrics_fail_closed(self) -> None:
        """NaN and infinity cannot silently produce a selected exemplar."""
        episodes = [
            self._episode("classic_head_on_corridor_low", 1, float("nan")),
            self._episode("classic_head_on_corridor_low", 2, float("inf")),
            self._episode("classic_head_on_corridor_low", 3, float("-inf")),
        ]
        with pytest.raises(
            _export_module.ExemplarExportInputError, match="no finite selection metric"
        ):
            _export_module.select_exemplars_for_planner(episodes, "goal")

    @staticmethod
    def _episode(scenario_id: str, seed: int, value: float) -> dict[str, Any]:
        """Build a minimally valid corridor episode for selection tests."""
        return {
            "scenario_id": scenario_id,
            "seed": seed,
            "episode_id": f"{scenario_id}_s{seed}",
            "status": "success",
            "metrics": {"path_efficiency": value},
        }


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

    def test_missing_generated_at_returns_none(self) -> None:
        # Per #4903 the marker date is provenance-pinned with no wall-clock
        # fallback: missing provenance must yield no date, not today's date.
        assert _export_module._extract_marker_date({}) is None

    def test_empty_generated_at_returns_none(self) -> None:
        metadata: dict[str, str] = {"generated_at_utc": ""}
        assert _export_module._extract_marker_date(metadata) is None


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


class TestPinGeneratedAt:
    """Test the pin_generated_at parameter of write_bundle."""

    def _make_minimal_record(self) -> dict[str, Any]:
        steps = [
            {
                "step": i,
                "time_s": float(i),
                "robot": {"position": [float(i), 0.0], "velocity": [1.0, 0.0], "heading": 0.0},
                "planner": {"selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0}},
                "pedestrians": [{"id": "p1", "position": [5.0, 0.0]}],
            }
            for i in range(3)
        ]
        return {
            "algorithm_metadata": {
                "simulation_step_trace": {"steps": steps},
                "algorithm": "goal",
            },
            "scenario_id": "classic_head_on_corridor_low",
            "seed": 20,
            "episode_id": "test_episode",
            "status": "collision",
        }

    def test_pin_generated_at_overrides_wall_clock(self, tmp_path: Path) -> None:
        """pin_generated_at replaces datetime.now in the written metadata."""
        pin = "2025-01-15T10:30:00+00:00"
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="goal",
            scenario_id="classic_head_on_corridor_low",
            seed=20,
            selection_mode="median",
            metric_value=1.0,
            episode_id="test_episode",
            status="collision",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
            pin_generated_at=pin,
        )
        meta = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["generated_at_utc"] == pin

    def test_pin_generated_at_byte_identical(self, tmp_path: Path) -> None:
        """pin_generated_at makes metadata.json and trace_series.json byte-identical."""
        pin = "2026-07-08T23:13:18.753884+00:00"
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="goal",
            scenario_id="classic_head_on_corridor_low",
            seed=20,
            selection_mode="median",
            metric_value=1.0,
            episode_id="test_episode",
            status="collision",
        )
        dirs = [tmp_path / "run1", tmp_path / "run2"]
        for d in dirs:
            d.mkdir()
            _export_module.write_bundle(
                episode_record=record,
                selection=sel,
                output_dir=d,
                pin_generated_at=pin,
            )
        for fname in ("metadata.json", "trace_series.json"):
            h1 = sha256_file(dirs[0] / fname)
            h2 = sha256_file(dirs[1] / fname)
            assert h1 == h2, f"{fname} not byte-identical across runs with pin_generated_at"

    def test_pin_none_uses_wall_clock(self, tmp_path: Path) -> None:
        """Without pin_generated_at, generated_at_utc is wall-clock (not None)."""
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="goal",
            scenario_id="classic_head_on_corridor_low",
            seed=20,
            selection_mode="median",
            metric_value=1.0,
            episode_id="test_episode",
            status="collision",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        meta = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        assert "2026" in meta["generated_at_utc"]


class TestSelectionReportDeterminism:
    """write_selection_report must be byte-stable when generated_at is pinned.

    Regression guard for the #4903 reproducibility gap: SELECTION_REPORT.md's
    ``Generated:`` line previously used wall-clock time, so two re-runs with the
    same ``--pin-generated-at`` diverged. The pinned ``generated_at`` must flow
    into the report so the whole export is byte-identical.
    """

    def _selections(self) -> list[Any]:
        return [
            _export_module.SelectedEpisode(
                planner="goal",
                scenario_id="classic_head_on_corridor_low",
                seed=24,
                selection_mode="best",
                metric_value=1.0,
                episode_id="test_episode",
                status="success",
            )
        ]

    def test_generated_at_pinned_in_report(self, tmp_path: Path) -> None:
        pin = "2026-07-09T12:00:00+00:00"
        _export_module.write_selection_report(
            self._selections(), tmp_path, marker_date="2026-07-09", generated_at=pin
        )
        report = (tmp_path / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert f"Generated: {pin}" in report

    def test_byte_identical_with_pinned_generated_at(self, tmp_path: Path) -> None:
        pin = "2026-07-09T12:00:00+00:00"
        dirs = [tmp_path / "r1", tmp_path / "r2"]
        for d in dirs:
            d.mkdir()
            _export_module.write_selection_report(
                self._selections(), d, marker_date="2026-07-09", generated_at=pin
            )
        assert sha256_file(dirs[0] / "SELECTION_REPORT.md") == sha256_file(
            dirs[1] / "SELECTION_REPORT.md"
        ), "SELECTION_REPORT.md not byte-identical across pinned re-runs"

    def test_wall_clock_fallback_when_unpinned(self, tmp_path: Path) -> None:
        """Without generated_at the report still renders (wall-clock fallback)."""
        _export_module.write_selection_report(self._selections(), tmp_path)
        report = (tmp_path / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert "Generated:" in report
