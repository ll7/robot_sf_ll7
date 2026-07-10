"""Tests for the issue #4848 group-crossing export script.

Covers the marker_date signature change, tuple-return _process_planner,
and full bundle fixture generation with evidence markers.  Code path is
identical to the tested 4891 export, but 4848 differs in scenario filter
(group_crossing) and selection strategy (no quality-aware step-count filter).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_sf.evidence.writers import extract_marker_date, sha256_file

_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "export_issue_4848_group_crossing_exemplars.py"
)
_spec = importlib.util.spec_from_file_location("export_issue_4848", _SCRIPT_PATH)
_export_module = importlib.util.module_from_spec(_spec)
sys.modules["export_issue_4848"] = _export_module
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
    """Test the select_exemplars_for_planner function for group-crossing."""

    def _make_episode(self, scenario_id: str, seed: int, path_eff: float) -> dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "seed": seed,
            "episode_id": f"{scenario_id}_s{seed}",
            "status": "success",
            "metrics": {"path_efficiency": path_eff},
        }

    def test_filters_group_crossing_only(self) -> None:
        episodes = [
            self._make_episode("classic_group_crossing_low", 1, 0.8),
            self._make_episode("classic_group_crossing_medium", 2, 0.9),
            self._make_episode("classic_group_crossing_high", 3, 0.95),
            self._make_episode("classic_doorway_low", 4, 0.7),  # excluded
        ]
        selected = _export_module.select_exemplars_for_planner(episodes, "goal")
        assert len(selected) == 3  # median, best, worst
        for sel in selected:
            assert "group_crossing" in sel.scenario_id

    def test_selects_median_best_worst(self) -> None:
        episodes = [
            self._make_episode("classic_group_crossing_low", 1, 0.5),
            self._make_episode("classic_group_crossing_low", 2, 0.7),
            self._make_episode("classic_group_crossing_low", 3, 0.9),
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

    def test_no_group_crossing(self) -> None:
        episodes = [self._make_episode("classic_doorway_low", 1, 0.8)]
        selected = _export_module.select_exemplars_for_planner(episodes, "goal")
        assert selected == []

    def test_no_quality_aware_filter(self) -> None:
        """4848 does NOT filter single-step episodes (unlike 4891).

        The 4848 script lacks the quality-aware step_count filter added in
        issue #4912, so all episodes with finite metrics participate regardless
        of step count.  This is a known difference documented in the issue.
        """
        episodes = [
            self._make_episode("classic_group_crossing_low", 1, 0.1),
            self._make_episode("classic_group_crossing_low", 2, 0.5),
            self._make_episode("classic_group_crossing_low", 3, 0.9),
        ]
        episodes[0]["metrics"]["step_count"] = 1  # single-step: NOT filtered
        selected = _export_module.select_exemplars_for_planner(episodes, "goal")
        assert len(selected) == 3
        selected_ids = {s.episode_id for s in selected}
        # Seed 1 (worst) should still be selectable since 4848 lacks the filter
        assert "classic_group_crossing_low_s1" in selected_ids


class TestDeriveTraceRows:
    """Test the derive_trace_rows function."""

    def _make_record(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "algorithm_metadata": {
                "simulation_step_trace": {"steps": steps},
                "algorithm": "goal",
            },
            "scenario_id": "classic_group_crossing_low",
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

    def test_multiple_steps_tracks_min_distance(self) -> None:
        steps = [
            {
                "step": i,
                "time_s": float(i),
                "robot": {"position": [float(i), 0.0], "velocity": [1.0, 0.0], "heading": 0.0},
                "planner": {"selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0}},
                "pedestrians": [{"id": "p1", "position": [3.0, 0.0]}],
            }
            for i in range(5)
        ]
        record = self._make_record(steps)
        result = _export_module.derive_trace_rows(record)
        assert len(result.trace_rows) == 5
        # Step 3 is closest to pedal position [3.0, 0.0] (distance 0)
        assert result.summary["global_min_robot_ped_distance_m"] == pytest.approx(0.0)
        assert result.summary["global_min_distance_step"] == 3


class TestWriteBundleFixture:
    """Test write_bundle produces contract-clean artifact files with markers.

    Creates a temporary fixture bundle and verifies all output files contain
    the expected AI-GENERATED / NEEDS-REVIEW markers per pr_contract_check rule 4.
    """

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
                "algorithm": "orca",
            },
            "scenario_id": "classic_group_crossing_low",
            "seed": 42,
            "episode_id": "classic_group_crossing_low_s42",
            "status": "success",
        }

    def test_bundle_writes_all_files(self, tmp_path: Path) -> None:
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="orca",
            scenario_id="classic_group_crossing_low",
            seed=42,
            selection_mode="best",
            metric_value=0.85,
            episode_id="classic_group_crossing_low_s42",
            status="success",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        expected_files = {
            "metadata.json",
            "trace_series.json",
            "trace_timeseries.csv",
            "min_distance_series.csv",
            "README.md",
            "SHA256SUMS",
        }
        actual = {f.name for f in bundle_dir.iterdir()}
        assert expected_files.issubset(actual)

    def test_json_has_review_marker(self, tmp_path: Path) -> None:
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="social_force",
            scenario_id="classic_group_crossing_medium",
            seed=10,
            selection_mode="median",
            metric_value=0.6,
            episode_id="classic_group_crossing_medium_s10",
            status="success",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        # metadata.json must contain review_marker
        meta_path = bundle_dir / "metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
        # trace_series.json must contain review_marker
        trace_path = bundle_dir / "trace_series.json"
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace_data["review_marker"] == "AI-GENERATED NEEDS-REVIEW"

    def test_csv_has_review_marker(self, tmp_path: Path) -> None:
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="orca",
            scenario_id="classic_group_crossing_low",
            seed=42,
            selection_mode="best",
            metric_value=0.85,
            episode_id="classic_group_crossing_low_s42",
            status="success",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        for csv_name in ("trace_timeseries.csv", "min_distance_series.csv"):
            content = (bundle_dir / csv_name).read_text(encoding="utf-8")
            assert content.startswith("# AI-GENERATED NEEDS-REVIEW\n")

    def test_readme_has_review_marker_with_date(self, tmp_path: Path) -> None:
        """README marker includes issue ref and provenance-pinned date (not wall-clock)."""
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="orca",
            scenario_id="classic_group_crossing_low",
            seed=42,
            selection_mode="best",
            metric_value=0.85,
            episode_id="classic_group_crossing_low_s42",
            status="success",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        metadata = _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        readme_path = bundle_dir / "README.md"
        readme = readme_path.read_text(encoding="utf-8")
        # Must start with the review marker comment
        assert readme.startswith("<!-- AI-GENERATED (robot_sf#4848")
        # Must include the provenance date
        marker_date = extract_marker_date(metadata)
        assert marker_date is not None
        assert f", {marker_date})" in readme
        # Must contain NEEDS-REVIEW
        assert "NEEDS-REVIEW" in readme
        # Must end with closing tag
        assert "<!-- /AI-GENERATED -->" in readme

    def test_sha256sums_has_marker(self, tmp_path: Path) -> None:
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="goal",
            scenario_id="classic_group_crossing_high",
            seed=99,
            selection_mode="worst",
            metric_value=0.3,
            episode_id="classic_group_crossing_high_s99",
            status="success",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        sha_path = bundle_dir / "SHA256SUMS"
        content = sha_path.read_text(encoding="utf-8")
        assert content.startswith("# AI-GENERATED NEEDS-REVIEW\n")
        # Does not self-reference
        lines_after_marker = content.split("\n", 1)[1]
        assert "SHA256SUMS" not in lines_after_marker

    def test_sha256sums_hashes_marked_files(self, tmp_path: Path) -> None:
        """SHA-256 values must match the marked file contents (not pre-marker originals)."""
        record = self._make_minimal_record()
        sel = _export_module.SelectedEpisode(
            planner="orca",
            scenario_id="classic_group_crossing_low",
            seed=42,
            selection_mode="best",
            metric_value=0.85,
            episode_id="classic_group_crossing_low_s42",
            status="success",
        )
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _export_module.write_bundle(
            episode_record=record,
            selection=sel,
            output_dir=bundle_dir,
        )
        sha_path = bundle_dir / "SHA256SUMS"
        sha_lines = sha_path.read_text(encoding="utf-8").strip().split("\n")
        # First line is the marker; rest are checksums
        for line in sha_lines[1:]:
            parts = line.split("  ", 1)
            assert len(parts) == 2
            digest, rel_path = parts
            # Resolve against bundle_dir since the SHA label may point outside tmp_path
            file_name = Path(rel_path).name
            file_path = bundle_dir / file_name
            assert file_path.exists(), f"SHA256SUMS references {rel_path} not in {bundle_dir}"
            assert sha256_file(file_path) == digest


class TestWriteSelectionReport:
    """Test write_selection_report marker_date integration."""

    def test_report_has_marker(self, tmp_path: Path) -> None:
        selections = [
            _export_module.SelectedEpisode(
                planner="goal",
                scenario_id="classic_group_crossing_low",
                seed=1,
                selection_mode="worst",
                metric_value=0.4,
                episode_id="test_s1",
                status="success",
            ),
        ]
        out = tmp_path / "report_dir"
        out.mkdir()
        _export_module.write_selection_report(selections, out, marker_date="2026-07-08")
        report = (out / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert "<!-- AI-GENERATED (robot_sf#4848, 2026-07-08) - NEEDS-REVIEW -->" in report

    def test_report_without_date_omits_date(self, tmp_path: Path) -> None:
        selections = [
            _export_module.SelectedEpisode(
                planner="orca",
                scenario_id="classic_group_crossing_medium",
                seed=2,
                selection_mode="best",
                metric_value=0.9,
                episode_id="test_s2",
                status="success",
            ),
        ]
        out = tmp_path / "report_dir"
        out.mkdir()
        _export_module.write_selection_report(selections, out, marker_date=None)
        report = (out / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert "<!-- AI-GENERATED (robot_sf#4848) - NEEDS-REVIEW -->" in report

    def test_report_header_is_marker(self, tmp_path: Path) -> None:
        selections = [
            _export_module.SelectedEpisode(
                planner="social_force",
                scenario_id="classic_group_crossing_high",
                seed=3,
                selection_mode="median",
                metric_value=0.7,
                episode_id="test_s3",
                status="success",
            ),
        ]
        out = tmp_path / "report_dir"
        out.mkdir()
        _export_module.write_selection_report(selections, out)
        report = (out / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert report.lstrip().startswith("<!-- AI-GENERATED")

    def test_report_has_closing_tag(self, tmp_path: Path) -> None:
        """SELECTION_REPORT.md ends with closing AI-GENERATED tag."""
        selections = [
            _export_module.SelectedEpisode(
                planner="goal",
                scenario_id="classic_group_crossing_low",
                seed=1,
                selection_mode="worst",
                metric_value=0.4,
                episode_id="test_s1",
                status="success",
            ),
        ]
        out = tmp_path / "report_dir"
        out.mkdir()
        _export_module.write_selection_report(selections, out, marker_date="2026-07-08")
        report = (out / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert "<!-- /AI-GENERATED -->" in report


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
                scenario_id="classic_group_crossing_low",
                seed=1,
                selection_mode="worst",
                metric_value=0.4,
                episode_id="test_s1",
                status="success",
            )
        ]

    def test_generated_at_pinned_in_report(self, tmp_path: Path) -> None:
        pin = "2026-07-09T12:00:00+00:00"
        out = tmp_path / "report_dir"
        out.mkdir()
        _export_module.write_selection_report(
            self._selections(), out, marker_date="2026-07-09", generated_at=pin
        )
        report = (out / "SELECTION_REPORT.md").read_text(encoding="utf-8")
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
        out = tmp_path / "report_dir"
        out.mkdir()
        _export_module.write_selection_report(self._selections(), out)
        report = (out / "SELECTION_REPORT.md").read_text(encoding="utf-8")
        assert "Generated:" in report


class TestProcessPlanner:
    """Verify the 4848 planner-processing input contract."""

    def test_missing_planner_dir_fails_closed(self, tmp_path: Path) -> None:
        """A missing planner directory raises the typed exporter input error."""
        campaign_root = tmp_path / "runs"
        campaign_root.mkdir()
        with pytest.raises(_export_module.ExemplarExportInputError, match="planner directory"):
            _export_module._process_planner("nonexistent", campaign_root, tmp_path / "output")


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
            self._episode("classic_group_crossing_low", 1, float("nan")),
            self._episode("classic_group_crossing_low", 2, float("inf")),
            self._episode("classic_group_crossing_low", 3, float("-inf")),
        ]
        with pytest.raises(
            _export_module.ExemplarExportInputError, match="no finite selection metric"
        ):
            _export_module.select_exemplars_for_planner(episodes, "goal")

    @staticmethod
    def _episode(scenario_id: str, seed: int, value: float) -> dict[str, Any]:
        """Build a minimally valid group-crossing episode for selection tests."""
        return {
            "scenario_id": scenario_id,
            "seed": seed,
            "episode_id": f"{scenario_id}_s{seed}",
            "status": "success",
            "metrics": {"path_efficiency": value},
        }


class TestBundleReproducibleByteIdentical:
    """Re-running the same export with the same inputs produces byte-identical outputs.

    Excludes generated_at_utc and git_commit (which change per run) by writing
    a fixed generated_at_utc into the episode record's metadata.
    """

    def _make_minimal_record(self, timestamp: str) -> dict[str, Any]:
        steps = [
            {
                "step": i,
                "time_s": float(i),
                "robot": {"position": [float(i), 0.0], "velocity": [1.0, 0.0], "heading": 0.0},
                "planner": {"selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0}},
                "pedestrians": [{"id": "p1", "position": [5.0, 0.0]}],
            }
            for i in range(2)
        ]
        return {
            "algorithm_metadata": {
                "simulation_step_trace": {"steps": steps},
                "algorithm": "goal",
            },
            "scenario_id": "classic_group_crossing_low",
            "seed": 1,
            "episode_id": "classic_group_crossing_low_s1",
            "status": "success",
        }

    def test_bundle_files_reproducible(self, tmp_path: Path) -> None:
        """Running write_bundle twice with same episode yields SHA-identical CSV files."""
        # Use a fixed git commit by monkey-patching (CSV output doesn't depend on it)
        record = self._make_minimal_record("2026-07-08T12:00:00+00:00")
        sel = _export_module.SelectedEpisode(
            planner="goal",
            scenario_id="classic_group_crossing_low",
            seed=1,
            selection_mode="best",
            metric_value=0.8,
            episode_id="classic_group_crossing_low_s1",
            status="success",
        )
        dirs = [tmp_path / "run1", tmp_path / "run2"]
        for d in dirs:
            d.mkdir()
            _export_module.write_bundle(
                episode_record=record,
                selection=sel,
                output_dir=d,
            )
        # CSV files are fully deterministic (no timestamps)
        for csv_name in ("trace_timeseries.csv", "min_distance_series.csv"):
            h1 = sha256_file(dirs[0] / csv_name)
            h2 = sha256_file(dirs[1] / csv_name)
            assert h1 == h2, f"{csv_name} not byte-identical across runs"

    def test_pin_generated_at_byte_identical_metadata(self, tmp_path: Path) -> None:
        """pin_generated_at makes metadata.json byte-identical across runs."""
        pin = "2026-07-08T23:13:18.753884+00:00"
        record = self._make_minimal_record(pin)
        sel = _export_module.SelectedEpisode(
            planner="goal",
            scenario_id="classic_group_crossing_low",
            seed=1,
            selection_mode="best",
            metric_value=0.8,
            episode_id="classic_group_crossing_low_s1",
            status="success",
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

    def test_pin_generated_at_overrides_wall_clock(self, tmp_path: Path) -> None:
        """pin_generated_at replaces datetime.now in the written metadata."""
        pin = "2025-01-15T10:30:00+00:00"
        record = self._make_minimal_record(pin)
        sel = _export_module.SelectedEpisode(
            planner="orca",
            scenario_id="classic_group_crossing_low",
            seed=1,
            selection_mode="median",
            metric_value=0.5,
            episode_id="classic_group_crossing_low_s1",
            status="success",
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


class TestExtractMarkerDateShared:
    """Test that the 4848 export uses the shared extract_marker_date (provenance-pinned)."""

    def test_marker_date_from_bundle_metadata(self) -> None:
        """extract_marker_date derives from generated_at_utc, not wall clock."""
        metadata = {
            "generated_at_utc": "2026-07-08T12:34:56+00:00",
        }
        date = extract_marker_date(metadata)
        assert date == "2026-07-08"

    def test_no_wall_clock_fallback(self) -> None:
        """Missing provenance must yield None, not today's date."""
        assert extract_marker_date({}) is None
        assert extract_marker_date({"generated_at_utc": ""}) is None


class TestConstants:
    """Test module constants are set correctly."""

    def test_scenario_set(self) -> None:
        assert "classic_group_crossing_low" in _export_module.GROUP_CROSSING_SCENARIOS
        assert "classic_group_crossing_medium" in _export_module.GROUP_CROSSING_SCENARIOS
        assert "classic_group_crossing_high" in _export_module.GROUP_CROSSING_SCENARIOS
        assert len(_export_module.GROUP_CROSSING_SCENARIOS) == 3

    def test_target_planners(self) -> None:
        assert _export_module.TARGET_PLANNERS == ["goal", "orca", "social_force"]

    def test_selection_modes(self) -> None:
        assert _export_module.SELECTION_MODES == ["median", "best", "worst"]

    def test_selection_metric(self) -> None:
        assert _export_module.SELECTION_METRIC == "path_efficiency"
