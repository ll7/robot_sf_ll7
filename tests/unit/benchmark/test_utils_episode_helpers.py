"""Test episode helper utilities in robot_sf.benchmark.utils."""

import json
import os
from unittest.mock import patch

from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    compute_episode_id,
    compute_fast_mode_and_cap,
    determine_episode_outcome,
    episode_identity_hash,
    format_episode_summary_table,
    format_overlay_text,
    index_existing,
)


class TestDetermineEpisodeOutcome:
    """Test determine_episode_outcome function."""

    def test_collision_outcome(self):
        """Test collision outcome detection."""
        info = {"collision": True, "success": False}
        assert determine_episode_outcome(info) == "collision"

    def test_success_outcome(self):
        """Test success outcome detection."""
        info = {"collision": False, "success": True}
        assert determine_episode_outcome(info) == "success"

    def test_timeout_outcome(self):
        """Test timeout outcome detection."""
        info = {"collision": False, "success": False, "timeout": True}
        assert determine_episode_outcome(info) == "timeout"

    def test_done_outcome(self):
        """Test done outcome as fallback."""
        info = {"collision": False, "success": False, "timeout": False}
        assert determine_episode_outcome(info) == "done"

    def test_empty_info(self):
        """Test empty info dict returns done."""
        assert determine_episode_outcome({}) == "done"

    def test_collision_priority(self):
        """Test collision takes priority over other outcomes."""
        info = {"collision": True, "success": True, "timeout": True}
        assert determine_episode_outcome(info) == "collision"


class TestFormatOverlayText:
    """Test format_overlay_text function."""

    def test_basic_overlay(self):
        """Test basic overlay text formatting."""
        result = format_overlay_text("test_scenario", 42, 100)
        assert result == "test_scenario | seed=42 | step=100"

    def test_overlay_with_outcome(self):
        """Test overlay text with outcome."""
        result = format_overlay_text("test_scenario", 42, 100, "success")
        assert result == "test_scenario | seed=42 | step=100 | success"

    def test_overlay_with_none_outcome(self):
        """Test overlay text with None outcome."""
        result = format_overlay_text("test_scenario", 42, 100, None)
        assert result == "test_scenario | seed=42 | step=100"

    def test_overlay_with_empty_outcome(self):
        """Test overlay text with empty outcome."""
        result = format_overlay_text("test_scenario", 42, 100, "")
        assert result == "test_scenario | seed=42 | step=100"

    def test_overlay_special_characters(self):
        """Test overlay with special characters in scenario name."""
        result = format_overlay_text("test-scenario_v2", 0, 999)
        assert result == "test-scenario_v2 | seed=0 | step=999"


class TestComputeFastModeAndCap:
    """Test compute_fast_mode_and_cap function."""

    def test_normal_mode(self):
        """Test normal mode without fast flags."""
        with patch.dict(os.environ, {}, clear=True):
            fast_mode, max_episodes = compute_fast_mode_and_cap(5)
            assert fast_mode is False
            assert max_episodes == 5

    def test_pytest_detection(self):
        """Test pytest detection enables fast mode."""
        env = {"PYTEST_CURRENT_TEST": "test_something.py::test_func"}
        with patch.dict(os.environ, env, clear=True):
            fast_mode, max_episodes = compute_fast_mode_and_cap(5)
            assert fast_mode is True
            assert max_episodes == 1

    def test_fast_demo_env_var(self):
        """Test ROBOT_SF_FAST_DEMO environment variable."""
        env = {"ROBOT_SF_FAST_DEMO": "1"}
        with patch.dict(os.environ, env, clear=True):
            fast_mode, max_episodes = compute_fast_mode_and_cap(5)
            assert fast_mode is True
            assert max_episodes == 1

    def test_fast_demo_zero_disabled(self):
        """Test ROBOT_SF_FAST_DEMO=0 keeps normal mode."""
        env = {"ROBOT_SF_FAST_DEMO": "0"}
        with patch.dict(os.environ, env, clear=True):
            fast_mode, max_episodes = compute_fast_mode_and_cap(5)
            assert fast_mode is False
            assert max_episodes == 5

    def test_fast_mode_single_episode_unchanged(self):
        """Test fast mode doesn't change single episode request."""
        env = {"ROBOT_SF_FAST_DEMO": "1"}
        with patch.dict(os.environ, env, clear=True):
            fast_mode, max_episodes = compute_fast_mode_and_cap(1)
            assert fast_mode is True
            assert max_episodes == 1

    def test_fast_demo_empty_string(self):
        """Test empty ROBOT_SF_FAST_DEMO treated as 0."""
        env = {"ROBOT_SF_FAST_DEMO": ""}
        with patch.dict(os.environ, env, clear=True):
            fast_mode, max_episodes = compute_fast_mode_and_cap(5)
            assert fast_mode is False
            assert max_episodes == 5


class TestFormatEpisodeSummaryTable:
    """Test format_episode_summary_table function."""

    def test_empty_rows(self):
        """Test empty rows returns placeholder."""
        result = format_episode_summary_table([])
        assert result == "(no episodes)"

    def test_single_episode(self):
        """Test single episode formatting."""
        rows = [
            {
                "scenario": "test",
                "seed": 42,
                "steps": 100,
                "outcome": "success",
                "recorded": True,
            }
        ]
        result = format_episode_summary_table(rows)
        assert "test" in result
        assert "42" in result
        assert "100" in result
        assert "success" in result
        assert "True" in result

    def test_multiple_episodes(self):
        """Test multiple episodes formatting."""
        rows = [
            {
                "scenario": "test1",
                "seed": 42,
                "steps": 100,
                "outcome": "success",
                "recorded": True,
            },
            {
                "scenario": "test2",
                "seed": 43,
                "steps": 200,
                "outcome": "collision",
                "recorded": False,
            },
        ]
        result = format_episode_summary_table(rows)
        assert "test1" in result
        assert "test2" in result
        assert "success" in result
        assert "collision" in result

    def test_column_alignment(self):
        """Test column alignment works correctly."""
        rows = [
            {
                "scenario": "short",
                "seed": 1,
                "steps": 10,
                "outcome": "done",
                "recorded": True,
            },
            {
                "scenario": "very_long_scenario_name",
                "seed": 12345,
                "steps": 9999,
                "outcome": "collision",
                "recorded": False,
            },
        ]
        result = format_episode_summary_table(rows)
        lines = result.strip().split("\n")
        # Check that all data lines have consistent column separation
        assert len(lines) >= 4  # header, separator, 2 data lines
        # Verify header and separator alignment
        header_line = lines[0]
        separator_line = lines[1]
        assert "|" in header_line
        assert "---" in separator_line

    def test_table_headers(self):
        """Test table includes expected headers."""
        rows = [
            {
                "scenario": "test",
                "seed": 1,
                "steps": 10,
                "outcome": "done",
                "recorded": True,
            }
        ]
        result = format_episode_summary_table(rows)
        assert "scenario" in result
        assert "seed" in result
        assert "steps" in result
        assert "outcome" in result
        assert "recorded" in result

    def test_iterator_input(self):
        """Test function works with iterators."""

        def episode_generator():
            """TODO docstring. Document this function."""
            yield {
                "scenario": "test",
                "seed": 1,
                "steps": 10,
                "outcome": "done",
                "recorded": True,
            }

        result = format_episode_summary_table(episode_generator())
        assert "test" in result
        assert "done" in result


class TestBenchmarkUtilsMetadata:
    """Test benchmark metadata helpers in utils."""

    def test_config_hash_is_deterministic(self):
        """Verify config hashing is order-independent to keep resumes stable."""
        first = _config_hash({"b": 2, "a": 1})
        second = _config_hash({"a": 1, "b": 2})
        assert first == second
        assert len(first) == 16

    def test_compute_episode_id_prioritizes_id(self):
        """Verify episode id uses scenario identifiers so resume keys stay readable."""
        scenario_params = {"id": "case_a", "name": "ignored", "scenario_id": "ignored"}
        assert compute_episode_id(scenario_params, 7) == "case_a--7"

    def test_compute_episode_id_fallbacks(self):
        """Verify episode id falls back consistently when identifiers are missing."""
        assert compute_episode_id({"name": "named"}, 3) == "named--3"
        assert compute_episode_id({"scenario_id": "legacy"}, 5) == "legacy--5"
        assert compute_episode_id({}, 9) == "unknown--9"

    def test_episode_identity_hash_shape(self):
        """Verify identity hash has stable length for manifest invalidation."""
        identity = episode_identity_hash()
        assert len(identity) == 12

    def test_index_existing_reads_episode_ids(self, tmp_path):
        """Verify resume scanning extracts episode ids and skips malformed lines."""
        out_path = tmp_path / "episodes.jsonl"
        lines = [
            {"episode_id": "a--1"},
            {"episode_id": "b--2"},
        ]
        out_path.write_text(
            "\n".join(
                [
                    json.dumps(lines[0]),
                    "not-json",
                    json.dumps({"other": 1}),
                    json.dumps(lines[1]),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        assert index_existing(out_path) == {"a--1", "b--2"}

    def test_git_hash_fallback_handles_errors(self):
        """Verify git hash fallback returns 'unknown' when git is unavailable."""
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            assert _git_hash_fallback() == "unknown"
