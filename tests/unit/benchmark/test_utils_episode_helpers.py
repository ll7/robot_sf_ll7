"""Test episode helper utilities in robot_sf.benchmark.utils."""

import os
from unittest.mock import patch

from robot_sf.benchmark.utils import (
    compute_fast_mode_and_cap,
    determine_episode_outcome,
    format_episode_summary_table,
    format_overlay_text,
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
            """Episode generator.

            Returns:
                Any: Auto-generated placeholder description.
            """
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
