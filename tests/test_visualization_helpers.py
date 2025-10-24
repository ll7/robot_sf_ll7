"""
Unit tests for visualization data processing helpers.

Tests the internal helper functions that process episode data,
filter episodes, and handle dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.benchmark.visualization import (
    VisualizationError,
    _check_dependencies,
    _check_matplotlib_available,
    _check_moviepy_available,
    _filter_episodes,
    _load_episodes,
)


class TestDependencyChecking:
    """Test dependency checking functions."""

    def test_check_matplotlib_available_success(self):
        """Test that matplotlib availability check passes when matplotlib is available."""
        # Should not raise an exception
        _check_matplotlib_available()

    def test_check_moviepy_available_success(self):
        """Test that moviepy availability check passes when moviepy is available."""
        # Should not raise an exception
        _check_moviepy_available()

    @patch.dict("sys.modules", {"matplotlib": None})
    def test_check_matplotlib_available_failure(self):
        """Test that matplotlib check fails when matplotlib is not available."""
        with pytest.raises(VisualizationError) as exc_info:
            _check_matplotlib_available()

        assert "matplotlib" in str(exc_info.value)
        assert exc_info.value.artifact_type == "dependency_check"

    @patch.dict("sys.modules", {"moviepy": None})
    def test_check_moviepy_available_failure(self):
        """Test that moviepy check fails when moviepy is not available."""
        with pytest.raises(VisualizationError) as exc_info:
            _check_moviepy_available()

        assert "moviepy" in str(exc_info.value)
        assert exc_info.value.artifact_type == "dependency_check"

    def test_check_dependencies_success(self):
        """Test that dependency check passes for available packages."""
        # Should not raise an exception
        _check_dependencies(["os", "json"])

    def test_check_dependencies_failure(self):
        """Test that dependency check fails for unavailable packages."""
        with pytest.raises(VisualizationError) as exc_info:
            _check_dependencies(["nonexistent_package_xyz"])

        assert "nonexistent_package_xyz" in str(exc_info.value)


class TestEpisodeLoading:
    """Test episode data loading functions."""

    def test_load_episodes_valid_file(self):
        """Test loading episodes from a valid JSONL file."""
        episodes = [
            {"episode_id": "ep1", "metrics": {"success": True}},
            {"episode_id": "ep2", "metrics": {"success": False}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
            temp_path = f.name

        try:
            loaded = _load_episodes(temp_path)
            assert len(loaded) == 2
            assert loaded[0]["episode_id"] == "ep1"
            assert loaded[1]["episode_id"] == "ep2"
        finally:
            Path(temp_path).unlink()

    def test_load_episodes_empty_file(self):
        """Test loading episodes from an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(VisualizationError) as exc_info:
                _load_episodes(temp_path)
            assert "No episode data found" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_load_episodes_invalid_json(self):
        """Test loading episodes with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("invalid json line\n")
            f.write('{"valid": "json"}\n')
            temp_path = f.name

        try:
            with pytest.raises(VisualizationError) as exc_info:
                _load_episodes(temp_path)
            assert "Failed to decode JSON" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_load_episodes_nonexistent_file(self):
        """Test loading episodes from a nonexistent file."""
        with pytest.raises(VisualizationError) as exc_info:
            _load_episodes("nonexistent_file.jsonl")
        assert "Failed to read episode file" in str(exc_info.value)


class TestEpisodeFiltering:
    """Test episode filtering functions."""

    def test_filter_episodes_no_filters(self):
        """Test filtering episodes with no filters applied."""
        episodes = [
            {
                "scenario_id": "scenario1",
                "scenario_params": {"algo": "socialforce"},
                "algo": "socialforce",
            },
            {
                "scenario_id": "scenario2",
                "scenario_params": {"algo": "random"},
                "algo": "random",
            },
        ]

        filtered = _filter_episodes(episodes, None, None)
        assert len(filtered) == 2
        assert filtered == episodes

    def test_filter_episodes_scenario_filter(self):
        """Test filtering episodes by scenario ID."""
        episodes = [
            {
                "scenario_id": "scenario1",
                "scenario_params": {"algo": "socialforce"},
                "algo": "socialforce",
            },
            {
                "scenario_id": "scenario2",
                "scenario_params": {"algo": "random"},
                "algo": "random",
            },
            {
                "scenario_id": "scenario1",
                "scenario_params": {"algo": "random"},
                "algo": "random",
            },
        ]

        filtered = _filter_episodes(episodes, "scenario1", None)
        assert len(filtered) == 2
        assert all(ep["scenario_id"] == "scenario1" for ep in filtered)

    def test_filter_episodes_baseline_filter(self):
        """Test filtering episodes by baseline algorithm."""
        episodes = [
            {
                "scenario_id": "scenario1",
                "scenario_params": {"algo": "socialforce"},
                "algo": "socialforce",
            },
            {
                "scenario_id": "scenario2",
                "scenario_params": {"algo": "random"},
                "algo": "random",
            },
            {
                "scenario_id": "scenario1",
                "scenario_params": {"algo": "random"},
                "algo": "random",
            },
        ]

        filtered = _filter_episodes(episodes, None, "socialforce")
        assert len(filtered) == 1
        assert filtered[0]["scenario_params"]["algo"] == "socialforce"

    def test_filter_episodes_both_filters(self):
        """Test filtering episodes by both scenario and baseline."""
        episodes = [
            {"scenario_id": "scenario1", "scenario_params": {"algo": "socialforce"}},
            {"scenario_id": "scenario2", "scenario_params": {"algo": "random"}},
            {"scenario_id": "scenario1", "scenario_params": {"algo": "random"}},
        ]

        filtered = _filter_episodes(episodes, "scenario1", "random")
        assert len(filtered) == 1
        assert filtered[0]["scenario_id"] == "scenario1"
        assert filtered[0]["scenario_params"]["algo"] == "random"

    def test_filter_episodes_no_matches(self):
        """Test filtering episodes when no episodes match the filters."""
        episodes = [
            {
                "scenario_id": "scenario1",
                "scenario_params": {"algo": "socialforce"},
                "algo": "socialforce",
            },
        ]

        with pytest.raises(VisualizationError) as exc_info:
            _filter_episodes(episodes, "nonexistent", None)
        assert "No episodes match the specified filters" in str(exc_info.value)

    def test_filter_episodes_missing_fields(self):
        """Test filtering episodes with missing scenario or baseline fields."""
        episodes = [
            {"scenario_id": "scenario1"},  # Missing scenario_params
            {"scenario_params": {"algo": "socialforce"}},  # Missing scenario_id
            {
                "scenario_id": "scenario2",
                "scenario_params": {"algo": "random"},
                "algo": "random",
            },
        ]

        # Should handle missing fields gracefully - no matches for scenario1 + socialforce
        with pytest.raises(VisualizationError) as exc_info:
            _filter_episodes(episodes, "scenario1", "socialforce")
        assert "No episodes match the specified filters" in str(exc_info.value)

        # Should find scenario2 with no baseline filter
        filtered = _filter_episodes(episodes, "scenario2", None)
        assert len(filtered) == 1
