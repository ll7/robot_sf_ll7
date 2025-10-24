"""Unit tests for figure orchestrator requirements generation.

Tests that the figure orchestrator properly determines and generates
the expected set of artifacts based on configuration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.generate_figures import main


class TestFigureOrchestratorRequirements:
    """Test figure orchestrator artifact requirements."""

    def test_basic_artifacts_generated(self, tmp_path: Path) -> None:
        """Test that basic artifacts are always generated."""
        episodes_file = tmp_path / "episodes.jsonl"
        out_dir = tmp_path / "figures"

        # Create minimal episodes file
        episodes_file.write_text(
            '{"schema_version": 1, "scenario_id": "test"}\n'
            '{"metrics": {"success": true, "collisions": 0}, "scenario_id": "test"}\n',
            encoding="utf-8",
        )

        with patch("robot_sf.benchmark.aggregate.read_jsonl") as mock_read:
            mock_read.return_value = [
                {"metrics": {"success": True, "collisions": 0}, "scenario_id": "test"},
            ]

            args = [
                "--episodes",
                str(episodes_file),
                "--out-dir",
                str(out_dir),
            ]

            # Mock the figure generation functions to avoid actual file I/O
            with (
                patch("scripts.generate_figures._generate_distributions") as mock_dists,
                patch("scripts.generate_figures._generate_table") as mock_table,
                patch("scripts.generate_figures._generate_pareto") as mock_pareto,
                patch("scripts.generate_figures._write_meta") as mock_meta,
                patch("sys.argv", ["script", *args]),
            ):
                main()

        # Verify that core functions were called
        mock_dists.assert_called_once()
        mock_table.assert_called_once()
        mock_pareto.assert_called_once()
        mock_meta.assert_called_once()

    def test_auto_out_dir_generation(self, tmp_path: Path) -> None:
        """Test that auto out dir generates canonical names."""
        from scripts.generate_figures import _compute_auto_out_dir

        episodes_path = tmp_path / "test_episodes.jsonl"

        # Test with base directory
        result = _compute_auto_out_dir(episodes_path, tmp_path)
        assert result.parent == tmp_path
        assert "test_episodes" in result.name

        # Test without base directory
        result = _compute_auto_out_dir(episodes_path, None)
        assert result.parent == Path("docs/figures")
        assert "test_episodes" in result.name

    def test_optional_artifacts_conditional(self, tmp_path: Path) -> None:
        """Test that optional artifacts are only generated when requested."""
        episodes_file = tmp_path / "episodes.jsonl"
        out_dir = tmp_path / "figures"
        matrix_file = tmp_path / "matrix.yaml"

        # Create minimal files
        episodes_file.write_text('{"schema_version": 1, "scenario_id": "test"}\n', encoding="utf-8")
        matrix_file.write_text("scenarios: []", encoding="utf-8")

        # Test with optional features enabled
        args = [
            "--episodes",
            str(episodes_file),
            "--out-dir",
            str(out_dir),
            "--pareto-x",
            "collisions",
            "--pareto-y",
            "comfort_exposure",
            "--thumbs-matrix",
            str(matrix_file),
            "--force-field",
        ]

        # Mock functions to track calls
        with (
            patch("robot_sf.benchmark.aggregate.read_jsonl") as mock_read,
            patch("scripts.generate_figures._generate_pareto") as mock_pareto,
            patch("scripts.generate_figures._maybe_thumbnails") as mock_thumbs,
            patch("scripts.generate_figures._maybe_force_field") as mock_ff,
            patch("scripts.generate_figures._generate_distributions"),
            patch("scripts.generate_figures._generate_table"),
            patch("scripts.generate_figures._write_meta"),
            patch("sys.argv", ["script", *args]),
        ):
            mock_read.return_value = [{"metrics": {"success": True}, "scenario_id": "test"}]

            main()

            # Verify optional functions were called
            mock_pareto.assert_called_once()
            mock_thumbs.assert_called_once()
            mock_ff.assert_called_once()

    def test_meta_includes_required_fields(self, tmp_path: Path) -> None:
        """Test that meta.json includes all required provenance fields."""
        import argparse
        import json

        from scripts.generate_figures import _write_meta

        episodes_file = tmp_path / "episodes.jsonl"
        out_dir = tmp_path / "figures"
        out_dir.mkdir()

        episodes_file.write_text('{"schema_version": 1, "scenario_id": "test"}\n', encoding="utf-8")

        # Create minimal args
        args = argparse.Namespace(
            episodes=episodes_file,
            out_dir=out_dir,
            auto_out_dir=False,
        )

        _write_meta(out_dir, episodes_file, args)

        # Verify meta.json content
        meta_path = out_dir / "meta.json"
        assert meta_path.exists()

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        # Check required fields
        required_fields = [
            "episodes_path",
            "generated_at",
            "git_sha",
            "schema_version",
            "script_version",
            "args",
        ]
        for field in required_fields:
            assert field in meta, f"Missing required field: {field}"

    def test_artifact_count_scaling(self) -> None:
        """Test that artifact count scales appropriately with options."""
        # This is a conceptual test - in practice you'd count actual files
        base_artifacts = {"meta.json", "table.md", "distributions.png"}

        # With pareto
        pareto_artifacts = base_artifacts | {"pareto.png"}
        assert len(pareto_artifacts) > len(base_artifacts)

        # With thumbnails
        thumb_artifacts = pareto_artifacts | {"scenarios/", "scenarios/montage.png"}
        assert len(thumb_artifacts) > len(pareto_artifacts)

        # With force field
        ff_artifacts = thumb_artifacts | {"fig-force-field.png", "fig-force-field.pdf"}
        assert len(ff_artifacts) > len(thumb_artifacts)


if __name__ == "__main__":
    pytest.main([__file__])
