"""Characterization wave 3 tests for scripts/replay_episode_figure.py.

Focuses on the argument contract (parse_args validation, defaults, choices).
NEW tests only, zero production changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.replay_episode_figure import main, parse_args

# ---------------------------------------------------------------------------
# parse_args validation
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for parse_args argument validation and defaults."""

    def test_required_args(self) -> None:
        """All required arguments must be provided."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_missing_episodes(self) -> None:
        """--episodes is required."""
        with pytest.raises(SystemExit):
            parse_args(["--episode-id", "ep1", "--outputs", "trajectory", "--out-dir", "/tmp"])

    def test_missing_episode_id(self) -> None:
        """--episode-id is required."""
        with pytest.raises(SystemExit):
            parse_args(["--episodes", "ep.jsonl", "--outputs", "trajectory", "--out-dir", "/tmp"])

    def test_missing_outputs(self) -> None:
        """--outputs is required."""
        with pytest.raises(SystemExit):
            parse_args(["--episodes", "ep.jsonl", "--episode-id", "ep1", "--out-dir", "/tmp"])

    def test_missing_out_dir(self) -> None:
        """--out-dir is required."""
        with pytest.raises(SystemExit):
            parse_args(["--episodes", "ep.jsonl", "--episode-id", "ep1", "--outputs", "trajectory"])

    def test_all_required_args(self) -> None:
        """All required args parsed correctly."""
        args = parse_args(
            [
                "--episodes",
                "episodes.jsonl",
                "--episode-id",
                "ep_001",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp/output",
            ]
        )
        assert args.episodes == Path("episodes.jsonl")
        assert args.episode_id == "ep_001"
        assert args.outputs == "trajectory"
        assert args.out_dir == Path("/tmp/output")

    def test_defaults(self) -> None:
        """Default values are correct."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
            ]
        )
        assert args.campaign_root is None
        assert args.scenario_matrix is None
        assert args.config_hash is None
        assert args.tolerance_m == 0.1
        assert args.frame_steps is None
        assert args.format == "png"
        assert args.no_determinism_check is False
        assert args.verbose is False

    def test_format_choices(self) -> None:
        """--format accepts only png, pdf, svg."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--format",
                "pdf",
            ]
        )
        assert args.format == "pdf"

        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--episodes",
                    "ep.jsonl",
                    "--episode-id",
                    "ep1",
                    "--outputs",
                    "trajectory",
                    "--out-dir",
                    "/tmp",
                    "--format",
                    "gif",
                ]
            )

    def test_tolerance_m_option(self) -> None:
        """--tolerance-m is parsed correctly."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--tolerance-m",
                "0.05",
            ]
        )
        assert args.tolerance_m == 0.05

    def test_frame_steps_option(self) -> None:
        """--frame-steps is parsed as string."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "filmstrip",
                "--out-dir",
                "/tmp",
                "--frame-steps",
                "0,10,20,30",
            ]
        )
        assert args.frame_steps == "0,10,20,30"

    def test_no_determinism_check_flag(self) -> None:
        """--no-determinism-check flag is parsed correctly."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--no-determinism-check",
            ]
        )
        assert args.no_determinism_check is True

    def test_verbose_flag(self) -> None:
        """--verbose flag is parsed correctly."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--verbose",
            ]
        )
        assert args.verbose is True

    def test_campaign_root_option(self) -> None:
        """--campaign-root is parsed as Path."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--campaign-root",
                "/campaign/root",
            ]
        )
        assert args.campaign_root == Path("/campaign/root")

    def test_scenario_matrix_option(self) -> None:
        """--scenario-matrix is parsed as Path."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--scenario-matrix",
                "/path/to/matrix.yaml",
            ]
        )
        assert args.scenario_matrix == Path("/path/to/matrix.yaml")

    def test_config_hash_option(self) -> None:
        """--config is parsed as string (config hash)."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                "/tmp",
                "--config",
                "abc123def",
            ]
        )
        assert args.config_hash == "abc123def"

    def test_all_options(self) -> None:
        """All options can be set simultaneously."""
        args = parse_args(
            [
                "--episodes",
                "ep.jsonl",
                "--episode-id",
                "ep_001",
                "--outputs",
                "still,filmstrip,trajectory",
                "--out-dir",
                "/tmp/output",
                "--campaign-root",
                "/campaign",
                "--scenario-matrix",
                "/matrix.yaml",
                "--config",
                "hash123",
                "--tolerance-m",
                "0.05",
                "--frame-steps",
                "0,5,10",
                "--format",
                "svg",
                "--no-determinism-check",
                "--verbose",
            ]
        )
        assert args.episodes == Path("ep.jsonl")
        assert args.episode_id == "ep_001"
        assert args.outputs == "still,filmstrip,trajectory"
        assert args.out_dir == Path("/tmp/output")
        assert args.campaign_root == Path("/campaign")
        assert args.scenario_matrix == Path("/matrix.yaml")
        assert args.config_hash == "hash123"
        assert args.tolerance_m == 0.05
        assert args.frame_steps == "0,5,10"
        assert args.format == "svg"
        assert args.no_determinism_check is True
        assert args.verbose is True


# ---------------------------------------------------------------------------
# main() argument validation
# ---------------------------------------------------------------------------


class TestMainArgumentValidation:
    """Tests for main() argument validation at the CLI level."""

    def test_invalid_output_type(self, tmp_path: Path) -> None:
        """Invalid output type returns exit code 1."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            '{"episode_id": "ep1", "scenario_id": "s", "seed": 1}\n',
            encoding="utf-8",
        )

        ret = main(
            [
                "--episodes",
                str(episodes_file),
                "--episode-id",
                "ep1",
                "--outputs",
                "invalid_type",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert ret == 1

    def test_missing_episodes_file(self, tmp_path: Path) -> None:
        """Missing episodes file returns exit code 1."""
        ret = main(
            [
                "--episodes",
                str(tmp_path / "nonexistent.jsonl"),
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert ret == 1

    def test_missing_episode_id_in_file(self, tmp_path: Path) -> None:
        """Non-existent episode ID in file returns exit code 1."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            '{"episode_id": "ep1", "scenario_id": "s", "seed": 1}\n',
            encoding="utf-8",
        )

        ret = main(
            [
                "--episodes",
                str(episodes_file),
                "--episode-id",
                "nonexistent",
                "--outputs",
                "trajectory",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert ret == 1

    def test_invalid_frame_steps_format(self, tmp_path: Path) -> None:
        """Invalid frame-steps format returns exit code 1."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "episode_id": "ep1",
                    "scenario_id": "s",
                    "seed": 1,
                    "replay_steps": [
                        {"t": 0, "x": 0, "y": 0, "heading": 0},
                        {"t": 1, "x": 1, "y": 0.5, "heading": 0.1},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        ret = main(
            [
                "--episodes",
                str(episodes_file),
                "--episode-id",
                "ep1",
                "--outputs",
                "filmstrip",
                "--out-dir",
                str(tmp_path / "out"),
                "--frame-steps",
                "0,abc,2",
            ]
        )
        assert ret == 1

    def test_multiple_output_types(self, tmp_path: Path) -> None:
        """Multiple comma-separated output types are accepted."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "episode_id": "ep1",
                    "scenario_id": "s",
                    "seed": 1,
                    "final_robot_position": [1.0, 0.5],
                    "replay_steps": [
                        {"t": 0, "x": 0, "y": 0, "heading": 0},
                        {"t": 1, "x": 1, "y": 0.5, "heading": 0.1},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        ret = main(
            [
                "--episodes",
                str(episodes_file),
                "--episode-id",
                "ep1",
                "--outputs",
                "still,trajectory",
                "--out-dir",
                str(tmp_path / "out"),
                "--no-determinism-check",
            ]
        )
        assert ret == 0

    def test_verbose_mode(self, tmp_path: Path) -> None:
        """Verbose mode does not affect exit code."""
        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text(
            json.dumps(
                {
                    "episode_id": "ep1",
                    "scenario_id": "s",
                    "seed": 1,
                    "final_robot_position": [1.0, 0.5],
                    "replay_steps": [
                        {"t": 0, "x": 0, "y": 0, "heading": 0},
                        {"t": 1, "x": 1, "y": 0.5, "heading": 0.1},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        ret = main(
            [
                "--episodes",
                str(episodes_file),
                "--episode-id",
                "ep1",
                "--outputs",
                "trajectory",
                "--out-dir",
                str(tmp_path / "out"),
                "--verbose",
                "--no-determinism-check",
            ]
        )
        assert ret == 0
