"""Tests for replay-bridge ↔ overlay integration (issue #4778).

Validates that replay_steps data feeds correctly into the overlay renderer
and that the ReplayEpisode → TrajectoryRow bridge works.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark.full_classic.replay import ReplayEpisode, ReplayStep
from robot_sf.benchmark.multi_planner_overlay import (
    MultiPlannerOverlayError,
    extract_trajectory_from_episode,
    trajectory_row_from_replay_episode,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# replay_steps extraction from episode records
# ---------------------------------------------------------------------------


class TestReplayStepsExtraction:
    """Test extracting trajectories from replay_steps in episode records."""

    def test_replay_steps_dict_format(self) -> None:
        """Extract from replay_steps with dict entries."""
        ep: dict[str, Any] = {
            "episode_id": "ep_rs1",
            "scenario_id": "corridor",
            "seed": 42,
            "algo": "orca",
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                {"t": 0.1, "x": 0.5, "y": 0.1, "heading": 0.2},
                {"t": 0.2, "x": 1.0, "y": 0.2, "heading": 0.3},
            ],
        }
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.positions == [(0.0, 0.0), (0.5, 0.1), (1.0, 0.2)]
        assert row.planner_key == "orca"
        assert row.episode_id == "ep_rs1"

    def test_replay_steps_list_format(self) -> None:
        """Extract from replay_steps with list/tuple entries (t, x, y, heading)."""
        ep: dict[str, Any] = {
            "episode_id": "ep_rs2",
            "scenario_id": "hallway",
            "seed": 7,
            "algo": "ppo",
            "replay_steps": [
                [0.0, 1.0, 2.0, 0.5],
                [0.1, 1.5, 2.1, 0.6],
            ],
        }
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.positions == [(1.0, 2.0), (1.5, 2.1)]

    def test_replay_steps_nonfinite_skipped(self) -> None:
        """Non-finite replay step coordinates are skipped."""
        ep: dict[str, Any] = {
            "episode_id": "ep_rs_nan",
            "scenario_id": "corridor",
            "seed": 1,
            "algo": "orca",
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                {"t": 0.1, "x": float("nan"), "y": 0.1, "heading": 0.0},
                {"t": 0.2, "x": 1.0, "y": 0.2, "heading": 0.0},
            ],
        }
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert len(row.positions) == 2

    def test_replay_steps_empty_returns_none(self) -> None:
        """Empty replay_steps list yields no trajectory."""
        ep: dict[str, Any] = {
            "episode_id": "ep_empty_rs",
            "scenario_id": "corridor",
            "seed": 1,
            "algo": "orca",
            "replay_steps": [],
        }
        assert extract_trajectory_from_episode(ep) is None

    def test_inline_trajectory_takes_precedence(self) -> None:
        """Inline trajectory fields take precedence over replay_steps."""
        ep: dict[str, Any] = {
            "episode_id": "ep_both",
            "scenario_id": "corridor",
            "seed": 1,
            "algo": "orca",
            "trajectory": {"robot_positions": [[0.0, 0.0], [1.0, 1.0]]},
            "replay_steps": [
                {"t": 0.0, "x": 5.0, "y": 5.0, "heading": 0.0},
                {"t": 0.1, "x": 6.0, "y": 6.0, "heading": 0.0},
            ],
        }
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.positions == [(0.0, 0.0), (1.0, 1.0)]

    def test_replay_steps_malformed_entries_skipped(self) -> None:
        """Malformed replay step entries are skipped gracefully."""
        ep: dict[str, Any] = {
            "episode_id": "ep_malformed",
            "scenario_id": "corridor",
            "seed": 1,
            "algo": "orca",
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                "invalid_step",
                {"t": 0.2, "x": 1.0, "y": 0.2, "heading": 0.0},
            ],
        }
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert len(row.positions) == 2

    def test_replay_steps_with_pedestrians_in_episode(self) -> None:
        """Pedestrians from the episode record are included with replay_steps."""
        ep: dict[str, Any] = {
            "episode_id": "ep_peds_rs",
            "scenario_id": "corridor",
            "seed": 1,
            "algo": "orca",
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                {"t": 0.1, "x": 1.0, "y": 0.0, "heading": 0.0},
            ],
            "pedestrians": [{"position": [0.5, 0.5]}, {"position": [0.3, 0.7]}],
        }
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.pedestrians == [(0.5, 0.5), (0.3, 0.7)]


# ---------------------------------------------------------------------------
# ReplayEpisode → TrajectoryRow bridge
# ---------------------------------------------------------------------------


class TestTrajectoryRowFromReplayEpisode:
    """Test converting ReplayEpisode to TrajectoryRow."""

    def test_basic_conversion(self) -> None:
        """Basic ReplayEpisode conversion produces correct TrajectoryRow."""
        replay = ReplayEpisode(
            episode_id="ep_bridge_1",
            scenario_id="corridor",
            steps=[
                ReplayStep(t=0.0, x=0.0, y=0.0, heading=0.0),
                ReplayStep(t=0.1, x=0.5, y=0.1, heading=0.2),
                ReplayStep(t=0.2, x=1.0, y=0.3, heading=0.4),
            ],
        )
        row = trajectory_row_from_replay_episode(replay, planner_key="orca")
        assert row.episode_id == "ep_bridge_1"
        assert row.scenario_id == "corridor"
        assert row.planner_key == "orca"
        assert row.positions == [(0.0, 0.0), (0.5, 0.1), (1.0, 0.3)]
        assert row.pedestrians is None

    def test_with_pedestrian_snapshot(self) -> None:
        """Last-step pedestrian positions are captured."""
        replay = ReplayEpisode(
            episode_id="ep_bridge_peds",
            scenario_id="hallway",
            steps=[
                ReplayStep(t=0.0, x=0.0, y=0.0, heading=0.0),
                ReplayStep(
                    t=0.1,
                    x=1.0,
                    y=0.0,
                    heading=0.0,
                    ped_positions=[(0.5, 0.5), (0.3, 0.7)],
                ),
            ],
        )
        row = trajectory_row_from_replay_episode(replay, planner_key="ppo")
        assert row.pedestrians == [(0.5, 0.5), (0.3, 0.7)]

    def test_empty_steps_raises(self) -> None:
        """ReplayEpisode with no steps raises MultiPlannerOverlayError."""
        replay = ReplayEpisode(
            episode_id="ep_empty",
            scenario_id="corridor",
            steps=[],
        )
        with pytest.raises(MultiPlannerOverlayError, match="no steps"):
            trajectory_row_from_replay_episode(replay)

    def test_nonfinite_ped_positions_excluded(self) -> None:
        """Non-finite pedestrian positions are excluded."""
        replay = ReplayEpisode(
            episode_id="ep_bad_peds",
            scenario_id="corridor",
            steps=[
                ReplayStep(
                    t=0.0,
                    x=0.0,
                    y=0.0,
                    heading=0.0,
                    ped_positions=[(0.5, 0.5), (float("nan"), 1.0), (float("inf"), 2.0)],
                ),
            ],
        )
        row = trajectory_row_from_replay_episode(replay)
        assert row.pedestrians == [(0.5, 0.5)]

    def test_unknown_planner_fallback(self) -> None:
        """Empty planner_key falls back to 'unknown_planner'."""
        replay = ReplayEpisode(
            episode_id="ep_unknown",
            scenario_id="corridor",
            steps=[ReplayStep(t=0.0, x=0.0, y=0.0, heading=0.0)],
        )
        row = trajectory_row_from_replay_episode(replay)
        assert row.planner_key == "unknown_planner"

    def test_source_path_preserved(self) -> None:
        """Source path is preserved in the TrajectoryRow."""
        replay = ReplayEpisode(
            episode_id="ep_src",
            scenario_id="corridor",
            steps=[ReplayStep(t=0.0, x=0.0, y=0.0, heading=0.0)],
        )
        row = trajectory_row_from_replay_episode(
            replay, planner_key="orca", source_path="/some/path.jsonl"
        )
        assert row.source_path == "/some/path.jsonl"


# ---------------------------------------------------------------------------
# Selection manifest → overlay integration (CLI)
# ---------------------------------------------------------------------------


class TestSelectionManifestCLI:
    """Test CLI --selection-manifest integration."""

    def test_cli_with_manifest(self, tmp_path: Path) -> None:
        """CLI renders overlay from selection manifest."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        # Create episodes JSONL
        ep_file = tmp_path / "episodes.jsonl"
        episodes = [
            {
                "episode_id": "ep_orca",
                "algo": "orca",
                "scenario_id": "corridor",
                "seed": 42,
                "trajectory": {"robot_positions": [[0, 0], [1, 1], [2, 0]]},
            },
            {
                "episode_id": "ep_ppo",
                "algo": "ppo",
                "scenario_id": "corridor",
                "seed": 42,
                "trajectory": {"robot_positions": [[0, 0], [0.5, 2], [2, 0]]},
            },
        ]
        ep_file.write_text(
            "\n".join(json.dumps(e) for e in episodes) + "\n",
            encoding="utf-8",
        )

        # Create selection manifest
        manifest = {
            "schema_version": "exemplar-selection.v1",
            "source_episodes": str(ep_file),
            "group_by": ["planner_key"],
            "metric": "path_efficiency",
            "selected": [
                {
                    "episode_id": "ep_orca",
                    "planner_key": "orca",
                    "scenario_id": "corridor",
                    "seed": 42,
                    "selection_mode": "median",
                    "selection_rank": 0,
                    "metric_value": 0.8,
                    "reason": "median within cell",
                },
                {
                    "episode_id": "ep_ppo",
                    "planner_key": "ppo",
                    "scenario_id": "corridor",
                    "seed": 42,
                    "selection_mode": "median",
                    "selection_rank": 0,
                    "metric_value": 0.7,
                    "reason": "median within cell",
                },
            ],
            "skipped_cells": [],
        }
        manifest_file = tmp_path / "selection_manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        out_dir = tmp_path / "overlay_out"
        exit_code = cli_main(
            [
                "--episodes",
                str(ep_file),
                "--selection-manifest",
                str(manifest_file),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        png_path = out_dir / "corridor_seed42.png"
        assert png_path.exists()

    def test_cli_manifest_missing_episodes(self, tmp_path: Path) -> None:
        """CLI returns error when manifest references episodes not in JSONL."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        ep_file.write_text(
            json.dumps(
                {
                    "episode_id": "ep_orca",
                    "algo": "orca",
                    "scenario_id": "corridor",
                    "seed": 42,
                    "trajectory": {"robot_positions": [[0, 0], [1, 1]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "exemplar-selection.v1",
            "selected": [
                {
                    "episode_id": "ep_orca",
                    "planner_key": "orca",
                    "scenario_id": "corridor",
                    "seed": 42,
                },
                {
                    "episode_id": "ep_missing",
                    "planner_key": "ppo",
                    "scenario_id": "corridor",
                    "seed": 42,
                },
            ],
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        out_dir = tmp_path / "out"
        exit_code = cli_main(
            [
                "--episodes",
                str(ep_file),
                "--selection-manifest",
                str(manifest_file),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 2

    def test_cli_manifest_with_allow_missing(self, tmp_path: Path) -> None:
        """CLI with --allow-missing renders available planners from manifest."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        ep_file.write_text(
            json.dumps(
                {
                    "episode_id": "ep_orca",
                    "algo": "orca",
                    "scenario_id": "corridor",
                    "seed": 42,
                    "trajectory": {"robot_positions": [[0, 0], [1, 1]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "exemplar-selection.v1",
            "selected": [
                {
                    "episode_id": "ep_orca",
                    "planner_key": "orca",
                    "scenario_id": "corridor",
                    "seed": 42,
                },
                {
                    "episode_id": "ep_missing",
                    "planner_key": "ppo",
                    "scenario_id": "corridor",
                    "seed": 42,
                },
            ],
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        out_dir = tmp_path / "out"
        exit_code = cli_main(
            [
                "--episodes",
                str(ep_file),
                "--selection-manifest",
                str(manifest_file),
                "--out-dir",
                str(out_dir),
                "--allow-missing",
            ]
        )
        assert exit_code == 0
        png_path = out_dir / "corridor_seed42.png"
        assert png_path.exists()

    def test_cli_manifest_empty_selected(self, tmp_path: Path) -> None:
        """CLI returns error when manifest has no selected episodes."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        ep_file.write_text(
            json.dumps({"episode_id": "ep1", "algo": "orca", "scenario_id": "s", "seed": 1}) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "exemplar-selection.v1",
            "selected": [],
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        exit_code = cli_main(
            [
                "--episodes",
                str(ep_file),
                "--selection-manifest",
                str(manifest_file),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == 1

    def test_cli_manifest_bad_schema(self, tmp_path: Path) -> None:
        """CLI returns error on unsupported manifest schema."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        ep_file.write_text(
            json.dumps({"episode_id": "ep1", "algo": "orca", "scenario_id": "s", "seed": 1}) + "\n",
            encoding="utf-8",
        )

        manifest = {"schema_version": "wrong.v0", "selected": []}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(SystemExit, match="1"):
            cli_main(
                [
                    "--episodes",
                    str(ep_file),
                    "--selection-manifest",
                    str(manifest_file),
                    "--out-dir",
                    str(tmp_path / "out"),
                ]
            )

    def test_cli_requires_args_without_manifest(self, tmp_path: Path) -> None:
        """CLI returns error when neither manifest nor required args provided."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        ep_file.write_text(
            json.dumps({"episode_id": "ep1", "algo": "orca", "scenario_id": "s", "seed": 1}) + "\n",
            encoding="utf-8",
        )

        exit_code = cli_main(["--episodes", str(ep_file), "--out-dir", str(tmp_path / "out")])
        assert exit_code == 1

    def test_cli_manifest_multiple_cells(self, tmp_path: Path) -> None:
        """CLI renders overlays for multiple cells in a manifest."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        records = [
            {
                "episode_id": "ep1",
                "algo": "orca",
                "scenario_id": "corridor",
                "seed": 1,
                "trajectory": {"robot_positions": [[0, 0], [1, 1]]},
            },
            {
                "episode_id": "ep2",
                "algo": "orca",
                "scenario_id": "hallway",
                "seed": 2,
                "trajectory": {"robot_positions": [[0, 0], [2, 2]]},
            },
        ]
        ep_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        manifest = {
            "schema_version": "exemplar-selection.v1",
            "selected": [
                {
                    "episode_id": "ep1",
                    "planner_key": "orca",
                    "scenario_id": "corridor",
                    "seed": 1,
                },
                {
                    "episode_id": "ep2",
                    "planner_key": "orca",
                    "scenario_id": "hallway",
                    "seed": 2,
                },
            ],
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        out_dir = tmp_path / "out"
        exit_code = cli_main(
            [
                "--episodes",
                str(ep_file),
                "--selection-manifest",
                str(manifest_file),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        assert (out_dir / "corridor_seed1.png").exists()
        assert (out_dir / "hallway_seed2.png").exists()
