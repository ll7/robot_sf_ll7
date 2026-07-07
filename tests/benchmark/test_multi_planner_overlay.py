"""Tests for multi-planner trajectory overlay (issue #4778).

Validates:
- Trajectory extraction from episode records with various layouts
- Episode selection per planner for a given scenario+seed
- Missing planner rows fail closed
- Overlay figure rendering produces expected outputs
- Planner colors match the shared palette
- Provenance sidecar is written
- Mismatched scenario/seed rows are rejected
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from robot_sf.benchmark.figures.style import planner_color, planner_palette
from robot_sf.benchmark.multi_planner_overlay import (
    MultiPlannerOverlayError,
    TrajectoryRow,
    build_overlay_figure,
    extract_trajectory_from_episode,
    load_episodes,
    select_episodes_for_overlay,
)


def _make_episode_with_trajectory(
    episode_id: str,
    planner: str,
    scenario_id: str,
    seed: int,
    positions: list[list[float]],
    *,
    layout: str = "trajectory.robot_positions",
    pedestrians: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create an episode record with embedded trajectory data."""
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": planner,
    }

    # Set up trajectory at the requested layout path
    parts = layout.split(".")
    root: dict[str, Any] = {}
    current = root
    last_key = parts[-1]
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[last_key] = positions

    # Merge into record
    _deep_merge(record, root)

    if pedestrians:
        record["pedestrians"] = pedestrians

    return record


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


class TestExtractTrajectory:
    """Test trajectory extraction from episode records."""

    def test_trajectory_robot_positions_layout(self) -> None:
        """Extract from trajectory.robot_positions layout."""
        positions = [[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]]
        ep = _make_episode_with_trajectory(
            "ep1",
            "orca",
            "corridor",
            42,
            positions,
            layout="trajectory.robot_positions",
        )
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.positions == [(0.0, 0.0), (0.5, 0.1), (1.0, 0.2)]
        assert row.planner_key == "orca"
        assert row.scenario_id == "corridor"
        assert row.seed == 42

    def test_robot_trajectory_positions_layout(self) -> None:
        """Extract from robot_trajectory.positions layout."""
        positions = [[0.0, 0.0], [1.0, 1.0]]
        ep = _make_episode_with_trajectory(
            "ep2",
            "ppo",
            "hallway",
            7,
            positions,
            layout="robot_trajectory.positions",
        )
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.positions == [(0.0, 0.0), (1.0, 1.0)]

    def test_no_trajectory_returns_none(self) -> None:
        """Episode without trajectory data returns None."""
        ep = {
            "episode_id": "ep_no_traj",
            "scenario_id": "corridor",
            "seed": 1,
            "algo": "orca",
            "metrics": {"collisions": 0},
        }
        assert extract_trajectory_from_episode(ep) is None

    def test_empty_positions_returns_none(self) -> None:
        """Episode with empty trajectory list returns None."""
        ep = _make_episode_with_trajectory(
            "ep_empty",
            "orca",
            "corridor",
            1,
            [],
            layout="trajectory.robot_positions",
        )
        assert extract_trajectory_from_episode(ep) is None

    def test_pedestrian_extraction(self) -> None:
        """Pedestrian positions are extracted when present."""
        positions = [[0.0, 0.0], [1.0, 0.0]]
        peds = [{"position": [0.5, 0.5]}]
        ep = _make_episode_with_trajectory(
            "ep_peds",
            "orca",
            "corridor",
            1,
            positions,
            pedestrians=peds,
        )
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.pedestrians == [(0.5, 0.5)]

    def test_planner_key_from_algo_field(self) -> None:
        """Planner key extracted from algo field."""
        positions = [[0.0, 0.0]]
        ep = _make_episode_with_trajectory(
            "ep_algo",
            "social_force",
            "corridor",
            1,
            positions,
        )
        row = extract_trajectory_from_episode(ep)
        assert row is not None
        assert row.planner_key == "social_force"


class TestSelectEpisodesForOverlay:
    """Test episode selection for overlay rendering."""

    def _make_episodes(
        self,
        planners: list[str],
        scenario_id: str = "corridor",
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Create episodes for multiple planners, same scenario+seed."""
        episodes = []
        for i, planner in enumerate(planners):
            positions = [[float(i), 0.0], [float(i) + 1, 0.5]]
            ep = _make_episode_with_trajectory(
                f"ep_{planner}",
                planner,
                scenario_id,
                seed,
                positions,
            )
            episodes.append(ep)
        return episodes

    def test_select_two_planners(self) -> None:
        """Two planners same scenario+seed both selected."""
        episodes = self._make_episodes(["orca", "ppo"])
        rows = select_episodes_for_overlay(
            episodes,
            scenario_id="corridor",
            seed=42,
            planner_keys=["orca", "ppo"],
        )
        assert len(rows) == 2
        assert "orca" in rows
        assert "ppo" in rows

    def test_missing_planner_fails_closed(self) -> None:
        """Missing planner raises MultiPlannerOverlayError."""
        episodes = self._make_episodes(["orca"])
        with pytest.raises(MultiPlannerOverlayError, match="No trajectory data"):
            select_episodes_for_overlay(
                episodes,
                scenario_id="corridor",
                seed=42,
                planner_keys=["orca", "ppo"],
            )

    def test_mismatched_scenario_rejected(self) -> None:
        """Episodes with wrong scenario_id are not selected."""
        episodes = self._make_episodes(["orca"], scenario_id="hallway")
        with pytest.raises(MultiPlannerOverlayError, match="No trajectory data"):
            select_episodes_for_overlay(
                episodes,
                scenario_id="corridor",
                seed=42,
                planner_keys=["orca"],
            )

    def test_mismatched_seed_rejected(self) -> None:
        """Episodes with wrong seed are not selected."""
        episodes = self._make_episodes(["orca"], seed=99)
        with pytest.raises(MultiPlannerOverlayError, match="No trajectory data"):
            select_episodes_for_overlay(
                episodes,
                scenario_id="corridor",
                seed=42,
                planner_keys=["orca"],
            )

    def test_no_trajectory_data_fails(self) -> None:
        """Episodes without trajectory data fail selection."""
        episodes = [
            {
                "episode_id": "ep_no_traj",
                "scenario_id": "corridor",
                "seed": 42,
                "algo": "orca",
                "metrics": {"collisions": 0},
            }
        ]
        with pytest.raises(MultiPlannerOverlayError, match="No trajectory data"):
            select_episodes_for_overlay(
                episodes,
                scenario_id="corridor",
                seed=42,
                planner_keys=["orca"],
            )


class TestBuildOverlayFigure:
    """Test overlay figure rendering."""

    def _make_rows(self, planners: list[str]) -> dict[str, TrajectoryRow]:
        """Create TrajectoryRow fixtures."""
        rows: dict[str, TrajectoryRow] = {}
        for i, pk in enumerate(planners):
            positions = [(float(i) + x * 0.5, float(x * 0.3)) for x in range(5)]
            rows[pk] = TrajectoryRow(
                episode_id=f"ep_{pk}",
                planner_key=pk,
                scenario_id="corridor",
                seed=42,
                positions=positions,
            )
        return rows

    def test_overlay_figure_produces_outputs(self, tmp_path: Path) -> None:
        """Two planners produces output files."""
        rows = self._make_rows(["orca", "ppo"])
        output_base = tmp_path / "corridor_seed42"
        saved = build_overlay_figure(rows, output_base=output_base, formats=("png",))
        assert len(saved) >= 1
        png_exists = tmp_path / "corridor_seed42.png"
        assert png_exists.exists()

    def test_overlay_provenance_sidecar_written(self, tmp_path: Path) -> None:
        """Provenance sidecar is created with correct fields."""
        rows = self._make_rows(["orca", "ppo"])
        output_base = tmp_path / "corridor_seed42"
        build_overlay_figure(rows, output_base=output_base, formats=("png",))
        provenance_path = tmp_path / "corridor_seed42.provenance.json"
        assert provenance_path.exists()
        data = json.loads(provenance_path.read_text())
        assert "planners" in data
        assert "scenario_id" in data
        assert data["scenario_id"] == "corridor"
        assert data["seed"] == 42

    def test_empty_rows_raises(self, tmp_path: Path) -> None:
        """Empty trajectory rows raises MultiPlannerOverlayError."""
        with pytest.raises(MultiPlannerOverlayError, match="No trajectory rows"):
            build_overlay_figure({}, output_base=tmp_path / "empty", formats=("png",))

    def test_planner_colors_match_palette(self) -> None:
        """Planner colors used match the shared palette."""
        palette = planner_palette()
        for key in ("orca", "ppo", "goal", "social_force"):
            color = planner_color(key)
            if key in palette:
                assert color == palette[key], f"{key} color mismatch"

    def test_single_planner_overlay(self, tmp_path: Path) -> None:
        """Single planner overlay still works."""
        rows = self._make_rows(["orca"])
        output_base = tmp_path / "single"
        saved = build_overlay_figure(rows, output_base=output_base, formats=("png",))
        assert len(saved) >= 1


class TestLoadEpisodes:
    """Test JSONL loading."""

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        """Load episodes from valid JSONL."""
        file = tmp_path / "episodes.jsonl"
        records = [
            {"episode_id": "ep1", "algo": "orca", "scenario_id": "s1", "seed": 1},
            {"episode_id": "ep2", "algo": "ppo", "scenario_id": "s1", "seed": 1},
        ]
        file.write_text(
            json.dumps(records[0]) + "\n" + json.dumps(records[1]) + "\n",
            encoding="utf-8",
        )
        episodes = load_episodes(file)
        assert len(episodes) == 2
        assert episodes[0]["episode_id"] == "ep1"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing file raises MultiPlannerOverlayError."""
        with pytest.raises(MultiPlannerOverlayError, match="not found"):
            load_episodes(tmp_path / "nonexistent.jsonl")

    def test_load_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines in JSONL are skipped."""
        file = tmp_path / "episodes.jsonl"
        file.write_text(
            json.dumps({"episode_id": "ep1"}) + "\n\n\n" + json.dumps({"episode_id": "ep2"}) + "\n",
            encoding="utf-8",
        )
        episodes = load_episodes(file)
        assert len(episodes) == 2


class TestCLI:
    """Test CLI entry point."""

    def test_cli_help(self) -> None:
        """Module exports are correct."""
        from robot_sf.benchmark.multi_planner_overlay import __all__

        assert "build_overlay_figure" in __all__

    def test_cli_missing_planner_returns_nonzero(self, tmp_path: Path) -> None:
        """CLI returns non-zero when a planner is missing."""
        from scripts.render_multi_planner_trajectory_overlay import main as cli_main

        ep_file = tmp_path / "episodes.jsonl"
        ep_file.write_text(
            json.dumps(
                {
                    "episode_id": "ep1",
                    "algo": "orca",
                    "scenario_id": "corridor",
                    "seed": 42,
                    "trajectory": {"robot_positions": [[0, 0], [1, 1]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # Request a planner that does not exist
        out_dir = tmp_path / "out"
        exit_code = cli_main(
            [
                "--episodes",
                str(ep_file),
                "--scenario-id",
                "corridor",
                "--seed",
                "42",
                "--planners",
                "orca,missing_planner",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 2
