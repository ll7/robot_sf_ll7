"""Tests for episode replay figure generation bridge.

These tests verify:
- Episode row loading and validation
- Replay episode construction from episode rows
- Determinism checking behavior
- Figure artifact generation (still, filmstrip, trajectory)
- Provenance sidecar generation
- Failure modes for corrupted/missing data
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.episode_replay_figure import (
    EpisodeRow,
    ProvenanceSidecar,
    build_replay_from_episode_row,
    check_determinism,
    compute_bytes_sha256,
    compute_file_sha256,
    generate_filmstrip,
    generate_still,
    generate_trajectory,
    load_episode_row,
    replay_episode_and_generate_figures,
    write_provenance_sidecar,
)
from robot_sf.benchmark.full_classic.replay import ReplayEpisode, ReplayStep


@pytest.fixture
def sample_episode_row_dict():
    """Sample episode row dictionary for testing."""
    return {
        "episode_id": "test_ep_001",
        "scenario_id": "crossing_easy",
        "seed": 42,
        "planner": "social_force",
        "campaign_id": "camp_123",
        "final_robot_position": [3.0, 1.5],
        "final_progress": 0.85,
        "success": True,
        "collision": False,
        "replay_steps": [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
            {"t": 1.0, "x": 1.0, "y": 0.5, "heading": 0.1},
            {"t": 2.0, "x": 2.0, "y": 1.0, "heading": 0.2},
            {"t": 3.0, "x": 3.0, "y": 1.5, "heading": 0.3},
        ],
        "replay_dt": 1.0,
        "replay_map_path": "/path/to/map.svg",
    }


@pytest.fixture
def sample_episode_row(sample_episode_row_dict):
    """Sample EpisodeRow instance."""
    return EpisodeRow.from_dict(sample_episode_row_dict)


@pytest.fixture
def sample_replay_episode(sample_episode_row_dict):
    """Sample ReplayEpisode instance."""
    steps = [
        ReplayStep(
            t=step["t"],
            x=step["x"],
            y=step["y"],
            heading=step["heading"],
        )
        for step in sample_episode_row_dict["replay_steps"]
    ]
    return ReplayEpisode(
        episode_id="test_ep_001",
        scenario_id="crossing_easy",
        steps=steps,
        dt=1.0,
        map_path="/path/to/map.svg",
    )


@pytest.fixture
def episodes_jsonl_file(sample_episode_row_dict):
    """Create temporary episodes JSONL file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as f:
        json.dump(sample_episode_row_dict, f)
        f.write("\n")
        other_ep = {
            "episode_id": "other_ep",
            "scenario_id": "other_scenario",
            "seed": 99,
        }
        json.dump(other_ep, f)
        f.write("\n")
        path = Path(f.name)
    yield path
    path.unlink()


class TestEpisodeRow:
    """Tests for EpisodeRow validation and construction."""

    def test_from_dict_valid(self, sample_episode_row_dict):
        """Test EpisodeRow creation from valid dictionary."""
        row = EpisodeRow.from_dict(sample_episode_row_dict)
        assert row.episode_id == "test_ep_001"
        assert row.scenario_id == "crossing_easy"
        assert row.seed == 42
        assert row.planner == "social_force"
        assert row.final_robot_position == (3.0, 1.5)
        assert row.success is True

    def test_from_dict_missing_required(self):
        """Test EpisodeRow creation fails with missing required fields."""
        data = {"episode_id": "test"}
        with pytest.raises(ValueError, match="missing required fields"):
            EpisodeRow.from_dict(data)

    def test_from_dict_missing_scenario_id(self):
        """Test EpisodeRow creation fails without scenario_id."""
        data = {"episode_id": "test", "seed": 1}
        with pytest.raises(ValueError, match="missing required fields"):
            EpisodeRow.from_dict(data)

    def test_from_dict_missing_seed(self):
        """Test EpisodeRow creation fails without seed."""
        data = {"episode_id": "test", "scenario_id": "scen"}
        with pytest.raises(ValueError, match="missing required fields"):
            EpisodeRow.from_dict(data)

    def test_from_dict_optional_fields(self):
        """Test EpisodeRow handles optional fields gracefully."""
        data = {
            "episode_id": "test",
            "scenario_id": "scen",
            "seed": 1,
            "planner_key": "pk",
            "algo": "algorithm",
            "config_hash": "abc123",
        }
        row = EpisodeRow.from_dict(data)
        assert row.planner_key == "pk"
        assert row.algo == "algorithm"
        assert row.config_hash == "abc123"


class TestLoadEpisodeRow:
    """Tests for loading episode rows from JSONL files."""

    def test_load_episode_found(self, episodes_jsonl_file):
        """Test loading episode by ID from JSONL."""
        row = load_episode_row(episodes_jsonl_file, "test_ep_001")
        assert row.episode_id == "test_ep_001"
        assert row.scenario_id == "crossing_easy"

    def test_load_episode_not_found(self, episodes_jsonl_file):
        """Test loading non-existent episode raises error."""
        with pytest.raises(ValueError, match="not found"):
            load_episode_row(episodes_jsonl_file, "nonexistent")

    def test_load_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_episode_row(Path("/nonexistent/path.jsonl"), "ep1")

    def test_load_invalid_json(self):
        """Test loading from file with invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("not valid json\n")
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_episode_row(path, "ep1")
        finally:
            path.unlink()


class TestBuildReplayFromEpisodeRow:
    """Tests for building ReplayEpisode from episode rows."""

    def test_build_with_replay_steps(self, sample_episode_row):
        """Test building ReplayEpisode with replay steps."""
        replay = build_replay_from_episode_row(sample_episode_row)
        assert replay is not None
        assert replay.episode_id == "test_ep_001"
        assert len(replay.steps) == 4
        assert replay.steps[0].t == 0.0
        assert replay.steps[-1].x == 3.0

    def test_build_without_replay_steps(self):
        """Test building ReplayEpisode without replay steps returns None."""
        row = EpisodeRow(
            episode_id="test",
            scenario_id="scen",
            seed=1,
        )
        replay = build_replay_from_episode_row(row)
        assert replay is None

    def test_build_with_empty_replay_steps(self):
        """Test building ReplayEpisode with empty replay steps returns None."""
        row = EpisodeRow(
            episode_id="test",
            scenario_id="scen",
            seed=1,
            raw={"replay_steps": []},
        )
        replay = build_replay_from_episode_row(row)
        assert replay is None

    def test_build_with_tuple_steps(self):
        """Test building ReplayEpisode with tuple-format steps."""
        row = EpisodeRow(
            episode_id="test",
            scenario_id="scen",
            seed=1,
            raw={
                "replay_steps": [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.5, 0.1),
                ]
            },
        )
        replay = build_replay_from_episode_row(row)
        assert replay is not None
        assert len(replay.steps) == 2


class TestCheckDeterminism:
    """Tests for determinism checking."""

    def test_check_pass_within_tolerance(self, sample_replay_episode, sample_episode_row):
        """Test determinism check passes when within tolerance."""
        sample_episode_row.final_robot_position = (3.0, 1.5)
        status, details = check_determinism(
            sample_replay_episode,
            sample_episode_row,
            tolerance_m=0.1,
        )
        assert status == "pass"
        assert "final_robot_position" in details["checks_passed"]

    def test_check_fail_outside_tolerance(self, sample_replay_episode, sample_episode_row):
        """Test determinism check fails when outside tolerance."""
        sample_episode_row.final_robot_position = (100.0, 100.0)
        status, details = check_determinism(
            sample_replay_episode,
            sample_episode_row,
            tolerance_m=0.1,
        )
        assert status == "fail"
        assert details["checks_failed"]
        assert "position_error_m" in details

    def test_check_not_evaluable_no_endpoint(self, sample_replay_episode):
        """Test determinism check not evaluable without endpoint data."""
        row = EpisodeRow(
            episode_id="test",
            scenario_id="scen",
            seed=1,
        )
        status, details = check_determinism(sample_replay_episode, row)
        assert status == "not_evaluable"
        assert "no evaluable endpoints" in details.get("reason", "")

    def test_check_not_evaluable_no_steps(self):
        """Test determinism check not evaluable without replay steps."""
        episode = ReplayEpisode(
            episode_id="test",
            scenario_id="scen",
            steps=[],
        )
        row = EpisodeRow(
            episode_id="test",
            scenario_id="scen",
            seed=1,
            final_robot_position=(0.0, 0.0),
        )
        status, details = check_determinism(episode, row)
        assert status == "not_evaluable"
        assert "no replay steps" in details.get("reason", "")


class TestFigureGeneration:
    """Tests for figure artifact generation."""

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_generate_still(self, sample_replay_episode, output_dir):
        """Test still frame generation."""
        artifact = generate_still(
            sample_replay_episode,
            step_idx=2,
            out_path=output_dir / "still.png",
            fmt="png",
        )
        assert artifact.artifact_type == "still"
        assert Path(artifact.path).exists()
        assert artifact.format == "png"
        assert len(artifact.sha256) == 64

    def test_generate_still_invalid_step(self, sample_replay_episode, output_dir):
        """Test still generation fails with invalid step index."""
        with pytest.raises(ValueError, match="out of range"):
            generate_still(
                sample_replay_episode,
                step_idx=100,
                out_path=output_dir / "still.png",
            )

    def test_generate_filmstrip(self, sample_replay_episode, output_dir):
        """Test filmstrip generation."""
        artifact = generate_filmstrip(
            sample_replay_episode,
            out_path=output_dir / "filmstrip.png",
            fmt="png",
        )
        assert artifact.artifact_type == "filmstrip"
        assert Path(artifact.path).exists()
        assert artifact.format == "png"

    def test_generate_filmstrip_custom_steps(self, sample_replay_episode, output_dir):
        """Test filmstrip with custom frame steps."""
        artifact = generate_filmstrip(
            sample_replay_episode,
            out_path=output_dir / "filmstrip.png",
            fmt="png",
            frame_steps=[0, 3],
        )
        assert artifact.artifact_type == "filmstrip"
        assert Path(artifact.path).exists()

    def test_generate_trajectory(self, sample_replay_episode, output_dir):
        """Test trajectory plot generation."""
        artifact = generate_trajectory(
            sample_replay_episode,
            out_path=output_dir / "trajectory.png",
            fmt="png",
        )
        assert artifact.artifact_type == "trajectory"
        assert Path(artifact.path).exists()
        assert artifact.format == "png"

    def test_generate_trajectory_pdf(self, sample_replay_episode, output_dir):
        """Test trajectory plot in PDF format."""
        artifact = generate_trajectory(
            sample_replay_episode,
            out_path=output_dir / "trajectory.pdf",
            fmt="pdf",
        )
        assert artifact.format == "pdf"
        assert Path(artifact.path).exists()

    def test_generate_trajectory_no_steps(self, output_dir):
        """Test trajectory generation fails without steps."""
        episode = ReplayEpisode(
            episode_id="test",
            scenario_id="scen",
            steps=[],
        )
        with pytest.raises(ValueError, match="No steps"):
            generate_trajectory(
                episode,
                out_path=output_dir / "trajectory.png",
            )


class TestProvenanceSidecar:
    """Tests for provenance sidecar generation."""

    def test_write_provenance_sidecar(self, tmp_path):
        """Test writing provenance sidecar."""
        sidecar = ProvenanceSidecar(
            campaign_id="camp_123",
            episode_id="ep_001",
            scenario_id="crossing",
            seed=42,
            planner_key="social_force",
            scenario_matrix_path="/path/matrix.json",
            campaign_config_hash="abc123",
            repo_commit="deadbeef",
            replay_command="python test.py",
            determinism_check_status="pass",
            determinism_tolerance=0.1,
            source_episodes_jsonl_path="/path/episodes.jsonl",
            source_episodes_jsonl_sha256="sha256hash",
            artifacts=[{"type": "trajectory", "path": "traj.png"}],
            generated_at="2024-01-01T00:00:00Z",
        )
        out_path = tmp_path / "provenance.json"
        write_provenance_sidecar(out_path, sidecar)

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["episode_id"] == "ep_001"
        assert data["seed"] == 42
        assert data["determinism_check_status"] == "pass"

    def test_compute_file_sha256(self, tmp_path):
        """Test file SHA-256 computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        sha = compute_file_sha256(test_file)
        assert len(sha) == 64
        assert isinstance(sha, str)

    def test_compute_bytes_sha256(self):
        """Test bytes SHA-256 computation."""
        sha = compute_bytes_sha256(b"hello world")
        assert len(sha) == 64


class TestReplayEpisodeAndGenerateFigures:
    """Integration tests for full replay and figure generation workflow."""

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow(self, sample_episode_row, output_dir, episodes_jsonl_file):
        """Test complete replay and figure generation workflow."""
        result = replay_episode_and_generate_figures(
            episode_row=sample_episode_row,
            outputs=["still", "trajectory"],
            out_dir=output_dir,
            tolerance_m=0.5,
            episodes_jsonl_path=episodes_jsonl_file,
        )

        assert result["episode_id"] == "test_ep_001"
        assert result["determinism_check_status"] == "pass"
        assert result["artifacts_generated"] == 2
        assert len(result["artifact_paths"]) == 2
        assert Path(result["provenance_sidecar"]).exists()
        assert Path(result["caption_fragment"]).exists()

    def test_full_workflow_determinism_pass(
        self,
        output_dir,
        episodes_jsonl_file,
    ):
        """Test workflow with determinism check passing."""
        row_dict = {
            "episode_id": "test_ep_002",
            "scenario_id": "crossing",
            "seed": 42,
            "final_robot_position": [3.0, 1.5],
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                {"t": 1.0, "x": 1.0, "y": 0.5, "heading": 0.1},
                {"t": 2.0, "x": 2.0, "y": 1.0, "heading": 0.2},
                {"t": 3.0, "x": 3.0, "y": 1.5, "heading": 0.3},
            ],
        }
        row = EpisodeRow.from_dict(row_dict)

        result = replay_episode_and_generate_figures(
            episode_row=row,
            outputs=["trajectory"],
            out_dir=output_dir,
            tolerance_m=0.5,
            episodes_jsonl_path=episodes_jsonl_file,
        )

        assert result["determinism_check_status"] == "pass"

    def test_full_workflow_determinism_fail(
        self,
        output_dir,
        episodes_jsonl_file,
    ):
        """Test workflow fails when determinism check fails."""
        row_dict = {
            "episode_id": "test_ep_003",
            "scenario_id": "crossing",
            "seed": 42,
            "final_robot_position": [100.0, 100.0],
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                {"t": 1.0, "x": 1.0, "y": 0.5, "heading": 0.1},
            ],
        }
        row = EpisodeRow.from_dict(row_dict)

        with pytest.raises(RuntimeError, match="Determinism check failed"):
            replay_episode_and_generate_figures(
                episode_row=row,
                outputs=["trajectory"],
                out_dir=output_dir,
                tolerance_m=0.1,
                episodes_jsonl_path=episodes_jsonl_file,
            )

    def test_full_workflow_no_determinism_check(
        self,
        sample_episode_row,
        output_dir,
        episodes_jsonl_file,
    ):
        """Test workflow with determinism check disabled."""
        result = replay_episode_and_generate_figures(
            episode_row=sample_episode_row,
            outputs=["trajectory"],
            out_dir=output_dir,
            no_determinism_check=True,
            episodes_jsonl_path=episodes_jsonl_file,
        )

        assert result["determinism_check_status"] == "skipped"

    def test_full_workflow_no_replay_steps(self, output_dir, episodes_jsonl_file):
        """Test workflow requires scenario matrix when replay steps absent."""
        row = EpisodeRow(
            episode_id="test_missing_replay",
            scenario_id="scen",
            seed=1,
        )

        with pytest.raises(ValueError, match="no replay_steps.*no scenario matrix"):
            replay_episode_and_generate_figures(
                episode_row=row,
                outputs=["trajectory"],
                out_dir=output_dir,
                episodes_jsonl_path=episodes_jsonl_file,
            )

    def test_full_workflow_insufficient_steps(self, output_dir, episodes_jsonl_file):
        """Test workflow fails with insufficient replay steps."""
        row = EpisodeRow(
            episode_id="test",
            scenario_id="scen",
            seed=1,
            raw={"replay_steps": [{"t": 0, "x": 0, "y": 0, "heading": 0}]},
        )

        with pytest.raises(ValueError, match="insufficient replay steps"):
            replay_episode_and_generate_figures(
                episode_row=row,
                outputs=["trajectory"],
                out_dir=output_dir,
                episodes_jsonl_path=episodes_jsonl_file,
            )

    def test_full_workflow_trajectory_only(
        self,
        sample_episode_row,
        output_dir,
        episodes_jsonl_file,
    ):
        """Test workflow with trajectory output only."""
        result = replay_episode_and_generate_figures(
            episode_row=sample_episode_row,
            outputs=["trajectory"],
            out_dir=output_dir,
            episodes_jsonl_path=episodes_jsonl_file,
        )

        assert result["artifacts_generated"] == 1
        assert any("trajectory" in p for p in result["artifact_paths"])


class TestCLI:
    """Tests for CLI entry point."""

    def test_cli_help(self):
        """Test CLI help output."""
        from scripts.replay_episode_figure import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_missing_episodes_file(self, tmp_path):
        """Test CLI with missing episodes file."""
        from scripts.replay_episode_figure import main

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

    def test_cli_missing_episode_id(self, tmp_path):
        """Test CLI with non-existent episode ID."""
        from scripts.replay_episode_figure import main as cli_main

        episodes_file = tmp_path / "episodes.jsonl"
        episodes_file.write_text('{"episode_id": "ep1", "scenario_id": "s", "seed": 1}\n')

        ret = cli_main(
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

    def test_cli_invalid_output_type(self, tmp_path, episodes_jsonl_file):
        """Test CLI with invalid output type."""
        from scripts.replay_episode_figure import main

        ret = main(
            [
                "--episodes",
                str(episodes_jsonl_file),
                "--episode-id",
                "test_ep_001",
                "--outputs",
                "invalid_type",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert ret == 1

    def test_cli_success(self, tmp_path, episodes_jsonl_file):
        """Test CLI successful execution."""
        from scripts.replay_episode_figure import main

        ret = main(
            [
                "--episodes",
                str(episodes_jsonl_file),
                "--episode-id",
                "test_ep_001",
                "--outputs",
                "trajectory",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert ret == 0

    def test_cli_with_frame_steps(self, tmp_path, episodes_jsonl_file):
        """Test CLI with custom frame steps."""
        from scripts.replay_episode_figure import main

        ret = main(
            [
                "--episodes",
                str(episodes_jsonl_file),
                "--episode-id",
                "test_ep_001",
                "--outputs",
                "filmstrip",
                "--out-dir",
                str(tmp_path / "out"),
                "--frame-steps",
                "0,2",
            ]
        )
        assert ret == 0

    def test_cli_with_format(self, tmp_path, episodes_jsonl_file):
        """Test CLI with PDF format."""
        from scripts.replay_episode_figure import main

        ret = main(
            [
                "--episodes",
                str(episodes_jsonl_file),
                "--episode-id",
                "test_ep_001",
                "--outputs",
                "trajectory",
                "--out-dir",
                str(tmp_path / "out"),
                "--format",
                "pdf",
            ]
        )
        assert ret == 0

    def test_cli_no_determinism_check(self, tmp_path, episodes_jsonl_file):
        """Test CLI with determinism check disabled."""
        from scripts.replay_episode_figure import main

        ret = main(
            [
                "--episodes",
                str(episodes_jsonl_file),
                "--episode-id",
                "test_ep_001",
                "--outputs",
                "trajectory",
                "--out-dir",
                str(tmp_path / "out"),
                "--no-determinism-check",
            ]
        )
        assert ret == 0

    def test_cli_end_to_end_full_workflow(self, tmp_path):
        """End-to-end test: complete workflow from episode row to rendered artifacts.

        This test validates the full command-to-artifact pipeline:
        1. Episode row loading from JSONL
        2. Replay figure generation (all output types)
        3. Artifact file creation and validation
        4. Provenance sidecar completeness
        5. Determinism check execution

        Corresponds to acceptance criterion: "One command maps an episode row to
        replay-derived figure artifacts with determinism check and provenance."
        """
        from scripts.replay_episode_figure import main

        # Create comprehensive episode row with full provenance
        episode_data = {
            "episode_id": "e2e_test_ep",
            "scenario_id": "crossing_scenario",
            "seed": 123,
            "planner": "social_force",
            "campaign_id": "test_campaign_001",
            "config_hash": "abc123def",
            "repo_commit": "a1b2c3d4",
            "final_robot_position": [5.0, 2.5],
            "final_progress": 0.75,
            "success": True,
            "collision": False,
            "replay_steps": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0},
                {"t": 1.0, "x": 1.0, "y": 0.5, "heading": 0.1, "speed": 1.0},
                {"t": 2.0, "x": 2.0, "y": 1.0, "heading": 0.2, "speed": 1.0},
                {"t": 3.0, "x": 3.0, "y": 1.5, "heading": 0.3, "speed": 1.0},
                {"t": 4.0, "x": 4.0, "y": 2.0, "heading": 0.4, "speed": 1.0},
                {"t": 5.0, "x": 5.0, "y": 2.5, "heading": 0.5, "speed": 0.0},
            ],
            "replay_dt": 1.0,
        }

        episodes_jsonl = tmp_path / "episodes.jsonl"
        import json

        with open(episodes_jsonl, "w") as f:
            json.dump(episode_data, f)
            f.write("\n")

        output_dir = tmp_path / "output"

        # Run CLI with all output types
        ret = main(
            [
                "--episodes",
                str(episodes_jsonl),
                "--episode-id",
                "e2e_test_ep",
                "--outputs",
                "still,filmstrip,trajectory",
                "--out-dir",
                str(output_dir),
                "--tolerance-m",
                "0.1",
            ]
        )
        assert ret == 0, "CLI should complete successfully"

        # Verify all expected artifact files exist
        # Note: still is generated at midpoint (len(steps)//2) when no frame_steps provided
        expected_artifacts = [
            "still_3.png",  # Midpoint of 6 steps (indices 0-5)
            "filmstrip.png",
            "trajectory.png",
        ]
        for artifact in expected_artifacts:
            artifact_path = output_dir / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be generated"
            assert artifact_path.stat().st_size > 0, f"Artifact {artifact} should not be empty"

        # Verify provenance sidecar exists and contains all required fields
        provenance_path = output_dir / "replay_provenance.json"
        assert provenance_path.exists(), "Provenance sidecar should be generated"

        with open(provenance_path) as f:
            provenance = json.load(f)

        # Required provenance fields per issue acceptance criteria
        required_fields = [
            "campaign_id",
            "episode_id",
            "scenario_id",
            "seed",
            "determinism_check_status",
            "artifacts",
            "generated_at",
        ]
        for field in required_fields:
            assert field in provenance, f"Provenance must contain {field}"

        # Verify provenance values match input
        assert provenance["episode_id"] == "e2e_test_ep"
        assert provenance["scenario_id"] == "crossing_scenario"
        assert provenance["seed"] == 123
        assert provenance["campaign_id"] == "test_campaign_001"
        assert provenance["determinism_check_status"] in ["pass", "fail", "not_evaluable"]

        # Verify artifacts list matches generated files
        artifact_names = [Path(a["path"]).name for a in provenance["artifacts"]]
        assert "still_3.png" in artifact_names
        assert "filmstrip.png" in artifact_names
        assert "trajectory.png" in artifact_names

        # Verify each artifact has metadata
        for artifact in provenance["artifacts"]:
            assert "type" in artifact
            assert "sha256" in artifact
            assert len(artifact["sha256"]) == 64  # SHA-256 hex digest length

        # Verify caption fragment exists
        caption_path = output_dir / "caption_fragment.tex"
        assert caption_path.exists(), "Caption fragment should be generated"

        with open(caption_path) as f:
            caption_content = f.read()
        assert len(caption_content) > 0, "Caption fragment should not be empty"
        assert "e2e_test_ep" in caption_content or "crossing_scenario" in caption_content


class TestResimulation:
    """Tests for deterministic re-simulation functionality."""

    @pytest.fixture
    def simple_scenario_matrix(self, tmp_path):
        """Create a simple scenario matrix for testing."""
        scenarios = [
            {
                "scenario_id": "test_scenario",
                "robot_start": [0.3, 3.0],
                "robot_goal": [9.7, 3.0],
                "n_agents": 1,
                "flow": "uni",
                "goal_topology": "point",
            }
        ]
        matrix_path = tmp_path / "scenarios.yaml"
        import yaml

        with open(matrix_path, "w") as f:
            yaml.dump(scenarios, f)
        return matrix_path

    def test_resolve_scenario_from_matrix_found(self, simple_scenario_matrix):
        """Test resolving scenario from matrix when found."""
        from robot_sf.benchmark.episode_replay_figure import _resolve_scenario_from_matrix

        scenario = _resolve_scenario_from_matrix("test_scenario", simple_scenario_matrix)
        assert scenario is not None
        assert scenario["scenario_id"] == "test_scenario"
        assert scenario["robot_start"] == [0.3, 3.0]

    def test_resolve_scenario_from_matrix_id_key(self, tmp_path):
        """Scenario matrices use id while episode rows use scenario_id."""
        import yaml

        from robot_sf.benchmark.episode_replay_figure import _resolve_scenario_from_matrix

        matrix_path = tmp_path / "scenarios.yaml"
        scenarios = [
            {
                "id": "test_scenario",
                "robot_start": [0.3, 3.0],
                "robot_goal": [9.7, 3.0],
                "n_agents": 0,
            }
        ]
        with open(matrix_path, "w") as f:
            yaml.dump(scenarios, f)

        scenario = _resolve_scenario_from_matrix("test_scenario", matrix_path)

        assert scenario is not None
        assert scenario["id"] == "test_scenario"

    def test_resolve_scenario_from_matrix_not_found(self, simple_scenario_matrix):
        """Test resolving scenario from matrix when not found."""
        from robot_sf.benchmark.episode_replay_figure import _resolve_scenario_from_matrix

        scenario = _resolve_scenario_from_matrix("nonexistent", simple_scenario_matrix)
        assert scenario is None

    def test_resolve_scenario_from_matrix_file_not_found(self, tmp_path):
        """Test resolving scenario from non-existent matrix file."""
        from robot_sf.benchmark.episode_replay_figure import _resolve_scenario_from_matrix

        with pytest.raises(FileNotFoundError, match="Scenario matrix not found"):
            _resolve_scenario_from_matrix("test", tmp_path / "nonexistent.yaml")

    def test_resimulate_episode_basic(self, simple_scenario_matrix):
        """Test basic episode re-simulation."""
        from robot_sf.benchmark.episode_replay_figure import (
            EpisodeRow,
            _resimulate_episode,
        )

        episode_row = EpisodeRow(
            episode_id="test_ep",
            scenario_id="test_scenario",
            seed=42,
            algo="simple_policy",
        )

        scenario = {
            "scenario_id": "test_scenario",
            "robot_start": [0.3, 3.0],
            "robot_goal": [9.7, 3.0],
            "n_agents": 1,
            "flow": "uni",
            "goal_topology": "point",
        }

        replay = _resimulate_episode(episode_row, scenario, horizon=10, dt=0.1)

        assert replay.episode_id == "test_ep"
        assert replay.scenario_id == "test_scenario"
        assert len(replay.steps) > 1
        assert replay.dt == 0.1

        # Check initial step
        assert replay.steps[0].t == 0.0
        assert replay.steps[0].x == pytest.approx(0.3)
        assert replay.steps[0].y == pytest.approx(3.0)

        # Check that robot moves toward goal
        final_x = replay.steps[-1].x
        assert final_x > 0.3  # Should have moved toward goal at x=9.7

    def test_resimulate_episode_with_pedestrians(self, simple_scenario_matrix):
        """Test re-simulation records pedestrian positions."""
        from robot_sf.benchmark.episode_replay_figure import (
            EpisodeRow,
            _resimulate_episode,
        )

        episode_row = EpisodeRow(
            episode_id="test_ep",
            scenario_id="test_scenario",
            seed=42,
        )

        scenario = {
            "scenario_id": "test_scenario",
            "robot_start": [0.3, 3.0],
            "robot_goal": [9.7, 3.0],
            "n_agents": 2,
            "flow": "uni",
            "goal_topology": "point",
        }

        replay = _resimulate_episode(episode_row, scenario, horizon=5, dt=0.1)

        # Should have pedestrian positions recorded
        for step in replay.steps:
            assert step.ped_positions is not None
            # The scenario generator may generate more than 2 pedestrians based on area
            assert len(step.ped_positions) >= 1

    def test_full_workflow_with_resimulation(self, tmp_path, simple_scenario_matrix):
        """Test full workflow using re-simulation when replay_steps absent."""
        from robot_sf.benchmark.episode_replay_figure import (
            EpisodeRow,
            replay_episode_and_generate_figures,
        )

        # Create an episode row WITHOUT replay_steps
        row_dict = {
            "episode_id": "test_resim_ep",
            "scenario_id": "test_scenario",
            "seed": 42,
            "final_robot_position": [5.0, 3.0],  # Expected approximate position
        }
        episode_row = EpisodeRow.from_dict(row_dict)

        # Create a dummy episodes JSONL file
        episodes_jsonl = tmp_path / "episodes.jsonl"
        import json

        with open(episodes_jsonl, "w") as f:
            json.dump(row_dict, f)
            f.write("\n")

        output_dir = tmp_path / "output"

        result = replay_episode_and_generate_figures(
            episode_row=episode_row,
            outputs=["trajectory"],
            out_dir=output_dir,
            tolerance_m=5.0,  # Large tolerance for simple policy
            episodes_jsonl_path=episodes_jsonl,
            scenario_matrix_path=simple_scenario_matrix,
        )

        assert result["episode_id"] == "test_resim_ep"
        assert result["artifacts_generated"] == 1
        assert Path(result["provenance_sidecar"]).exists()

        # Check that provenance shows resimulation occurred
        with open(result["provenance_sidecar"]) as f:
            provenance = json.load(f)
        assert provenance["resimulated"] is True
        assert provenance["scenario_id"] == "test_scenario"

    def test_full_workflow_resimulation_determinism_pass(self, tmp_path):
        """Test re-simulation determinism check passes with correct endpoint."""
        from robot_sf.benchmark.episode_replay_figure import (
            EpisodeRow,
            replay_episode_and_generate_figures,
        )

        # Create scenario matrix
        scenarios = [
            {
                "scenario_id": "test_scenario",
                "robot_start": [0.3, 3.0],
                "robot_goal": [9.7, 3.0],
                "n_agents": 0,  # No pedestrians for deterministic behavior
                "flow": "uni",
                "goal_topology": "point",
            }
        ]
        matrix_path = tmp_path / "scenarios.yaml"
        import yaml

        with open(matrix_path, "w") as f:
            yaml.dump(scenarios, f)

        # Create episode row with expected final position after the finite 100-step replay horizon.
        row_dict = {
            "episode_id": "test_det_ep",
            "scenario_id": "test_scenario",
            "seed": 42,
            "final_robot_position": [9.417570463519, 3.0],
        }
        episode_row = EpisodeRow.from_dict(row_dict)

        episodes_jsonl = tmp_path / "episodes.jsonl"
        import json

        with open(episodes_jsonl, "w") as f:
            json.dump(row_dict, f)
            f.write("\n")

        output_dir = tmp_path / "output"

        result = replay_episode_and_generate_figures(
            episode_row=episode_row,
            outputs=["trajectory"],
            out_dir=output_dir,
            tolerance_m=0.01,
            episodes_jsonl_path=episodes_jsonl,
            scenario_matrix_path=matrix_path,
        )

        assert result["determinism_check_status"] == "pass"

    def test_resimulation_missing_scenario_matrix_error(self, tmp_path):
        """Test re-simulation fails without scenario matrix."""
        from robot_sf.benchmark.episode_replay_figure import (
            EpisodeRow,
            replay_episode_and_generate_figures,
        )

        # Create episode row WITHOUT replay_steps
        row_dict = {
            "episode_id": "test_ep",
            "scenario_id": "test_scenario",
            "seed": 42,
        }
        episode_row = EpisodeRow.from_dict(row_dict)

        episodes_jsonl = tmp_path / "episodes.jsonl"
        import json

        with open(episodes_jsonl, "w") as f:
            json.dump(row_dict, f)
            f.write("\n")

        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="no replay_steps.*no scenario matrix"):
            replay_episode_and_generate_figures(
                episode_row=episode_row,
                outputs=["trajectory"],
                out_dir=output_dir,
                episodes_jsonl_path=episodes_jsonl,
            )

    def test_resimulation_scenario_not_in_matrix(self, tmp_path):
        """Test re-simulation fails when scenario not in matrix."""
        from robot_sf.benchmark.episode_replay_figure import (
            EpisodeRow,
            replay_episode_and_generate_figures,
        )

        # Create empty scenario matrix
        scenarios = []
        matrix_path = tmp_path / "scenarios.yaml"
        import yaml

        with open(matrix_path, "w") as f:
            yaml.dump(scenarios, f)

        # Create episode row for non-existent scenario
        row_dict = {
            "episode_id": "test_ep",
            "scenario_id": "nonexistent_scenario",
            "seed": 42,
        }
        episode_row = EpisodeRow.from_dict(row_dict)

        episodes_jsonl = tmp_path / "episodes.jsonl"
        import json

        with open(episodes_jsonl, "w") as f:
            json.dump(row_dict, f)
            f.write("\n")

        output_dir = tmp_path / "output"

        # The error can come from either our lookup or the scenario generator
        with pytest.raises(ValueError, match="Scenario.*(not found|missing|config)"):
            replay_episode_and_generate_figures(
                episode_row=episode_row,
                outputs=["trajectory"],
                out_dir=output_dir,
                episodes_jsonl_path=episodes_jsonl,
                scenario_matrix_path=matrix_path,
            )
