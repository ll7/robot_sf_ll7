"""Tests for CrowdNav HEIGHT source-harness probing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.probe_crowdnav_height_source_harness import _render_markdown, run_probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    """Write text while creating parent directories for fake repositories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    """Create the minimal HEIGHT repository layout required by probe tests."""
    _write(repo_root / "requirements.txt", "gym\nmatplotlib\n")
    _write(
        repo_root / "crowd_nav" / "configs" / "config.py",
        """
class Empty:
    pass

class Config:
    def __init__(self):
        self.env = Empty()
        self.robot = Empty()
        self.sim = Empty()
        self.action_space = Empty()
        self.env.env_name = 'CrowdSim3DTbObs-v0'
        self.env.scenario = 'circle_crossing'
        self.env.mode = 'sim'
        self.robot.policy = 'selfAttn_merge_srnn_lidar'
        self.sim.human_num = 5
        self.sim.static_obs = True
        self.action_space.kinematics = 'unicycle'
""",
    )


def test_run_probe_blocks_on_missing_assets(tmp_path: Path) -> None:
    """Missing source files should produce an actionable blocked verdict."""
    report = run_probe(
        tmp_path / "missing_repo",
        model_dir="trained_models/HEIGHT",
        checkpoint="237400.pt",
        timeout_seconds=1,
    )

    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "missing_assets"
    assert "missing_repo/test.py" in (report.failure_summary or "")


def test_run_probe_captures_missing_dependency_and_checkpoint_status(tmp_path: Path) -> None:
    """A failing source entrypoint should surface missing deps and local checkpoint status."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    _write(repo_root / "test.py", "raise ModuleNotFoundError(\"No module named 'gym'\")\n")

    report = run_probe(
        repo_root,
        model_dir="trained_models/HEIGHT",
        checkpoint="237400.pt",
        timeout_seconds=5,
    )

    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "source_entrypoint"
    assert report.failure_summary == "missing python dependency: gym"
    assert report.checkpoint_status == "missing_local_checkpoint"
    assert report.source_contract["robot_policy"] == "selfAttn_merge_srnn_lidar"
    assert report.source_contract["action_space_kinematics"] == "unicycle"


def test_run_probe_records_requested_metadata(tmp_path: Path) -> None:
    """Probe reports should record issue and repository metadata explicitly."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    _write(repo_root / "test.py", "print('ok')\n")

    report = run_probe(
        repo_root,
        model_dir="trained_models/HEIGHT",
        checkpoint="237400.pt",
        timeout_seconds=5,
        issue=1394,
        repo_remote_url="https://github.com/Shuijing725/CrowdNav_HEIGHT",
    )

    assert report.issue == 1394
    assert report.repo_remote_url == "https://github.com/Shuijing725/CrowdNav_HEIGHT"


def test_render_markdown_includes_probe_summary(tmp_path: Path) -> None:
    """Markdown rendering should expose the verdict, invocation, and checkpoint status."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    _write(repo_root / "test.py", "print('ok')\n")

    report = run_probe(
        repo_root,
        model_dir="trained_models/HEIGHT",
        checkpoint="237400.pt",
        timeout_seconds=5,
    )
    markdown = _render_markdown(report)

    assert "CrowdNav HEIGHT Source Harness Probe" in markdown
    assert "trained_models/HEIGHT" in markdown
    assert "missing_local_checkpoint" in markdown
    assert f"`{report.verdict}`" in markdown
