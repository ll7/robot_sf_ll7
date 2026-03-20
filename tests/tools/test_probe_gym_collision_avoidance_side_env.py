"""Tests for the gym-collision-avoidance side-environment probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.tools import probe_gym_collision_avoidance_side_env as probe


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(repo_root / "README.md", "# stub\n")
    _write(
        repo_root / "gym_collision_avoidance" / "experiments" / "src" / "example.py",
        "print('example')\n",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "tests" / "test_collision_avoidance.py",
        "def test_example_script():\n    assert True\n",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "envs" / "policies" / "GA3CCADRLPolicy.py",
        "class GA3CCADRLPolicy: ...\n",
    )


def test_run_probe_requires_side_env_python(tmp_path: Path) -> None:
    """Missing side-env interpreters should fail fast with a concrete error."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    with pytest.raises(FileNotFoundError, match="Side-environment interpreter missing"):
        probe.run_probe(repo_root, tmp_path / "missing-python", timeout_seconds=1)


def test_run_probe_preserves_venv_symlink_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe must execute the venv entrypoint itself, not its resolved base interpreter."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_root = tmp_path / "side" / ".venv" / "bin"
    side_env_root.mkdir(parents=True)
    side_env_python = side_env_root / "python"
    side_env_python.write_text("", encoding="utf-8")

    seen_commands: list[list[str]] = []

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        seen_commands.append(command)
        return probe.CommandResult(
            name=name,
            command=command,
            returncode=0,
            failure_summary=None,
            stdout_tail="ok",
            stderr_tail="",
        )

    monkeypatch.setattr(probe, "_run_command", fake_run)
    probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert seen_commands
    assert all(command[0] == str(side_env_python) for command in seen_commands)


def test_run_probe_reports_tkagg_blocker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """TkAgg backend failures should become the primary blocked verdict."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "python"
    side_env_python.write_text("", encoding="utf-8")

    results = {
        "side_env_versions": probe.CommandResult(
            name="side_env_versions",
            command=["python", "-c", "versions"],
            returncode=0,
            failure_summary=None,
            stdout_tail='{"gym": "0.26.2", "tensorflow": "2.21.0"}',
            stderr_tail="",
        ),
        "learned_policy_import": probe.CommandResult(
            name="learned_policy_import",
            command=["python", "-c", "ga3c"],
            returncode=0,
            failure_summary=None,
            stdout_tail="ga3c_ready",
            stderr_tail="",
        ),
        "upstream_example": probe.CommandResult(
            name="upstream_example",
            command=["python", "example.py"],
            returncode=1,
            failure_summary="upstream macOS visualization path forces TkAgg backend",
            stdout_tail="",
            stderr_tail="ImportError: Failed to import tkagg backend",
        ),
        "pytest_example_collection": probe.CommandResult(
            name="pytest_example_collection",
            command=["python", "-m", "pytest"],
            returncode=1,
            failure_summary="upstream macOS visualization path forces TkAgg backend",
            stdout_tail="",
            stderr_tail="ImportError: Failed to import tkagg backend",
        ),
    }

    monkeypatch.setattr(
        probe,
        "_run_command",
        lambda name, command, cwd, timeout_seconds: results[name],
    )
    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert report.verdict == "source harness still blocked"
    assert report.failure_stage == "upstream_example"
    assert report.failure_summary == "upstream macOS visualization path forces TkAgg backend"


def test_run_probe_marks_success_when_example_and_pytest_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful example and pytest stages should mark the side harness reproducible."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "python"
    side_env_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        probe,
        "_run_command",
        lambda name, command, cwd, timeout_seconds: probe.CommandResult(
            name=name,
            command=command,
            returncode=0,
            failure_summary=None,
            stdout_tail="ok",
            stderr_tail="",
        ),
    )

    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)
    assert report.verdict == "source harness reproducible in side environment"
    assert report.failure_stage is None


def test_detect_failure_summary_prefers_tkagg() -> None:
    """TkAgg-specific failures should be summarized ahead of generic import noise."""
    summary = probe._detect_failure_summary("", "ImportError: Failed to import tkagg backend")
    assert summary == "upstream macOS visualization path forces TkAgg backend"


def test_render_markdown_mentions_narrower_follow_up(tmp_path: Path) -> None:
    """Blocked markdown output should preserve the conservative wrapper recommendation."""
    """Successful example and pytest stages should mark the side harness reproducible."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    report = probe.ProbeReport(
        issue=641,
        repo_root=str(repo_root),
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        side_env_python=str(tmp_path / "python"),
        verdict="source harness still blocked",
        failure_stage="upstream_example",
        failure_summary="upstream macOS visualization path forces TkAgg backend",
        source_contract=probe._extract_source_contract(),
        commands=[],
    )
    markdown = probe._render_markdown(report)
    assert "Verdict: `source harness still blocked`" in markdown
    assert "GA3C-CADRL learned-policy import path now reproduce successfully" in markdown
    assert "A Robot SF wrapper is still not justified" in markdown
