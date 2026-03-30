"""Tests for the gym-collision-avoidance headless reproduction probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.tools import probe_gym_collision_avoidance_headless_reproduction as probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(repo_root / "README.md", "# stub\n")
    _write(
        repo_root / "gym_collision_avoidance" / "experiments" / "src" / "example.py",
        "print('example')\n",
    )


def test_run_probe_requires_side_env_python(tmp_path: Path) -> None:
    """Missing side-env interpreters should fail fast with a concrete error."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    with pytest.raises(FileNotFoundError, match="Side-environment interpreter missing"):
        probe.run_probe(repo_root, tmp_path / "missing-python", timeout_seconds=1)


def test_run_probe_marks_success_when_final_headless_stage_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The staged probe should mark success when the final no-animation launcher completes."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    results = {
        "headless_tkagg_redirect": probe.CommandResult(
            name="headless_tkagg_redirect",
            command=["python", "-c", "stage1"],
            returncode=1,
            failure_summary="legacy gym passive checker expects numpy.bool8",
            stdout_tail="",
            stderr_tail="AttributeError: module 'numpy' has no attribute 'bool8'",
        ),
        "headless_plus_numpy_bool8_alias": probe.CommandResult(
            name="headless_plus_numpy_bool8_alias",
            command=["python", "-c", "stage2"],
            returncode=1,
            failure_summary="legacy moviepy/imageio ffmpeg download path blocks headless reset animation",
            stdout_tail="",
            stderr_tail="OSError: Unable to download 'ffmpeg-osx-v3.2.4'",
        ),
        "headless_plus_numpy_bool8_alias_no_animation": probe.CommandResult(
            name="headless_plus_numpy_bool8_alias_no_animation",
            command=["python", "-c", "stage3"],
            returncode=0,
            failure_summary=None,
            stdout_tail="All agents finished!\nExperiment over.\n",
            stderr_tail="",
        ),
    }

    monkeypatch.setattr(
        probe,
        "_run_command",
        lambda name, command, cwd, timeout_seconds: results[name],
    )
    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert report.verdict == "headless source harness reproducible"
    assert report.failure_stage is None
    assert report.commands[-1].stdout_tail.endswith("Experiment over.\n")


def test_run_probe_preserves_venv_symlink_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe must execute the venv entrypoint itself, not a resolved base interpreter."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
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


def test_detect_failure_summary_handles_stage_specific_failures() -> None:
    """Known headless blockers should collapse into stable failure summaries."""
    assert (
        probe._detect_failure_summary("", "AttributeError: module 'numpy' has no attribute 'bool8'")
        == "legacy gym passive checker expects numpy.bool8"
    )
    assert (
        probe._detect_failure_summary("", "OSError: Unable to download 'ffmpeg-osx-v3.2.4'")
        == "legacy moviepy/imageio ffmpeg download path blocks headless reset animation"
    )


def test_render_markdown_records_wrapper_justification(tmp_path: Path) -> None:
    """Successful markdown output should explicitly justify the next wrapper/parity step."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    report = probe.ProbeReport(
        issue=659,
        repo_root=str(repo_root),
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        side_env_python=str(tmp_path / "python"),
        verdict="headless source harness reproducible",
        failure_stage=None,
        failure_summary=None,
        source_contract=probe._extract_source_contract(),
        shims=["shim a", "shim b", "shim c"],
        commands=[],
    )
    markdown = probe._render_markdown(report)
    assert "Verdict: `headless source harness reproducible`" in markdown
    assert "a wrapper/parity issue is now justified" in markdown.lower()


def test_main_returns_failure_when_final_stage_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI should exit non-zero when the final staged reproduction remains blocked."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda repo_root, side_env_python, timeout_seconds: probe.ProbeReport(
            issue=659,
            repo_root=str(repo_root),
            repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
            side_env_python=str(side_env_python),
            verdict="still blocked beyond visualization",
            failure_stage="headless_plus_numpy_bool8_alias_no_animation",
            failure_summary="blocked",
            source_contract=probe._extract_source_contract(),
            shims=[],
            commands=[],
        ),
    )

    args = [
        "--repo-root",
        str(repo_root),
        "--side-env-python",
        str(side_env_python),
        "--output-json",
        str(tmp_path / "out.json"),
        "--output-md",
        str(tmp_path / "out.md"),
    ]
    assert probe.main(args) == 1
