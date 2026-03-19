"""Tests for SoNIC source-harness probing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.probe_sonic_source_harness import _render_markdown, run_probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(repo_root / "Dockerfile", "FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel\n")
    _write(repo_root / "gst_updated" / "requirements.txt", "gym==0.26.0\nnumpy==1.26.4\n")
    _write(repo_root / "Python-RVO2" / "requirements.txt", "Cython==0.21.1\n")
    _write(
        repo_root / "trained_models" / "SoNIC_GST" / "arguments.py",
        """
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='CrowdSimPredRealGST-v0')
    return parser.parse_args()
""",
    )
    _write(
        repo_root / "trained_models" / "SoNIC_GST" / "configs" / "config.py",
        """
class Empty:
    pass

class Config:
    def __init__(self):
        self.env = Empty()
        self.robot = Empty()
        self.humans = Empty()
        self.sim = Empty()
        self.action_space = Empty()
        self.env.use_wrapper = True
        self.robot.policy = 'selfAttn_merge_srnn'
        self.humans.policy = 'orca'
        self.robot.sensor = 'coordinates'
        self.sim.predict_method = 'inferred'
        self.action_space.kinematics = 'holonomic'
""",
    )
    _write(
        repo_root / "trained_models" / "SoNIC_GST" / "checkpoints" / "05207.pt",
        "stub\n",
    )


def test_run_probe_blocks_on_missing_assets(tmp_path: Path) -> None:
    """Missing source files should produce an actionable blocked verdict."""
    report = run_probe(
        tmp_path / "missing_repo", model_name="SoNIC_GST", checkpoint=None, timeout_seconds=1
    )
    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "missing_assets"
    assert "missing_repo/test.py" in report.failure_summary


def test_run_probe_captures_missing_dependency_from_entrypoint(tmp_path: Path) -> None:
    """A failing source entrypoint should surface the missing module in the report."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    _write(
        repo_root / "test.py",
        "raise ModuleNotFoundError(\"No module named 'gym'\")\n",
    )

    report = run_probe(repo_root, model_name="SoNIC_GST", checkpoint="05207.pt", timeout_seconds=5)
    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "source_entrypoint"
    assert report.failure_summary == "missing python dependency: gym"
    assert report.source_contract["action_kinematics"] == "holonomic"
    assert report.training_defaults["env_name"] == "CrowdSimPredRealGST-v0"


def test_run_probe_blocks_cleanly_when_checkpoint_dir_is_empty(tmp_path: Path) -> None:
    """An empty checkpoints dir should produce a blocked report, not a traceback."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    (repo_root / "trained_models" / "SoNIC_GST" / "checkpoints" / "05207.pt").unlink()
    _write(repo_root / "test.py", "print('ok')\n")

    report = run_probe(repo_root, model_name="SoNIC_GST", checkpoint=None, timeout_seconds=5)

    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "missing_checkpoint"
    assert "No .pt checkpoints found" in (report.failure_summary or "")


def test_run_probe_metadata_extraction_failure_does_not_overwrite_probe_result(
    monkeypatch, tmp_path: Path
) -> None:
    """Metadata import failures should not replace the already-determined probe verdict."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    _write(
        repo_root / "test.py",
        "raise ModuleNotFoundError(\"No module named 'gym'\")\n",
    )

    monkeypatch.setattr(
        "scripts.tools.probe_sonic_source_harness._extract_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(2)),
    )
    monkeypatch.setattr(
        "scripts.tools.probe_sonic_source_harness._load_args_defaults",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad args import")),
    )

    report = run_probe(repo_root, model_name="SoNIC_GST", checkpoint="05207.pt", timeout_seconds=5)

    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "source_entrypoint"
    assert report.failure_summary == "missing python dependency: gym"
    assert report.source_contract == {}
    assert report.training_defaults == {}


def test_render_markdown_includes_probe_summary(tmp_path: Path) -> None:
    """Markdown rendering should expose the verdict, invocation, and failure summary."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    _write(repo_root / "test.py", "print('ok')\n")

    report = run_probe(repo_root, model_name="SoNIC_GST", checkpoint="05207.pt", timeout_seconds=5)
    markdown = _render_markdown(report)
    assert "SoNIC Source Harness Probe" in markdown
    assert "trained_models/SoNIC_GST" in markdown
    assert f"`{report.verdict}`" in markdown
    assert "Docker base image" in markdown
