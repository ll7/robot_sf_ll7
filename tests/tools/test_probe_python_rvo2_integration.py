"""Tests for the Python-RVO2 integration probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools import probe_python_rvo2_integration as probe


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_collects_upstream_and_adapter_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful upstream/example validation should yield a viable prototype verdict."""
    monkeypatch.setattr(probe, "_validate_required_files", lambda root: {"example": "example.py"})
    monkeypatch.setattr(
        probe,
        "_run_upstream_example",
        lambda root: {
            "returncode": 0,
            "stdout_preview": ["Running simulation"],
            "stderr_preview": [],
        },
    )
    monkeypatch.setattr(
        probe,
        "_probe_adapter",
        lambda: {
            "linear_velocity": 0.5,
            "angular_velocity": 0.1,
            "upstream_reference": {"repo_url": "https://example.invalid/rvo2", "commit": "abc"},
            "planner_kinematics": {
                "upstream_command_space": "velocity_vector_xy",
                "benchmark_command_space": "unicycle_vw",
                "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            },
        },
    )

    report = probe.build_report(Path("third_party/python-rvo2"))

    assert report["verdict"] == "viable benchmark prototype"
    assert report["upstream_example"]["returncode"] == 0
    assert report["adapter_probe"]["planner_kinematics"]["projection_policy"] == (
        "heading_safe_velocity_to_unicycle_vw"
    )


def test_build_report_marks_nonzero_example_as_not_yet_viable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing upstream example should block the prototype verdict."""
    monkeypatch.setattr(probe, "_validate_required_files", lambda root: {"example": "example.py"})
    monkeypatch.setattr(
        probe,
        "_run_upstream_example",
        lambda root: {"returncode": 1, "stdout_preview": [], "stderr_preview": ["boom"]},
    )
    monkeypatch.setattr(
        probe,
        "_probe_adapter",
        lambda: {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
            "upstream_reference": {},
            "planner_kinematics": {},
        },
    )

    report = probe.build_report(Path("third_party/python-rvo2"))

    assert report["verdict"] == "not yet viable"


def test_validate_required_files_rejects_missing_inputs(tmp_path: Path) -> None:
    """Missing vendored files should fail fast with a useful error."""
    _write(tmp_path / "README.md", "readme")
    _write(tmp_path / "UPSTREAM.md", "upstream")
    _write(tmp_path / "LICENSE", "license")

    with pytest.raises(FileNotFoundError, match="example.py"):
        probe._validate_required_files(tmp_path)


def test_render_markdown_mentions_projection_policy() -> None:
    """Markdown output should expose the adapter projection contract explicitly."""
    markdown = probe.render_markdown(
        {
            "verdict": "viable benchmark prototype",
            "vendored_root": "third_party/python-rvo2",
            "upstream_example": {
                "returncode": 0,
                "stdout_preview": ["Running simulation"],
                "stderr_preview": [],
            },
            "adapter_probe": {
                "linear_velocity": 0.5,
                "angular_velocity": 0.1,
                "upstream_reference": {
                    "repo_url": "https://example.invalid/rvo2",
                    "commit": "abc",
                    "adapter_boundary": "Use upstream solver first.",
                },
                "planner_kinematics": {
                    "upstream_command_space": "velocity_vector_xy",
                    "benchmark_command_space": "unicycle_vw",
                    "projection_policy": "heading_safe_velocity_to_unicycle_vw",
                },
            },
        }
    )

    assert "Verdict: `viable benchmark prototype`" in markdown
    assert "projection policy: `heading_safe_velocity_to_unicycle_vw`" in markdown
