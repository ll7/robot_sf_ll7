"""Smoke the platform commands copied from rendered Markdown."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DOC = ROOT / "docs/context/issue_2040_artifact_publication_workflow.md"
TRACE_DOC = ROOT / "docs/debug_visualization.md"
CAMPAIGN_FIXTURE = ROOT / "docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06"
TRACE_FIXTURE = (
    ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)


def test_rendered_artifact_publication_snippet_runs_with_local_fixture(tmp_path: Path) -> None:
    """The canonical publication command should work with a disposable output directory."""
    command = _command_from_markdown(
        ARTIFACT_DOC,
        "uv run python scripts/tools/compile_benchmark_artifacts.py ",
    )
    output_dir = tmp_path / "publication_candidates" / "fixture_campaign"
    command = _replace_args(
        command,
        {
            "output/benchmarks/camera_ready/<campaign_id>": str(CAMPAIGN_FIXTURE),
            "output/benchmarks/publication_candidates/<campaign_id>": str(output_dir),
            "<campaign_id>_publication_candidates": "rendered_snippet_fixture",
        },
    )

    result = _run_snippet(command)

    assert result.returncode == 0, _failure(result)
    assert (output_dir / "artifact_catalog.yaml").is_file()
    assert (output_dir / "checksums.sha256").is_file()


def test_rendered_trace_viewer_snippet_runs_with_local_fixture(tmp_path: Path) -> None:
    """The canonical trace-viewer command should create a static viewer from a fixture."""
    command = _command_from_markdown(
        TRACE_DOC,
        "uv run python -m robot_sf.render.trace_viewer ",
    )
    output_dir = tmp_path / "trace_viewer"
    command = _replace_args(
        command,
        {
            "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json": str(
                TRACE_FIXTURE
            ),
            "output/trace_viewer": str(output_dir),
        },
    )

    result = _run_snippet(command)

    assert result.returncode == 0, _failure(result)
    assert (output_dir / "index.html").is_file()
    assert (output_dir / "viewer.js").is_file()
    assert (output_dir / "scene.json").is_file()


def _command_from_markdown(path: Path, anchor: str) -> str:
    """Extract the first shell fence containing a canonical command anchor."""
    markdown = path.read_text(encoding="utf-8")
    fences = re.finditer(
        r"(?ms)^[ \t]*```(?:bash|sh)\s*\n(?P<body>.*?)^[ \t]*```\s*$",
        markdown,
    )
    for fence in fences:
        body = fence.group("body")
        if anchor in body:
            return re.sub(r"\\\n\s*", " ", body).strip()
    raise AssertionError(f"No shell fence found for {anchor!r} in {path}")


def _replace_args(command: str, replacements: dict[str, str]) -> str:
    """Replace documented placeholders with shell-quoted local paths."""
    for source, target in replacements.items():
        command = command.replace(source, shlex.quote(target))
    return command


def _run_snippet(command: str) -> subprocess.CompletedProcess[str]:
    """Run one extracted shell snippet from the repository root."""
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        executable="/bin/bash",
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _failure(result: subprocess.CompletedProcess[str]) -> str:
    """Return bounded command output for an actionable assertion failure."""
    return f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
