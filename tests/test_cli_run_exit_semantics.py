"""CLI contract tests for benchmark run exit semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark import cli as bench_cli

if TYPE_CHECKING:
    from pathlib import Path


def _write_minimal_matrix(tmp_path: Path) -> Path:
    """Write a tiny matrix file accepted by the CLI parser."""
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "cli-exit-smoke",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]
    matrix_path.write_text(yaml.safe_dump(scenarios), encoding="utf-8")
    return matrix_path


def _run_args(matrix_path: Path, out_path: Path) -> list[str]:
    """Build a minimal `robot_sf_bench run` argv payload."""
    return [
        "run",
        "--matrix",
        str(matrix_path),
        "--out",
        str(out_path),
        "--schema",
        "robot_sf/benchmark/schemas/episode.schema.v1.json",
    ]


def test_cli_run_returns_non_zero_for_zero_written_with_scheduled_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """All-failed/zero-written runs must return a non-zero code for CI gating."""
    matrix_path = _write_minimal_matrix(tmp_path)
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr(
        bench_cli,
        "run_batch",
        lambda **_kwargs: {
            "total_jobs": 2,
            "written": 0,
            "failures": [{"scenario_id": "s1"}, {"scenario_id": "s2"}],
            "out_path": str(out_path),
        },
    )

    rc = bench_cli.cli_main(_run_args(matrix_path, out_path))
    assert rc == 2


def test_cli_run_allows_resume_noop_without_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resume no-op runs (no scheduled jobs) should still return success."""
    matrix_path = _write_minimal_matrix(tmp_path)
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr(
        bench_cli,
        "run_batch",
        lambda **_kwargs: {
            "total_jobs": 0,
            "written": 0,
            "failures": [],
            "out_path": str(out_path),
        },
    )

    rc = bench_cli.cli_main(_run_args(matrix_path, out_path))
    assert rc == 0


def test_cli_run_allows_partial_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Runs with at least one written episode should return success."""
    matrix_path = _write_minimal_matrix(tmp_path)
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr(
        bench_cli,
        "run_batch",
        lambda **_kwargs: {
            "total_jobs": 3,
            "written": 1,
            "failures": [{"scenario_id": "s2"}, {"scenario_id": "s3"}],
            "out_path": str(out_path),
        },
    )

    rc = bench_cli.cli_main(_run_args(matrix_path, out_path))
    assert rc == 0
