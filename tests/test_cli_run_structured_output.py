"""Tests for structured CLI run output and external log-noise controls."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark import cli as bench_cli

if TYPE_CHECKING:
    from pathlib import Path


SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _write_matrix(tmp_path: Path) -> Path:
    """Write a minimal matrix used by CLI parser tests."""
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "cli-structured-smoke",
                    "density": "low",
                    "flow": "uni",
                    "obstacle": "open",
                    "groups": 0.0,
                    "speed_var": "low",
                    "goal_topology": "point",
                    "robot_context": "embedded",
                    "repeats": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    return matrix_path


def _args(matrix: Path, out: Path) -> list[str]:
    """Build baseline argv for `robot_sf_bench run` tests."""
    return [
        "run",
        "--matrix",
        str(matrix),
        "--out",
        str(out),
        "--schema",
        SCHEMA_PATH,
    ]


def test_structured_json_emits_summary_event(tmp_path: Path, monkeypatch, capsys) -> None:
    """`--structured-output json` should emit a final summary object."""
    matrix = _write_matrix(tmp_path)
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr(
        bench_cli,
        "run_batch",
        lambda **_kwargs: {
            "total_jobs": 1,
            "written": 1,
            "failures": [],
            "out_path": str(out_path),
        },
    )

    rc = bench_cli.cli_main([*_args(matrix, out_path), "--structured-output", "json"])
    assert rc == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["event"] == "benchmark.run.summary"
    assert payload["exit_code"] == 0
    assert payload["written"] == 1


def test_structured_jsonl_emits_failure_events_then_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """`--structured-output jsonl` should emit per-failure events and final summary."""
    matrix = _write_matrix(tmp_path)
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.setattr(
        bench_cli,
        "run_batch",
        lambda **_kwargs: {
            "total_jobs": 2,
            "written": 0,
            "failures": [
                {"scenario_id": "s1", "seed": 1, "error": "boom"},
                {"scenario_id": "s2", "seed": 2, "error": "boom2"},
            ],
            "out_path": str(out_path),
        },
    )

    rc = bench_cli.cli_main([*_args(matrix, out_path), "--structured-output", "jsonl"])
    assert rc == 2

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0]["event"] == "benchmark.run.failure"
    assert lines[1]["event"] == "benchmark.run.failure"
    assert lines[2]["event"] == "benchmark.run.summary"
    assert lines[2]["exit_code"] == 2


def test_external_log_noise_auto_sets_suppression_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Auto mode should suppress noisy external logs unless DEBUG is requested."""
    matrix = _write_matrix(tmp_path)
    out_path = tmp_path / "episodes.jsonl"

    monkeypatch.delenv("PYGAME_HIDE_SUPPORT_PROMPT", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OPENCV_LOG_LEVEL", raising=False)

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

    rc = bench_cli.cli_main([*_args(matrix, out_path), "--external-log-noise", "auto"])
    assert rc == 0
    assert os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT") == "1"
    assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "2"
    assert os.environ.get("OPENCV_LOG_LEVEL") == "ERROR"
