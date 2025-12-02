"""Module test_cli_run_snqi_from auto-generated docstring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_minimal_matrix(path: Path) -> None:
    """Write minimal matrix.

    Args:
        path: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    scenarios = [
        {
            "id": "cli-run-snqi-from",
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
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def _write_minimal_baseline(path: Path) -> None:
    """Write minimal baseline.

    Args:
        path: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
        "curvature_mean": {"med": 0.0, "p95": 1.0},
    }
    path.write_text(json.dumps(baseline), encoding="utf-8")


def test_cli_run_with_snqi_weights_from(tmp_path: Path, capsys):
    """Test cli run with snqi weights from.

    Args:
        tmp_path: Auto-generated placeholder description.
        capsys: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Prepare matrix and baseline
    matrix_path = tmp_path / "matrix.yaml"
    _write_minimal_matrix(matrix_path)

    baseline_path = tmp_path / "baseline.json"
    _write_minimal_baseline(baseline_path)

    # Create a fake optimization report with recommended weights
    report = {
        "results": {
            "recommended": {
                "weights": {
                    "w_success": 1.0,
                    "w_time": 0.1,
                    "w_collisions": 1.0,
                    "w_near": 0.5,
                    "w_comfort": 0.2,
                    "w_force_exceed": 0.3,
                    "w_jerk": 0.1,
                    "w_curvature": 0.1,
                },
            },
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    out_jsonl = tmp_path / "episodes.jsonl"

    rc = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_jsonl),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--horizon",
            "6",
            "--dt",
            "0.1",
            "--snqi-weights-from",
            str(report_path),
            "--snqi-baseline",
            str(baseline_path),
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"CLI run failed: {cap.err}"

    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "metrics" in rec and isinstance(rec["metrics"], dict)
    assert "snqi" in rec["metrics"], "Expected metrics.snqi when weights-from/baseline provided"
