"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_run_with_snqi_flags(tmp_path: Path, capsys):
    # Minimal scenario matrix YAML
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "cli-run-snqi",
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

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    # Write simple SNQI weights JSON
    weights = {
        "w_success": 1.0,
        "w_time": 0.1,
        "w_collisions": 1.0,
        "w_near": 0.5,
        "w_comfort": 0.2,
        "w_force_exceed": 0.3,
        "w_jerk": 0.1,
        "w_curvature": 0.1,
    }
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(weights), encoding="utf-8")

    # Write simple baseline stats JSON (med/p95 per metric)
    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
        "curvature_mean": {"med": 0.0, "p95": 1.0},
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

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
            "--snqi-weights",
            str(weights_path),
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
    assert "snqi" in rec["metrics"], "Expected metrics.snqi when weights/baseline provided"
