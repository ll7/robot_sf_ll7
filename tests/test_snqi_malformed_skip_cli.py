"""CLI test verifying malformed JSONL line skip counting appears in output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(30)
@pytest.mark.parametrize("script_kind", ["optimization", "recompute"])
def test_snqi_cli_malformed_skip_count(tmp_path: Path, script_kind: str):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        script_kind: TODO docstring.
    """
    episodes_path = tmp_path / "episodes.jsonl"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "out.json"

    # Craft file with malformed + valid lines
    lines = [
        "{",  # malformed
        "not json",  # malformed
        json.dumps(
            {
                "scenario_id": "ok1",
                "metrics": {
                    "success": 1.0,
                    "time_to_goal_norm": 0.5,
                    "collisions": 0,
                    "near_misses": 1,
                    "comfort_exposure": 0.2,
                    "force_exceed_events": 0,
                    "jerk_mean": 0.1,
                },
            },
        ),
        json.dumps(
            {
                "scenario_id": "ok2",
                "metrics": {
                    "success": 0.0,
                    "time_to_goal_norm": 0.9,
                    "collisions": 1,
                    "near_misses": 3,
                    "comfort_exposure": 0.4,
                    "force_exceed_events": 1,
                    "jerk_mean": 0.3,
                },
            },
        ),
    ]
    episodes_path.write_text("\n".join(lines), encoding="utf-8")

    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 1.0, "p95": 3.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.1, "p95": 0.6},
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    if script_kind == "optimization":
        cmd = [
            sys.executable,
            "scripts/snqi_weight_optimization.py",
            "--episodes",
            str(episodes_path),
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_path),
            "--method",
            "grid",
            "--grid-resolution",
            "2",
        ]
    else:
        cmd = [
            sys.executable,
            "scripts/recompute_snqi_weights.py",
            "--episodes",
            str(episodes_path),
            "--baseline",
            str(baseline_path),
            "--strategy",
            "balanced",
            "--output",
            str(output_path),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=Path.cwd())
    assert proc.returncode == 0, proc.stderr or proc.stdout

    data = json.loads(output_path.read_text(encoding="utf-8"))

    # Summary and metadata should both contain skipped count = 2
    if script_kind == "optimization":
        skipped_summary = data.get("summary", {}).get("skipped_malformed_lines")
        skipped_meta = data.get("_metadata", {}).get("skipped_malformed_lines")
    else:
        # recompute script currently only stores skipped in metadata if integrated later; for now just ensure success path
        skipped_summary = data.get("summary", {}).get("skipped_malformed_lines")
        skipped_meta = data.get("_metadata", {}).get("skipped_malformed_lines")

    assert skipped_meta == 2, f"Expected 2 skipped lines (metadata), got {skipped_meta}"
    # summary presence optional; if present should match
    if skipped_summary is not None:
        assert skipped_summary == 2
