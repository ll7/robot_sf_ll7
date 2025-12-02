"""Schema/key snapshot test for SNQI optimization script.

Validates that the JSON structure produced by `snqi_weight_optimization.py` with a
small deterministic dataset matches a stored snapshot of expected key layouts.

Focuses on key presence / structural contract rather than exact numeric values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from robot_sf.benchmark.snqi.schema import validate_snqi  # type: ignore

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "snqi_optimization_keys_snapshot.json"


def _make_dataset(tmp: Path) -> tuple[Path, Path]:
    """Make dataset.

    Args:
        tmp: Auto-generated placeholder description.

    Returns:
        tuple[Path, Path]: Auto-generated placeholder description.
    """
    episodes_path = tmp / "episodes.jsonl"
    baseline_path = tmp / "baseline.json"
    episodes = [
        {
            "scenario_id": "s1",
            "metrics": {
                "success": 1.0,
                "time_to_goal_norm": 0.5,
                "collisions": 0,
                "near_misses": 1,
                "comfort_exposure": 0.3,
                "force_exceed_events": 0,
                "jerk_mean": 0.15,
            },
        },
        {
            "scenario_id": "s2",
            "metrics": {
                "success": 0.0,
                "time_to_goal_norm": 0.9,
                "collisions": 1,
                "near_misses": 3,
                "comfort_exposure": 0.5,
                "force_exceed_events": 1,
                "jerk_mean": 0.35,
            },
        },
    ]
    with episodes_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 1.0, "p95": 3.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.1, "p95": 0.6},
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return episodes_path, baseline_path


def _run_script(episodes: Path, baseline: Path, out: Path) -> dict:
    """Run script.

    Args:
        episodes: Auto-generated placeholder description.
        baseline: Auto-generated placeholder description.
        out: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    cmd = [
        sys.executable,
        "scripts/snqi_weight_optimization.py",
        "--episodes",
        str(episodes),
        "--baseline",
        str(baseline),
        "--output",
        str(out),
        "--method",
        "grid",
        "--grid-resolution",
        "2",
        "--seed",
        "123",
        "--validate",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=Path.cwd())
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    # Internal validation (already run with --validate but double check)
    validate_snqi(data, kind="optimization", check_finite=True)
    return data


def test_snqi_optimization_schema_snapshot(tmp_path: Path):
    """Test snqi optimization schema snapshot.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    episodes, baseline = _make_dataset(tmp_path)
    out_file = tmp_path / "out.json"
    data = _run_script(episodes, baseline, out_file)

    # Load snapshot and compare key structure
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    top_level_keys = sorted(data.keys())
    assert all(k in top_level_keys for k in snapshot["top_level_keys"]), (
        f"Missing expected top-level keys. Expected superset of {snapshot['top_level_keys']}, got {top_level_keys}"
    )

    # Recommended keys
    rec = data.get("recommended", {})
    for k in snapshot["recommended_keys"]:
        assert k in rec, f"Recommended section missing key {k}"

    meta = data.get("_metadata", {})
    for k in snapshot["metadata_keys_subset"]:
        assert k in meta, f"_metadata missing key {k}"

    summary = data.get("summary", {})
    for k in snapshot["summary_keys_subset"]:
        assert k in summary, f"summary missing key {k}"

    # Ensure baseline_missing_metric_count present even if zero
    assert summary.get("baseline_missing_metric_count", 0) == meta.get(
        "baseline_missing_metric_count",
        0,
    )
