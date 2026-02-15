"""Integration check: real metrics + replay capture produced by run_full_benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def _read_first_record(path: Path) -> dict:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.

    Returns:
        TODO docstring.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    return {}


def test_replay_and_metrics_present(config_factory, tmp_path):
    """Ensure full-benchmark smoke runs emit replay and core metric artifacts.

    Args:
        config_factory: Fixture that builds orchestrator config objects.
        tmp_path: Temporary output root for the benchmark run.
    """
    cfg = config_factory(
        smoke=True,
        capture_replay=True,
        disable_videos=True,
        horizon_override=15,
        output_root=str(tmp_path / "real_metrics"),
    )
    cfg.smoke_horizon_cap = 15
    manifest = run_full_benchmark(cfg)
    episodes_path = Path(manifest.output_root) / "episodes" / "episodes.jsonl"
    record = _read_first_record(episodes_path)
    assert record, "episode record should be written"
    replay_steps = record.get("replay_steps") or []
    assert len(replay_steps) >= 1
    replay_forces = record.get("replay_ped_forces") or []
    assert len(replay_forces) == len(replay_steps)
    metrics = record.get("metrics") or {}
    assert "time_to_goal" in metrics
    assert "avg_speed" in metrics
    assert metrics.get("success_rate") is not None
    assert "collision_rate" in metrics
    assert record.get("replay_dt") is not None
    assert record.get("replay_map_path")
    assert isinstance(record.get("scenario_params"), dict)
