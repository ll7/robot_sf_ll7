from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.summary import summarize_to_plots


def _write_sample_jsonl(path: Path) -> None:
    records = [
        {
            "episode_id": "e1",
            "scenario_id": "s1",
            "seed": 0,
            "metrics": {"min_distance": 0.42, "avg_speed": 0.8},
        },
        {
            "episode_id": "e2",
            "scenario_id": "s1",
            "seed": 1,
            "metrics": {"min_distance": 0.31, "avg_speed": 1.2},
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_summary_creates_pngs(tmp_path: Path):
    src = tmp_path / "episodes.jsonl"
    _write_sample_jsonl(src)
    out_dir = tmp_path / "figs"
    outs = summarize_to_plots(src, out_dir)
    # We expect two images (min_distance + avg_speed)
    assert len(outs) == 2
    for p in outs:
        assert Path(p).exists()
