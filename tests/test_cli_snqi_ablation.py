from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _write_episodes(path: Path) -> None:
    recs = [
        {
            "episode_id": "e1",
            "scenario_id": "sc-1",
            "seed": 0,
            "scenario_params": {"algo": "A"},
            "metrics": {"success": 1.0, "time_to_goal_norm": 0.5, "collisions": 1.0},
        },
        {
            "episode_id": "e2",
            "scenario_id": "sc-1",
            "seed": 1,
            "scenario_params": {"algo": "B"},
            "metrics": {"success": 0.8, "time_to_goal_norm": 0.4, "collisions": 0.0},
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_cli_snqi_ablate_md(tmp_path: Path, capsys):
    episodes = tmp_path / "episodes.jsonl"
    _write_episodes(episodes)
    weights = {"w_success": 2.0, "w_time": 1.0, "w_collisions": 2.0}
    baseline = {"collisions": {"med": 0.0, "p95": 2.0}}
    w_path = tmp_path / "weights.json"
    b_path = tmp_path / "baseline.json"
    w_path.write_text(json.dumps(weights), encoding="utf-8")
    b_path.write_text(json.dumps(baseline), encoding="utf-8")

    out = tmp_path / "ablation.md"
    rc = cli_main(
        [
            "snqi-ablate",
            "--in",
            str(episodes),
            "--out",
            str(out),
            "--snqi-weights",
            str(w_path),
            "--snqi-baseline",
            str(b_path),
            "--format",
            "md",
        ],
    )
    capsys.readouterr()
    assert rc == 0
    content = out.read_text(encoding="utf-8")
    assert content.startswith("| Rank | Group | base_mean")
    assert content.endswith("\n")
