"""Test that --table-tex produces a LaTeX table file.

Uses a tiny synthetic episodes file (1 record) to keep runtime minimal.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_figures_table_tex(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    episodes = tmp_path / "eps.jsonl"
    rec = {
        "episode_id": "e0",
        "scenario_id": "scnA",
        "scenario_params": {"algo": "dummy"},
        "algo": "dummy",
        "seed": 0,
        "metrics": {
            "success": 1.0,
            "time_to_goal_norm": 0.4,
            "collisions": 0.0,
            "comfort_exposure": 0.0,
        },
    }
    episodes.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    out_dir = tmp_path / "figs"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_figures.py",
        "--episodes",
        str(episodes),
        "--out-dir",
        str(out_dir),
        "--dmetrics",
        "collisions,comfort_exposure",
        "--table-metrics",
        "collisions,comfort_exposure",
        "--no-pareto",
        "--table-tex",
    ]
    subprocess.check_call(cmd, env={**os.environ, "MPLBACKEND": "Agg"})

    assert (out_dir / "baseline_table.md").exists()
    tex_file = out_dir / "baseline_table.tex"
    assert tex_file.exists(), "Expected LaTeX table not generated"
    content = tex_file.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in content and "\\end{tabular}" in content
