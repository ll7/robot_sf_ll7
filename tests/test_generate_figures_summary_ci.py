"""Test table generation from pre-computed summary with CI columns.

Creates a synthetic summary JSON matching expected structure and invokes
generate_figures with --table-summary and --table-include-ci to ensure
CI columns appear in Markdown and LaTeX outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_table_from_summary_with_ci(tmp_path: Path):
    # Minimal episodes (still required for other steps, but table will use summary)
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    episodes = tmp_path / "eps.jsonl"
    episodes.write_text("{}\n", encoding="utf-8")  # dummy line; table not derived from it

    summary = {
        "algoA": {
            "collisions": {"mean": 0.1, "median": 0.0, "p95": 0.3, "mean_ci": [0.05, 0.2]},
            "comfort_exposure": {"mean": 0.0, "median": 0.0, "p95": 0.0, "mean_ci": [0.0, 0.0]},
        },
        "algoB": {
            "collisions": {"mean": 0.2, "median": 0.0, "p95": 0.4, "mean_ci": [0.1, 0.3]},
            "comfort_exposure": {"mean": 0.01, "median": 0.0, "p95": 0.02, "mean_ci": [0.0, 0.02]},
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

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
        "--no-pareto",
        "--dmetrics",
        "collisions",  # keep fast
        "--table-metrics",
        "collisions,comfort_exposure",
        "--table-summary",
        str(summary_path),
        "--table-stats",
        "mean,median",
        "--table-include-ci",
        "--table-tex",
    ]
    subprocess.check_call(cmd, env={**os.environ, "MPLBACKEND": "Agg"})

    md = (out_dir / "baseline_table.md").read_text(encoding="utf-8")
    tex = (out_dir / "baseline_table.tex").read_text(encoding="utf-8")

    # Expect columns for collisions_mean and collisions_mean_ci_low/high
    assert "collisions_mean_ci_low" in md
    assert "collisions_mean_ci_high" in md
    # LaTeX escapes underscores
    assert ("collisions_mean_ci_low" in tex) or ("collisions\\_mean\\_ci\\_low" in tex)
    assert ("collisions_mean_ci_high" in tex) or ("collisions\\_mean\\_ci\\_high" in tex)
