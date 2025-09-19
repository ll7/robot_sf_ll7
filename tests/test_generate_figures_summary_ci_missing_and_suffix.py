"""Tests for missing CI warning and custom CI suffix handling.

1. Missing CI: Provide summary without *_ci arrays; ensure warning emitted and columns blank.
2. Custom suffix: Use --ci-column-suffix ci95 and confirm column names reflect suffix.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _base_cmd(episodes: Path, out_dir: Path) -> list[str]:
    return [
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
        "collisions",
        "--table-metrics",
        "collisions",
        "--table-stats",
        "mean",
        "--table-include-ci",
        "--table-tex",
    ]


def test_missing_ci_warning(tmp_path: Path):
    episodes = tmp_path / "eps.jsonl"
    episodes.write_text("{}\n", encoding="utf-8")
    # Summary without mean_ci
    summary = {"algoA": {"collisions": {"mean": 0.2}}}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out_dir = tmp_path / "figs_missing"
    cmd = _base_cmd(episodes, out_dir) + ["--table-summary", str(summary_path)]
    # Capture stderr for warning
    proc = subprocess.run(
        cmd, env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, check=True
    )
    assert "Missing CI arrays" in proc.stderr
    md = (out_dir / "baseline_table.md").read_text(encoding="utf-8")
    # Column exists but blank cell after header line
    assert "collisions_mean_ci_low" in md


def test_custom_ci_suffix(tmp_path: Path):
    episodes = tmp_path / "eps.jsonl"
    episodes.write_text("{}\n", encoding="utf-8")
    summary = {"algoA": {"collisions": {"mean": 0.2, "mean_ci": [0.1, 0.3]}}}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out_dir = tmp_path / "figs_suffix"
    cmd = _base_cmd(episodes, out_dir) + [
        "--table-summary",
        str(summary_path),
        "--ci-column-suffix",
        "ci95",
    ]
    subprocess.check_call(cmd, env={**os.environ, "MPLBACKEND": "Agg"})
    md = (out_dir / "baseline_table.md").read_text(encoding="utf-8")
    assert "collisions_mean_ci95_low" in md
    tex = (out_dir / "baseline_table.tex").read_text(encoding="utf-8")
    assert ("collisions_mean_ci95_low" in tex) or ("collisions\\_mean\\_ci95\\_low" in tex)
