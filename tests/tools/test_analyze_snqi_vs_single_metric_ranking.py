"""Tests for SNQI-vs-single-metric ranking ablation."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from scripts.tools import analyze_snqi_vs_single_metric_ranking

if TYPE_CHECKING:
    from pathlib import Path


def _write_campaign_table(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal campaign_table.csv for the ablation CLI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "planner_key",
        "algo",
        "planner_group",
        "kinematics",
        "benchmark_success",
        "success_mean",
        "collisions_mean",
        "near_misses_mean",
        "snqi_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_analysis_reports_top_ties_without_decisive_winner_change() -> None:
    """A tied single-metric top should not be reported as a decisive winner change."""
    rows = [
        {
            "planner_key": "alpha",
            "algo": "alpha",
            "planner_group": "core",
            "kinematics": "differential_drive",
            "success_mean": 1.0,
            "collisions_mean": 0.0,
            "near_misses_mean": 0.0,
            "snqi_mean": 0.4,
        },
        {
            "planner_key": "zulu",
            "algo": "zulu",
            "planner_group": "core",
            "kinematics": "differential_drive",
            "success_mean": 1.0,
            "collisions_mean": 0.0,
            "near_misses_mean": 0.0,
            "snqi_mean": 0.8,
        },
    ]

    payload = analyze_snqi_vs_single_metric_ranking._build_analysis(rows)

    assert payload["snqi_order"] == ["zulu", "alpha"]
    assert payload["comparisons"]["success_mean"]["order"] == ["alpha", "zulu"]
    assert payload["comparisons"]["success_mean"]["top_tied"] is True
    assert payload["comparisons"]["success_mean"]["winner_agrees_with_snqi"] is False
    assert payload["interpretation"]["decisive_single_metric_winner_disagreement"] is False
    assert payload["interpretation"]["any_single_metric_top_tied"] is True
    assert payload["interpretation"]["verdict"] == "snqi_mostly_consistent"


def test_cli_writes_tie_metadata_and_utf8_artifacts(tmp_path: Path) -> None:
    """CLI output should include top-tie metadata in JSON and Markdown artifacts."""
    campaign_root = tmp_path / "campaign"
    reports_dir = campaign_root / "reports"
    _write_campaign_table(
        reports_dir / "campaign_table.csv",
        [
            {
                "planner_key": "alpha",
                "algo": "alpha",
                "planner_group": "core",
                "kinematics": "differential_drive",
                "benchmark_success": "true",
                "success_mean": 1.0,
                "collisions_mean": 0.0,
                "near_misses_mean": 0.0,
                "snqi_mean": 0.4,
            },
            {
                "planner_key": "zulu",
                "algo": "zulu",
                "planner_group": "core",
                "kinematics": "differential_drive",
                "benchmark_success": "true",
                "success_mean": 1.0,
                "collisions_mean": 0.0,
                "near_misses_mean": 0.0,
                "snqi_mean": 0.8,
            },
        ],
    )

    exit_code = analyze_snqi_vs_single_metric_ranking.main(
        ["--campaign-root", str(campaign_root), "--core-only"]
    )

    assert exit_code == 0
    json_path = reports_dir / "snqi_vs_single_metric_ranking.json"
    md_path = reports_dir / "snqi_vs_single_metric_ranking.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["comparisons"]["success_mean"]["top_tied"] is True
    assert "| Metric | Kendall tau vs SNQI | Spearman rho vs SNQI | Winner agrees | Top tied |" in (
        md_path.read_text(encoding="utf-8")
    )
