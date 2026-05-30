"""Tests for the scenario seed-sensitivity analyzer."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from scripts.tools.analyze_scenario_seed_sensitivity import (
    build_seed_sensitivity_analysis,
    main,
    select_top_planners,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write CSV fixture rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_campaign(root: Path) -> None:
    """Create a compact campaign fixture with two scenarios and four top planners."""
    _write_csv(
        root / "reports" / "campaign_table.csv",
        [
            _planner("alpha", "0.90", "0.10", "0.40", "1.0"),
            _planner("beta", "0.90", "0.05", "0.50", "1.0"),
            _planner("gamma", "0.80", "0.00", "0.30", "3.0"),
            _planner("delta", "0.70", "0.00", "0.20", "2.0"),
            _planner("failed", "1.00", "0.00", "0.10", "0.0", benchmark_success="false"),
        ],
    )
    rows: list[dict[str, str]] = []
    for scenario_id in ("sensitive", "stable"):
        for seed in (111, 112, 113):
            for planner in ("alpha", "beta", "gamma", "delta"):
                success = "1.0"
                collision = "0.0"
                if scenario_id == "sensitive" and seed == 111:
                    success = "0.0"
                    collision = "1.0"
                rows.append(
                    {
                        "episode_id": f"{scenario_id}-{seed}-{planner}",
                        "scenario_id": scenario_id,
                        "planner_key": planner,
                        "kinematics": "differential_drive",
                        "algo": planner,
                        "seed": str(seed),
                        "repeat_index": "0",
                        "success": success,
                        "collision": collision,
                        "near_miss": "0.0",
                        "time_to_goal": "0.5",
                        "snqi": "0.0",
                    }
                )
    _write_csv(root / "reports" / "seed_episode_rows.csv", rows)


def _planner(
    key: str,
    success: str,
    collision: str,
    time_to_goal: str,
    near_misses: str,
    *,
    benchmark_success: str = "true",
) -> dict[str, str]:
    """Build a campaign_table fixture row."""
    return {
        "planner_key": key,
        "algo": key,
        "planner_group": "core",
        "kinematics": "differential_drive",
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "benchmark_success": benchmark_success,
        "success_mean": success,
        "collisions_mean": collision,
        "near_misses_mean": near_misses,
        "time_to_goal_norm_mean": time_to_goal,
        "snqi_mean": "0.0",
    }


def test_select_top_planners_uses_success_then_safety_tie_breakers(tmp_path: Path) -> None:
    """Planner selection should be deterministic and fail-closed rows should not win."""
    _write_campaign(tmp_path)

    planners = select_top_planners(tmp_path / "reports" / "campaign_table.csv", top_count=4)

    assert [planner.planner_key for planner in planners] == ["beta", "alpha", "gamma", "delta"]


def test_select_top_planners_uses_late_tie_breakers(tmp_path: Path) -> None:
    """Time-to-goal, near misses, and planner key should break deeper ranking ties."""
    _write_csv(
        tmp_path / "reports" / "campaign_table.csv",
        [
            _planner("zeta", "0.90", "0.05", "0.40", "1.0"),
            _planner("alpha", "0.90", "0.05", "0.40", "1.0"),
            _planner("ttg_winner", "0.90", "0.05", "0.30", "5.0"),
            _planner("near_winner", "0.90", "0.05", "0.40", "0.5"),
        ],
    )

    planners = select_top_planners(tmp_path / "reports" / "campaign_table.csv", top_count=4)

    assert [planner.planner_key for planner in planners] == [
        "ttg_winner",
        "near_winner",
        "alpha",
        "zeta",
    ]


def test_build_seed_sensitivity_analysis_classifies_scenarios(tmp_path: Path) -> None:
    """The analyzer should separate shared hard seeds from stable all-success scenarios."""
    _write_campaign(tmp_path)

    analysis = build_seed_sensitivity_analysis(campaign_root=tmp_path, top_planner_count=4)

    rows = {row["scenario_id"]: row for row in analysis["scenario_rows"]}
    assert analysis["classification_counts"] == {
        "seed_sensitive": 1,
        "not_seed_sensitive": 1,
        "inconclusive": 0,
    }
    assert rows["sensitive"]["classification"] == "seed_sensitive"
    assert rows["sensitive"]["hard_seeds"] == [111]
    assert rows["sensitive"]["easy_seeds"] == [112, 113]
    assert rows["stable"]["classification"] == "not_seed_sensitive"
    assert analysis["selected_planners"][0]["planner_key"] == "beta"


def test_build_seed_sensitivity_analysis_marks_missing_cells_inconclusive(
    tmp_path: Path,
) -> None:
    """Missing selected planner/seed cells should be visible instead of treated as success."""
    _write_campaign(tmp_path)
    path = tmp_path / "reports" / "seed_episode_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    filtered = [
        row
        for row in rows
        if not (row["scenario_id"] == "stable" and row["planner_key"] == "delta")
    ]
    _write_csv(path, filtered)

    analysis = build_seed_sensitivity_analysis(campaign_root=tmp_path, top_planner_count=4)

    rows_by_scenario = {row["scenario_id"]: row for row in analysis["scenario_rows"]}
    assert rows_by_scenario["stable"]["classification"] == "inconclusive"
    assert rows_by_scenario["stable"]["missing_cells"] == [
        "111:delta",
        "112:delta",
        "113:delta",
    ]


def test_main_writes_reviewable_outputs(tmp_path: Path) -> None:
    """CLI output should include JSON, scenario CSV, seed CSV, and Markdown artifacts."""
    campaign_root = tmp_path / "campaign"
    output_dir = tmp_path / "out"
    _write_campaign(campaign_root)

    assert main(["--campaign-root", str(campaign_root), "--output-dir", str(output_dir)]) == 0

    payload = json.loads((output_dir / "seed_sensitivity_analysis.json").read_text())
    assert payload["schema_version"] == "scenario-seed-sensitivity.v1"
    assert (output_dir / "scenario_seed_sensitivity.csv").is_file()
    assert (output_dir / "seed_difficulty_summary.csv").is_file()
    assert (
        "Scenario Seed Sensitivity Analysis"
        in (output_dir / "seed_sensitivity_analysis.md").read_text()
    )
