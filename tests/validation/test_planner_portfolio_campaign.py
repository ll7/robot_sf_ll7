"""Tests for planner portfolio campaign utilities."""

from __future__ import annotations

import yaml

from scripts.validation import run_planner_portfolio_campaign as campaign


def test_load_grid_parses_valid_variants(tmp_path) -> None:
    """Grid loader should keep only variants with name+algo."""
    payload = {
        "variants": [
            {"name": "risk", "algo": "risk_dwa"},
            {"name": "", "algo": "mppi_social"},
            {"name": "x"},
        ]
    }
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump(payload), encoding="utf-8")

    variants = campaign._load_grid(grid)
    assert len(variants) == 1
    assert variants[0]["algo"] == "risk_dwa"


def test_rank_key_prefers_higher_success() -> None:
    """Ranking should prioritize hard success then global success then clearance."""
    hard_a = campaign.EvalResult(
        "a",
        "risk_dwa",
        "hard",
        1,
        0.4,
        0.0,
        0.0,
        0.2,
        0.3,
        {"success": 1},
        0.5,
        0.0,
        0.01,
        "a",
    )
    global_a = campaign.EvalResult(
        "a",
        "risk_dwa",
        "global",
        1,
        0.6,
        0.0,
        0.0,
        0.1,
        0.2,
        {"success": 1},
        0.2,
        0.0,
        0.01,
        "a",
    )
    hard_b = campaign.EvalResult(
        "b",
        "mppi_social",
        "hard",
        1,
        0.2,
        0.0,
        0.0,
        0.4,
        0.5,
        {"collision": 1},
        2.0,
        0.0,
        0.01,
        "b",
    )
    global_b = campaign.EvalResult(
        "b",
        "mppi_social",
        "global",
        1,
        0.9,
        0.0,
        0.0,
        0.3,
        0.4,
        {"success": 1},
        2.0,
        0.0,
        0.01,
        "b",
    )

    assert campaign._rank_key(hard_a, global_a) > campaign._rank_key(hard_b, global_b)


def test_write_progress_report_writes_json_and_md(tmp_path) -> None:
    """Progress report should emit both JSON and markdown with ranking table."""
    ranked = [
        {
            "candidate": "cand_a",
            "algo": "prediction_planner",
            "hard": {"success_rate": 0.5, "collision_reason_rate": 0.1, "max_steps_rate": 0.3},
            "global": {"success_rate": 0.4, "collision_reason_rate": 0.2, "max_steps_rate": 0.4},
        }
    ]
    json_path, md_path = campaign._write_progress_report(
        output_dir=tmp_path,
        scenario_matrix="configs/scenarios/classic_interactions.yaml",
        hard_seed_manifest="configs/benchmarks/predictive_hard_seeds_v1.yaml",
        portfolio_grid="configs/benchmarks/portfolio_sweep_grid_v1.yaml",
        ranked=ranked,
    )

    assert json_path.exists()
    assert md_path.exists()
    assert "Hard MaxSteps" in md_path.read_text(encoding="utf-8")
