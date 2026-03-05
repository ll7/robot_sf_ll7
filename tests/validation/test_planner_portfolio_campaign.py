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
    hard_a = campaign.EvalResult("a", "risk_dwa", "hard", 1, 0.4, 0.0, 0.0, 0.5, 0.0, "a")
    global_a = campaign.EvalResult("a", "risk_dwa", "global", 1, 0.6, 0.0, 0.0, 0.2, 0.0, "a")
    hard_b = campaign.EvalResult("b", "mppi_social", "hard", 1, 0.2, 0.0, 0.0, 2.0, 0.0, "b")
    global_b = campaign.EvalResult("b", "mppi_social", "global", 1, 0.9, 0.0, 0.0, 2.0, 0.0, "b")

    assert campaign._rank_key(hard_a, global_a) > campaign._rank_key(hard_b, global_b)
