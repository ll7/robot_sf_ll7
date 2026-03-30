"""Contract tests for the hybrid portfolio benchmark comparison config."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_hybrid_portfolio_compare_uses_same_full_predictive_model_as_baseline() -> None:
    """The portfolio comparison should isolate strategy, not predictive-model selection."""
    repo_root = Path(__file__).resolve().parents[2]
    hybrid_config = yaml.safe_load(
        (repo_root / "configs" / "algos" / "hybrid_portfolio_camera_ready.yaml").read_text(
            encoding="utf-8",
        ),
    )
    baseline_config = yaml.safe_load(
        (repo_root / "configs" / "algos" / "prediction_planner_camera_ready.yaml").read_text(
            encoding="utf-8",
        ),
    )

    assert hybrid_config["predictive_model_id"] == "predictive_proxy_selected_v2_full"
    assert hybrid_config["predictive_model_id"] == baseline_config["predictive_model_id"]
