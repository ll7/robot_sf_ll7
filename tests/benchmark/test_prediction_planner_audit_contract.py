"""Contract tests for prediction-planner audit claims."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)
from robot_sf.benchmark.predictive_planner_config import (
    build_predictive_planner_algo_config,
    load_predictive_planner_algo_config,
)


def test_prediction_planner_readiness_is_experimental_and_checkpoint_dependent() -> None:
    """Prediction planner should remain experimental and require a trained checkpoint."""
    spec = get_algorithm_readiness("prediction_planner")
    assert spec is not None
    assert spec.canonical_name == "prediction_planner"
    assert spec.tier == "experimental"
    assert "RGL-inspired" in spec.note
    assert "trained checkpoint" in spec.note

    allowed = require_algorithm_allowed(
        algo="prediction_planner",
        benchmark_profile="experimental",
        ppo_paper_ready=False,
    )
    assert allowed == spec

    with pytest.raises(ValueError, match="blocked by profile 'baseline-safe'"):
        require_algorithm_allowed(
            algo="prediction_planner",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
        )


def test_prediction_planner_metadata_exposes_adapter_contract() -> None:
    """Benchmark metadata should describe prediction_planner as an adapter-backed learner."""
    meta = enrich_algorithm_metadata(
        algo="prediction_planner",
        metadata={"status": "ok"},
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]

    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "predictive_model_based_adapter"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["supports_native_commands"] is False
    assert planner["supports_adapter_commands"] is True
    assert planner["execution_mode"] == "adapter"
    assert planner["adapter_name"] == "PredictionPlannerAdapter"
    assert planner["robot_kinematics"] == "differential_drive"
    assert planner["adapter_active"] is True


def test_prediction_planner_camera_ready_config_matches_registry_contract() -> None:
    """Canonical config should point at the current registry-backed benchmark model."""
    config = load_predictive_planner_algo_config()
    assert config["predictive_model_id"] == "predictive_proxy_selected_v2_full"
    assert config["predictive_sequence_search_enabled"] is True
    assert config["predictive_sequence_segments"] == 3
    assert config["predictive_sequence_branch_factor"] == 5
    assert config["predictive_phase_logic_enabled"] is True

    registry = yaml.safe_load(
        Path("model/registry.yaml").read_text(encoding="utf-8"),
    )
    model_ids = {entry["model_id"] for entry in registry.get("models", [])}
    assert config["predictive_model_id"] in model_ids


def test_build_predictive_planner_algo_config_prefers_explicit_checkpoint_override() -> None:
    """Runtime checkpoint override should replace registry selection for direct eval paths."""
    config = build_predictive_planner_algo_config(
        checkpoint_path="output/tmp/predictive_planner/checkpoints/test.pt",
        device="cpu",
    )

    assert config["predictive_checkpoint_path"].endswith("test.pt")
    assert config["predictive_device"] == "cpu"
    assert "predictive_model_id" not in config
