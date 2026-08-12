"""Contract tests for prediction-planner audit claims."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)
from robot_sf.benchmark.predictive.predictive_planner_config import (
    build_predictive_planner_algo_config,
    infer_predictive_checkpoint_feature_schema_name,
    load_predictive_planner_algo_config,
)
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    predictive_feature_schema_metadata,
)
from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    save_predictive_checkpoint,
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
        (Path(__file__).resolve().parents[2] / "model" / "registry.yaml").read_text(
            encoding="utf-8",
        ),
    )
    model_ids = {entry["model_id"] for entry in registry.get("models", [])}
    assert config["predictive_model_id"] in model_ids


def test_load_predictive_planner_algo_config_rejects_non_mapping(tmp_path: Path) -> None:
    """Custom predictive planner configs must retain the mapping contract."""
    config_path = tmp_path / "invalid_predictive_config.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a mapping"):
        load_predictive_planner_algo_config(config_path)


def test_infer_predictive_checkpoint_feature_schema_handles_unusable_payloads(
    tmp_path: Path,
) -> None:
    """Checkpoint schema inference should fail closed and support the legacy config field."""
    missing = tmp_path / "missing.pt"
    assert not missing.exists()
    from_missing = build_predictive_planner_algo_config(
        checkpoint_path=missing,
        device=None,
    )
    assert from_missing["predictive_feature_schema_name"] == "predictive_legacy_v1"
    assert from_missing["predictive_device"] == "cpu"

    empty = tmp_path / "empty.pt"
    empty.touch()
    from_empty = build_predictive_planner_algo_config(checkpoint_path=empty)
    assert from_empty["predictive_feature_schema_name"] == "predictive_legacy_v1"

    non_mapping = tmp_path / "non_mapping.pt"
    torch.save(["not-a-mapping"], non_mapping)
    assert infer_predictive_checkpoint_feature_schema_name(non_mapping) is None

    fallback = tmp_path / "fallback.pt"
    torch.save(
        {
            "feature_schema": {"name": ""},
            "config": {"feature_schema_name": "  predictive_custom_v2  "},
        },
        fallback,
    )
    assert infer_predictive_checkpoint_feature_schema_name(fallback) == "predictive_custom_v2"

    no_schema = tmp_path / "no_schema.pt"
    torch.save({"feature_schema": "not-a-mapping", "config": {}}, no_schema)
    assert infer_predictive_checkpoint_feature_schema_name(no_schema) is None


def test_build_predictive_planner_algo_config_preserves_custom_schema_and_applies_overrides(
    tmp_path: Path,
) -> None:
    """Config-path, no-device, and explicit override inputs should compose predictably."""
    config_path = tmp_path / "custom_predictive_config.yaml"
    config_path.write_text(
        "predictive_model_id: custom_model\npredictive_feature_schema_name: config_schema\n",
        encoding="utf-8",
    )

    config = build_predictive_planner_algo_config(
        config_path=config_path,
        device=None,
        overrides={"predictive_device": "cuda", "custom_marker": "covered"},
    )

    assert config["predictive_model_id"] == "custom_model"
    assert config["predictive_feature_schema_name"] == "config_schema"
    assert config["predictive_device"] == "cuda"
    assert config["custom_marker"] == "covered"


def test_prediction_planner_metadata_overrides_expose_search_and_uncertainty_modes() -> None:
    """Map-runner metadata should surface predictive planner mode selection explicitly."""
    from robot_sf.benchmark.map_runner import _prediction_planner_metadata_overrides

    probabilistic = _prediction_planner_metadata_overrides(
        {
            "predictive_uncertainty_mode": "heuristic_gaussian",
            "predictive_risk_sample_count": 5,
            "predictive_risk_objective": "cvar",
            "predictive_mcts_enabled": True,
        },
    )
    assert probabilistic["prediction_mode"] == "probabilistic"
    assert probabilistic["predictive_uncertainty_mode"] == "heuristic_gaussian"
    assert probabilistic["predictive_risk_objective"] == "cvar"
    assert probabilistic["predictive_risk_sample_count"] == 5
    assert probabilistic["predictive_search_mode"] == "mcts_lite"

    deterministic = _prediction_planner_metadata_overrides(
        {
            "predictive_sequence_search_enabled": True,
            "predictive_uncertainty_mode": "deterministic",
            "predictive_risk_sample_count": 1,
        },
    )
    assert deterministic["prediction_mode"] == "deterministic"
    assert deterministic["predictive_uncertainty_mode"] == "deterministic"
    assert deterministic["predictive_risk_sample_count"] == 1
    assert deterministic["predictive_search_mode"] == "sequence_beam"


def test_build_predictive_planner_algo_config_prefers_explicit_checkpoint_override() -> None:
    """Runtime checkpoint override should replace registry selection for direct eval paths."""
    config = build_predictive_planner_algo_config(
        checkpoint_path="output/tmp/predictive_planner/checkpoints/test.pt",
        device="cpu",
    )

    assert config["predictive_checkpoint_path"].endswith("test.pt")
    assert config["predictive_device"] == "cpu"
    assert config["predictive_feature_schema_name"] == "predictive_legacy_v1"
    assert "predictive_model_id" not in config


def test_build_predictive_planner_algo_config_uses_checkpoint_schema(tmp_path: Path) -> None:
    """Runtime checkpoint overrides should carry the checkpoint's feature-schema contract."""
    feature_schema = predictive_feature_schema_metadata(
        model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        ego_conditioning=False,
    )
    model = PredictiveTrajectoryModel(
        PredictiveModelConfig(
            input_dim=int(feature_schema["input_dim"]),
            feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        )
    )
    checkpoint = tmp_path / "obstacle_predictive_model.pt"
    save_predictive_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        epoch=1,
        feature_schema_metadata=feature_schema,
    )

    config = build_predictive_planner_algo_config(checkpoint_path=checkpoint, device="cpu")

    assert config["predictive_checkpoint_path"] == str(checkpoint)
    assert config["predictive_feature_schema_name"] == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA
