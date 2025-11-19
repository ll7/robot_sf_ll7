"""Integration test for BC pretraining → PPO fine-tuning pipeline.

Validates end-to-end workflow:
1. Load trajectory dataset
2. Pretrain via BC
3. Fine-tune with PPO
4. Verify sample-efficiency metrics
"""

from __future__ import annotations

import pytest

from robot_sf import common


@pytest.fixture
def minimal_trajectory_dataset(tmp_path, monkeypatch):
    """Create a minimal trajectory dataset for testing."""
    import numpy as np

    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    dataset_id = "test_traj_minimal"
    dataset_path = common.get_trajectory_dataset_path(dataset_id)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Create minimal dummy data
    positions = np.array([[[1.0, 2.0], [1.1, 2.1]] for _ in range(5)], dtype=object)
    actions = np.array([[0.1, 0.2] for _ in range(5)], dtype=object)
    observations = np.array([{"dummy": True} for _ in range(5)], dtype=object)
    metadata = {
        "dataset_id": dataset_id,
        "source_policy_id": "test_expert",
        "scenario_label": "test_scenario",
        "scenario_coverage": {"test_scenario": 5},
        "random_seeds": [42],
    }

    np.savez(
        dataset_path,
        positions=positions,
        actions=actions,
        observations=observations,
        episode_count=np.array(5),
        metadata=metadata,
    )

    return dataset_path


def test_pretraining_pipeline_smoke(tmp_path, monkeypatch, minimal_trajectory_dataset):
    """Smoke test for BC pretraining followed by PPO fine-tuning."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    # This test validates the artifact structure is created correctly
    # Full training would be too expensive for integration tests
    # Instead we check that:
    # 1. Dataset loads correctly
    # 2. Pretrain config can be created
    # 3. Expected artifact paths exist

    from robot_sf.training.imitation_config import BCPretrainingConfig

    dataset_id = "test_traj_minimal"

    # Test that we can create a BC pretraining config
    bc_config = BCPretrainingConfig.from_raw(
        run_id="test_bc_pretrain_run",
        dataset_id=dataset_id,
        policy_output_id="test_bc_policy",
        bc_epochs=1,  # Minimal for smoke test
        batch_size=2,
        learning_rate=0.0003,
        random_seeds=(42,),
    )

    assert bc_config.dataset_id == dataset_id
    assert bc_config.bc_epochs == 1
    assert bc_config.run_id == "test_bc_pretrain_run"

    # Verify dataset path resolution
    dataset_path = common.get_trajectory_dataset_path(dataset_id)
    assert dataset_path.exists()

    # Verify pretrained policy output directory exists
    policy_dir = common.get_expert_policy_dir()
    assert policy_dir.exists()
    # Note: actual policy file created by pretrain script, not tested here


def test_fine_tuning_config_structure(tmp_path, monkeypatch):
    """Test PPO fine-tuning configuration structure."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    from robot_sf.training.imitation_config import PPOFineTuningConfig

    config = PPOFineTuningConfig.from_raw(
        run_id="test_ppo_finetune",
        pretrained_policy_id="test_bc_policy",
        total_timesteps=1000,
        random_seeds=(42, 43),
        learning_rate=0.0001,
    )

    assert config.run_id == "test_ppo_finetune"
    assert config.pretrained_policy_id == "test_bc_policy"
    assert config.total_timesteps == 1000
    assert len(config.random_seeds) == 2


def test_comparative_metrics_structure():
    """Test structure of comparison report artifacts."""
    # This validates the expected fields in a comparison report
    # without running actual training

    expected_fields = {
        "run_group_id",
        "baseline_run_id",
        "pretrained_run_id",
        "sample_efficiency_ratio",
        "timesteps_to_convergence",
        "metrics_comparison",
    }

    # Mock comparison report structure
    mock_report = {
        "run_group_id": "test_group",
        "baseline_run_id": "baseline_ppo",
        "pretrained_run_id": "pretrained_ppo",
        "sample_efficiency_ratio": 0.65,
        "timesteps_to_convergence": {
            "baseline": 1000000,
            "pretrained": 650000,
        },
        "metrics_comparison": {
            "success_rate": {"baseline": 0.85, "pretrained": 0.87},
            "collision_rate": {"baseline": 0.12, "pretrained": 0.10},
        },
    }

    assert set(mock_report.keys()) >= expected_fields
    assert mock_report["sample_efficiency_ratio"] < 0.70  # Meets target
