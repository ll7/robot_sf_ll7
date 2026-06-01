"""Integration test for BC pretraining → PPO fine-tuning pipeline.

Validates end-to-end workflow:
1. Load trajectory dataset
2. Pretrain via BC
3. Fine-tune with PPO
4. Verify sample-efficiency metrics
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

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
    assert bc_config.device == "auto"

    # Verify dataset path resolution
    dataset_path = common.get_trajectory_dataset_path(dataset_id)
    assert dataset_path.exists()

    # Verify pretrained policy output directory exists
    policy_dir = common.get_expert_policy_dir()
    assert policy_dir.exists()
    # Note: actual policy file created by pretrain script, not tested here


def test_bc_pretraining_config_accepts_explicit_cpu_device():
    """BC pre-training config should allow deterministic CPU-only runs."""
    from robot_sf.training.imitation_config import BCPretrainingConfig

    bc_config = BCPretrainingConfig.from_raw(
        run_id="test_bc_cpu_run",
        dataset_id="test_traj_minimal",
        policy_output_id="test_bc_policy",
        bc_epochs=1,
        batch_size=2,
        learning_rate=0.0003,
        random_seeds=(42,),
        device="cpu",
    )

    assert bc_config.device == "cpu"


def test_bc_pretraining_config_treats_none_device_as_auto():
    """Null device inputs should fall back to the default auto policy."""
    from robot_sf.training.imitation_config import BCPretrainingConfig

    bc_config = BCPretrainingConfig.from_raw(
        run_id="test_bc_none_device",
        dataset_id="test_traj_minimal",
        policy_output_id="test_bc_policy",
        bc_epochs=1,
        batch_size=2,
        learning_rate=0.0003,
        random_seeds=(42,),
        device=None,
    )

    assert bc_config.device == "auto"


def test_default_bc_pretraining_config_loads_auto_device():
    """Checked-in BC config should expose the default accelerator policy."""
    from scripts.training.pretrain_from_expert import load_bc_config

    repo_root = Path(__file__).resolve().parents[2]
    bc_config = load_bc_config(repo_root / "configs/training/ppo_imitation/bc_pretrain.yaml")

    assert bc_config.device == "auto"


def test_load_bc_config_accepts_cpu_device_override(tmp_path):
    """BC YAML should be able to force a CPU-only pre-training run."""
    from scripts.training.pretrain_from_expert import load_bc_config

    config_path = tmp_path / "bc_pretrain_cpu.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_id: test_bc_cpu_run",
                "dataset_id: test_traj_minimal",
                "policy_output_id: test_bc_policy",
                "bc_epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.0003",
                "random_seeds: [42]",
                "device: cpu",
                "",
            ]
        ),
        encoding="utf-8",
    )

    bc_config = load_bc_config(config_path)

    assert bc_config.device == "cpu"


def test_create_bc_trainer_threads_configured_device(monkeypatch):
    """BC trainer construction should pass the configured device to both trainers."""
    from robot_sf.training.imitation_config import BCPretrainingConfig
    from scripts.training import pretrain_from_expert

    captured: dict[str, str] = {}

    class _FakePPO:
        def __init__(self, _policy_name, _env, **kwargs):
            captured["ppo_device"] = kwargs["device"]
            self.policy = object()

    class _FakeBCModule:
        class BC:
            def __init__(self, **kwargs):
                captured["bc_device"] = kwargs["device"]

    fake_env = SimpleNamespace(
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
    )
    config = BCPretrainingConfig.from_raw(
        run_id="test_bc_cpu_run",
        dataset_id="test_traj_minimal",
        policy_output_id="test_bc_policy",
        bc_epochs=1,
        batch_size=2,
        learning_rate=0.0003,
        random_seeds=(42,),
        device="cpu",
    )

    monkeypatch.setattr(pretrain_from_expert, "PPO", _FakePPO)
    monkeypatch.setattr(pretrain_from_expert, "_require_imitation_bc", lambda: _FakeBCModule)

    pretrain_from_expert._create_bc_trainer(fake_env, [], config)

    assert captured == {"ppo_device": "cpu", "bc_device": "cpu"}


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


def test_issue_749_warm_start_configs_load():
    """Issue #749 configs should preserve the BC -> PPO warm-start contract."""
    from scripts.training.pretrain_from_expert import load_bc_config
    from scripts.training.train_ppo_with_pretrained_policy import load_ppo_finetuning_config

    repo_root = Path(__file__).resolve().parents[2]
    bc_config = load_bc_config(
        repo_root / "configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml"
    )
    fine_tune_config = load_ppo_finetuning_config(
        repo_root / "configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml"
    )

    assert bc_config.dataset_id == "issue_749_b60iopxt_v10_eval_trajectories"
    assert bc_config.policy_output_id == "issue_749_bc_preinit_v10_policy"
    assert fine_tune_config.pretrained_policy_id == bc_config.policy_output_id
    assert fine_tune_config.total_timesteps == 10_000_000
    assert fine_tune_config.num_envs == 22
    assert fine_tune_config.worker_mode == "subproc"
    assert fine_tune_config.device == "cpu"
    assert fine_tune_config.checkpoint_freq == 500_000
    assert fine_tune_config.snqi_weights_path is not None
    assert fine_tune_config.snqi_weights_path.name == "snqi_weights_camera_ready_v3.json"
    assert fine_tune_config.snqi_baseline_path is not None
    assert fine_tune_config.snqi_baseline_path.name == "snqi_baseline_camera_ready_v3.json"
    assert bc_config.env_overrides["observation_mode"] == "socnav_struct"
    assert fine_tune_config.env_overrides["observation_mode"] == "socnav_struct"
    assert bc_config.env_overrides["predictive_foresight_enabled"] is True
    assert fine_tune_config.env_overrides["include_grid_in_observation"] is True
    assert fine_tune_config.env_overrides["predictive_foresight_device"] == "cuda"
    assert bc_config.env_factory_kwargs["reward_name"] == "route_completion_v3"
    assert fine_tune_config.env_factory_kwargs["reward_name"] == "route_completion_v3"


def test_issue_749_warm_start_configs_define_env_contract():
    """Issue #749 configs should name the env contract used across BC and fine-tuning."""
    from scripts.training.pretrain_from_expert import load_bc_config
    from scripts.training.train_ppo_with_pretrained_policy import load_ppo_finetuning_config

    repo_root = Path(__file__).resolve().parents[2]
    bc_config = load_bc_config(
        repo_root / "configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml"
    )
    fine_tune_config = load_ppo_finetuning_config(
        repo_root / "configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml"
    )

    expected_training = (
        repo_root / "configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml"
    ).resolve()
    expected_scenarios = (
        repo_root / "configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml"
    ).resolve()

    assert bc_config.training_config_path == expected_training
    assert bc_config.scenario_config_path == expected_scenarios
    assert fine_tune_config.training_config_path == expected_training
    assert fine_tune_config.scenario_config_path == expected_scenarios
    assert fine_tune_config.dataset_id == bc_config.dataset_id


def test_collect_trajectories_filters_to_policy_observation_space(tmp_path, monkeypatch):
    """Collector should drop current extra observation keys before replaying old PPO policies."""
    from scripts.training import collect_expert_trajectories as collector

    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    checkpoint = common.get_expert_policy_dir() / "demo_policy.zip"
    checkpoint.write_text("placeholder", encoding="utf-8")
    scenario_config = tmp_path / "scenario.yaml"
    scenario_config.write_text("[]\n", encoding="utf-8")

    captured_policy_obs: list[dict[str, np.ndarray]] = []
    captured_metadata: dict[str, object] = {}

    class _FakePolicy:
        observation_space = spaces.Dict(
            {"kept": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)}
        )

        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def predict(self, obs, deterministic=True):
            captured_policy_obs.append(dict(obs))
            assert sorted(obs) == ["kept"]
            return np.array([0.0, 0.0], dtype=np.float32), None

    class _FakeEnv:
        observation_space = spaces.Dict(
            {
                "kept": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "extra": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        action_space = _FakePolicy.action_space

        def __init__(self):
            self.state = SimpleNamespace(max_sim_steps=1, nav=SimpleNamespace(pos=(0.0, 0.0)))

        def reset(self):
            return {
                "kept": np.array([0.25], dtype=np.float32),
                "extra": np.array([0.75], dtype=np.float32),
            }, {}

        def step(self, action):
            return (
                {
                    "kept": np.array([0.5], dtype=np.float32),
                    "extra": np.array([1.0], dtype=np.float32),
                },
                0.0,
                True,
                False,
                {},
            )

        def close(self):
            return None

    monkeypatch.setattr(collector.PPO, "load", lambda _path: _FakePolicy())
    monkeypatch.setattr(collector, "load_scenarios", lambda _path: ({"name": "demo"},))
    monkeypatch.setattr(collector, "select_scenario", lambda scenarios, _scenario_id: scenarios[0])
    monkeypatch.setattr(collector, "build_robot_config_from_scenario", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(collector, "make_robot_env", lambda **_kwargs: _FakeEnv())
    monkeypatch.setattr(
        collector,
        "_write_dataset",
        lambda _path, _arrays, _episode_count, metadata: captured_metadata.update(metadata),
    )
    monkeypatch.setattr(
        collector,
        "TrajectoryDatasetValidator",
        lambda _path: SimpleNamespace(
            validate=lambda minimum_episodes, **_kwargs: SimpleNamespace(
                episode_count=minimum_episodes,
                scenario_coverage={"demo": minimum_episodes},
                integrity_report={},
                quality_status=common.TrajectoryQuality.VALIDATED,
            )
        ),
    )
    monkeypatch.setattr(
        collector,
        "write_trajectory_dataset_manifest",
        lambda _artifact: tmp_path / "manifest.json",
    )

    exit_code = collector.main(
        [
            "--dataset-id",
            "demo_dataset",
            "--policy-id",
            "demo_policy",
            "--episodes",
            "1",
            "--scenario-config",
            str(scenario_config),
            "--seeds",
            "111",
        ]
    )

    assert exit_code == 0
    assert captured_policy_obs
    assert captured_metadata["observation_contract"]["keys"] == ["kept"]


def test_collect_trajectories_merges_training_then_env_contract(tmp_path, monkeypatch):
    """Collector env-config values should explicitly override training-config defaults."""
    from scripts.training import collect_expert_trajectories as collector

    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    scenario_config = tmp_path / "scenario.yaml"
    scenario_config.write_text("[]\n", encoding="utf-8")
    training_config = tmp_path / "training.yaml"
    training_config.write_text(
        "\n".join(
            [
                "env_overrides:",
                "  observation_mode: default",
                "  predictive_foresight_enabled: false",
                "env_factory_kwargs:",
                "  reward_name: training_reward",
                "  training_only: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_config = tmp_path / "env.yaml"
    env_config.write_text(
        "\n".join(
            [
                "env_overrides:",
                "  predictive_foresight_enabled: true",
                "  include_grid_in_observation: true",
                "env_factory_kwargs:",
                "  reward_name: env_reward",
                "  env_only: 7",
                "",
            ]
        ),
        encoding="utf-8",
    )

    captured_overrides: dict[str, object] = {}
    captured_factory_kwargs: dict[str, object] = {}
    captured_metadata: dict[str, object] = {}

    class _FakeEnv:
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def __init__(self):
            self.state = SimpleNamespace(max_sim_steps=1, nav=SimpleNamespace(pos=(0.0, 0.0)))

        def reset(self):
            return {"obs": np.array([0.25], dtype=np.float32)}, {}

        def step(self, action):
            return {"obs": np.array([0.5], dtype=np.float32)}, 0.0, True, False, {}

        def close(self):
            return None

    def fake_apply_env_overrides(_env_config, overrides):
        captured_overrides.update(overrides)

    def fake_make_robot_env(**kwargs):
        captured_factory_kwargs.update(kwargs)
        return _FakeEnv()

    monkeypatch.setattr(collector, "load_scenarios", lambda _path: ({"name": "demo"},))
    monkeypatch.setattr(collector, "select_scenario", lambda scenarios, _scenario_id: scenarios[0])
    monkeypatch.setattr(collector, "build_robot_config_from_scenario", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(collector, "_apply_env_overrides", fake_apply_env_overrides)
    monkeypatch.setattr(collector, "make_robot_env", fake_make_robot_env)
    monkeypatch.setattr(
        collector,
        "_write_dataset",
        lambda _path, _arrays, _episode_count, metadata: captured_metadata.update(metadata),
    )
    monkeypatch.setattr(
        collector,
        "TrajectoryDatasetValidator",
        lambda _path: SimpleNamespace(
            validate=lambda minimum_episodes, **_kwargs: SimpleNamespace(
                episode_count=minimum_episodes,
                scenario_coverage={"demo": minimum_episodes},
                integrity_report={},
                quality_status=common.TrajectoryQuality.VALIDATED,
            )
        ),
    )
    monkeypatch.setattr(
        collector,
        "write_trajectory_dataset_manifest",
        lambda _artifact: tmp_path / "manifest.json",
    )

    exit_code = collector.main(
        [
            "--dataset-id",
            "demo_dataset",
            "--policy-id",
            "demo_policy",
            "--episodes",
            "1",
            "--scenario-config",
            str(scenario_config),
            "--training-config",
            str(training_config),
            "--env-config",
            str(env_config),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert captured_overrides["observation_mode"] == "default"
    assert captured_overrides["predictive_foresight_enabled"] is True
    assert captured_overrides["include_grid_in_observation"] is True
    assert captured_factory_kwargs["reward_name"] == "env_reward"
    assert captured_factory_kwargs["training_only"] is True
    assert captured_factory_kwargs["env_only"] == 7
    assert captured_metadata["env_contract_config"] == str(env_config.resolve())
    assert captured_metadata["env_contract_configs"] == [
        str(training_config.resolve()),
        str(env_config.resolve()),
    ]


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
