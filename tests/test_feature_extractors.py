"""
Test suite for alternative feature extractors.

This module tests all new feature extractors to ensure they work correctly
with the robot environment and StableBaselines3 integration.
"""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from robot_sf.feature_extractors import (
    AttentionFeatureExtractor,
    LightweightCNNExtractor,
    MLPFeatureExtractor,
)
from robot_sf.feature_extractors.config import (
    FeatureExtractorConfig,
    FeatureExtractorPresets,
    FeatureExtractorType,
    create_feature_extractor_config,
)
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


class TestFeatureExtractors:
    """Test cases for feature extractors."""

    @pytest.fixture
    def observation_space(self):
        """Create a mock observation space for testing."""
        # Typical observation space from robot environment
        drive_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32)
        rays_space = spaces.Box(low=0, high=10, shape=(5, 64), dtype=np.float32)

        return spaces.Dict({OBS_DRIVE_STATE: drive_state_space, OBS_RAYS: rays_space})

    @pytest.fixture
    def sample_observation(self, observation_space):
        """Create a sample observation for testing."""
        return {
            OBS_DRIVE_STATE: th.randn(2, 5, 5),  # Batch size 2
            OBS_RAYS: th.rand(2, 5, 64) * 10,  # Batch size 2
        }

    def test_mlp_extractor_initialization(self, observation_space):
        """Test MLP feature extractor initialization."""
        extractor = MLPFeatureExtractor(observation_space)

        assert isinstance(extractor, MLPFeatureExtractor)
        assert extractor.features_dim > 0
        assert hasattr(extractor, "ray_extractor")
        assert hasattr(extractor, "drive_state_extractor")

    def test_mlp_extractor_forward(self, observation_space, sample_observation):
        """Test MLP feature extractor forward pass."""
        extractor = MLPFeatureExtractor(observation_space)

        features = extractor(sample_observation)

        assert isinstance(features, th.Tensor)
        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] == extractor.features_dim
        assert not th.isnan(features).any()

    def test_mlp_extractor_custom_params(self, observation_space):
        """Test MLP extractor with custom parameters."""
        extractor = MLPFeatureExtractor(
            observation_space,
            ray_hidden_dims=[256, 128],
            drive_hidden_dims=[64, 32],
            dropout_rate=0.2,
        )

        assert extractor.features_dim > 0
        # Should have more parameters than default
        total_params = sum(p.numel() for p in extractor.parameters())
        assert total_params > 0

    def test_attention_extractor_initialization(self, observation_space):
        """Test attention feature extractor initialization."""
        extractor = AttentionFeatureExtractor(observation_space)

        assert isinstance(extractor, AttentionFeatureExtractor)
        assert extractor.features_dim > 0
        assert hasattr(extractor, "ray_embedding")
        assert hasattr(extractor, "attention_layers")
        assert hasattr(extractor, "drive_state_extractor")

    def test_attention_extractor_forward(self, observation_space, sample_observation):
        """Test attention feature extractor forward pass."""
        extractor = AttentionFeatureExtractor(observation_space)

        features = extractor(sample_observation)

        assert isinstance(features, th.Tensor)
        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] == extractor.features_dim
        assert not th.isnan(features).any()

    def test_attention_extractor_custom_params(self, observation_space):
        """Test attention extractor with custom parameters."""
        extractor = AttentionFeatureExtractor(
            observation_space, embed_dim=128, num_heads=8, num_layers=3, dropout_rate=0.1
        )

        assert extractor.features_dim > 0
        # Check that attention layers are created
        assert len(extractor.attention_layers) == 3
        assert len(extractor.layer_norms) == 3

    def test_lightweight_cnn_extractor_initialization(self, observation_space):
        """Test lightweight CNN feature extractor initialization."""
        extractor = LightweightCNNExtractor(observation_space)

        assert isinstance(extractor, LightweightCNNExtractor)
        assert extractor.features_dim > 0
        assert hasattr(extractor, "ray_extractor")
        assert hasattr(extractor, "drive_state_extractor")

    def test_lightweight_cnn_extractor_forward(self, observation_space, sample_observation):
        """Test lightweight CNN feature extractor forward pass."""
        extractor = LightweightCNNExtractor(observation_space)

        features = extractor(sample_observation)

        assert isinstance(features, th.Tensor)
        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] == extractor.features_dim
        assert not th.isnan(features).any()

    def test_lightweight_cnn_custom_params(self, observation_space):
        """Test lightweight CNN with custom parameters."""
        extractor = LightweightCNNExtractor(
            observation_space, num_filters=[64, 32, 16], kernel_sizes=[7, 5, 3], dropout_rate=0.15
        )

        assert extractor.features_dim > 0
        total_params = sum(p.numel() for p in extractor.parameters())
        assert total_params > 0

    def test_all_extractors_same_interface(self, observation_space, sample_observation):
        """Test that all extractors follow the same interface."""
        extractors = [
            MLPFeatureExtractor(observation_space),
            AttentionFeatureExtractor(observation_space),
            LightweightCNNExtractor(observation_space),
        ]

        for extractor in extractors:
            # All should have features_dim
            assert hasattr(extractor, "features_dim")
            assert extractor.features_dim > 0

            # All should process the same observation format
            features = extractor(sample_observation)
            assert isinstance(features, th.Tensor)
            assert features.shape[0] == 2  # Batch size
            assert features.shape[1] == extractor.features_dim


class TestFeatureExtractorConfig:
    """Test feature extractor configuration system."""

    def test_config_creation(self):
        """Test creating extractor configurations."""
        config = FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.MLP, params={"ray_hidden_dims": [128, 64]}
        )

        assert config.extractor_type == FeatureExtractorType.MLP
        assert config.params["ray_hidden_dims"] == [128, 64]
        assert config.get_extractor_class() == MLPFeatureExtractor

    def test_config_policy_kwargs(self):
        """Test generating policy kwargs from config."""
        config = FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.ATTENTION, params={"embed_dim": 64, "num_heads": 4}
        )

        kwargs = config.get_policy_kwargs()

        assert "features_extractor_class" in kwargs
        assert "features_extractor_kwargs" in kwargs
        assert kwargs["features_extractor_class"] == AttentionFeatureExtractor
        assert kwargs["features_extractor_kwargs"]["embed_dim"] == 64
        assert kwargs["features_extractor_kwargs"]["num_heads"] == 4

    def test_preset_configurations(self):
        """Test predefined preset configurations."""
        presets = [
            FeatureExtractorPresets.dynamics_original(),
            FeatureExtractorPresets.dynamics_no_conv(),
            FeatureExtractorPresets.mlp_small(),
            FeatureExtractorPresets.mlp_large(),
            FeatureExtractorPresets.attention_small(),
            FeatureExtractorPresets.attention_large(),
            FeatureExtractorPresets.lightweight_cnn(),
        ]

        for preset in presets:
            assert isinstance(preset, FeatureExtractorConfig)
            assert isinstance(preset.extractor_type, FeatureExtractorType)
            assert isinstance(preset.params, dict)

    def test_create_config_function(self):
        """Test the create_feature_extractor_config function."""
        # Test with string
        config1 = create_feature_extractor_config("mlp", ray_hidden_dims=[128])
        assert config1.extractor_type == FeatureExtractorType.MLP
        assert config1.params["ray_hidden_dims"] == [128]

        # Test with enum
        config2 = create_feature_extractor_config(FeatureExtractorType.ATTENTION, embed_dim=32)
        assert config2.extractor_type == FeatureExtractorType.ATTENTION
        assert config2.params["embed_dim"] == 32


@pytest.mark.slow
class TestIntegrationWithStableBaselines3:
    """Integration tests with StableBaselines3 and robot environment."""

    def test_mlp_extractor_with_ppo(self):
        """Test MLP extractor integration with PPO."""
        # Create minimal environment
        config = EnvSettings()
        config.sim_config.time_per_step_in_secs = 0.1
        config.sim_config.sim_time_in_secs = 10

        def make_env():
            """TODO docstring. Document this function."""
            return RobotEnv(config)

        env = make_vec_env(make_env, n_envs=1)

        # Test with MLP extractor
        policy_kwargs = {
            "features_extractor_class": MLPFeatureExtractor,
            "features_extractor_kwargs": {"ray_hidden_dims": [64, 32]},
        }

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            gamma=0.95,
            learning_rate=3e-4,
            verbose=0,
        )

        # Test that model can be created without errors
        assert model is not None
        assert hasattr(model.policy, "features_extractor")
        assert isinstance(model.policy.features_extractor, MLPFeatureExtractor)

        # Test a few training steps
        model.learn(total_timesteps=32)

        env.close()

    def test_attention_extractor_with_ppo(self):
        """Test attention extractor integration with PPO."""
        config = EnvSettings()
        config.sim_config.time_per_step_in_secs = 0.02
        config.sim_config.sim_time_in_secs = 2

        def make_env():
            """TODO docstring. Document this function."""
            return RobotEnv(config)

        env = make_vec_env(make_env, n_envs=1)

        policy_kwargs = {
            "features_extractor_class": AttentionFeatureExtractor,
            "features_extractor_kwargs": {"embed_dim": 32, "num_heads": 2},
        }

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            gamma=0.95,
            learning_rate=3e-4,
            verbose=0,
        )

        assert model is not None
        assert isinstance(model.policy.features_extractor, AttentionFeatureExtractor)

        model.learn(total_timesteps=32)
        env.close()

    def test_lightweight_cnn_extractor_with_ppo(self):
        """Test lightweight CNN extractor integration with PPO."""
        config = EnvSettings()
        config.sim_config.time_per_step_in_secs = 0.1
        config.sim_config.sim_time_in_secs = 10

        def make_env():
            """TODO docstring. Document this function."""
            return RobotEnv(config)

        env = make_vec_env(make_env, n_envs=1)

        policy_kwargs = {
            "features_extractor_class": LightweightCNNExtractor,
            "features_extractor_kwargs": {"num_filters": [16, 8]},
        }

        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

        assert model is not None
        assert isinstance(model.policy.features_extractor, LightweightCNNExtractor)

        model.learn(total_timesteps=32)
        env.close()

    def test_config_with_ppo(self):
        """Test using configuration system with PPO."""
        config = EnvSettings()
        config.sim_config.time_per_step_in_secs = 0.1
        config.sim_config.sim_time_in_secs = 10

        def make_env():
            """TODO docstring. Document this function."""
            return RobotEnv(config)

        env = make_vec_env(make_env, n_envs=1)

        # Use preset configuration
        extractor_config = FeatureExtractorPresets.mlp_small()
        policy_kwargs = extractor_config.get_policy_kwargs()

        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

        assert model is not None
        assert isinstance(model.policy.features_extractor, MLPFeatureExtractor)

        model.learn(total_timesteps=32)
        env.close()


class TestParameterCounting:
    """Test parameter counting for different extractors."""

    @pytest.fixture
    def observation_space(self):
        """Create observation space for parameter counting tests."""
        drive_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32)
        rays_space = spaces.Box(low=0, high=10, shape=(5, 64), dtype=np.float32)

        return spaces.Dict({OBS_DRIVE_STATE: drive_state_space, OBS_RAYS: rays_space})

    def test_parameter_counts_different(self, observation_space):
        """Test that different extractors have different parameter counts."""
        extractors = {
            "mlp_small": MLPFeatureExtractor(
                observation_space, ray_hidden_dims=[32], drive_hidden_dims=[16]
            ),
            "mlp_large": MLPFeatureExtractor(
                observation_space, ray_hidden_dims=[256, 128], drive_hidden_dims=[64, 32]
            ),
            "attention_small": AttentionFeatureExtractor(
                observation_space, embed_dim=32, num_layers=1
            ),
            "attention_large": AttentionFeatureExtractor(
                observation_space, embed_dim=128, num_layers=3
            ),
            "lightweight_cnn": LightweightCNNExtractor(observation_space, num_filters=[16, 8]),
        }

        param_counts = {}
        for name, extractor in extractors.items():
            param_counts[name] = sum(p.numel() for p in extractor.parameters())

        # All should have different parameter counts
        assert len(set(param_counts.values())) == len(param_counts)

        # Small versions should have fewer parameters than large versions
        assert param_counts["mlp_small"] < param_counts["mlp_large"]
        assert param_counts["attention_small"] < param_counts["attention_large"]

        print(f"Parameter counts: {param_counts}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
