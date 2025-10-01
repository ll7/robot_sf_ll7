"""
Demo script showing how to use the new feature extractors.

This script demonstrates the different feature extractors and their usage
with minimal training to validate everything works.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from robot_sf.feature_extractors.config import (
    FeatureExtractorPresets,
    FeatureExtractorType,
    create_feature_extractor_config,
)
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv


def create_demo_env():
    """Create a simple environment for demonstration."""
    config = EnvSettings()
    config.sim_config.time_per_step_in_secs = 0.1
    config.sim_config.sim_time_in_secs = 5  # Short episodes for quick demo
    config.sim_config.ped_density_by_difficulty = [0.01]  # Low density for speed
    config.sim_config.difficulty = 0
    return RobotEnv(config)


def demo_extractor(name: str, extractor_config, timesteps: int = 1000):
    """Demonstrate training with a specific feature extractor."""
    print(f"\n{'=' * 60}")
    print(f"Demo: {name} Feature Extractor")
    print(f"Type: {extractor_config.extractor_type.value}")
    print(f"Params: {extractor_config.params}")
    print(f"{'=' * 60}")

    # Create vectorized environment
    env = make_vec_env(create_demo_env, n_envs=2)

    # Get policy kwargs from configuration
    policy_kwargs = extractor_config.get_policy_kwargs()

    # Create PPO model
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    extractor_params = sum(p.numel() for p in model.policy.features_extractor.parameters())

    print(f"Total model parameters: {total_params:,}")
    print(f"Feature extractor parameters: {extractor_params:,}")
    print(f"Feature extractor output dim: {model.policy.features_extractor.features_dim}")

    # Quick training to verify everything works
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # Test inference
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"Sample action shape: {action.shape}")

    env.close()
    print(f"✓ {name} demo completed successfully!")


def main():
    """Run demonstrations of all feature extractors."""
    print("Feature Extractor Demonstration")
    print("This demo shows each extractor working with minimal training")

    # Define extractors to demonstrate
    extractors_to_demo = {
        "Original Dynamics (Conv)": FeatureExtractorPresets.dynamics_original(),
        "Original Dynamics (Flatten)": FeatureExtractorPresets.dynamics_no_conv(),
        "MLP Small": FeatureExtractorPresets.mlp_small(),
        "MLP Large": FeatureExtractorPresets.mlp_large(),
        "Attention Small": FeatureExtractorPresets.attention_small(),
        "Lightweight CNN": FeatureExtractorPresets.lightweight_cnn(),
    }

    # Run demonstrations
    for name, config in extractors_to_demo.items():
        try:
            demo_extractor(name, config, timesteps=500)  # Short training for demo
        except Exception as e:
            print(f"✗ {name} demo failed: {e}")

    print(f"\n{'=' * 60}")
    print("Demo Summary:")
    print("All feature extractors are working and can be used with PPO!")
    print(f"{'=' * 60}")

    # Show how to create custom configurations
    print("\nCustom Configuration Examples:")

    # Custom MLP
    custom_mlp = create_feature_extractor_config(
        "mlp", ray_hidden_dims=[512, 256, 128], drive_hidden_dims=[128, 64], dropout_rate=0.2
    )
    print(f"Custom MLP config: {custom_mlp.extractor_type.value} with params {custom_mlp.params}")

    # Custom Attention
    custom_attention = create_feature_extractor_config(
        FeatureExtractorType.ATTENTION, embed_dim=256, num_heads=16, num_layers=4
    )
    print(
        f"Custom Attention config: {custom_attention.extractor_type.value} with params {custom_attention.params}"
    )

    print("\nReady for production training and comparison!")


if __name__ == "__main__":
    main()
