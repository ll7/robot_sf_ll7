"""
Multi-extractor training script for comparing feature extraction approaches.

This script trains PPO models with different feature extractors and collects
metrics for statistical comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.feature_extractors.config import (
    FeatureExtractorConfig,
    FeatureExtractorPresets,
)
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.tb_logging import DrivingMetricsCallback


class MultiExtractorTraining:
    """
    Training manager for comparing multiple feature extractors.
    """

    def __init__(
        self,
        output_dir: str = "./results/multi_extractor_training",
        n_envs: int = 8,
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10_000,
        save_freq: int = 50_000,
        n_eval_episodes: int = 10,
        difficulty: int = 2,
        ped_densities: Optional[List[float]] = None,
    ):
        """
        Initialize the multi-extractor training setup.

        Args:
            output_dir: Directory to save results and models
            n_envs: Number of parallel environments
            total_timesteps: Total training timesteps per extractor
            eval_freq: Frequency of evaluation (in timesteps)
            save_freq: Frequency of model saving (in timesteps)
            n_eval_episodes: Number of episodes for evaluation
            difficulty: Environment difficulty level
            ped_densities: List of pedestrian densities by difficulty
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.difficulty = difficulty
        self.ped_densities = ped_densities or [0.01, 0.02, 0.04, 0.08]

        # Results storage
        self.results: Dict[str, Dict] = {}

    def create_env(self):
        """Create environment factory function."""
        def make_env():
            config = EnvSettings()
            config.sim_config.ped_density_by_difficulty = self.ped_densities
            config.sim_config.difficulty = self.difficulty
            return RobotEnv(config)
        return make_env

    def train_with_extractor(
        self,
        extractor_config: FeatureExtractorConfig,
        name: str,
        verbose: int = 1
    ) -> Dict:
        """
        Train a model with a specific feature extractor.

        Args:
            extractor_config: Feature extractor configuration
            name: Name for this training run
            verbose: Verbosity level for training

        Returns:
            Dictionary with training results and metrics
        """
        print(f"\n{'='*60}")
        print(f"Training with {name} feature extractor")
        print(f"Type: {extractor_config.extractor_type.value}")
        print(f"Params: {extractor_config.params}")
        print(f"{'='*60}")

        # Create output directory for this extractor
        extractor_dir = self.output_dir / name
        extractor_dir.mkdir(exist_ok=True)

        # Create environments
        env_factory = self.create_env()
        train_env = make_vec_env(env_factory, n_envs=self.n_envs, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(env_factory, n_envs=1)

        # Get policy kwargs from extractor config
        policy_kwargs = extractor_config.get_policy_kwargs()

        # Create model
        model = PPO(
            "MultiInputPolicy",
            train_env,
            tensorboard_log=str(extractor_dir / "tensorboard"),
            policy_kwargs=policy_kwargs,
            verbose=verbose
        )

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq // self.n_envs,
            save_path=str(extractor_dir / "checkpoints"),
            name_prefix=f"ppo_{name}"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(extractor_dir / "best_model"),
            log_path=str(extractor_dir / "eval_logs"),
            eval_freq=self.eval_freq // self.n_envs,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            render=False
        )

        metrics_callback = DrivingMetricsCallback(self.n_envs)

        callback = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

        # Record training start time
        start_time = time.time()

        try:
            # Train the model
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback,
                progress_bar=True
            )

            training_time = time.time() - start_time

            # Save final model
            model.save(extractor_dir / "final_model")

            # Collect results
            results = {
                "name": name,
                "extractor_type": extractor_config.extractor_type.value,
                "extractor_params": extractor_config.params,
                "training_time": training_time,
                "total_timesteps": self.total_timesteps,
                "final_reward": getattr(eval_callback, 'last_mean_reward', None),
                "best_reward": getattr(eval_callback, 'best_mean_reward', None),
                "n_envs": self.n_envs,
                "completed": True,
                "timestamp": datetime.now().isoformat()
            }

            # Count model parameters
            total_params = sum(p.numel() for p in model.policy.parameters())
            trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

            results.update({
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params)
            })

            print("Training completed successfully!")
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Total parameters: {total_params:,}")
            print(f"Best mean reward: {results.get('best_reward', 'N/A')}")

        except Exception as e:
            print(f"Training failed with error: {e}")
            results = {
                "name": name,
                "extractor_type": extractor_config.extractor_type.value,
                "extractor_params": extractor_config.params,
                "completed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        finally:
            # Clean up environments
            train_env.close()
            eval_env.close()

        return results

    def run_comparison(
        self,
        extractor_configs: Dict[str, FeatureExtractorConfig],
        save_individual_results: bool = True
    ) -> Dict:
        """
        Run training comparison across multiple feature extractors.

        Args:
            extractor_configs: Dictionary mapping names to extractor configs
            save_individual_results: Whether to save individual results

        Returns:
            Complete results dictionary
        """
        print("Starting multi-extractor training comparison")
        print(f"Extractors to test: {list(extractor_configs.keys())}")
        print(f"Total timesteps per extractor: {self.total_timesteps:,}")
        print(f"Output directory: {self.output_dir}")

        comparison_start = time.time()

        for name, config in extractor_configs.items():
            print(f"\nStarting training for {name}...")

            try:
                result = self.train_with_extractor(config, name)
                self.results[name] = result

                if save_individual_results:
                    result_file = self.output_dir / name / "training_results.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)

            except Exception as e:
                print(f"Failed to train {name}: {e}")
                self.results[name] = {
                    "name": name,
                    "completed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # Save complete results
        total_time = time.time() - comparison_start
        complete_results = {
            "comparison_metadata": {
                "total_time": total_time,
                "start_timestamp": datetime.now().isoformat(),
                "total_timesteps_per_extractor": self.total_timesteps,
                "n_envs": self.n_envs,
                "n_eval_episodes": self.n_eval_episodes,
                "difficulty": self.difficulty,
                "ped_densities": self.ped_densities
            },
            "results": self.results
        }

        results_file = self.output_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Comparison completed in {total_time:.2f} seconds")
        print(f"Results saved to: {results_file}")
        self._print_summary()

        return complete_results

    def _print_summary(self):
        """Print a summary of results."""
        print("\nTraining Summary:")
        print(f"{'Extractor':<20} {'Completed':<12} {'Best Reward':<12} {'Parameters':<12} {'Time (s)':<10}")
        print(f"{'-'*76}")

        for name, result in self.results.items():
            completed = "✓" if result.get("completed", False) else "✗"
            best_reward = f"{result.get('best_reward', 'N/A'):>8.3f}" if result.get('best_reward') else "N/A"
            params = f"{result.get('total_parameters', 0):>9,}" if result.get('total_parameters') else "N/A"
            train_time = f"{result.get('training_time', 0):>8.1f}" if result.get('training_time') else "N/A"

            print(f"{name:<20} {completed:<12} {best_reward:<12} {params:<12} {train_time:<10}")


def main():
    """Main function for running multi-extractor training."""
    # Define the extractors to compare
    extractors_to_test = {
        "dynamics_original": FeatureExtractorPresets.dynamics_original(),
        "dynamics_no_conv": FeatureExtractorPresets.dynamics_no_conv(),
        "mlp_small": FeatureExtractorPresets.mlp_small(),
        "mlp_large": FeatureExtractorPresets.mlp_large(),
        "attention_small": FeatureExtractorPresets.attention_small(),
        "lightweight_cnn": FeatureExtractorPresets.lightweight_cnn(),
    }

    # Create training manager
    trainer = MultiExtractorTraining(
        output_dir="./results/feature_extractor_comparison",
        n_envs=8,  # Reduced for faster testing
        total_timesteps=500_000,  # Reduced for faster testing
        eval_freq=10_000,
        save_freq=50_000,
        n_eval_episodes=5,  # Reduced for faster testing
    )

    # Run comparison
    results = trainer.run_comparison(extractors_to_test)

    print("Multi-extractor training comparison completed!")
    return results


if __name__ == "__main__":
    main()
