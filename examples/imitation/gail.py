"""
https://imitation.readthedocs.io/en/latest/algorithms/gail.html
"""

import time
from typing import List

import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from tqdm import tqdm

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import RobotEnv


# Create a callback to track GAIL training progress with tqdm
class GAILProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.current_step = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="GAIL Training")

    def _on_step(self):
        self.current_step += 1
        self.pbar.update(1)
        # You can add more info here if needed
        return True

    def _on_training_end(self):
        self.pbar.close()
        self.pbar = None


# Custom monitoring for rollout collection
def monitor_rollout_collection(
    min_episodes: int = 60,
    max_wait_seconds: int = 3600,  # 1 hour timeout
) -> List[Trajectory]:
    """
    A custom function to collect rollouts with progress monitoring.

    Args:
        min_episodes: Minimum number of episodes to collect
        max_wait_seconds: Maximum time to wait before timing out

    Returns:
        List of collected trajectories
    """
    # This is a dummy placeholder - the actual implementation would be in the main block
    pass


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    n_envs = 1
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    target_episodes = 60

    # Create a wrapper function that already includes the RolloutInfoWrapper
    def make_env():
        config = EnvSettings()
        config.sim_config.ped_density_by_difficulty = ped_densities
        config.sim_config.difficulty = difficulty
        env = RobotEnv(config)
        return RolloutInfoWrapper(env)

    # Use make_vec_env from stable_baselines3
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=SEED)
    env = VecMonitor(env)  # Add monitoring wrapper

    expert = PPO.load("./model/ppo_model_retrained_10m_2025-02-01.zip", env=env)

    print("Collecting expert rollouts...")

    # Create a manual progress bar
    pbar = tqdm(total=target_episodes, desc="Collecting rollouts")
    collected_rollouts = []

    # Set up the sampler
    sampler = rollout.make_sample_until(min_timesteps=None, min_episodes=target_episodes)

    # Call rollout without callback
    start_time = time.time()
    collected_episodes = 0

    # Use the more basic version that lets us monitor ourself
    rollouts = []
    while sampler(rollouts):
        # Generate one trajectory
        traj = rollout.generate_trajectory(expert, env, deterministic_policy=False)
        rollouts.append(traj)

        # Update progress bar
        collected_episodes += 1
        pbar.update(1)
        pbar.set_postfix(
            ep_length=len(traj),
            ep_return=float(sum(traj.rews)),
            time=f"{time.time() - start_time:.1f}s",
        )

    # Close the progress bar
    pbar.close()

    print(f"Collected {len(rollouts)} expert rollouts in {time.time() - start_time:.1f} seconds")

    policy_kwargs = dict(features_extractor_class=DynamicsExtractor)
    learner = PPO(
        env=env,
        policy="MultiInputPolicy",
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=5,
        seed=SEED,
        tensorboard_log="./logs/gail/",
        policy_kwargs=policy_kwargs,
    )

    # Create a simple reward network for dictionary observation spaces
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Use smaller buffer sizes if memory is an issue with dictionary observations
    print("Initializing GAIL trainer...")
    try:
        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=64,  # Reduced from 1024
            gen_replay_buffer_capacity=128,  # Reduced from 512
            n_disc_updates_per_round=4,  # Reduced from 8
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )
    except Exception as e:
        print(f"Error initializing GAIL: {e}")
        # If that doesn't work, we might need a custom ReplayBuffer implementation
        # that better handles dictionary observation spaces
        raise

    # Evaluate the learner before training
    print("Evaluating learner before training...")
    learner_rewards_before_training, _ = evaluate_policy(
        learner,
        env,
        10,  # Reduced from 100 for faster testing
        return_episode_rewards=True,
    )

    # Train the learner with a progress bar
    print("Training with GAIL...")
    training_timesteps = 20000
    gail_progress_callback = GAILProgressBarCallback(training_timesteps)
    gail_trainer.train(training_timesteps, callback=gail_progress_callback)

    print("Evaluating learner after training...")
    learner_rewards_after_training, _ = evaluate_policy(
        learner,
        env,
        10,  # Reduced from 100 for faster testing
        return_episode_rewards=True,
    )

    print(f"Mean reward before training: {np.mean(learner_rewards_before_training):.2f}")
    print(f"Mean reward after training: {np.mean(learner_rewards_after_training):.2f}")

    # Save the trained model
    learner.save("./model/gail_model")
    print("Model saved to ./model/gail_model")
