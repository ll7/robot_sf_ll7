"""
https://imitation.readthedocs.io/en/latest/algorithms/gail.html
"""

import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
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


# Create a tqdm callback for monitoring rollout progress
class TqdmCallback:
    def __init__(self, total_episodes=60, desc="Collecting rollouts"):
        self.pbar = tqdm(total=total_episodes, desc=desc)
        self.episodes_completed = 0

    def __call__(self, trajectory):
        self.episodes_completed += 1
        self.pbar.update(1)
        self.pbar.set_postfix(ep_length=len(trajectory), ep_return=float(sum(trajectory.rews)))
        return False  # Continue collecting

    def close(self):
        self.pbar.close()


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


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    n_envs = 1  # Keeping at 1 for simplicity with the observation space issues
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

    # Create the progress tracking callback
    progress_callback = TqdmCallback(total_episodes=target_episodes)

    try:
        # Use the callback during rollout collection
        rollouts = rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=target_episodes),
            rng=np.random.default_rng(SEED),
            callback=progress_callback,
        )
    finally:
        # Make sure we close the progress bar
        progress_callback.close()

    print(f"Collected {len(rollouts)} expert rollouts")

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
