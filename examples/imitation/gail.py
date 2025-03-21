"""
https://imitation.readthedocs.io/en/latest/algorithms/gail.html
"""

import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from loguru import logger
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


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    n_envs = 1  # We need to use 1 for the rollout collection
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    target_episodes = 10  # Reduced for testing

    # For vectorized environments like this, directly create the environment
    def make_env():
        config = EnvSettings()
        config.sim_config.ped_density_by_difficulty = ped_densities
        config.sim_config.difficulty = difficulty
        env = RobotEnv(config)
        return RolloutInfoWrapper(env)

    # Create the environment
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=SEED)
    env = VecMonitor(env)

    # Load the expert model
    try:
        logger.info("Loading expert model...")
        expert = PPO.load("./model/ppo_model_retrained_10m_2025-02-01.zip", env=env)
        logger.info("Expert model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading expert model: {e}")
        logger.warning("Creating a basic model for testing...")
        # Create a basic model if the expert can't be loaded
        expert = PPO("MultiInputPolicy", env, verbose=1)
        expert.learn(total_timesteps=1000)  # Just train it a tiny bit
        logger.info("Basic expert model created.")

    logger.info("Collecting expert rollouts...")

    # Let's try the direct rollout approach first
    try:
        logger.info("Attempting to use rollout() function...")
        rollouts = rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_episodes=target_episodes),
            rng=np.random.default_rng(SEED),
        )
        logger.info(f"Successfully collected {len(rollouts)} rollouts using rollout()")
    except Exception as e:
        logger.warning(f"Error using rollout(): {e}")
        logger.info("Falling back to manual collection...")

        # Try the manual approach
        rollouts = []
        pbar = tqdm(total=target_episodes, desc="Collecting rollouts")

        # Set up sampler
        sampler = rollout.make_sample_until(min_episodes=target_episodes)

        try:
            # We need to use the generate_trajectory function
            while sampler(rollouts):
                try:
                    traj = rollout.generate_trajectory(expert, env, deterministic_policy=False)
                    rollouts.append(traj)
                    pbar.update(1)
                    pbar.set_postfix(ep_length=len(traj), ep_return=float(sum(traj.rews)))
                except Exception as e:
                    logger.error(f"Error generating trajectory: {e}")
                    # Try resetting the environment and continue
                    env.reset()
                    continue

            pbar.close()
            logger.info(f"Successfully collected {len(rollouts)} rollouts manually")
        except Exception as e:
            logger.error(f"Error in manual collection: {e}")
            raise

    # If we still don't have rollouts, we can't continue
    if not rollouts:
        logger.error("Failed to collect any rollouts. Exiting.")
        exit(1)

    # Print stats about the collected rollouts
    logger.info(f"Collected {len(rollouts)} expert rollouts")
    ep_lengths = [len(traj) for traj in rollouts]
    ep_returns = [float(sum(traj.rews)) for traj in rollouts]
    logger.info(
        f"Episode lengths: min={min(ep_lengths)}, mean={np.mean(ep_lengths):.1f}, max={max(ep_lengths)}"
    )
    logger.info(
        f"Episode returns: min={min(ep_returns):.2f}, mean={np.mean(ep_returns):.2f}, max={max(ep_returns):.2f}"
    )

    # Create PPO learner
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

    # Create reward network
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Initialize GAIL
    logger.info("Initializing GAIL trainer...")
    try:
        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=64,
            gen_replay_buffer_capacity=128,
            n_disc_updates_per_round=4,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )
        logger.info("GAIL trainer initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing GAIL: {e}")

        # If we encounter issues, try to debug by inspecting the rollouts
        if rollouts:
            traj = rollouts[0]
            logger.info(
                f"First trajectory info: obs shape={traj.obs.shape}, acts shape={traj.acts.shape}"
            )
            logger.info(f"First trajectory obs type: {type(traj.obs)}")
            logger.info(f"First trajectory acts type: {type(traj.acts)}")

        raise

    # Evaluate pre-training
    logger.info("Evaluating learner before training...")
    learner_rewards_before_training, _ = evaluate_policy(
        learner,
        env,
        10,
        return_episode_rewards=True,
    )

    # Train with GAIL
    logger.info("Training with GAIL...")
    training_timesteps = 5000  # Reduced for testing
    gail_progress_callback = GAILProgressBarCallback(training_timesteps)

    try:
        gail_trainer.train(training_timesteps, callback=gail_progress_callback)
        logger.info("GAIL training completed successfully!")
    except Exception as e:
        logger.error(f"Error during GAIL training: {e}")
        raise

    # Evaluate post-training
    logger.info("Evaluating learner after training...")
    learner_rewards_after_training, _ = evaluate_policy(
        learner,
        env,
        10,
        return_episode_rewards=True,
    )

    # Report results
    logger.info(f"Mean reward before training: {np.mean(learner_rewards_before_training):.2f}")
    logger.info(f"Mean reward after training: {np.mean(learner_rewards_after_training):.2f}")

    # Save the model
    learner.save("./model/gail_model")
    logger.info("Model saved to ./model/gail_model")
