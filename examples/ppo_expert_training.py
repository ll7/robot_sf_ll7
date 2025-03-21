"""
WIP, currently not working!
To accelerate the training of your Proximal Policy Optimization (PPO) algorithm in the CartPole environment using Stable Baselines3 (SB3), you can employ **Behavior Cloning** to pre-train your model on expert demonstrations. This approach allows your agent to learn from high-quality episodes before fine-tuning through reinforcement learning, thereby improving sample efficiency and reducing training time.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

try:
    # Try to import imitation library (you may need to install this with pip)
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    HAS_IMITATION = True
except ImportError:
    logger.warning("The 'imitation' package is not installed. Using manual behavior cloning.")
    HAS_IMITATION = False


def create_expert_model(env_id="CartPole-v1", timesteps=30_000, n_envs=1):
    """
    Create and train an expert PPO model.

    Args:
        env_id (str): The gym environment ID
        timesteps (int): Number of timesteps to train for
        n_envs (int): Number of parallel environments

    Returns:
        PPO: Trained expert model
    """
    # Create vectorized environment
    env = make_vec_env(env_id, n_envs=n_envs)

    # Initialize the model
    expert_model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    logger.info(f"Training expert model on {env_id} for {timesteps} timesteps...")
    expert_model.learn(total_timesteps=timesteps)

    return expert_model, env


def generate_expert_trajectories(model, env, n_trajectories=100):
    """
    Generate expert trajectories using a trained model.

    Args:
        model: The trained expert model
        env: The environment to generate trajectories in
        n_trajectories (int): Number of trajectories to generate

    Returns:
        dict: Expert trajectory data with 'obs' and 'actions' keys
    """
    expert_data = {"obs": [], "actions": [], "rewards": [], "dones": []}

    obs = env.reset()
    episode_rewards = 0
    episodes_completed = 0

    logger.info(f"Generating {n_trajectories} expert trajectories...")

    while episodes_completed < n_trajectories:
        action, _ = model.predict(obs, deterministic=True)

        # Store observation and action
        expert_data["obs"].append(obs.copy())
        expert_data["actions"].append(action.copy())

        # Take a step in the environment
        next_obs, reward, done, info = env.step(action)

        # Store reward and done flag
        expert_data["rewards"].append(reward)
        expert_data["dones"].append(done)

        # Update for next iteration
        obs = next_obs
        episode_rewards += reward[0]

        if done:
            obs = env.reset()
            episodes_completed += 1
            logger.info(
                f"Episode {episodes_completed}/{n_trajectories} completed with reward: {episode_rewards}"
            )
            episode_rewards = 0

    # Convert lists to numpy arrays
    for key in expert_data:
        expert_data[key] = (
            np.concatenate(expert_data[key]) if key != "dones" else np.array(expert_data[key])
        )

    logger.info(f"Generated trajectories with {len(expert_data['obs'])} steps")

    return expert_data


def save_expert_data(expert_data, env_id="CartPole-v1"):
    """
    Save expert data to a properly organized directory.

    Args:
        expert_data (dict): Dictionary containing expert trajectory data
        env_id (str): Environment ID for naming

    Returns:
        str: Path to the saved data file
    """
    # Create directory structure
    base_dir = "examples/expert_models"
    env_dir = os.path.join(base_dir, env_id)
    data_dir = os.path.join(env_dir, "trajectories")
    model_dir = os.path.join(env_dir, "models")

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trajectories data
    data_file = os.path.join(data_dir, f"expert_{env_id}_{timestamp}.npz")
    np.savez(data_file, **expert_data)

    logger.info(f"Expert data saved to {data_file}")

    return data_file


def save_expert_model(model, env_id="CartPole-v1"):
    """
    Save the trained expert model.

    Args:
        model: The trained expert model
        env_id (str): Environment ID for naming

    Returns:
        str: Path to the saved model
    """
    # Create directory structure if it doesn't exist
    base_dir = "examples/expert_models"
    env_dir = os.path.join(base_dir, env_id)
    model_dir = os.path.join(env_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_file = os.path.join(model_dir, f"expert_{env_id}_{timestamp}")
    model.save(model_file)

    logger.info(f"Expert model saved to {model_file}")

    return model_file


def load_expert_data(expert_data_path=None, env_id="CartPole-v1"):
    """
    Load expert data from a file, or find the latest expert data file.

    Args:
        expert_data_path (str, optional): Path to the expert data file
        env_id (str): Environment ID for finding latest data if path not provided

    Returns:
        tuple: (observations, actions, dones) arrays from the expert data
    """
    # If no data path provided, find the latest expert data
    if expert_data_path is None:
        base_dir = Path(f"examples/expert_models/{env_id}/trajectories")
        if not base_dir.exists():
            raise FileNotFoundError(f"No expert data directory found at {base_dir}")

        expert_files = list(base_dir.glob("expert_*.npz"))
        if not expert_files:
            raise FileNotFoundError(f"No expert data files found in {base_dir}")

        # Get the most recent file
        expert_data_path = str(
            sorted(expert_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        )
        logger.info(f"Using latest expert data file: {expert_data_path}")

    # Load expert data
    logger.info(f"Loading expert data from {expert_data_path}")
    expert_data = np.load(expert_data_path)
    observations = expert_data["obs"]
    actions = expert_data["actions"]

    # Process dones
    if "dones" in expert_data:
        # Flatten dones to ensure it's 1D
        dones = np.array(expert_data["dones"]).flatten()
        logger.info(f"Dones shape before flattening: {expert_data['dones'].shape}")
        logger.info(f"Dones shape after flattening: {dones.shape}")
    else:
        # If dones not available, assume only the last step is done
        dones = np.zeros(len(observations), dtype=bool)
        episode_ends = np.where(expert_data.get("episode_ends", [len(observations) - 1]))[0]
        dones[episode_ends] = True

    # Make sure actions are properly shaped
    if len(actions.shape) > 2:
        actions = actions.reshape(len(observations), -1)

    logger.info(f"Observations shape: {observations.shape}")
    logger.info(f"Actions shape: {actions.shape}")

    return observations, actions, dones


def prepare_transitions(observations, actions, dones):
    """
    Prepare transitions for the imitation library.

    Args:
        observations (np.ndarray): Expert observations
        actions (np.ndarray): Expert actions
        dones (np.ndarray): Done flags for each transition

    Returns:
        Transitions: Transitions object for the imitation library
    """
    # Create empty infos array with same length as observations
    infos = [{} for _ in range(len(observations))]

    # Calculate next_obs
    next_obs = np.zeros_like(observations)
    for i in range(len(observations) - 1):
        if not dones[i]:
            next_obs[i] = observations[i + 1]
        else:
            # For terminal states, next_obs doesn't matter but shouldn't be all zeros
            next_obs[i] = observations[i]
    # Handle the last observation
    next_obs[-1] = observations[-1]

    logger.info(f"Next_obs shape: {next_obs.shape}")

    # Create transitions object
    transitions = Transitions(
        obs=observations,
        acts=actions,
        infos=infos,
        dones=dones,
        next_obs=next_obs,
    )

    return transitions


def behavior_clone_with_imitation(model, transitions, epochs, batch_size, lr):
    """
    Use the imitation library to perform behavior cloning.

    Args:
        model (PPO): PPO model to train
        transitions (Transitions): Transitions for training
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate

    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Create BC trainer with the PPO policy
        bc_trainer = BC(
            observation_space=model.observation_space,
            action_space=model.action_space,
            policy=model.policy,
            batch_size=batch_size,
            optimizer_kwargs={"lr": lr},
            rng=np.random.RandomState(0),
        )

        # Train using behavior cloning
        bc_trainer.train(transitions, n_epochs=epochs)
        return True
    except Exception as e:
        logger.error(f"Error with imitation library: {e}")
        logger.warning("Falling back to manual behavior cloning implementation")
        return False


def manual_behavior_cloning(
    model, observations, actions, env, epochs, batch_size, lr, gradient_steps
):
    """
    Manually implement behavior cloning when the imitation library is not available.

    Args:
        model (PPO): PPO model to train
        observations (np.ndarray): Expert observations
        actions (np.ndarray): Expert actions
        env: Training environment
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        gradient_steps (int): Number of gradient steps per epoch
    """

    # Create a custom optimizer for the policy
    optimizer = optim.Adam(model.policy.parameters(), lr=lr)

    # Prepare data
    if env.action_space.__class__.__name__ == "Discrete":
        actions = actions.flatten()
        logger.info("Using CrossEntropyLoss for discrete action space")
        loss_fn = nn.CrossEntropyLoss()
    else:
        logger.info("Using MSELoss for continuous action space")
        loss_fn = nn.MSELoss()

    # Create dataset
    dataset = list(zip(observations, actions))

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []

        # Shuffle dataset
        np.random.shuffle(dataset)

        # Process in batches
        n_batches = min(gradient_steps, len(dataset) // batch_size)

        for batch_idx in range(n_batches):
            # Get batch
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]

            # Extract observations and actions
            batch_obs = np.array([x[0] for x in batch])
            batch_actions = np.array([x[1] for x in batch])

            # Manual behavior cloning step
            model.policy.set_training_mode(True)
            obs_tensor = torch.FloatTensor(batch_obs)

            try:
                # Get actions from policy
                actions_tensor = model.policy.forward(obs_tensor)[0]

                # Calculate loss based on action space
                if env.action_space.__class__.__name__ == "Discrete":
                    act_tensor = torch.LongTensor(batch_actions)
                    loss = loss_fn(actions_tensor, act_tensor)
                else:
                    act_tensor = torch.FloatTensor(batch_actions)
                    loss = loss_fn(actions_tensor, act_tensor)

                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            except Exception as e:
                logger.error(f"Error during manual behavior cloning: {e}")
                logger.error(
                    "Consider installing the 'imitation' package for better behavior cloning support"
                )
                raise

        # Log progress
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")


def pretrain_ppo_with_behavior_cloning(
    env_id="CartPole-v1",
    expert_data_path=None,
    epochs=10,
    batch_size=64,
    lr=0.0003,
    gradient_steps=100,
):
    """
    Pre-train a PPO model on expert demonstrations using behavior cloning.

    Args:
        env_id (str): The gym environment ID
        expert_data_path (str): Path to the expert trajectory data (.npz file)
            If None, will try to find the latest expert data file
        epochs (int): Number of training epochs for behavior cloning
        batch_size (int): Batch size for training
        lr (float): Learning rate for the optimizer
        gradient_steps (int): Number of gradient steps per epoch

    Returns:
        PPO: Pre-trained model ready for fine-tuning
    """
    global HAS_IMITATION  # Make it clear we're using the global variable

    logger.info("Starting behavior cloning pre-training...")

    # Load expert data
    observations, actions, dones = load_expert_data(expert_data_path, env_id)

    # Create environment and initialize model
    env = make_vec_env(env_id, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr)

    # Attempt to use imitation library if available
    use_imitation = HAS_IMITATION
    if use_imitation:
        logger.info("Using imitation library for behavior cloning")
        transitions = prepare_transitions(observations, actions, dones)
        success = behavior_clone_with_imitation(model, transitions, epochs, batch_size, lr)
        if not success:
            use_imitation = False

    # Fall back to manual implementation if needed
    if not use_imitation:
        logger.info("Using manual behavior cloning implementation")
        manual_behavior_cloning(
            model, observations, actions, env, epochs, batch_size, lr, gradient_steps
        )

    logger.info("Behavior cloning pre-training complete")
    return model


def finetune_pretrained_model(pretrained_model, env_id="CartPole-v1", timesteps=50000, n_envs=1):
    """
    Fine-tune a pre-trained model with PPO reinforcement learning.

    Args:
        pretrained_model: The pre-trained PPO model
        env_id (str): The gym environment ID
        timesteps (int): Number of timesteps to train for
        n_envs (int): Number of parallel environments

    Returns:
        PPO: Fine-tuned model
    """
    # Create vectorized environment
    env = make_vec_env(env_id, n_envs=n_envs)

    # Set the pre-trained policy in a new model with the same environment
    model = PPO("MlpPolicy", env, verbose=1)
    model.policy = pretrained_model.policy
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=model.learning_rate, **model.policy.optimizer_kwargs
    )

    logger.info(f"Fine-tuning model with PPO for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    logger.info("Fine-tuning complete")
    return model


def evaluate_model(model, env_id="CartPole-v1", n_eval_episodes=100):
    """
    Evaluate a trained model.

    Args:
        model: The trained model to evaluate
        env_id (str): The gym environment ID
        n_eval_episodes (int): Number of episodes to evaluate

    Returns:
        float: Mean reward
    """

    # Create environment for evaluation
    eval_env = make_vec_env(env_id, n_envs=1)

    # Evaluate the model
    logger.info(f"Evaluating model over {n_eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)

    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward


def main_with_pretraining():
    """
    Main function that demonstrates the full pipeline:
    1. Train expert model
    2. Generate expert trajectories
    3. Pre-train new model with behavior cloning
    4. Fine-tune model with PPO
    5. Evaluate and save final model
    """
    # Configuration
    env_id = "CartPole-v1"
    expert_training_timesteps = 30_000
    expert_trajectories = 100
    pretraining_epochs = 10
    finetuning_timesteps = 50_000

    # Step 1 & 2: Train expert and generate trajectories
    expert_model, env = create_expert_model(env_id, expert_training_timesteps)
    expert_data = generate_expert_trajectories(expert_model, env, expert_trajectories)

    # Save expert data and model
    expert_data_path = save_expert_data(expert_data, env_id)
    _expert_model_path = save_expert_model(expert_model, env_id)

    # Step 3: Pre-train new model with behavior cloning
    pretrained_model = pretrain_ppo_with_behavior_cloning(
        env_id=env_id, expert_data_path=expert_data_path, epochs=pretraining_epochs
    )

    # Step 4: Fine-tune model with PPO
    finetuned_model = finetune_pretrained_model(
        pretrained_model, env_id=env_id, timesteps=finetuning_timesteps
    )

    # Step 5: Evaluate final model
    mean_reward = evaluate_model(finetuned_model, env_id)
    logger.info(f"Mean reward after fine-tuning: {mean_reward:.2f}")

    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"examples/expert_models/{env_id}"
    model_dir = os.path.join(base_dir, "pretrained_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"pretrained_{env_id}_{timestamp}")
    finetuned_model.save(model_path)

    logger.info(f"Final model saved to {model_path}")
    logger.info("Full training pipeline completed successfully!")


if __name__ == "__main__":
    # Use the new main function with pretraining
    main_with_pretraining()
