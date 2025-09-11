#!/usr/bin/env python3
"""
Demonstration of the new JSONL recording and playback functionality.

This script shows how to use the new per-episode JSONL recording system
and enhanced interactive playback with episode boundaries.

Features demonstrated:
    - JSONL recording with per-episode files
    - Batch episode recording
    - Interactive playback with trajectory clearing
    - Directory-based batch playback
    - Legacy pickle file compatibility
"""

import tempfile
from pathlib import Path

import loguru
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.render.interactive_playback import load_and_play_jsonl_interactively
from robot_sf.render.jsonl_playback import JSONLPlaybackLoader

logger = loguru.logger


def demo_jsonl_recording():
    """Demonstrate JSONL recording with multiple episodes."""
    logger.info("=== JSONL Recording Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Recording to temporary directory: {temp_dir}")
        
        # Create environment with JSONL recording enabled
        env = RobotEnv(
            recording_enabled=True,
            use_jsonl_recording=True,
            recording_dir=temp_dir,
            suite_name="demo",
            scenario_name="multi_episode", 
            algorithm_name="random_walk",
            recording_seed=42
        )
        
        # Record multiple episodes
        num_episodes = 3
        steps_per_episode = [5, 7, 4]  # Different lengths
        
        for episode_idx in range(num_episodes):
            logger.info(f"Recording episode {episode_idx + 1}/{num_episodes}")
            
            # Reset environment (starts new episode)
            env.reset()
            
            # Run some steps
            for step in range(steps_per_episode[episode_idx]):
                action = env.action_space.sample()
                env.step(action)
                logger.debug(f"  Step {step + 1}/{steps_per_episode[episode_idx]}")
            
            # End episode explicitly
            env.end_episode_recording()
        
        # Clean up
        env.close_recorder()
        
        # Check generated files
        temp_path = Path(temp_dir)
        jsonl_files = sorted(temp_path.glob("*.jsonl"))
        meta_files = sorted(temp_path.glob("*.meta.json"))
        
        logger.info(f"Generated {len(jsonl_files)} JSONL files:")
        for jsonl_file in jsonl_files:
            logger.info(f"  - {jsonl_file.name}")
            
        logger.info(f"Generated {len(meta_files)} metadata files:")
        for meta_file in meta_files:
            logger.info(f"  - {meta_file.name}")
        
        # Demonstrate batch loading
        logger.info("Loading batch for verification...")
        loader = JSONLPlaybackLoader()
        batch = loader.load_directory(temp_dir)
        
        logger.info(f"Loaded batch: {batch.total_episodes} episodes, {batch.total_steps} total steps")
        for i, episode in enumerate(batch.episodes):
            logger.info(f"  Episode {episode.episode_id}: {len(episode.states)} states")
        
        return temp_dir


def demo_interactive_playback():
    """Demonstrate interactive playback with episode boundaries.""" 
    logger.info("=== Interactive Playback Demo ===")
    
    # For this demo, we'll use the test pickle file if available
    test_pickle = Path("test_pygame/recordings/2024-06-04_08-39-59.pkl")
    
    if test_pickle.exists():
        logger.info(f"Demo: Loading legacy pickle file: {test_pickle}")
        logger.info("This would open an interactive window (disabled in demo)")
        # load_and_play_jsonl_interactively(test_pickle)  # Commented out for demo
        
        # Instead, just load and show info
        loader = JSONLPlaybackLoader()
        episode, map_def = loader.load_single_episode(test_pickle)
        logger.info(f"Loaded legacy file: {len(episode.states)} states")
        logger.info(f"Reset points detected: {len(episode.reset_points)} points at indices {episode.reset_points}")
    else:
        logger.info("Test pickle file not found - skipping interactive demo")


def demo_batch_analysis():
    """Demonstrate batch analysis capabilities."""
    logger.info("=== Batch Analysis Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a small batch of recordings
        for ep in range(2):
            env = RobotEnv(
                recording_enabled=True,
                use_jsonl_recording=True,
                recording_dir=temp_dir,
                suite_name="analysis",
                scenario_name="batch_test",
                algorithm_name="demo",
                recording_seed=ep
            )
            
            env.reset()
            
            # Record a few steps
            for _ in range(3):
                action = env.action_space.sample()
                env.step(action)
            
            env.end_episode_recording()
            env.close_recorder()
        
        # Analyze the batch
        loader = JSONLPlaybackLoader()
        batch = loader.load_directory(temp_dir)
        
        logger.info("Batch Analysis Results:")
        logger.info(f"  Total episodes: {batch.total_episodes}")
        logger.info(f"  Total steps: {batch.total_steps}")
        logger.info(f"  Average steps per episode: {batch.total_steps / batch.total_episodes:.1f}")
        
        # Analyze robot positions across episodes
        all_robot_positions = []
        for episode in batch.episodes:
            episode_positions = []
            for state in episode.states:
                if hasattr(state, 'robot_pose') and state.robot_pose:
                    pos = state.robot_pose[0]
                    episode_positions.append((pos[0], pos[1]))
            all_robot_positions.extend(episode_positions)
            logger.info(f"  Episode {episode.episode_id}: {len(episode_positions)} robot positions")
        
        if all_robot_positions:
            x_positions = [pos[0] for pos in all_robot_positions]
            y_positions = [pos[1] for pos in all_robot_positions]
            logger.info(f"  Robot position range: X[{min(x_positions):.2f}, {max(x_positions):.2f}], Y[{min(y_positions):.2f}, {max(y_positions):.2f}]")


def main():
    """Main demonstration function."""
    logger.info("🤖 RobotSF JSONL Recording & Playback Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demo 1: JSONL Recording
        demo_jsonl_recording()
        print()
        
        # Demo 2: Interactive Playback  
        demo_interactive_playback()
        print()
        
        # Demo 3: Batch Analysis
        demo_batch_analysis()
        print()
        
        logger.info("✅ All demonstrations completed successfully!")
        logger.info("")
        logger.info("Key benefits of the new system:")
        logger.info("  • Per-episode files prevent 'teleport' trails")
        logger.info("  • Episode metadata enables better analysis")
        logger.info("  • Directory-based batch processing")
        logger.info("  • Backward compatibility with pickle files")
        logger.info("  • Streaming JSONL format for large datasets")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()