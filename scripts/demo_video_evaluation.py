#!/usr/bin/env python3
"""
Demonstration script showing how to use the video evaluation system.

This script provides examples of how to:
1. Set up the evaluation environment
2. Load a trained model
3. Run evaluation with video recording
4. Analyze the results

Usage:
    python scripts/demo_video_evaluation.py
"""

import json
from pathlib import Path

from scripts.evaluate_with_video import video_evaluation_series


def _check_model_availability(model_path: str) -> bool:
    """Check if model exists and show available alternatives."""
    if Path(model_path).exists():
        return True

    print(f"Model not found at {model_path}")
    print("Available models:")
    model_dir = Path("model")
    if model_dir.exists():
        for model_file in model_dir.glob("*.zip"):
            print(f"  - {model_file}")
    else:
        print("  No model directory found")
    return False


def _display_results(results: dict, output_dir: str) -> None:
    """Display evaluation results and video file information."""
    print("\n=== Evaluation Results ===")
    for level, metrics in results.items():
        print(f"\nDifficulty Level {level}:")
        print(f"  Success Rate: {metrics.success_rate:.2%}")
        print(f"  Collision Rate: {metrics.collision_rate:.2%}")
        print(f"  Timeout Rate: {metrics.timeout_rate:.2%}")
        print(f"  Average Episode Length: {metrics.avg_episode_length:.1f}")

        # Show termination reason breakdown
        if hasattr(metrics, "termination_reasons"):
            print("  Termination Reasons:")
            for reason, count in metrics.termination_reasons.items():
                print(f"    {reason}: {count}")

    # Check video files
    video_dir = Path(output_dir) / "videos"
    if video_dir.exists():
        print("\n=== Video Files Created ===")
        for subdir in video_dir.iterdir():
            if subdir.is_dir():
                video_count = len(list(subdir.glob("*.mp4")))
                print(f"  {subdir.name}: {video_count} videos")

    print(f"\nEvaluation complete! Check {output_dir} for results.")


def demo_basic_evaluation():
    """Demonstrate basic video evaluation."""
    print("=== Basic Video Evaluation Demo ===")

    # Configuration
    model_path = "model/ppo_model_retrained_10m_2025-02-01.zip"
    output_dir = "evaluation_results"

    # Check if model exists
    if not _check_model_availability(model_path):
        return

    try:
        # Import here to avoid issues if not all dependencies are available
        import numpy as np
        from gymnasium import spaces

        from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
        from robot_sf.robot.differential_drive import DifferentialDriveSettings
        from scripts.evaluate_with_video import GymAdapterSettings, VideoEvalSettings

        # Run evaluation
        print(f"Running evaluation with model: {model_path}")
        print(f"Output directory: {output_dir}")

        # Create observation and action spaces (simplified for demo)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(94,), dtype=np.float64)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

        # Create settings with minimal configuration
        gym_config = GymAdapterSettings(
            obs_space=obs_space,
            action_space=action_space,
            obs_timesteps=1,
            squeeze_obs=True,
            cut_2nd_target_angle=True,
            return_dict=False,
        )

        settings = VideoEvalSettings(
            num_episodes=5,  # Reduced for demo
            ped_densities=[0.01, 0.02],  # Only test easier levels
            vehicle_config=DifferentialDriveSettings(),
            prf_config=PedRobotForceConfig(),
            gym_config=gym_config,
            video_output_dir=output_dir,
        )

        results = video_evaluation_series(
            model_path=model_path,
            settings=settings,
        )

        _display_results(results, output_dir)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


def demo_analysis():
    """Demonstrate how to analyze saved evaluation results."""
    print("\n=== Results Analysis Demo ===")

    results_file = Path("evaluation_results/evaluation_metrics.json")
    if not results_file.exists():
        print(f"No results file found at {results_file}")
        print("Run the evaluation first!")
        return

    try:
        with open(results_file) as f:
            data = json.load(f)

        print("=== Saved Evaluation Results ===")
        print(f"Evaluation Date: {data.get('timestamp', 'Unknown')}")
        print(f"Model: {data.get('model_path', 'Unknown')}")

        metrics = data.get("metrics", {})
        for level, level_data in metrics.items():
            print(f"\nLevel {level}:")
            for key, value in level_data.items():
                if isinstance(value, float):
                    if key.endswith("_rate"):
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error analyzing results: {e}")


def main():
    """Main demonstration function."""
    print("Video Evaluation System Demo")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("robot_sf").exists():
        print("Error: Please run this script from the robot_sf_ll7 root directory")
        return

    # Run demos
    demo_basic_evaluation()
    demo_analysis()

    print("\n" + "=" * 40)
    print("Demo complete!")
    print("\nNext steps:")
    print("1. Check the 'evaluation_results' directory for videos and metrics")
    print("2. Modify the evaluation parameters in evaluate_with_video.py as needed")
    print("3. Use the test_video_recording.py script to test without a trained model")


if __name__ == "__main__":
    main()
