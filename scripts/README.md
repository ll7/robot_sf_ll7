# Video Evaluation System for Robot RL Policies

This directory contains scripts for evaluating trained reinforcement learning policies with video recording capabilities. The system records videos of simulation episodes and organizes them by termination reason for easy analysis.

## Features

- **Video Recording**: Records videos of all evaluation episodes
- **Termination Labeling**: Organizes videos by termination reason:
  - Route completion (successful navigation)
  - Pedestrian collision
  - Obstacle collision  
  - Robot collision
  - Timeout (max steps exceeded)
- **Metrics Collection**: Saves comprehensive evaluation metrics to JSON
- **Multi-Difficulty**: Evaluates across different difficulty levels
- **Model Compatibility**: Supports both PPO and A2C models
- **Flexible Configuration**: Customizable evaluation parameters

## Scripts

### `evaluate_with_video.py`
Main evaluation script with video recording functionality.

**Usage:**
```bash
python scripts/evaluate_with_video.py
```

**Key Features:**
- Loads trained RL models (PPO/A2C)
- Records videos for each episode
- Organizes videos in subdirectories by termination reason
- Saves evaluation metrics to `evaluation_videos/evaluation_metrics.json`
- Handles dictionary observation spaces correctly

### `test_video_recording.py`
Test script for verifying video recording without a trained model.

**Usage:**
```bash
python scripts/test_video_recording.py
```

**Purpose:**
- Tests video recording functionality with random policy
- Verifies environment setup and video file creation
- Useful for debugging video recording issues

### `demo_video_evaluation.py`
Demonstration script showing how to use the evaluation system.

**Usage:**
```bash
python scripts/demo_video_evaluation.py
```

**Features:**
- Shows complete evaluation workflow
- Demonstrates results analysis
- Provides usage examples

## Configuration

### Model Requirements
- Trained RL model file (`.zip` format)
- Compatible with PPO and A2C from stable-baselines3
- Model should be trained with dictionary observation space

### Environment Settings
The evaluation uses these default settings:
- **Observation timesteps**: 1
- **Observation adaptation**: Flattened for non-dict models
- **Video FPS**: 30
- **Recording**: All episodes (can be configured to failures only)

### Customizing Evaluation

Modify `VideoEvalSettings` in `evaluate_with_video.py`:

```python
settings = VideoEvalSettings(
    num_episodes=20,  # Episodes per difficulty level
    ped_densities=[0.01, 0.02, 0.03],  # Difficulty levels
    vehicle_config=DifferentialDriveSettings(),
    prf_config=PedRobotForceConfig(),
    gym_config=gym_adapter_settings,
    video_output_dir="my_evaluation_results",
    video_fps=60.0,  # Higher quality videos
    record_all_episodes=False,  # Only record failures
)
```

## Output Structure

```
evaluation_videos/
├── evaluation_metrics.json          # Comprehensive metrics
├── videos/
│   ├── route_completion/            # Successful episodes
│   │   ├── episode_001_level_1.mp4
│   │   └── ...
│   ├── pedestrian_collision/        # Pedestrian collisions
│   │   ├── episode_003_level_1.mp4
│   │   └── ...
│   ├── obstacle_collision/          # Obstacle collisions
│   ├── robot_collision/             # Robot-robot collisions
│   └── timeout/                     # Episodes that timed out
└── logs/                            # Detailed logs (if enabled)
```

## Metrics

The system tracks comprehensive metrics:

- **Success Rate**: Percentage of episodes reaching the goal
- **Collision Rates**: Breakdown by collision type
- **Timeout Rate**: Episodes exceeding maximum steps
- **Episode Length**: Average steps per episode
- **Termination Reasons**: Count of each termination type

Example metrics output:
```json
{
  "timestamp": "2024-12-15T10:30:45",
  "model_path": "model/ppo_model_retrained_10m_2025-02-01.zip",
  "metrics": {
    "difficulty_1": {
      "success_rate": 0.85,
      "collision_rate": 0.10,
      "timeout_rate": 0.05,
      "avg_episode_length": 142.3,
      "termination_reasons": {
        "route_completion": 17,
        "pedestrian_collision": 2,
        "timeout": 1
      }
    }
  }
}
```

## Requirements

### Python Dependencies
- `stable-baselines3`: RL model loading
- `gymnasium`: Environment interface
- `moviepy`: Video recording
- `numpy`: Numerical operations
- `tqdm`: Progress bars

### Environment Setup
- Enable debug mode for video recording
- Sufficient disk space for video files
- GPU recommended for faster evaluation

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install stable-baselines3 gymnasium moviepy numpy tqdm
   ```

2. **Video Recording Fails**
   ```bash
   # Test with the test script first
   python scripts/test_video_recording.py
   ```

3. **Model Loading Issues**
   - Verify model file exists
   - Check model was trained with compatible observation space
   - Ensure model type (PPO/A2C) matches expectations

4. **Memory Issues**
   - Reduce `num_episodes` in settings
   - Lower video FPS
   - Record only failures (`record_all_episodes=False`)

### Debug Mode
Enable detailed logging by modifying the environment settings:
```python
env_settings = EnvSettings(
    debug=True,  # Enables video recording
    verbose=True,  # Additional logging
)
```

## Advanced Usage

### Custom Observation Spaces
Modify `GymAdapterSettings` for different observation formats:
```python
gym_config = GymAdapterSettings(
    obs_space=custom_obs_space,
    action_space=custom_action_space,
    return_dict=True,  # Keep dictionary format
    squeeze_obs=False,  # Don't flatten observations
)
```

### Integration with Training
Use the evaluation system to monitor training progress:
```python
# Evaluate periodically during training
if episode % 1000 == 0:
    video_evaluation_series(
        model_path=f"models/checkpoint_{episode}.zip",
        settings=eval_settings
    )
```

## Contributing

When extending the evaluation system:
1. Follow the existing code structure
2. Add comprehensive error handling
3. Update documentation
4. Test with both PPO and A2C models
5. Verify video recording works correctly

## License

This evaluation system is part of the robot_sf project and follows the same license terms.
