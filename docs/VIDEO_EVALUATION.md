# Video Recording Evaluation Script

## Overview

The `evaluate_with_video.py` script extends the standard policy evaluation functionality by recording videos of simulation episodes and organizing them by termination reasons. This allows for detailed analysis of policy behavior across different scenarios.

## Features

- **Automatic Video Recording**: Records every episode of policy evaluation
- **Termination-Based Organization**: Videos are automatically organized into directories based on how episodes end:
  - `route_complete/` - Successfully completed episodes
  - `pedestrian_collision/` - Episodes ending in pedestrian collision
  - `obstacle_collision/` - Episodes ending in obstacle collision  
  - `robot_collision/` - Episodes ending in robot-robot collision
  - `timeout/` - Episodes that exceeded maximum time steps
  - `unknown/` - Episodes with unidentified termination reasons

- **Multi-Difficulty Evaluation**: Evaluates across multiple difficulty levels (pedestrian densities)
- **Comprehensive Metrics**: Saves detailed evaluation metrics to JSON file
- **Descriptive Filenames**: Videos are named with episode number, difficulty, and termination reason

## Requirements

- Trained RL model (PPO or A2C format)
- MoviePy library for video creation (`pip install moviepy`)
- FFmpeg for video encoding (MoviePy dependency)
- Sufficient disk space for video storage

## Usage

### Basic Usage

1. **Update Model Path**: Edit `main()` function in `evaluate_with_video.py` to point to your trained model:
   ```python
   model_path = "./model/your_model.zip"
   ```

2. **Run Evaluation**:
   ```bash
   python scripts/evaluate_with_video.py
   ```

### Advanced Configuration

Modify the `VideoEvalSettings` in `main()` to customize evaluation:

```python
settings = VideoEvalSettings(
    num_episodes=20,                              # Episodes per difficulty level
    ped_densities=[0.00, 0.02, 0.08, 0.20],     # Difficulty levels
    vehicle_config=vehicle_config,                # Robot configuration
    prf_config=prf_config,                       # Force configuration
    gym_config=gym_settings,                     # Observation/action spaces
    video_output_dir="evaluation_videos",        # Output directory
    video_fps=30.0,                              # Video frame rate
    record_all_episodes=True,                    # Record all vs. failures only
)
```

## Output Structure

```
evaluation_videos/
├── evaluation_results.json                     # Metrics summary
├── difficulty_0/                               # Low difficulty (no pedestrians)
│   ├── route_complete/
│   │   ├── ep_001_diff_0_route_complete_20250121_143022.mp4
│   │   └── ep_002_diff_0_route_complete_20250121_143045.mp4
│   └── timeout/
│       └── ep_003_diff_0_timeout_20250121_143108.mp4
├── difficulty_1/                               # Medium-low difficulty
│   ├── route_complete/
│   ├── pedestrian_collision/
│   │   ├── ep_005_diff_1_pedestrian_collision_20250121_143145.mp4
│   │   └── ep_007_diff_1_pedestrian_collision_20250121_143201.mp4
│   └── obstacle_collision/
└── difficulty_2/                               # Higher difficulty levels
    └── ...
```

## Video Analysis

### Successful Episodes (`route_complete/`)
- Study optimal navigation strategies
- Identify efficient path planning
- Analyze pedestrian avoidance behaviors

### Collision Episodes
- **`pedestrian_collision/`**: Understand failure modes with dynamic obstacles
- **`obstacle_collision/`**: Identify issues with static obstacle avoidance
- **`robot_collision/`**: Multi-robot coordination problems (if applicable)

### Timeout Episodes (`timeout/`)
- Investigate scenarios where robot gets stuck
- Identify inefficient navigation patterns
- Spot areas needing improved exploration

## Configuration Parameters

### Model and Environment
- **Model Path**: Path to saved PPO/A2C model file
- **Observation Space**: Must match trained model's input format
- **Action Space**: Continuous control space for robot movement
- **Vehicle Config**: Robot physical parameters (radius, speeds, etc.)

### Video Recording
- **Frame Rate**: Higher FPS = smoother video, larger files
- **Output Directory**: Where to save organized video files
- **Record All**: `True` to record everything, `False` for failures only

### Evaluation Settings
- **Episodes per Difficulty**: Balance between statistical significance and time
- **Difficulty Levels**: Pedestrian densities to test robustness
- **Termination Criteria**: Automatic based on environment feedback

## Troubleshooting

### Common Issues

1. **MoviePy Not Available**
   ```bash
   pip install moviepy
   # On some systems you may also need:
   brew install ffmpeg  # macOS
   apt-get install ffmpeg  # Ubuntu
   ```

2. **Model Loading Errors**
   - Verify model path exists
   - Ensure observation/action spaces match training configuration
   - Check model format (PPO vs A2C)

3. **Large Video Files**
   - Reduce `video_fps` (e.g., 10-15 FPS instead of 30)
   - Set `record_all_episodes=False` to only record failures
   - Reduce `num_episodes` for initial testing

4. **Slow Evaluation**
   - Decrease number of episodes per difficulty
   - Reduce video frame rate
   - Test with fewer difficulty levels initially

### Performance Tips

- Start with small test runs (2-3 episodes) to verify setup
- Monitor disk space during evaluation
- Use `test_video_recording.py` to verify functionality without full evaluation

## Testing

Use the provided test script to verify functionality:

```bash
python scripts/test_video_recording.py
```

This script:
- Tests video recording components without needing a trained model
- Verifies termination reason detection
- Checks environment setup and rendering
- Confirms video filename generation

## Integration with Analysis Pipeline

The generated videos and metrics can be used for:
- Policy behavior analysis
- Failure mode identification  
- Training data augmentation
- Presentation and demonstration
- Debugging specific scenarios
- Comparative analysis between models

The JSON metrics file contains quantitative results, while videos provide qualitative insights into policy behavior.
