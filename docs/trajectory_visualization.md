# Trajectory Visualization Feature

This document describes the trajectory visualization feature added to the interactive playback system in RobotSF.

## Overview

The trajectory visualization feature allows you to display movement trails of entities (robot, pedestrians, ego pedestrian) during playback of recorded simulation states. This helps visualize the path taken by entities over time.

## Features

- **Real-time trajectory display**: See entity trails as you navigate through simulation frames
- **Multiple entity support**: Visualize trajectories for robots, pedestrians, and ego pedestrians
- **Configurable trail length**: Adjust how many previous positions to show in the trail
- **Color-coded trails**: Different colors for different entity types
- **Interactive controls**: Toggle display on/off and adjust settings during playback

## Controls

### Trajectory Controls
- **V**: Toggle trajectory display on/off
- **B**: Increase trail length (+20 points)
- **C**: Decrease trail length (-20 points)  
- **X**: Clear all trajectory histories

### Existing Playback Controls
- **Space**: Play/pause
- **Period (.)**: Next frame
- **Comma (,)**: Previous frame
- **N**: Jump to first frame
- **M**: Jump to last frame
- **K**: Speed up playback
- **J**: Slow down playback
- **H**: Show help with all controls

## Usage

### Basic Usage

```python
from robot_sf.render.interactive_playback import InteractivePlayback, load_states

# Load recorded states
states, map_def = load_states("path/to/recording.pkl")

# Create interactive playback
playback = InteractivePlayback(states, map_def)

# Enable trajectories (optional - can also be toggled with 'V' key)
playback.show_trajectories = True
playback.max_trajectory_length = 100

# Run the playback
playback.run()
```

### Demo Script

A complete demo is available in `examples/trajectory_demo.py`:

```bash
python examples/trajectory_demo.py path/to/recording.pkl
```

## Visual Elements

### Trajectory Colors
- **Robot trajectory**: Blue (`RGB(0, 100, 255)`)
- **Pedestrian trajectories**: Light red (`RGB(255, 100, 100)`)
- **Ego pedestrian trajectory**: Magenta (`RGB(200, 0, 200)`)

### Trail Properties
- **Default trail length**: 100 points
- **Trail length range**: 10 - 500 points
- **Trail width**: 2-3 pixels depending on entity type

## Implementation Details

### Key Components

1. **Trajectory Storage**: Uses `collections.deque` with configurable maximum length
2. **Trajectory Update**: Updates position history as frames are processed
3. **Trajectory Rendering**: Draws connected lines between historical positions
4. **Frame Navigation**: Rebuilds trajectories when jumping between frames

### Data Structures

```python
# Trajectory storage
robot_trajectory: Deque[Tuple[float, float]]
ped_trajectories: Dict[int, Deque[Tuple[float, float]]]  
ego_ped_trajectory: Deque[Tuple[float, float]]
```

### Performance Considerations

- Trajectory length is limited to prevent memory issues
- Trajectories are only updated when display is enabled
- Efficient deque operations for adding/removing positions

## Configuration

### Trail Length
- **Default**: 100 points
- **Minimum**: 10 points  
- **Maximum**: 500 points
- **Adjustment**: ±20 points per key press

### Display Options
- **Toggle**: On/off via 'V' key or programmatically
- **Clear**: Remove all history via 'X' key
- **Real-time update**: Trajectories update as frames advance

## Technical Notes

### Frame Navigation Behavior
- **Forward navigation**: Incrementally adds positions to trajectories
- **Backward navigation**: Rebuilds trajectories from beginning to current frame
- **Frame jumping**: Rebuilds trajectories to maintain accuracy

### Memory Management
- Fixed-size deques prevent unlimited memory growth
- Trajectory clearing allows manual memory cleanup
- Configurable trail length for memory/visual trade-off

## Testing

The feature includes comprehensive tests in `test_trajectory_feature.py`:

```bash
python test_trajectory_feature.py
```

Tests cover:
- Trajectory updates and storage
- Length limiting functionality
- Trajectory clearing
- Display toggle behavior

## Future Enhancements

Possible future improvements:
- Trajectory persistence across playback sessions
- Customizable colors for individual entities
- Trajectory export functionality
- Advanced trail effects (fading, thickness variation)
- Trajectory analysis tools integration