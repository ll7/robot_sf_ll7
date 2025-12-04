# Quickstart: Using Extended Occupancy Grids

**Date**: 2025-12-04  
**Phase**: Phase 1 (Design & Contracts)  
**Audience**: Developers integrating occupancy grids into environments or using them for planning

---

## Installation & Setup

No additional dependencies required beyond existing robot_sf setup.

```bash
# Ensure you have the latest environment synced
uv sync --all-extras
source .venv/bin/activate
```

---

## 1. Basic Grid Generation & Querying

### Create an Environment with Occupancy Grid Observation

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig, GridConfig

# Define grid configuration
grid_config = GridConfig(
    size_m=(10.0, 10.0),        # 10m x 10m grid
    resolution_m=0.1,            # 0.1m per cell (100x100 grid)
    frame="ego",                 # Robot-relative coordinates
    occupancy_type="binary",     # Binary occupancy (0 or 1)
    enabled_channels=["static_obstacles", "pedestrians"],
    include_static_obstacles=True,
    include_pedestrians=True,
)

# Create environment with grid observation
config = RobotSimulationConfig(
    use_occupancy_grid=True,
    grid_config=grid_config,
    debug=True,  # Enable visualization
)

env = make_robot_env(config=config)
obs = env.reset(seed=42)

# Grid observation is included in obs dict
print("Occupancy grid shape:", obs["occupancy_grid"].shape)
# Output: (2, 100, 100) -> [channels, height, width]
```

### Inspect Grid Channels

```python
# obs["occupancy_grid"] is a numpy array with shape (num_channels, height, width)
grid_array = obs["occupancy_grid"]

# Channel 0: static obstacles
# Channel 1: pedestrians

obstacles_channel = grid_array[0]
pedestrians_channel = grid_array[1]

print("Obstacles occupied cells:", (obstacles_channel > 0).sum())
print("Pedestrians occupied cells:", (pedestrians_channel > 0).sum())
```

### Step and Update Grid

```python
for step in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Grid updates automatically with pedestrian positions
    current_grid = obs["occupancy_grid"]
    print(f"Step {step}: pedestrians channel max occupancy = {current_grid[1].max()}")
    
    if terminated or truncated:
        break
```

---

## 2. Point-of-Interest Queries for Spawn Validation

### Check if a Location is Safe

```python
from robot_sf.nav.occupancy import POIQuery, query_occupancy

# Get the internal grid object (depends on implementation)
# Option 1: If exposed via env API
internal_grid = env.unwrapped.occupancy_grid

# Option 2: Reconstruct from observation array (if not directly exposed)
# This requires knowledge of grid config; simpler to use Option 1

# Query a specific point in world coordinates
query = POIQuery(
    query_type="point",
    world_x=5.0,
    world_y=5.0,
    grid_frame="world",  # Query in world frame
)

result = query_occupancy(internal_grid, query)

if result.safe_to_spawn:
    print("Location (5.0, 5.0) is safe for spawning")
else:
    print(f"Location occupied: {result.occupancy_fraction * 100:.1f}% occupied")
```

### Check if a Circular Region is Free

```python
# Validate a spawn region (e.g., robot requires 1m clearance)
spawn_query = POIQuery(
    query_type="circle",
    world_x=3.0,
    world_y=3.0,
    radius_m=1.0,  # 1m radius
    grid_frame="world",
)

spawn_result = query_occupancy(internal_grid, spawn_query)

if spawn_result.safe_to_spawn:
    print("Spawn region is clear; robot can be placed here")
else:
    print(f"Spawn region blocked: {spawn_result.occupancy_fraction * 100:.1f}% occupied")
    print(f"Obstacles: {spawn_result.channel_results.get('static_obstacles', 0) * 100:.1f}%")
    print(f"Pedestrians: {spawn_result.channel_results.get('pedestrians', 0) * 100:.1f}%")
```

### Validate Multiple Spawn Candidates

```python
def find_safe_spawn(grid, candidates, required_clearance_m):
    """Find first safe spawn point from candidates."""
    for x, y in candidates:
        query = POIQuery(
            query_type="circle",
            world_x=x,
            world_y=y,
            radius_m=required_clearance_m,
            grid_frame="world",
        )
        result = query_occupancy(grid, query)
        if result.safe_to_spawn:
            return (x, y)
    return None

# Test with multiple candidates
spawn_candidates = [
    (2.0, 2.0),
    (8.0, 8.0),
    (5.0, 5.0),
    (1.0, 1.0),
]

safe_spawn = find_safe_spawn(internal_grid, spawn_candidates, required_clearance_m=0.5)
if safe_spawn:
    print(f"Safe spawn found at: {safe_spawn}")
else:
    print("No safe spawn location available")
```

---

## 3. Training with Occupancy Grid Observations

### Use with Stable Baselines3

```python
from stable_baselines3 import PPO

# Create env (from Section 1)
env = make_robot_env(config=config)

# PPO will learn from occupancy grid automatically
model = PPO(
    "CnnPolicy",  # Use CNN policy to process grid observations
    env,
    verbose=1,
    device="auto",
)

# Train
model.learn(total_timesteps=100000)

# Evaluate
obs = env.reset(seed=123)
for step in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Custom Feature Extractor (Optional)

```python
from stable_baselines3.common.policies import CnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN
import torch.nn as nn

class OccupancyGridFeatureExtractor(nn.Module):
    """Custom feature extractor for occupancy grids."""
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__()
        # Simple CNN: 3x3 conv → 3x3 conv → flatten
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),  # 2 channels (obstacles, pedestrians)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.net(observations)

# Use with PPO (set in policy_kwargs)
policy_kwargs = {
    "features_extractor_class": OccupancyGridFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
}
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000)
```

---

## 4. Visualization in Pygame

### Enable Grid Visualization

```python
from robot_sf.gym_env.environment_factory import make_robot_env

config = RobotSimulationConfig(
    use_occupancy_grid=True,
    grid_config=grid_config,
    debug=True,  # Enable interactive visualization
)

env = make_robot_env(config=config)
env.reset(seed=42)

# Interactive visualization available in debug mode
# - Grid overlay shows on the pygame window
# - Yellow tinted cells: obstacles
# - Transparent cells: free space
# - Other colors: pedestrians (or other channels)

obs = env.reset()
for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # env handles visualization automatically in debug mode
    # Press 'g' to toggle grid visibility (example; depends on implementation)
    # Press 'c' to toggle specific channels (example)
    
    if terminated or truncated:
        obs = env.reset()
```

### Headless Visualization Recording

```python
# Record environment with grid visualization to video
from robot_sf.gym_env.environment_factory import make_robot_env

config = RobotSimulationConfig(
    use_occupancy_grid=True,
    grid_config=grid_config,
    debug=True,
    record_episodes=True,  # Record to mp4 or similar
    output_dir="output/recordings/",
)

env = make_robot_env(config=config)
env.reset(seed=42)

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Video saved with grid overlay visible
print("Recording saved to output/recordings/")
```

---

## 5. Custom Configuration Examples

### World-Frame Grid (for Global Planning)

```python
world_grid_config = GridConfig(
    size_m=(20.0, 20.0),       # Larger world view
    resolution_m=0.5,          # Coarser resolution (0.5m per cell)
    frame="world",             # Global coordinates (not ego-rotated)
    occupancy_type="continuous",  # Soft occupancy
    enabled_channels=["static_obstacles"],  # Only static obstacles (faster)
    include_pedestrians=False,
)

env = make_robot_env(config=RobotSimulationConfig(
    use_occupancy_grid=True,
    grid_config=world_grid_config,
))
```

### High-Resolution Ego-Frame (for Precise Local Planning)

```python
high_res_grid_config = GridConfig(
    size_m=(5.0, 5.0),         # Smaller local view
    resolution_m=0.05,         # Very fine (5cm per cell) → 100x100 grid
    frame="ego",               # Robot-relative
    occupancy_type="continuous",
    enabled_channels=["static_obstacles", "pedestrians"],
)

# Note: High resolution means slower grid generation
# Target: <5ms for 10m grid at 0.1m resolution
# 5m at 0.05m → 100x100 grid → similar cost
```

### Pedestrians Only (for Collision Avoidance Training)

```python
ped_only_config = GridConfig(
    size_m=(10.0, 10.0),
    resolution_m=0.1,
    frame="ego",
    occupancy_type="binary",
    enabled_channels=["pedestrians"],  # Only pedestrians
    include_pedestrians=True,
    include_static_obstacles=False,
)
```

---

## 6. Common Patterns

### Pattern: Augment Observations with Grid

```python
# Combine grid with other observations (e.g., lidar, goal direction)
class GridAugmentedObserver:
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        obs = self.env.reset()
        return self._augment(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info
    
    def _augment(self, obs):
        # obs already includes "occupancy_grid" from environment
        # Add custom fields if needed
        obs["grid_info"] = {
            "obstacles": obs["occupancy_grid"][0],
            "pedestrians": obs["occupancy_grid"][1],
        }
        return obs
```

### Pattern: Reward Shaping with Grid Occupancy

```python
def grid_aware_reward(obs, grid_config):
    """Penalize high occupancy (crowded regions)."""
    pedestrian_channel = obs["occupancy_grid"][1]
    crowding_penalty = -0.1 * pedestrian_channel.mean()  # Avg occupancy
    return crowding_penalty

# Use in environment wrapper
class GridAwareRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        grid_reward = grid_aware_reward(obs, self.env.unwrapped.grid_config)
        reward += grid_reward
        return obs, reward, terminated, truncated, info
```

---

## 7. Troubleshooting

### Grid Not Updating Pedestrians
**Issue**: Pedestrian channel stays zero despite pedestrians in scene.
**Solution**: Ensure `include_pedestrians=True` in GridConfig; verify FastPysfWrapper is active (check sim backend).

### Out-of-Bounds Query Error
**Issue**: POI query outside grid bounds raises exception.
**Solution**: Check `result.is_within_bounds`; handle gracefully or clamp query coordinates.

### Performance: Grid Generation Slow
**Issue**: Grid generation exceeds 5ms budget.
**Solution**: Reduce resolution_m (e.g., 0.1m → 0.2m) or grid size_m. Profile with `cProfile`.

### Visualization Showing Grid Overlay
**Issue**: Grid not visible in pygame visualization.
**Solution**: Ensure `debug=True` in config; verify grid channels are enabled in visualization state.

---

## Next Steps

1. **Run Examples**: Try scripts from `examples/` directory
2. **Write Tests**: Follow patterns in `tests/test_occupancy_*.py`
3. **Customize Config**: Experiment with different grid_config parameters
4. **Integrate into Your Workflow**: Use grids in your training/planning pipeline
5. **Refer to Full Docs**: See `docs/dev/occupancy/` for advanced topics

---

## References

- **Specification**: `specs/1382-extend-occupancy-grid/spec.md`
- **Data Model**: `specs/1382-extend-occupancy-grid/data-model.md`
- **Dev Guide**: `docs/dev_guide.md` (environment factory, unified config)
- **Gymnasium Docs**: https://gymnasium.farama.org/ (observation spaces, Box, CnnPolicy)
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/ (RL training with visual observations)
