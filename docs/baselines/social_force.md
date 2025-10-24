# Social Force Baseline

The Social Force baseline (`baseline_sf`) implements the classical Social Force model for pedestrian dynamics as a navigation planner. This model treats navigation as a physical system where agents are subject to attractive forces toward goals and repulsive forces from obstacles and other agents.

## Quick Start

### Python API

```python
from robot_sf.baselines import get_baseline
from robot_sf.baselines.social_force import SFPlannerConfig, Observation

# Create planner with default configuration
SocialForcePlanner = get_baseline("baseline_sf")
planner = SocialForcePlanner({}, seed=42)

# Create observation
obs = Observation(
    dt=0.1,
    robot={
        "position": [0.0, 0.0],
        "velocity": [0.0, 0.0], 
        "goal": [5.0, 5.0],
        "radius": 0.3
    },
    agents=[
        {"position": [2.0, 1.0], "velocity": [0.5, 0.0], "radius": 0.35}
    ],
    obstacles=[]
)

# Get action
action = planner.step(obs)
print(f"Action: {action}")  # {"vx": 0.8, "vy": 0.9}

planner.close()
```

### Command Line Interface

```bash
# Run benchmark with Social Force baseline
robot_sf_bench run \
    --matrix scenarios/test_scenarios.yaml \
    --algo baseline_sf \
    --out results/social_force_results.jsonl

# Use custom configuration
robot_sf_bench run \
    --matrix scenarios/test_scenarios.yaml \
    --algo baseline_sf \
    --config configs/baselines/social_force_custom.yaml \
    --out results/social_force_custom.jsonl
```

## Configuration

### Basic Configuration

```python
from robot_sf.baselines.social_force import SFPlannerConfig

config = SFPlannerConfig(
    # Action space: "velocity" for (vx, vy) or "unicycle" for (v, omega)
    action_space="velocity",
    
    # Speed limits
    v_max=2.0,           # Maximum linear velocity [m/s]
    omega_max=1.0,       # Maximum angular velocity [rad/s]
    
    # Goal behavior
    desired_speed=1.0,   # Preferred speed toward goal [m/s]
    tau=0.5,            # Acceleration time constant [s]
    
    # Social forces (agent repulsion)
    A=5.1,              # Force amplitude [N]
    B=0.35,             # Interaction range [m]
    lambda_anisotropy=2.0,  # Forward vs backward strength
    
    # Safety
    safety_clamp=True,   # Enforce velocity limits
    noise_std=0.0       # Force noise (0 = deterministic)
)

planner = SocialForcePlanner(config, seed=42)
```

### YAML Configuration

Create a YAML file (e.g., `my_sf_config.yaml`):

```yaml
# Social Force configuration
action_space: "velocity"
v_max: 1.5
desired_speed: 1.0
tau: 0.4

# Social force parameters
A: 6.0          # Stronger repulsion
B: 0.3          # Shorter range
lambda_anisotropy: 2.5

# Obstacle avoidance
A_obs: 15.0     # Strong wall repulsion
B_obs: 0.05     # Sharp force profile

# Numerical settings
clip_force: true
max_force: 100.0
safety_clamp: true
noise_std: 0.0  # Deterministic
```

Load and use:

```python
import yaml
from robot_sf.baselines.social_force import SocialForcePlanner

with open("my_sf_config.yaml") as f:
    config_dict = yaml.safe_load(f)

planner = SocialForcePlanner(config_dict, seed=42)
```

## Parameter Guide

### Speed and Kinematics

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `v_max` | 2.0 | Maximum linear velocity [m/s] | 0.5 - 3.0 |
| `omega_max` | 1.0 | Maximum angular velocity [rad/s] | 0.5 - 2.0 |
| `desired_speed` | 1.0 | Preferred movement speed [m/s] | 0.5 - 2.0 |
| `tau` | 0.5 | Acceleration time constant [s] | 0.2 - 1.0 |

### Social Force Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `A` | 5.1 | Social force amplitude [N] | 1.0 - 20.0 |
| `B` | 0.35 | Interaction decay length [m] | 0.1 - 1.0 |
| `lambda_anisotropy` | 2.0 | Directional asymmetry factor | 1.0 - 5.0 |
| `cutoff_radius` | 10.0 | Maximum interaction distance [m] | 3.0 - 20.0 |

### Obstacle Forces

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `A_obs` | 10.0 | Obstacle force amplitude [N] | 5.0 - 50.0 |
| `B_obs` | 0.0 | Obstacle force range [m] | 0.0 - 0.2 |

### Parameter Effects

**Higher `A` values:**
- ✅ Stronger avoidance of other agents
- ❌ May cause oscillations or overly conservative behavior

**Lower `B` values:**
- ✅ Sharper force profiles, more decisive avoidance
- ❌ May cause abrupt direction changes

**Higher `tau` values:**
- ✅ Smoother acceleration, more natural motion
- ❌ Slower response to obstacles and goals

**Higher `lambda_anisotropy` values:**
- ✅ More realistic forward-focused avoidance
- ❌ May ignore threats from behind

## Action Spaces

### Velocity Commands (`action_space="velocity"`)

Outputs direct velocity commands:

```python
action = {"vx": 1.2, "vy": 0.3}  # [m/s]
```

**Use for:**
- Holonomic robots (can move in any direction)
- Simulation environments with direct velocity control
- Simple integration schemes

### Unicycle Commands (`action_space="unicycle"`)

Outputs linear and angular velocity:

```python
action = {"v": 1.5, "omega": 0.2}  # [m/s, rad/s]
```

**Use for:**
- Car-like robots, differential drive robots
- Non-holonomic systems
- More realistic robot kinematics

## Common Usage Patterns

### Basic Goal Navigation

```python
# Simple point-to-point navigation
config = SFPlannerConfig(
    action_space="velocity",
    desired_speed=1.0,
    v_max=1.5
)

planner = SocialForcePlanner(config, seed=42)
# ... use with observations containing goal positions
```

### Crowded Environment Navigation

```python
# More conservative parameters for crowded spaces
config = SFPlannerConfig(
    desired_speed=0.8,    # Slower in crowds
    A=8.0,               # Stronger avoidance
    B=0.4,               # Wider comfort zone
    tau=0.3              # Quicker response
)

planner = SocialForcePlanner(config, seed=42)
```

### Aggressive Navigation

```python
# Faster, more direct navigation (use with caution)
config = SFPlannerConfig(
    desired_speed=1.5,
    A=3.0,               # Weaker avoidance
    B=0.2,               # Narrower personal space
    lambda_anisotropy=3.0 # Focus on forward obstacles
)

planner = SocialForcePlanner(config, seed=42)
```

### Stochastic Behavior

```python
# Add noise for more human-like behavior
config = SFPlannerConfig(
    noise_std=0.2,       # Small amount of noise
    # ... other parameters
)

planner = SocialForcePlanner(config, seed=42)
```

## Troubleshooting

### Common Issues

**Robot doesn't move or moves very slowly:**
- Check that `desired_speed > 0` and `v_max > desired_speed`
- Verify goal position is different from robot position
- Reduce `A` if social forces are too strong

**Robot oscillates or shows erratic behavior:**
- Reduce `A` (force amplitude)
- Increase `B` (smoother force profile) 
- Increase `tau` (gentler acceleration)
- Enable `clip_force` and reduce `max_force`

**Robot collides with obstacles:**
- Increase `A_obs` (stronger obstacle repulsion)
- Decrease `B_obs` (sharper obstacle forces)
- Ensure obstacles are properly defined in observation

**Robot ignores pedestrians:**
- Increase `A` (social force amplitude)
- Check that agents are within `cutoff_radius`
- Verify agent positions are correctly provided

**Non-deterministic behavior:**
- Set `noise_std=0.0`
- Use consistent `seed` values
- Ensure observation data is deterministic

### Performance Tuning

**For real-time applications:**
- Reduce `cutoff_radius` to limit interaction calculations
- Use fewer agents in scenarios
- Consider simplified force models

**For accuracy:**
- Use smaller timesteps (`dt`)
- Increase `cutoff_radius` for longer-range interactions
- Tune force parameters based on real pedestrian data

## Limitations and Caveats

### Model Limitations

1. **Local minima**: May get stuck in local minima with complex obstacle configurations
2. **Oscillations**: Can exhibit oscillatory behavior with certain parameter combinations
3. **No path planning**: Purely reactive, doesn't plan around large obstacles
4. **Velocity assumptions**: Assumes agents maintain constant velocity between observations

### Implementation Limitations

1. **Obstacle representation**: Limited to simple geometric obstacles (lines, circles)
2. **No group dynamics**: Doesn't model pedestrian groups or social interactions
3. **Single robot**: Designed for single-agent scenarios
4. **No learning**: Parameters are static, no adaptation to environment

### When to Use Alternatives

Consider other baselines when:
- **Complex environments**: Use path planning methods (A*, RRT)
- **Dynamic obstacles**: Use predictive methods (ORCA, RVO)
- **Learning required**: Use reinforcement learning approaches
- **Multi-robot**: Use dedicated multi-agent algorithms

## Integration with Benchmark System

### Episode Metadata

The planner automatically provides metadata for benchmark episodes:

```python
metadata = planner.get_metadata()
# {
#     "algorithm": "social_force",
#     "config": {...},  # Full configuration
#     "config_hash": "abc123..."  # Reproducibility hash
# }
```

### Baseline Statistics

Use with the benchmark system to generate baseline statistics:

```bash
# Generate baseline stats for SNQI normalization
robot_sf_bench baseline \
    --matrix scenarios/validation_scenarios.yaml \
    --out baseline_stats.json \
    --algo baseline_sf
```

### Reproducibility

For reproducible benchmarks:
1. Always set `seed` parameter
2. Use `noise_std=0.0` for deterministic behavior
3. Save configuration with version hashes
4. Document parameter choices

## References

- **Original Paper**: Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. Physical review E, 51(4), 4282.
- **Implementation**: Based on PySocialForce library
- **Design Documentation**: See `docs/dev/design/social_force_wrapper.md`