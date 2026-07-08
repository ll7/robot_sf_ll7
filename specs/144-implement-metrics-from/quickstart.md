# Quickstart: Using Paper 2306.16740v4 Metrics

**Purpose**: Get started with the new paper metrics in 5 minutes
**Audience**: Researchers evaluating social navigation algorithms
**Date**: October 23, 2025

## Overview

This feature adds 22 standardized metrics from paper 2306.16740v4 to evaluate robot social navigation performance. Metrics are organized into two categories:

- **NHT (Navigation/Hard Task)**: 11 metrics for basic navigation performance
- **SHT (Social/Human-aware Task)**: 11 metrics for social interaction quality

## Quick Example

```python
from robot_sf.benchmark.metrics import (
    EpisodeData,
    success_rate,
    collision_count,
    velocity_avg,
    space_compliance,
)
import numpy as np

# Create episode data from your simulation
episode_data = EpisodeData(
    robot_pos=robot_trajectory,  # (T, 2) array
    robot_vel=robot_velocities,  # (T, 2) array
    robot_acc=robot_accelerations,  # (T, 2) array
    peds_pos=pedestrian_positions,  # (T, K, 2) array
    ped_forces=pedestrian_forces,  # (T, K, 2) array
    goal=np.array([10.0, 10.0]),  # Target position
    dt=0.1,  # Simulation timestep
    reached_goal_step=45,  # Goal reached at step 45
)

# Compute metrics
success = success_rate(episode_data, horizon=100)
collisions = collision_count(episode_data)
avg_vel = velocity_avg(episode_data)
compliance = space_compliance(episode_data, threshold=0.5)

print(f"Success: {success:.2f}")
print(f"Collisions: {collisions:.0f}")
print(f"Average Velocity: {avg_vel:.2f} m/s")
print(f"Space Compliance: {compliance:.2%}")
```

**Output**:
```
Success: 1.00
Collisions: 0
Average Velocity: 1.23 m/s
Space Compliance: 15.22%
```

---

## Installation

No additional installation required - metrics are part of the `robot_sf` package:

```bash
# Ensure you have the latest version
cd /path/to/robot_sf_ll7
git checkout 144-implement-metrics-from
uv sync
```

---

## Computing All Metrics

### Option 1: Call Individual Functions

```python
from robot_sf.benchmark import metrics

# NHT Metrics
nht_metrics = {
    "success": metrics.success_rate(data, horizon=100),
    "collisions": metrics.collision_count(data),
    "wall_collisions": metrics.wall_collisions(data),
    "agent_collisions": metrics.agent_collisions(data),
    "human_collisions": metrics.human_collisions(data),
    "timeout": metrics.timeout(data, horizon=100),
    "failure_to_progress": metrics.failure_to_progress(data),
    "stalled_time": metrics.stalled_time(data),
    "time_to_goal": metrics.time_to_goal(data),
    "path_length": metrics.path_length(data),
    "spl": metrics.success_path_length(data, horizon=100, optimal_length=14.14),
}

# SHT Metrics
sht_metrics = {
    "velocity_min": metrics.velocity_min(data),
    "velocity_avg": metrics.velocity_avg(data),
    "velocity_max": metrics.velocity_max(data),
    "acceleration_min": metrics.acceleration_min(data),
    "acceleration_avg": metrics.acceleration_avg(data),
    "acceleration_max": metrics.acceleration_max(data),
    "jerk_min": metrics.jerk_min(data),
    "jerk_avg": metrics.jerk_avg(data),
    "jerk_max": metrics.jerk_max(data),
    "clearing_distance_min": metrics.clearing_distance_min(data),
    "clearing_distance_avg": metrics.clearing_distance_avg(data),
    "space_compliance": metrics.space_compliance(data),
    "distance_to_human_min": metrics.distance_to_human_min(data),
    "time_to_collision_min": metrics.time_to_collision_min(data),
}

all_metrics = {**nht_metrics, **sht_metrics}
```

### Option 2: Batch Computation Helper (Future)

```python
# Coming soon: compute_all_paper_metrics() helper
from robot_sf.benchmark.metrics import compute_all_paper_metrics

metrics_dict = compute_all_paper_metrics(
    data=episode_data,
    horizon=100,
    optimal_path_length=14.14,
)
```

---

## Integration with Benchmark Runner

The metrics integrate seamlessly with the existing benchmark infrastructure:

```python
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci

# Run benchmark episodes
run_batch(
    scenarios=scenario_configs,
    out_path="results/episodes.jsonl",
    workers=4,
)

# Aggregate results with confidence intervals
summary = compute_aggregates_with_ci(
    episodes_path="results/episodes.jsonl",
    group_by="scenario_params.algo",
    bootstrap_samples=1000,
    bootstrap_confidence=0.95,
)

# Access metric statistics
for algo, stats in summary.items():
    print(f"\n{algo}:")
    print(f"  Success Rate: {stats['success']['mean']:.2%} ± {stats['success']['ci_width']:.2%}")
    print(f"  Avg Velocity: {stats['velocity_avg']['mean']:.2f} m/s")
    print(f"  Space Compliance: {stats['space_compliance']['mean']:.2%}")
```

---

## Handling Optional Data

Some metrics require additional data not present in all scenarios:

### Wall Collisions (requires obstacles)

```python
from robot_sf.benchmark.metrics import EpisodeData

# Provide obstacle positions
obstacles = np.array([
    [5.0, 5.0],   # Wall point 1
    [5.0, 15.0],  # Wall point 2
    # ... more obstacle points
])

episode_data = EpisodeData(
    # ... standard fields ...
    obstacles=obstacles,  # NEW: Enable wall collision detection
)

wall_collisions = metrics.wall_collisions(episode_data)
# Returns actual count if obstacles provided, 0.0 otherwise
```

### Agent Collisions (requires other robots)

```python
# Multi-robot scenario
other_agents_positions = np.random.rand(100, 3, 2)  # T=100, J=3 robots, 2D

episode_data = EpisodeData(
    # ... standard fields ...
    other_agents_pos=other_agents_positions,  # NEW: Multi-agent support
)

agent_collisions = metrics.agent_collisions(episode_data)
# Returns actual count if other agents provided, 0.0 otherwise
```

### Graceful Degradation

If optional fields are not provided, metrics return safe defaults:

```python
# Without obstacles
data_no_obstacles = EpisodeData(...)  # obstacles=None (default)

metrics.wall_collisions(data_no_obstacles)       # Returns 0.0
metrics.clearing_distance_min(data_no_obstacles)  # Returns NaN
```

---

## Understanding Metric Values

### NHT Metrics Interpretation

| Metric | Good Value | Bad Value | Units |
|--------|-----------|-----------|-------|
| Success | 1.0 | 0.0 | boolean |
| Collisions | 0 | > 0 | count |
| Time to Goal | Lower | Higher | seconds |
| Path Length | Lower | Higher | meters |
| SPL | 1.0 | 0.0 | ratio |
| Stalled Time | 0 | High | seconds |

### SHT Metrics Interpretation

| Metric | Preferred Range | Units |
|--------|----------------|-------|
| Velocity Avg | 0.8-1.5 | m/s |
| Acceleration Max | < 2.0 | m/s² |
| Jerk Avg | < 1.0 | m/s³ |
| Space Compliance | < 0.3 | ratio (lower = better) |
| Distance to Human Min | > 0.5 | meters |
| Time to Collision | > 5.0 | seconds |

**Note**: "Good" ranges depend on scenario complexity and robot capabilities.

---

## Common Patterns

### Computing Metrics for Multiple Episodes

```python
import json

results = []
for episode in episode_list:
    metrics_dict = {
        "episode_id": episode.id,
        "success": metrics.success_rate(episode.data, horizon=episode.horizon),
        "velocity_avg": metrics.velocity_avg(episode.data),
        "space_compliance": metrics.space_compliance(episode.data),
        # ... more metrics ...
    }
    results.append(metrics_dict)

# Save to JSON
with open("metrics_output.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Filtering Valid Metrics (Excluding NaN)

```python
import math

def filter_valid_metrics(metrics_dict):
    """Remove NaN values from metrics dictionary."""
    return {
        k: v for k, v in metrics_dict.items()
        if not (isinstance(v, float) and math.isnan(v))
    }

valid_metrics = filter_valid_metrics(all_metrics)
```

### Comparing Algorithms

```python
from collections import defaultdict
import numpy as np

# Collect metrics by algorithm
algo_metrics = defaultdict(list)

for episode in episodes:
    algo_metrics[episode.algorithm].append({
        "success": metrics.success_rate(episode.data, horizon=100),
        "velocity_avg": metrics.velocity_avg(episode.data),
    })

# Compute means
for algo, episodes_metrics in algo_metrics.items():
    success_rates = [m["success"] for m in episodes_metrics]
    velocities = [m["velocity_avg"] for m in episodes_metrics]
    
    print(f"{algo}:")
    print(f"  Success: {np.mean(success_rates):.2%}")
    print(f"  Avg Velocity: {np.mean(velocities):.2f} m/s")
```

---

## Troubleshooting

### "NaN values in results"

**Cause**: Missing data or insufficient trajectory length.

**Solutions**:
- Check episode completed successfully (not truncated)
- Verify pedestrian data present (K > 0) for social metrics
- Ensure trajectory has multiple timesteps (T > 1) for velocity/acceleration metrics

### "Metrics don't match paper values"

**Possible causes**:
1. Different threshold parameters (check defaults in `robot_sf/benchmark/constants.py`)
2. Different collision distance definition
3. Edge case handling differences

**Debug approach**:
```python
# Print intermediate values
print(f"Trajectory length: {data.robot_pos.shape[0]}")
print(f"Pedestrians: {data.peds_pos.shape[1]}")
print(f"Goal reached at: {data.reached_goal_step}")
print(f"Collision threshold: {metrics.D_COLL}")
```

### "Performance too slow"

**Expected performance**: < 100ms per episode for 50 pedestrians

**If slower**:
- Profile specific metrics using `cProfile`
- Check trajectory length (very long episodes may be slow)
- Verify NumPy vectorization is working (no Python loops)

---

## Next Steps

- **Full Documentation**: See `docs/benchmark.md` for detailed metric formulas
- **Advanced Usage**: See `examples/demo_aggregate.py` for batch processing
- **Testing**: See `tests/test_metrics.py` for usage examples
- **Paper Reference**: See paper 2306.16740v4 Table 1 for formal definitions

## Getting Help

- Check metric docstrings: `help(metrics.success_rate)`
- Review test cases: `tests/test_metrics.py`
- See existing metrics: `robot_sf/benchmark/metrics.py`
- File issues on GitHub with minimal reproduction example
