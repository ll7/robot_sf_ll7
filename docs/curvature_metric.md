# Curvature Path Metric Implementation

## Overview

The curvature path metric has been implemented to complement the existing jerk metric for assessing path smoothness in robot navigation. This metric provides insights into the sharpness of turns and overall path curvature.

## Mathematical Foundation

The curvature metric uses the cross product formula for 2D paths:

```
κ = |v × a| / |v|³
```

Where:
- `v` is the velocity vector (computed from position differences)
- `a` is the acceleration vector (computed from velocity differences)
- `κ` is the curvature at each point

For discrete trajectory data, velocities and accelerations are computed using finite differences:
- `v(t) = (pos(t+1) - pos(t)) / dt`
- `a(t) = (v(t+1) - v(t)) / dt`

## Implementation Details

### Function: `curvature_mean(data: EpisodeData) -> float`

**Location**: `robot_sf/benchmark/metrics.py`

**Parameters**:
- `data`: EpisodeData container with robot trajectory information

**Returns**:
- Mean curvature magnitude over the trajectory
- Returns 0.0 for trajectories with fewer than 4 timesteps
- Returns 0.0 if all velocity magnitudes are near zero

**Algorithm**:
1. Extract robot positions from episode data
2. Compute velocities using finite differences: `v_t = (pos_{t+1} - pos_t) / dt`
3. Compute accelerations using finite differences: `a_t = (v_{t+1} - v_t) / dt`
4. For each point with sufficient velocity magnitude:
   - Calculate cross product: `|v_x * a_y - v_y * a_x|`
   - Compute curvature: `κ = cross_product / |v|³`
5. Return the mean of all computed curvature values

### Integration Points

1. **Metrics Orchestration**:
   - Added to `METRIC_NAMES` list
   - Called in `compute_all_metrics()` function
   - Exported in `__all__` list

2. **SNQI Scoring**:
   - Added `w_curvature` weight parameter
   - Integrated curvature normalization using baseline statistics
   - Curvature contributes as a penalty term (higher curvature = lower score)

3. **Baseline Statistics**:
   - Added to `DEFAULT_METRICS` in `baseline_stats.py`
   - Enables proper normalization for SNQI scoring

## Usage Examples

### Basic Usage
```python
from robot_sf.benchmark.metrics import curvature_mean, EpisodeData

# Compute curvature for an episode
curvature = curvature_mean(episode_data)
print(f"Mean path curvature: {curvature}")
```

### In Complete Metrics Analysis
```python
from robot_sf.benchmark.metrics import compute_all_metrics

# Get all metrics including curvature
metrics = compute_all_metrics(episode_data, horizon=100)
print(f"Curvature: {metrics['curvature_mean']}")
print(f"Jerk: {metrics['jerk_mean']}")
```

### In SNQI Scoring
```python
from robot_sf.benchmark.metrics import snqi

weights = {
    "w_success": 1.0,
    "w_jerk": 0.2,
    "w_curvature": 0.3,  # Weight for curvature penalty
    # ... other weights
}

score = snqi(metrics, weights, baseline_stats)
```

## Test Coverage

### Test Cases
1. **Circular Path**: Validates known curvature (κ = 1/radius)
2. **Straight Line**: Validates zero curvature for linear motion
3. **Insufficient Points**: Edge case handling for <4 timesteps
4. **SNQI Integration**: Validates proper scoring integration

### Test Functions
- `test_curvature_mean()`: Tests circular path with known curvature
- `test_curvature_mean_straight_line()`: Tests zero curvature for straight motion
- `test_curvature_mean_insufficient_points()`: Tests edge case handling
- `test_snqi_scoring()`: Tests SNQI integration (updated to include curvature)

## Expected Values

### Interpretation
- **Low curvature (0.0 - 0.5)**: Smooth, straight paths with gentle turns
- **Medium curvature (0.5 - 2.0)**: Moderate turning behavior
- **High curvature (>2.0)**: Sharp turns, erratic movement patterns

### Reference Values
- **Straight line**: κ = 0.0
- **Circle of radius R**: κ = 1/R
- **Typical robot navigation**: κ = 0.1 - 1.0 (depending on scenario)

## Relationship to Jerk

Both metrics assess path smoothness but capture different aspects:

- **Jerk**: Measures smoothness of acceleration changes (comfort, control stability)
- **Curvature**: Measures sharpness of path geometry (efficiency, naturalness)

Together, they provide a comprehensive assessment of path quality for robot navigation evaluation.