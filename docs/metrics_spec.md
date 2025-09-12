# Robot-SF Metrics Specification

## Overview

This document provides formal definitions for all metrics computed in the Robot-SF simulation environment, with particular focus on social comfort and interaction metrics used for SNQI stability and sensitivity analysis.

## Symbol Table

| Symbol | Description |
|--------|-------------|
| `r_t` | Robot position at timestep t |
| `p_{j,t}` | Position of pedestrian j at timestep t |
| `F_{j,t}` | Net social force vector on pedestrian j at timestep t |
| `J_t` | Set of pedestrians present at timestep t |
| `T` | Total number of timesteps in episode |
| `P` | Total number of pedestrians |
| `d_{r,j}(t)` | Euclidean distance between robot and pedestrian j at timestep t |

## Metrics Definitions

### 1. Mean Interpersonal Distance (MID)

**Definition**: The average distance between the robot and all pedestrians across all valid timesteps in an episode.

**Formula**:
```
MID = (Σ_t Σ_{j ∈ J_t} ||r_t - p_{j,t}||₂) / (Σ_t |J_t|)
```

**Computation**:
1. For each timestep t, compute distances `d_{r,j}(t) = ||r_t - p_{j,t}||₂` for all pedestrians j present
2. Sum all distances across all timesteps and pedestrians
3. Divide by total count of robot-pedestrian pairs across all timesteps
4. Skip timesteps where `|J_t| = 0` (no pedestrians present)

**Edge Cases**:
- If no pedestrians are present for the entire episode (`Σ_t |J_t| = 0`), return `NaN`
- If a pedestrian appears/disappears mid-episode, include only timesteps where present
- Downstream aggregation uses `nanmean()` to handle NaN values gracefully

### 2. Per-Pedestrian Force Quantiles

**Definition**: Statistical distribution (quantiles) of force magnitudes experienced by each pedestrian, aggregated across the episode.

**Per-Pedestrian Force Magnitude**:
```
M_j = {||F_{j,t}||₂ | t ∈ T_j}
```
where `T_j` is the set of timesteps when pedestrian j is present.

**Episode Quantiles**:
```
Q_j(q) = Quantile_q(M_j)  for q ∈ {0.50, 0.90, 0.95}
```

**Episode Aggregate**:
```
Force_q = mean_j Q_j(q)  (mean across all pedestrians)
```

**Output Keys**:
- `ped_force_q50`: 50th percentile (median) of force magnitudes
- `ped_force_q90`: 90th percentile of force magnitudes  
- `ped_force_q95`: 95th percentile of force magnitudes

**Computation**:
1. For each pedestrian j, collect force magnitudes `||F_{j,t}||₂` for all timesteps where j is present
2. Apply `np.nan_to_num(forces, copy=False)` to handle NaN/inf values before magnitude computation
3. Compute quantiles using `np.nanpercentile()` for each pedestrian
4. Take mean of quantiles across all pedestrians

**Edge Cases**:
- If no pedestrians are present for entire episode, return `NaN` for all quantiles
- If a pedestrian has ≤1 force samples, its quantiles equal its single magnitude value (or `NaN` if no samples)
- Force arrays containing NaN are cleaned with `np.nan_to_num()` before processing
- Use `np.nanpercentile()` for robustness against remaining NaN values

## Implementation Guidelines

### Vectorization Requirements
- All computations must be O(T × P) vectorized operations
- Avoid Python loops over timesteps or pedestrians
- Use numpy broadcasting where possible

### Numerical Stability
- Handle division by zero gracefully (return NaN)
- Use `np.nan_to_num()` for force preprocessing
- Use `np.nanpercentile()` and `nanmean()` for robust aggregation

### Data Collection
- Collect data during each simulation timestep
- Store accumulated data per episode
- Reset accumulators at episode boundaries

## Integration Points

### Data Sources
- **Robot positions**: `simulator.robot_pos[0]` (assuming single robot)
- **Pedestrian positions**: `simulator.ped_pos` 
- **Force data**: `simulator.pysf_sim.compute_forces()`

### Output Integration
- New metrics added to `meta_dict()` return values
- Compatible with existing `PedEnvMetrics` update mechanism
- Follows existing naming conventions and data types

## Testing Requirements

### Unit Tests
- Happy path with multiple pedestrians and varying forces
- Edge case: no pedestrians for entire episode → expected NaNs
- Edge case: single pedestrian, constant distance → mean equals that distance
- Edge case: force array with injected NaN → verify handling
- Edge case: pedestrian appears/disappears mid-episode → correct timestep filtering

### Performance Tests
- Verify O(T × P) complexity
- Benchmark with large T, P arrays (optional micro-benchmark)
- No regression in existing metrics computation time