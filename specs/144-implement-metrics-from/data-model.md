# Data Model: Metrics from Paper 2306.16740v4

**Purpose**: Define data structures for paper metrics implementation
**Date**: October 23, 2025
**Status**: Complete

## Core Entities

### EpisodeData (Extended)

**Location**: `robot_sf/benchmark/metrics.py`

**Purpose**: Container for episode trajectory data with optional collision detection support

**Changes**: Add two optional fields to existing dataclass

```python
@dataclass
class EpisodeData:
    """Container for a single episode trajectory.

    Attributes
    ----------
    robot_pos : (T,2) array
        Robot position at each timestep
    robot_vel : (T,2) array
        Robot velocity at each timestep
    robot_acc : (T,2) array
        Robot acceleration at each timestep
    peds_pos : (T,K,2) array
        Pedestrian positions at each timestep (K pedestrians)
    ped_forces : (T,K,2) array
        Social force magnitudes per pedestrian
    goal : (2,) array
        Robot target position
    dt : float
        Simulation timestep duration (seconds)
    reached_goal_step : int | None
        First step index where robot reached goal (None if not reached)
    force_field_grid : dict[str, np.ndarray] | None
        Optional pre-sampled force field (keys: X, Y, Fx, Fy)
    obstacles : np.ndarray | None
        NEW: (M, 2) array of obstacle/wall positions for collision detection
        Used by: wall_collisions (WC), clearing_distance (CD)
        Default: None (metrics return 0.0 or NaN)
    other_agents_pos : np.ndarray | None
        NEW: (T, J, 2) array of other robot positions for multi-agent scenarios
        Used by: agent_collisions (AC)
        Default: None (metric returns 0.0)
    """
    
    robot_pos: np.ndarray
    robot_vel: np.ndarray
    robot_acc: np.ndarray
    peds_pos: np.ndarray
    ped_forces: np.ndarray
    goal: np.ndarray
    dt: float
    reached_goal_step: int | None = None
    force_field_grid: dict[str, np.ndarray] | None = None
    obstacles: np.ndarray | None = None  # NEW
    other_agents_pos: np.ndarray | None = None  # NEW
```

**Validation Rules**:
- `robot_pos`, `robot_vel`, `robot_acc` must have shape `(T, 2)` where T ≥ 1
- `peds_pos` must have shape `(T, K, 2)` where K ≥ 0 (no pedestrians valid)
- `ped_forces` must match `peds_pos` shape
- `goal` must have shape `(2,)`
- `dt` must be > 0
- If `obstacles` provided, must have shape `(M, 2)` where M ≥ 0
- If `other_agents_pos` provided, must have shape `(T, J, 2)` where J ≥ 0
- `reached_goal_step` must be in range `[0, T)` if not None

**State Transitions**: N/A (immutable data container)

---

## Metric Function Signatures

All metric functions follow the pattern established in `robot_sf/benchmark/metrics.py`:

### Pure Functions (No Additional Parameters)

```python
def metric_name(data: EpisodeData) -> float:
    """Short description.
    
    Long description including formula, units, and edge case handling.
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
        
    Returns
    -------
    float
        Metric value with specified units (or NaN if undefined)
        
    Notes
    -----
    - Edge case behavior documented here
    - Paper reference: [paper name] equation X
    """
```

### Functions with Additional Parameters

```python
def metric_name(data: EpisodeData, *, horizon: int, threshold: float = DEFAULT) -> float:
    """Short description with parameter explanation.
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    horizon : int
        Maximum episode length (timesteps)
    threshold : float, optional
        Detection threshold (default: DEFAULT from constants)
        
    Returns
    -------
    float
        Metric value
    """
```

---

## Metric Catalog

### NHT (Navigation/Hard Task) Metrics

| Metric | Function Name | Return Type | Parameters | Units | Range |
|--------|--------------|-------------|------------|-------|-------|
| Success | `success_rate` | float | `horizon: int` | boolean | [0, 1] |
| Collision | `collision_count` | float | - | count | [0, ∞) |
| Wall Collisions | `wall_collisions` | float | `threshold: float` | count | [0, ∞) |
| Agent Collisions | `agent_collisions` | float | `threshold: float` | count | [0, ∞) |
| Human Collisions | `human_collisions` | float | `threshold: float` | count | [0, ∞) |
| Timeout | `timeout` | float | `horizon: int` | boolean | [0, 1] |
| Failure to Progress | `failure_to_progress` | float | `distance_threshold: float, time_threshold: float` | count | [0, ∞) |
| Stalled Time | `stalled_time` | float | `velocity_threshold: float` | seconds | [0, ∞) |
| Time to Goal | `time_to_goal` | float | - | seconds | [0, ∞) |
| Path Length | `path_length` | float | - | meters | [0, ∞) |
| Success Path Length | `success_path_length` | float | `horizon: int, optimal_length: float` | ratio | [0, 1] |

### SHT (Social/Human-aware Task) Metrics

| Metric | Function Name | Return Type | Parameters | Units | Range |
|--------|--------------|-------------|------------|-------|-------|
| Velocity Min | `velocity_min` | float | - | m/s | [-∞, ∞) |
| Velocity Avg | `velocity_avg` | float | - | m/s | [-∞, ∞) |
| Velocity Max | `velocity_max` | float | - | m/s | [-∞, ∞) |
| Acceleration Min | `acceleration_min` | float | - | m/s² | [-∞, ∞) |
| Acceleration Avg | `acceleration_avg` | float | - | m/s² | [-∞, ∞) |
| Acceleration Max | `acceleration_max` | float | - | m/s² | [-∞, ∞) |
| Jerk Min | `jerk_min` | float | - | m/s³ | [-∞, ∞) |
| Jerk Avg | `jerk_avg` | float | - | m/s³ | [-∞, ∞) |
| Jerk Max | `jerk_max` | float | - | m/s³ | [-∞, ∞) |
| Clearing Distance Min | `clearing_distance_min` | float | - | meters | [0, ∞) |
| Clearing Distance Avg | `clearing_distance_avg` | float | - | meters | [0, ∞) |
| Space Compliance | `space_compliance` | float | `threshold: float` | ratio | [0, 1] |
| Distance to Human Min | `distance_to_human_min` | float | - | meters | [0, ∞) |
| Time to Collision | `time_to_collision_min` | float | - | seconds | [0, ∞) |
| Aggregated Time | `aggregated_time` | float | `agent_ids: list[int]` | seconds | [0, ∞) |

---

## Internal Helper Functions

### Pedestrian Velocity Computation

```python
def _compute_ped_velocities(peds_pos: np.ndarray, dt: float) -> np.ndarray:
    """Compute pedestrian velocities from positions via finite difference.
    
    Parameters
    ----------
    peds_pos : np.ndarray
        (T, K, 2) array of pedestrian positions
    dt : float
        Timestep duration (seconds)
        
    Returns
    -------
    np.ndarray
        (T, K, 2) array of velocities (first timestep uses forward difference)
    """
```

### Jerk Computation

```python
def _compute_jerk(robot_acc: np.ndarray, dt: float) -> np.ndarray:
    """Compute jerk (acceleration derivative) via finite difference.
    
    Parameters
    ----------
    robot_acc : np.ndarray
        (T, 2) array of robot accelerations
    dt : float
        Timestep duration (seconds)
        
    Returns
    -------
    np.ndarray
        (T, 2) array of jerk values
    """
```

### Distance Matrix (Optional Optimization)

```python
def _compute_distance_matrix(data: EpisodeData) -> np.ndarray:
    """Compute robot-pedestrian distance matrix.
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
        
    Returns
    -------
    np.ndarray
        (T, K) array of distances from robot to each pedestrian
    """
```

---

## Edge Case Handling Specification

| Scenario | Behavior | Rationale |
|----------|----------|-----------|
| No pedestrians (K=0) | Count metrics → 0.0, Distance metrics → NaN | Nothing to count; distance undefined |
| Single timestep (T=1) | Velocity/accel/jerk → NaN, Time → 0.0 | Insufficient data; instant completion |
| Empty trajectory (T=0) | All metrics → NaN | No data to process |
| Goal not reached | `success_rate` → 0.0, `timeout` → 1.0 | Explicit failure |
| No obstacles | `wall_collisions` → 0.0, `clearing_distance` → NaN | Nothing to collide with; distance undefined |
| No other agents | `agent_collisions` → 0.0 | Nothing to collide with |
| Zero velocity threshold | `stalled_time` → total time | Robot always below threshold |
| Infinite threshold | Compliance metrics → 1.0 | Never violated |

---

## Relationships

```
EpisodeData (1) ──computes──> (22) MetricValue
                │
                ├──requires (optional)──> Obstacles (wall collisions, clearing distance)
                └──requires (optional)──> OtherAgents (agent collisions)

MetricValue ──aggregates into──> SummaryStatistics (mean, median, CI)
```

---

## Migration Notes

**Backward Compatibility**: 
- New `obstacles` and `other_agents_pos` fields are optional (default None)
- Existing code creating `EpisodeData` without these fields continues to work
- Metrics gracefully degrade when optional fields absent (return 0.0 or NaN)

**Schema Impact**:
- No changes to `episode.schema.v1.json` required
- New metric names added as additional keys in `metrics` object
- All metric values are numeric (float), conforming to existing schema

**Testing Impact**:
- Existing tests unaffected (new fields optional)
- New tests needed for 22 metric functions
- Integration tests should verify optional field handling
