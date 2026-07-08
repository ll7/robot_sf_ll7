# Research: Metrics from Paper 2306.16740v4

**Purpose**: Resolve unknowns and establish implementation patterns for 22 social navigation metrics
**Date**: October 23, 2025
**Status**: Complete

## Research Tasks

### 1. EpisodeData Sufficiency for New Metrics

**Question**: Does the existing `EpisodeData` dataclass provide all necessary fields for computing the 22 paper metrics?

**Current EpisodeData Structure** (from `robot_sf/benchmark/metrics.py`):
```python
@dataclass
class EpisodeData:
    robot_pos: np.ndarray      # (T,2) robot positions
    robot_vel: np.ndarray      # (T,2) robot velocities
    robot_acc: np.ndarray      # (T,2) robot accelerations
    peds_pos: np.ndarray       # (T,K,2) pedestrian positions
    ped_forces: np.ndarray     # (T,K,2) social force per pedestrian
    goal: np.ndarray           # (2,) target position
    dt: float                  # timestep duration
    reached_goal_step: int | None  # first step reaching goal
    force_field_grid: dict[str, np.ndarray] | None  # optional force field
```

**Analysis by Metric Category**:

#### NHT Metrics Coverage:
- ✅ **Success (S)**: Can use `reached_goal_step` (already used by existing `success()` function)
- ⚠️ **Collision (C)**: Need collision detection - currently inferred from min distance < threshold
- ⚠️ **Wall Collisions (WC)**: Need wall/obstacle positions (not in EpisodeData)
- ⚠️ **Agent Collisions (AC)**: Need other robot positions (not in EpisodeData - only pedestrians)
- ⚠️ **Human Collisions (HC)**: Can use pedestrian proximity (via `peds_pos`)
- ⚠️ **Timeout (TO)**: Need horizon parameter and timeout flag (not in EpisodeData)
- ⚠️ **Failure to progress (FP)**: Can compute from `robot_pos` + distance/time thresholds
- ⚠️ **Stalled time (ST)**: Can compute from `robot_vel` + threshold
- ✅ **Time (T)**: Can use `reached_goal_step * dt`
- ✅ **Path length (PL)**: Can compute from `robot_pos` (similar to existing `path_efficiency`)
- ✅ **SPL**: Can compute from PL + shortest path + success

#### SHT Metrics Coverage:
- ✅ **Velocity (V_min/avg/max)**: Have `robot_vel`
- ✅ **Acceleration (A_min/avg/max)**: Have `robot_acc`
- ⚠️ **Jerk (J_min/avg/max)**: Need to compute from `robot_acc` (derivative)
- ⚠️ **Clearing distance (CD_min/avg)**: Need obstacle positions (walls, static objects)
- ✅ **Space compliance (SC)**: Can compute from `peds_pos` (min distance to pedestrians)
- ✅ **DH_min**: Can compute from `peds_pos` (min distance to any pedestrian)
- ⚠️ **TTC**: Need pedestrian velocities for linear projection (only have positions)
- ⚠️ **Aggregated Time (AT)**: Need multi-agent goal data (cooperative agents set)

**Decision**: EpisodeData is **partially sufficient**. Need to:
1. Add optional fields for wall/obstacle data
2. Add episode metadata (horizon, timeout flag)
3. Compute pedestrian velocities from position differences
4. Document which metrics require extended data vs. are computable now

**Rationale**: Minimize breaking changes by making new fields optional with defaults

**Alternatives Considered**:
- Create new EpisodeDataExtended: rejected (breaks compatibility)
- Pass additional parameters to each metric function: rejected (inconsistent API)
- Compute missing data on-the-fly: accepted for pedestrian velocities, jerk

---

### 2. Collision Detection Strategy

**Question**: How should we detect and categorize collisions (wall, agent, human)?

**Current Approach** (from existing `collisions()` metric):
```python
def collisions(data: EpisodeData) -> float:
    """Count timesteps where min pedestrian distance < D_COLL."""
    # Uses robot_sf.benchmark.constants.COLLISION_DIST
```

**Analysis**:
- Current implementation only detects pedestrian collisions via distance threshold
- Uses constant `D_COLL` from `robot_sf/benchmark/constants.py`
- Does not distinguish collision types

**Decision**: Implement hierarchical collision detection:
1. **Human Collisions (HC)**: Use existing pattern with `peds_pos` and `D_COLL` threshold
2. **Wall Collisions (WC)**: Require optional obstacle/wall data in EpisodeData or episode metadata
3. **Agent Collisions (AC)**: Require optional other-agent data (if multi-robot scenarios supported)
4. **Total Collisions (C)**: Sum of WC + AC + HC where data available

**Rationale**: 
- Maintains backward compatibility with existing single-robot, pedestrian-only scenarios
- Returns 0 or NaN for collision types when relevant data unavailable
- Aligns with constitution's requirement for graceful degradation

**Implementation Notes**:
- Add `obstacles: np.ndarray | None` field to EpisodeData (optional)
- Add `other_agents_pos: np.ndarray | None` field to EpisodeData (optional)
- Document that WC/AC return 0.0 when obstacle/agent data not provided

---

### 3. Timeout and Progress Tracking

**Question**: How to track timeout and failure-to-progress without modifying episode runner?

**Current State**:
- Existing `success()` metric uses `reached_goal_step` and `horizon` parameter
- No explicit timeout flag in EpisodeData
- Episode runner may have timeout logic but not exposed to metrics

**Decision**: Pass `horizon` as parameter to relevant metrics
- **Timeout (TO)**: `1.0 if reached_goal_step is None else 0.0` (assume None means timeout/truncation)
- **Failure to progress (FP)**: Compute from trajectory - count intervals where robot doesn't approach goal for threshold time
- Metrics requiring horizon take it as explicit parameter (like existing `success()` and `time_to_goal_norm()`)

**Rationale**:
- Aligns with existing patterns (`success(data, horizon=...)`)
- No EpisodeData changes needed
- Clear semantics: None reached_goal_step = didn't reach goal (timeout or other failure)

---

### 4. Pedestrian Velocity and Jerk Computation

**Question**: How to compute pedestrian velocities and robot jerk when only positions/velocities available?

**Decision**: Compute from finite differences:

**Pedestrian Velocity**:
```python
def _compute_ped_velocities(peds_pos: np.ndarray, dt: float) -> np.ndarray:
    """Compute pedestrian velocities from positions via finite difference.
    
    Args:
        peds_pos: (T, K, 2) array of pedestrian positions
        dt: timestep duration
        
    Returns:
        (T, K, 2) array with velocities (first timestep = 0)
    """
    if peds_pos.shape[0] < 2:
        return np.zeros_like(peds_pos)
    
    # Forward difference: v[t] = (pos[t+1] - pos[t]) / dt
    vel = np.zeros_like(peds_pos)
    vel[:-1] = (peds_pos[1:] - peds_pos[:-1]) / dt
    vel[-1] = vel[-2]  # duplicate last velocity
    return vel
```

**Robot Jerk**:
```python
def _compute_jerk(robot_acc: np.ndarray, dt: float) -> np.ndarray:
    """Compute jerk (derivative of acceleration) via finite difference.
    
    Args:
        robot_acc: (T, 2) array of robot accelerations
        dt: timestep duration
        
    Returns:
        (T, 2) array of jerk values
    """
    if robot_acc.shape[0] < 2:
        return np.zeros_like(robot_acc)
    
    jerk = np.zeros_like(robot_acc)
    jerk[:-1] = (robot_acc[1:] - robot_acc[:-1]) / dt
    jerk[-1] = jerk[-2]
    return jerk
```

**Rationale**: 
- Standard numerical differentiation approach
- Maintains consistency with how velocities/accelerations likely computed in simulation
- Simple, deterministic, no new dependencies

---

### 5. Edge Case Handling Patterns

**Question**: How to handle edge cases consistently across all metrics?

**Analysis of Existing Patterns**:
```python
# Pattern 1: Return NaN when data insufficient
def min_distance(data: EpisodeData) -> float:
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    # ... compute ...

# Pattern 2: Return 0 when meaningless
def collisions(data: EpisodeData) -> float:
    if data.peds_pos.shape[1] == 0:
        return 0.0
    # ... compute ...

# Pattern 3: Return safe default (1.0 for ratios)
def path_efficiency(...) -> float:
    if data.robot_pos.shape[0] < 2:
        return 1.0
    # ... compute ...
```

**Decision**: Adopt semantic-based pattern:
- **Counts (C, WC, AC, HC, FP)**: Return `0.0` when data unavailable (nothing to count)
- **Distances (CD, DH)**: Return `float("nan")` when undefined (can't measure what doesn't exist)
- **Ratios/Probabilities (S, TO, SC, SPL)**: Return appropriate limit (0 or 1) or NaN based on semantics
- **Statistics (V/A/J min/avg/max, T, PL)**: Return `float("nan")` for empty trajectories
- **Time metrics (ST, T, AT, TTC)**: Return `0.0` for instant/empty trajectories

**Rationale**: 
- Makes aggregation meaningful (NaN excluded from means; 0s contribute)
- Aligns with paper definitions and existing codebase patterns
- Clear error signaling vs. valid zero values

---

### 6. Performance Optimization Strategy

**Question**: How to meet < 100ms per episode target for 50 pedestrians?

**Analysis**:
- Existing metrics use vectorized NumPy operations
- Current pattern: one function per metric (clean, testable)
- Potential bottleneck: repeated distance computations

**Decision**: 
1. Keep one-function-per-metric pattern (readability, testability)
2. Use memoization/caching only if profiling shows bottleneck
3. Leverage existing NumPy vectorization
4. Compute shared intermediate values once where beneficial (e.g., distance matrix)

**Implementation Note**:
Consider adding optional internal helper:
```python
def _compute_distance_matrix(data: EpisodeData) -> np.ndarray:
    """Cache computation of robot-pedestrian distances (T, K)."""
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    return np.linalg.norm(diffs, axis=2)
```

**Rationale**:
- Premature optimization avoided
- Pattern established for future optimization if needed
- Metrics remain independently callable

---

### 7. Integration with Existing Benchmark Infrastructure

**Question**: How to integrate new metrics without breaking existing workflows?

**Analysis of Current Integration Points**:
1. `robot_sf/benchmark/runner.py` - calls metric functions
2. `robot_sf/benchmark/aggregate.py` - aggregates results
3. Schema validation in `robot_sf/benchmark/schemas/`
4. Existing tests in `tests/test_metrics.py`

**Decision**: 
- Add new metric functions to `robot_sf/benchmark/metrics.py` following existing pattern
- Export new functions in module's `__all__` if present
- Ensure metric names match schema expectations (lowercase_with_underscores)
- Add tests in `tests/test_metrics.py` alongside existing metric tests
- Update documentation in `docs/benchmark.md`

**No Breaking Changes Required**:
- Schema already supports arbitrary numeric metrics via `additionalProperties`
- Aggregation already handles any numeric metric
- Runner can call new metrics selectively

---

## Summary of Decisions

### EpisodeData Extensions (Optional Fields)
```python
@dataclass
class EpisodeData:
    # ... existing fields ...
    obstacles: np.ndarray | None = None  # (M, 2) obstacle/wall positions
    other_agents_pos: np.ndarray | None = None  # (T, J, 2) other robot positions
```

### Helper Functions (Internal)
```python
def _compute_ped_velocities(peds_pos, dt) -> np.ndarray: ...
def _compute_jerk(robot_acc, dt) -> np.ndarray: ...
def _compute_distance_matrix(data) -> np.ndarray: ...  # optional optimization
```

### Metric Implementation Categories

**Category 1: Ready to Implement** (10 metrics)
- Success (S), Time (T), Path Length (PL), SPL
- Velocity stats (V_min/avg/max)
- Acceleration stats (A_min/avg/max)
- Human distance (DH_min)

**Category 2: Need Jerk Computation** (3 metrics)
- Jerk stats (J_min/avg/max)

**Category 3: Need Pedestrian Velocities** (2 metrics)
- Space Compliance (SC)
- Time to Collision (TTC)

**Category 4: Need Optional Data** (5 metrics)
- Wall Collisions (WC) - needs obstacles
- Agent Collisions (AC) - needs other_agents_pos
- Clearing Distance (CD) - needs obstacles
- Collision total (C) - sum of WC+AC+HC
- Aggregated Time (AT) - needs multi-agent goal data

**Category 5: Computable from Trajectory** (2 metrics)
- Failure to Progress (FP)
- Stalled Time (ST)

### Testing Strategy
- Unit tests for each metric with synthetic data
- Edge case tests (empty trajectories, no pedestrians, single timestep)
- Integration test with real episode data
- Performance benchmark (ensure < 100ms target)

### Documentation Requirements
- Add metric formulas and units to `docs/benchmark.md`
- Update `docs/README.md` to link metrics documentation
- Add docstrings to each metric function (formula, units, edge cases)
- Note which metrics require optional EpisodeData fields
