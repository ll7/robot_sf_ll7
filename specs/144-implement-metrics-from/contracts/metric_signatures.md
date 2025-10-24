# Metric Function Contracts

**Purpose**: API contracts for paper 2306.16740v4 metrics
**Format**: Python type signatures with behavioral contracts
**Date**: October 23, 2025

## Contract Principles

1. **Pure Functions**: No side effects, deterministic output for given input
2. **Defensive Validation**: Validate inputs, return NaN/0.0 for invalid cases
3. **Clear Documentation**: Docstring must include formula, units, edge cases
4. **Type Safety**: Use type hints for all parameters and returns
5. **Performance**: O(T*K) maximum complexity where T=timesteps, K=pedestrians

---

## NHT (Navigation/Hard Task) Metrics

### success_rate

```python
def success_rate(data: EpisodeData, *, horizon: int) -> float:
    """Binary success indicator (1.0 if goal reached before horizon, else 0.0).
    
    Formula: S = 1 if (reached_goal_step < horizon) AND (collisions == 0) else 0
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with reached_goal_step field
    horizon : int
        Maximum allowed timesteps
        
    Returns
    -------
    float
        1.0 if successful, 0.0 otherwise
        
    Behavioral Contract
    -------------------
    - Returns 0.0 if data.reached_goal_step is None
    - Returns 0.0 if data.reached_goal_step >= horizon
    - Returns 0.0 if any collisions detected (HC > 0)
    - Returns 1.0 only if goal reached collision-free before horizon
    
    Edge Cases
    ----------
    - Empty trajectory (T=0): returns 0.0
    - Robot starts at goal: returns 1.0 if no collisions
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Success (S)
    """
```

### collision_count

```python
def collision_count(data: EpisodeData) -> float:
    """Total collision count (sum of wall, agent, and human collisions).
    
    Formula: C = WC + AC + HC
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with optional collision data
        
    Returns
    -------
    float
        Total number of collisions detected
        
    Behavioral Contract
    -------------------
    - Returns sum of wall_collisions + agent_collisions + human_collisions
    - If obstacles is None: WC = 0
    - If other_agents_pos is None: AC = 0
    - HC always computed from peds_pos
    
    Edge Cases
    ----------
    - No pedestrians: returns WC + AC (may be 0)
    - No obstacles/agents: returns HC only
    - All None: returns 0.0
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Collision (C)
    """
```

### wall_collisions

```python
def wall_collisions(
    data: EpisodeData, 
    *, 
    threshold: float = D_COLL
) -> float:
    """Count collisions with walls/obstacles.
    
    Formula: WC = sum_t I(min_m ||robot_pos[t] - obstacles[m]|| < threshold)
    where I is indicator function
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with optional obstacles field
    threshold : float, optional
        Collision distance threshold (default: D_COLL from constants)
        
    Returns
    -------
    float
        Number of timesteps with wall collision
        
    Behavioral Contract
    -------------------
    - Returns 0.0 if data.obstacles is None
    - Returns 0.0 if obstacles array is empty (M=0)
    - Collision detected when min distance to any obstacle < threshold
    - Counts unique timesteps (not obstacle-timestep pairs)
    
    Edge Cases
    ----------
    - No obstacles: returns 0.0
    - Robot teleports: may skip obstacles
    - threshold = 0: only counts exact contact
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Wall Collisions (WC)
    """
```

### agent_collisions

```python
def agent_collisions(
    data: EpisodeData,
    *,
    threshold: float = D_COLL
) -> float:
    """Count collisions with other agents/robots.
    
    Formula: AC = sum_t I(min_j ||robot_pos[t] - agents[t,j]|| < threshold)
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with optional other_agents_pos field
    threshold : float, optional
        Collision distance threshold (default: D_COLL from constants)
        
    Returns
    -------
    float
        Number of timesteps with agent collision
        
    Behavioral Contract
    -------------------
    - Returns 0.0 if data.other_agents_pos is None
    - Returns 0.0 if no other agents (J=0)
    - Collision detected when min distance to any agent < threshold
    - Counts unique timesteps
    
    Edge Cases
    ----------
    - Single robot scenario: returns 0.0
    - Self-collision: excluded (j ≠ ego_robot)
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Agent Collisions (AC)
    """
```

### human_collisions

```python
def human_collisions(
    data: EpisodeData,
    *,
    threshold: float = D_COLL
) -> float:
    """Count collisions with pedestrians/humans.
    
    Formula: HC = sum_t I(min_k ||robot_pos[t] - peds_pos[t,k]|| < threshold)
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with peds_pos field
    threshold : float, optional
        Collision distance threshold (default: D_COLL from constants)
        
    Returns
    -------
    float
        Number of timesteps with pedestrian collision
        
    Behavioral Contract
    -------------------
    - Returns 0.0 if no pedestrians (K=0)
    - Collision detected when min distance to any pedestrian < threshold
    - Counts unique timesteps (not pedestrian-timestep pairs)
    - Matches behavior of existing collisions() metric
    
    Edge Cases
    ----------
    - No pedestrians: returns 0.0
    - Multiple simultaneous collisions: counts as 1 per timestep
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Human Collisions (HC)
    """
```

### timeout

```python
def timeout(data: EpisodeData, *, horizon: int) -> float:
    """Binary indicator for timeout failure (1.0 if timeout, else 0.0).
    
    Formula: TO = 1 if reached_goal_step is None else 0
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with reached_goal_step field
    horizon : int
        Maximum allowed timesteps
        
    Returns
    -------
    float
        1.0 if episode timed out, 0.0 otherwise
        
    Behavioral Contract
    -------------------
    - Returns 1.0 if data.reached_goal_step is None
    - Returns 0.0 if goal was reached (even if >= horizon)
    - Assumes None means truncation/timeout
    
    Edge Cases
    ----------
    - Immediate success: returns 0.0
    - Other failure modes: returns 1.0 (conservative)
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Timeout Before reaching goal (TO)
    """
```

### failure_to_progress

```python
def failure_to_progress(
    data: EpisodeData,
    *,
    distance_threshold: float = 0.1,
    time_threshold: float = 5.0
) -> float:
    """Count failure-to-progress events (robot doesn't approach goal).
    
    Formula: FP = count of intervals where robot doesn't reduce distance to goal
             by distance_threshold within time_threshold window
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with robot_pos and goal
    distance_threshold : float, optional
        Minimum required progress (meters, default: 0.1m)
    time_threshold : float, optional
        Time window for progress check (seconds, default: 5.0s)
        
    Returns
    -------
    float
        Number of failure-to-progress events detected
        
    Behavioral Contract
    -------------------
    - Computes distance to goal at each timestep
    - Slides window of time_threshold duration
    - Counts windows where distance reduction < distance_threshold
    - Returns 0.0 for trajectories shorter than time_threshold
    
    Edge Cases
    ----------
    - T*dt < time_threshold: returns 0.0
    - Robot at goal: returns 0.0
    - Oscillating robot: may count multiple events
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Failure to progress (FP)
    """
```

### stalled_time

```python
def stalled_time(
    data: EpisodeData,
    *,
    velocity_threshold: float = 0.05
) -> float:
    """Total time robot speed is below threshold (seconds).
    
    Formula: ST = sum_t I(||robot_vel[t]|| < threshold) * dt
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with robot_vel and dt
    velocity_threshold : float, optional
        Speed threshold for stalling (m/s, default: 0.05)
        
    Returns
    -------
    float
        Total stalled time in seconds
        
    Behavioral Contract
    -------------------
    - Computes speed magnitude at each timestep
    - Counts timesteps where speed < threshold
    - Multiplies count by dt for total time
    - Returns 0.0 if never below threshold
    
    Edge Cases
    ----------
    - Stationary robot: returns T * dt (total time)
    - threshold = 0: only counts exactly stationary
    - Single timestep: returns dt if below threshold
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Stalled time (ST)
    """
```

### time_to_goal

```python
def time_to_goal(data: EpisodeData) -> float:
    """Time from start to goal (seconds, or NaN if not reached).
    
    Formula: T = reached_goal_step * dt (if reached, else NaN)
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with reached_goal_step and dt
        
    Returns
    -------
    float
        Time to goal in seconds, or NaN if goal not reached
        
    Behavioral Contract
    -------------------
    - Returns reached_goal_step * dt if goal reached
    - Returns NaN if reached_goal_step is None
    - Includes time of goal-reaching timestep
    
    Edge Cases
    ----------
    - Goal not reached: returns NaN
    - Immediate success (step 0): returns 0.0
    - Truncated episode: returns NaN
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Time to reach goal (T)
    """
```

### path_length

```python
def path_length(data: EpisodeData) -> float:
    """Total path length traveled by robot (meters).
    
    Formula: PL = sum_{t=0}^{T-1} ||robot_pos[t+1] - robot_pos[t]||
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with robot_pos
        
    Returns
    -------
    float
        Total distance traveled in meters
        
    Behavioral Contract
    -------------------
    - Sums Euclidean distances between consecutive positions
    - Uses full trajectory (not just to goal)
    - Returns 0.0 for single-timestep trajectories
    - Never returns NaN (0.0 minimum)
    
    Edge Cases
    ----------
    - Stationary robot: returns 0.0
    - Teleportation: includes discontinuous jumps
    - T=1: returns 0.0
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Path length (PL)
    """
```

### success_path_length

```python
def success_path_length(
    data: EpisodeData,
    *,
    horizon: int,
    optimal_length: float
) -> float:
    """Success weighted by path efficiency (normalized inverse path length).
    
    Formula: SPL = S * (optimal_length / max(actual_length, optimal_length))
    where S is success indicator (0 or 1)
    
    Parameters
    ----------
    data : EpisodeData
        Episode trajectory
    horizon : int
        Maximum allowed timesteps
    optimal_length : float
        Shortest path length (straight-line distance to goal)
        
    Returns
    -------
    float
        SPL metric in range [0, 1]
        
    Behavioral Contract
    -------------------
    - Returns 0.0 if not successful (S=0)
    - Returns S * efficiency where efficiency ∈ [0, 1]
    - Efficiency = optimal / max(actual, optimal)
    - Clips efficiency to 1.0 (never > 1)
    
    Edge Cases
    ----------
    - optimal_length = 0: returns S (perfect efficiency)
    - actual < optimal: returns S (clipped to 1.0)
    - Failure: returns 0.0 regardless of path
    
    Paper Reference
    ---------------
    Table 1, NHT Metrics, Success weighted by path length (SPL)
    """
```

---

## SHT (Social/Human-aware Task) Metrics

### Velocity Statistics

```python
def velocity_min(data: EpisodeData) -> float:
    """Minimum linear velocity magnitude (m/s).
    
    Formula: V_min = min_t ||robot_vel[t]||
    
    Returns: float in [-∞, ∞) or NaN if T=0
    Paper Reference: Table 1, SHT Metrics, Velocity-based features
    """

def velocity_avg(data: EpisodeData) -> float:
    """Average linear velocity magnitude (m/s).
    
    Formula: V_avg = mean_t ||robot_vel[t]||
    
    Returns: float in [-∞, ∞) or NaN if T=0
    Paper Reference: Table 1, SHT Metrics, Velocity-based features
    """

def velocity_max(data: EpisodeData) -> float:
    """Maximum linear velocity magnitude (m/s).
    
    Formula: V_max = max_t ||robot_vel[t]||
    
    Returns: float in [-∞, ∞) or NaN if T=0
    Paper Reference: Table 1, SHT Metrics, Velocity-based features
    """
```

### Acceleration Statistics

```python
def acceleration_min(data: EpisodeData) -> float:
    """Minimum linear acceleration magnitude (m/s²).
    
    Formula: A_min = min_t ||robot_acc[t]||
    
    Returns: float in [-∞, ∞) or NaN if T=0
    Paper Reference: Table 1, SHT Metrics, Linear acceleration based features
    """

def acceleration_avg(data: EpisodeData) -> float:
    """Average linear acceleration magnitude (m/s²).
    
    Formula: A_avg = mean_t ||robot_acc[t]||
    
    Returns: float in [-∞, ∞) or NaN if T=0
    Paper Reference: Table 1, SHT Metrics, Linear acceleration based features
    """

def acceleration_max(data: EpisodeData) -> float:
    """Maximum linear acceleration magnitude (m/s²).
    
    Formula: A_max = max_t ||robot_acc[t]||
    
    Returns: float in [-∞, ∞) or NaN if T=0
    Paper Reference: Table 1, SHT Metrics, Linear acceleration based features
    """
```

### Jerk Statistics

```python
def jerk_min(data: EpisodeData) -> float:
    """Minimum jerk magnitude (m/s³).
    
    Formula: J_min = min_t ||d(robot_acc[t])/dt||
    Computed via: finite difference of acceleration
    
    Returns: float in [-∞, ∞) or NaN if T < 2
    Paper Reference: Table 1, SHT Metrics, Movement jerk
    """

def jerk_avg(data: EpisodeData) -> float:
    """Average jerk magnitude (m/s³).
    
    Formula: J_avg = mean_t ||d(robot_acc[t])/dt||
    
    Returns: float in [-∞, ∞) or NaN if T < 2
    Paper Reference: Table 1, SHT Metrics, Movement jerk
    """

def jerk_max(data: EpisodeData) -> float:
    """Maximum jerk magnitude (m/s³).
    
    Formula: J_max = max_t ||d(robot_acc[t])/dt||
    
    Returns: float in [-∞, ∞) or NaN if T < 2
    Paper Reference: Table 1, SHT Metrics, Movement jerk
    """
```

### Distance and Compliance Metrics

```python
def clearing_distance_min(data: EpisodeData) -> float:
    """Minimum distance to obstacles (meters).
    
    Formula: CD_min = min_t min_m ||robot_pos[t] - obstacles[m]||
    
    Returns: float in [0, ∞) or NaN if obstacles is None
    Paper Reference: Table 1, SHT Metrics, Clearing distance
    """

def clearing_distance_avg(data: EpisodeData) -> float:
    """Average minimum distance to obstacles (meters).
    
    Formula: CD_avg = mean_t min_m ||robot_pos[t] - obstacles[m]||
    
    Returns: float in [0, ∞) or NaN if obstacles is None
    Paper Reference: Table 1, SHT Metrics, Clearing distance
    """

def space_compliance(
    data: EpisodeData,
    *,
    threshold: float = 0.5
) -> float:
    """Ratio of trajectory within personal space threshold.
    
    Formula: SC = (# timesteps where min_k distance < threshold) / T
    
    Parameters
    ----------
    threshold : float, optional
        Personal space radius (meters, default: 0.5m)
    
    Returns: float in [0, 1] or NaN if K=0
    Paper Reference: Table 1, SHT Metrics, Space compliance (SC)
    """

def distance_to_human_min(data: EpisodeData) -> float:
    """Minimum distance to any human/pedestrian (meters).
    
    Formula: DH_min = min_t min_k ||robot_pos[t] - peds_pos[t,k]||
    
    Returns: float in [0, ∞) or NaN if K=0
    Paper Reference: Table 1, SHT Metrics, Minimum distance to human
    """

def time_to_collision_min(data: EpisodeData) -> float:
    """Minimum time to collision with pedestrian (seconds).
    
    Formula: TTC = min_{t,k} (d / ||v_rel||) where d = distance, v_rel = relative velocity
    Assumes linear extrapolation of current velocities
    
    Returns: float in [0, ∞) or NaN if K=0 or no collision trajectory
    Paper Reference: Table 1, SHT Metrics, Minimum time to collision
    """

def aggregated_time(
    data: EpisodeData,
    *,
    agent_goal_times: dict[int, int]
) -> float:
    """Time for cooperative agents to reach goals (seconds).
    
    Formula: AT = max_j (goal_time[j]) for j in cooperative set
    
    Parameters
    ----------
    agent_goal_times : dict[int, int]
        Mapping from agent ID to goal-reaching timestep
    
    Returns: float in [0, ∞) or NaN if no agents in set
    Paper Reference: Table 1, SHT Metrics, Aggregated Time
    """
```

---

## Contract Verification Checklist

- [ ] All 22 metric functions have type hints
- [ ] All functions have comprehensive docstrings (formula, params, returns, edge cases)
- [ ] Edge case behavior explicitly documented
- [ ] Return types match data-model.md specification
- [ ] Parameter names match paper notation where feasible
- [ ] Default values sourced from constants or documented
- [ ] No breaking changes to existing EpisodeData contract
- [ ] NaN vs 0.0 semantics clear and consistent
- [ ] Paper references included in docstrings
