# Implementation Tasks: Metrics from Paper 2306.16740v4

**Branch**: `144-implement-metrics-from`  
**Generated**: October 23, 2025  
**Total Tasks**: 35  
**Estimated Effort**: Medium (22 metric functions + tests + docs)

## Overview

This document provides a step-by-step implementation plan for adding 22 social navigation metrics from paper 2306.16740v4. Tasks are organized by user story to enable independent implementation and testing.

**User Stories**:
- **US1 (P1)**: Compute Paper-Specific Metrics - Core metric implementation
- **US2 (P2)**: Validate Against Paper Baselines - Test with synthetic scenarios
- **US3 (P3)**: Export Metrics in Standard Format - Integration verification

**Implementation Strategy**: MVP-first approach - complete US1 for immediate value, then incrementally add US2 and US3.

---

## Phase 1: Setup & Infrastructure

**Goal**: Prepare codebase for metric implementation

### T001: [Setup] Extend EpisodeData with optional fields
**File**: `robot_sf/benchmark/metrics.py`
**Story**: Foundation for all stories
**Dependencies**: None
**Parallelizable**: No

Add two optional fields to the `EpisodeData` dataclass:
```python
@dataclass
class EpisodeData:
    # ... existing fields ...
    obstacles: np.ndarray | None = None  # (M, 2) for wall collisions
    other_agents_pos: np.ndarray | None = None  # (T, J, 2) for agent collisions
```

**Acceptance**:
- [ ] Fields added with correct type hints
- [ ] Default values set to None
- [ ] Docstring updated with field descriptions
- [ ] No breaking changes to existing EpisodeData usage

---

### T002: [Setup] Add internal helper functions
**File**: `robot_sf/benchmark/metrics.py`
**Story**: Foundation for all stories
**Dependencies**: T001
**Parallelizable**: No

Implement three internal helper functions:

```python
def _compute_ped_velocities(peds_pos: np.ndarray, dt: float) -> np.ndarray:
    """Compute pedestrian velocities from positions via finite difference."""
    # See research.md for implementation details

def _compute_jerk(robot_acc: np.ndarray, dt: float) -> np.ndarray:
    """Compute jerk (acceleration derivative) via finite difference."""
    # See research.md for implementation details

def _compute_distance_matrix(data: EpisodeData) -> np.ndarray:
    """Compute robot-pedestrian distance matrix (T, K)."""
    # Optional optimization helper
```

**Acceptance**:
- [ ] All three functions implemented with correct signatures
- [ ] Functions handle edge cases (T=0, T=1, K=0)
- [ ] Docstrings include formula, parameters, returns
- [ ] Prefix with underscore (internal use)

---

## Phase 2: User Story 1 - Compute Paper-Specific Metrics (P1)

**Goal**: Implement all 22 metrics from paper Table 1

**Independent Test Criteria**: 
- All 22 metric functions callable with valid EpisodeData
- Each metric returns correct type (float) and handles edge cases
- Unit tests verify formulas match paper definitions

### NHT Metrics (Navigation/Hard Task)

#### T003: [US1] Implement success_rate metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def success_rate(data: EpisodeData, *, horizon: int) -> float:
    """Binary success indicator (1.0 if goal reached before horizon without collisions, else 0.0).
    
    Formula: S = 1 if (reached_goal_step < horizon) AND (collisions == 0) else 0
    Units: boolean [0,1]
    """
```

**Acceptance**:
- [ ] Function signature matches contract
- [ ] Returns 1.0 for successful episodes
- [ ] Returns 0.0 for failures (timeout, collisions, not reached)
- [ ] Docstring includes formula, units, edge cases, paper reference

---

#### T004: [US1] Implement collision_count metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T005, T006, T007
**Parallelizable**: No (depends on WC, AC, HC)

```python
def collision_count(data: EpisodeData) -> float:
    """Total collision count (sum of wall, agent, human collisions).
    
    Formula: C = WC + AC + HC
    Units: collision count [0,∞)
    """
```

**Acceptance**:
- [ ] Returns sum of all collision types
- [ ] Returns 0.0 when all collision data unavailable
- [ ] Gracefully handles None obstacles/other_agents

---

#### T005: [US1] Implement wall_collisions metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def wall_collisions(data: EpisodeData, *, threshold: float = D_COLL) -> float:
    """Count collisions with walls/obstacles.
    
    Formula: WC = sum_t I(min_m ||robot_pos[t] - obstacles[m]|| < threshold)
    Units: collision count [0,∞)
    """
```

**Acceptance**:
- [ ] Returns 0.0 if obstacles is None
- [ ] Counts timesteps with min distance < threshold
- [ ] Uses D_COLL constant from robot_sf.benchmark.constants

---

#### T006: [US1] Implement agent_collisions metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def agent_collisions(data: EpisodeData, *, threshold: float = D_COLL) -> float:
    """Count collisions with other agents/robots.
    
    Formula: AC = sum_t I(min_j ||robot_pos[t] - agents[t,j]|| < threshold)
    Units: collision count [0,∞)
    """
```

**Acceptance**:
- [ ] Returns 0.0 if other_agents_pos is None
- [ ] Counts timesteps with min distance to any agent < threshold
- [ ] Handles J=0 case (no other agents)

---

#### T007: [US1] Implement human_collisions metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def human_collisions(data: EpisodeData, *, threshold: float = D_COLL) -> float:
    """Count collisions with pedestrians/humans.
    
    Formula: HC = sum_t I(min_k ||robot_pos[t] - peds_pos[t,k]|| < threshold)
    Units: collision count [0,∞)
    """
```

**Acceptance**:
- [ ] Returns 0.0 if no pedestrians (K=0)
- [ ] Counts timesteps with min distance < threshold
- [ ] Matches behavior of existing collisions() metric

---

#### T008: [US1] Implement timeout metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def timeout(data: EpisodeData, *, horizon: int) -> float:
    """Binary indicator for timeout failure.
    
    Formula: TO = 1 if reached_goal_step is None else 0
    Units: timeout [0,1]
    """
```

**Acceptance**:
- [ ] Returns 1.0 if reached_goal_step is None
- [ ] Returns 0.0 if goal was reached
- [ ] Simple binary logic

---

#### T009: [US1] Implement failure_to_progress metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def failure_to_progress(
    data: EpisodeData,
    *,
    distance_threshold: float = 0.1,
    time_threshold: float = 5.0
) -> float:
    """Count failure-to-progress events.
    
    Formula: Count intervals where robot doesn't reduce distance to goal
             by distance_threshold within time_threshold window
    Units: failure count [0,∞)
    """
```

**Acceptance**:
- [ ] Computes distance to goal at each timestep
- [ ] Slides window of time_threshold duration
- [ ] Counts windows where progress < distance_threshold
- [ ] Returns 0.0 for short trajectories (T*dt < time_threshold)

---

#### T010: [US1] Implement stalled_time metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def stalled_time(data: EpisodeData, *, velocity_threshold: float = 0.05) -> float:
    """Total time robot speed is below threshold.
    
    Formula: ST = sum_t I(||robot_vel[t]|| < threshold) * dt
    Units: seconds [0,∞)
    """
```

**Acceptance**:
- [ ] Computes speed magnitude at each timestep
- [ ] Counts timesteps below threshold
- [ ] Multiplies by dt for time duration
- [ ] Returns 0.0 if never stalled

---

#### T011: [US1] Implement time_to_goal metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def time_to_goal(data: EpisodeData) -> float:
    """Time from start to goal.
    
    Formula: T = reached_goal_step * dt (if reached, else NaN)
    Units: seconds [0,∞)
    """
```

**Acceptance**:
- [ ] Returns reached_goal_step * dt if goal reached
- [ ] Returns NaN if reached_goal_step is None
- [ ] Simple multiplication

---

#### T012: [US1] Implement path_length metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def path_length(data: EpisodeData) -> float:
    """Total path length traveled by robot.
    
    Formula: PL = sum_{t=0}^{T-1} ||robot_pos[t+1] - robot_pos[t]||
    Units: meters [0,∞)
    """
```

**Acceptance**:
- [ ] Sums Euclidean distances between consecutive positions
- [ ] Uses full trajectory (not just to goal)
- [ ] Returns 0.0 for single-timestep trajectories
- [ ] Similar to existing path_efficiency metric

---

#### T013: [US1] Implement success_path_length metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T003, T012
**Parallelizable**: No (depends on success_rate, path_length)

```python
def success_path_length(
    data: EpisodeData,
    *,
    horizon: int,
    optimal_length: float
) -> float:
    """Success weighted by path efficiency.
    
    Formula: SPL = S * (optimal_length / max(actual_length, optimal_length))
    Units: ratio [0,1]
    """
```

**Acceptance**:
- [ ] Returns 0.0 if not successful
- [ ] Computes efficiency = optimal / max(actual, optimal)
- [ ] Clips efficiency to 1.0
- [ ] Calls success_rate and path_length internally

---

### SHT Metrics (Social/Human-aware Task)

#### T014: [US1] Implement velocity statistics (V_min, V_avg, V_max)
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

Implement three functions:
```python
def velocity_min(data: EpisodeData) -> float:
    """Minimum linear velocity magnitude. Units: m/s [-∞,∞)"""

def velocity_avg(data: EpisodeData) -> float:
    """Average linear velocity magnitude. Units: m/s [-∞,∞)"""

def velocity_max(data: EpisodeData) -> float:
    """Maximum linear velocity magnitude. Units: m/s [-∞,∞)"""
```

**Acceptance**:
- [ ] All three functions implemented
- [ ] Compute ||robot_vel[t]|| magnitude at each timestep
- [ ] Return NaN for empty trajectories (T=0)
- [ ] Use np.min, np.mean, np.max on magnitudes

---

#### T015: [US1] Implement acceleration statistics (A_min, A_avg, A_max)
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

Implement three functions:
```python
def acceleration_min(data: EpisodeData) -> float:
    """Minimum linear acceleration magnitude. Units: m/s² [-∞,∞)"""

def acceleration_avg(data: EpisodeData) -> float:
    """Average linear acceleration magnitude. Units: m/s² [-∞,∞)"""

def acceleration_max(data: EpisodeData) -> float:
    """Maximum linear acceleration magnitude. Units: m/s² [-∞,∞)"""
```

**Acceptance**:
- [ ] All three functions implemented
- [ ] Compute ||robot_acc[t]|| magnitude
- [ ] Return NaN for T=0
- [ ] Mirror velocity statistics pattern

---

#### T016: [US1] Implement jerk statistics (J_min, J_avg, J_max)
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T002
**Parallelizable**: Yes [P]

Implement three functions using _compute_jerk helper:
```python
def jerk_min(data: EpisodeData) -> float:
    """Minimum jerk magnitude. Units: m/s³ [-∞,∞)"""

def jerk_avg(data: EpisodeData) -> float:
    """Average jerk magnitude. Units: m/s³ [-∞,∞)"""

def jerk_max(data: EpisodeData) -> float:
    """Maximum jerk magnitude. Units: m/s³ [-∞,∞)"""
```

**Acceptance**:
- [ ] Call _compute_jerk(data.robot_acc, data.dt)
- [ ] Compute magnitude of jerk vectors
- [ ] Return NaN for T < 2
- [ ] Apply min/mean/max statistics

---

#### T017: [US1] Implement clearing_distance statistics (CD_min, CD_avg)
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

Implement two functions:
```python
def clearing_distance_min(data: EpisodeData) -> float:
    """Minimum distance to obstacles. Units: meters [0,∞)"""

def clearing_distance_avg(data: EpisodeData) -> float:
    """Average minimum distance to obstacles. Units: meters [0,∞)"""
```

**Acceptance**:
- [ ] Return NaN if obstacles is None
- [ ] Compute min_m ||robot_pos[t] - obstacles[m]|| for each t
- [ ] Apply min or mean over timesteps
- [ ] Handle M=0 (no obstacles)

---

#### T018: [US1] Implement space_compliance metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T002
**Parallelizable**: Yes [P]

```python
def space_compliance(data: EpisodeData, *, threshold: float = 0.5) -> float:
    """Ratio of trajectory within personal space threshold.
    
    Formula: SC = (# timesteps where min_k distance < threshold) / T
    Units: ratio [0,1]
    """
```

**Acceptance**:
- [ ] Return NaN if no pedestrians (K=0)
- [ ] Compute min distance to pedestrians at each timestep
- [ ] Count timesteps where min_dist < threshold
- [ ] Divide by total timesteps T
- [ ] Default threshold 0.5m

---

#### T019: [US1] Implement distance_to_human_min metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def distance_to_human_min(data: EpisodeData) -> float:
    """Minimum distance to any human/pedestrian.
    
    Formula: DH_min = min_t min_k ||robot_pos[t] - peds_pos[t,k]||
    Units: meters [0,∞)
    """
```

**Acceptance**:
- [ ] Return NaN if K=0
- [ ] Compute distance matrix (T, K)
- [ ] Return global minimum
- [ ] Similar to existing min_distance metric

---

#### T020: [US1] Implement time_to_collision_min metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T002
**Parallelizable**: Yes [P]

```python
def time_to_collision_min(data: EpisodeData) -> float:
    """Minimum time to collision with pedestrian.
    
    Formula: TTC = min_{t,k} (d / ||v_rel||)
    Units: seconds [0,∞)
    """
```

**Acceptance**:
- [ ] Return NaN if K=0 or no collision trajectory
- [ ] Call _compute_ped_velocities to get pedestrian velocities
- [ ] Compute relative velocity v_rel = v_robot - v_ped
- [ ] Compute TTC = distance / ||v_rel|| for approaching pairs
- [ ] Return minimum TTC across all (t, k) pairs

---

#### T021: [US1] Implement aggregated_time metric
**File**: `robot_sf/benchmark/metrics.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

```python
def aggregated_time(
    data: EpisodeData,
    *,
    agent_goal_times: dict[int, int]
) -> float:
    """Time for cooperative agents to reach goals.
    
    Formula: AT = max_j (goal_time[j]) for j in cooperative set
    Units: seconds [0,∞)
    """
```

**Acceptance**:
- [ ] Return NaN if agent_goal_times is empty
- [ ] Find max goal-reaching timestep
- [ ] Multiply by dt for time duration
- [ ] Handle None values in dict

---

### US1 Testing

#### T022: [US1] Add unit tests for all metrics
**File**: `tests/test_metrics.py`
**Dependencies**: T003-T021
**Parallelizable**: No (tests implementation)

Add comprehensive unit tests for all 22 metrics:

**Test Structure**:
- Normal cases: Valid episode data, metrics compute correctly
- Edge cases: Empty trajectories, no pedestrians, single timestep
- Boundary cases: Zero thresholds, perfect navigation, worst-case collisions
- Type checks: All metrics return float
- NaN handling: Verify when NaN is returned vs 0.0

**Acceptance**:
- [ ] At least 2 test cases per metric (normal + edge)
- [ ] 100% coverage of new metric functions
- [ ] Tests use synthetic EpisodeData fixtures
- [ ] Tests verify formula correctness with known inputs
- [ ] All tests pass with pytest

---

#### T023: [US1] Add integration test with real episode data
**File**: `tests/test_benchmark_integration.py`
**Dependencies**: T022
**Parallelizable**: Yes [P]

Create integration test using actual episode from benchmark:

```python
def test_paper_metrics_integration():
    """Verify all 22 metrics can be computed on real episode."""
    # Load or generate realistic episode
    # Compute all metrics
    # Verify all return valid numeric values
    # Verify no errors or exceptions
```

**Acceptance**:
- [ ] Test loads realistic episode data
- [ ] All 22 metrics called and complete successfully
- [ ] Results are valid floats or NaN (documented)
- [ ] Performance < 100ms per episode verified

---

#### T024: [US1] Update documentation with metric formulas
**File**: `docs/benchmark.md`
**Dependencies**: T003-T021
**Parallelizable**: Yes [P]

Add comprehensive metric documentation:

**Content**:
- NHT Metrics section with all 11 formulas, units, ranges
- SHT Metrics section with all 11 formulas, units, ranges
- Table: Name | Formula | Units | Range | Paper Ref
- Edge case behavior documentation
- Usage examples for each category

**Acceptance**:
- [ ] All 22 metrics documented
- [ ] Formulas match paper Table 1
- [ ] Units and ranges clearly specified
- [ ] Code examples included
- [ ] Links to paper reference

---

#### T025: [US1] Update docs/README.md index
**File**: `docs/README.md`
**Dependencies**: T024
**Parallelizable**: No (after benchmark.md update)

Add link to new metrics documentation:

```markdown
- [Benchmark Metrics](benchmark.md) - Social navigation metrics including paper 2306.16740v4
```

**Acceptance**:
- [ ] Link added to docs index
- [ ] Proper section placement (near existing benchmark docs)
- [ ] Concise description

---

## Phase 3: User Story 2 - Validate Against Paper Baselines (P2)

**Goal**: Ensure metric implementations match paper definitions

**Independent Test Criteria**:
- Synthetic test cases with known outcomes validate metric formulas
- Edge cases return expected values
- Metrics match theoretical predictions within tolerance

### T026: [US2] Create synthetic test fixtures
**File**: `tests/fixtures/episode_data.py`
**Dependencies**: T001
**Parallelizable**: Yes [P]

Create reusable synthetic EpisodeData for testing:

**Fixtures**:
- Perfect navigation (straight line, no collisions, success)
- Collision scenario (robot hits pedestrian)
- Timeout scenario (doesn't reach goal)
- Stalled robot (low velocity)
- Zero pedestrians (isolated navigation)
- Single timestep (edge case)

**Acceptance**:
- [ ] At least 6 synthetic fixtures created
- [ ] Each fixture has documented expected metric values
- [ ] Fixtures cover normal and edge cases
- [ ] Fixtures are reusable across test files

---

#### T027: [US2] Add validation tests with known outcomes
**File**: `tests/test_metrics_validation.py`
**Dependencies**: T026
**Parallelizable**: Yes [P]

Test metrics against fixtures with theoretical predictions:

```python
def test_perfect_navigation_metrics():
    """Perfect navigation should have S=1, C=0, high efficiency."""
    data = perfect_navigation_fixture()
    assert success_rate(data, horizon=100) == 1.0
    assert collision_count(data) == 0.0
    assert success_path_length(data, horizon=100, optimal_length=10.0) > 0.95
```

**Acceptance**:
- [ ] Tests for all 6 fixtures
- [ ] Each test verifies multiple metrics
- [ ] Assertions use 5% tolerance where applicable
- [ ] Tests document expected vs actual values

---

#### T028: [US2] Add edge case regression tests
**File**: `tests/test_metrics_validation.py`
**Dependencies**: T026
**Parallelizable**: Yes [P]

Comprehensive edge case coverage:

**Test Cases**:
- Zero pedestrians → social metrics return NaN or 0.0
- Single timestep → velocity/acceleration metrics return NaN
- Robot at goal (T=0 path) → path_length = 0, SPL handling
- Infinite threshold → compliance metrics = 1.0
- No obstacles → wall_collisions = 0.0

**Acceptance**:
- [ ] All edge cases from spec.md covered
- [ ] NaN vs 0.0 semantics verified
- [ ] No exceptions raised
- [ ] Graceful degradation confirmed

---

## Phase 4: User Story 3 - Export Metrics in Standard Format (P3)

**Goal**: Verify integration with existing benchmark infrastructure

**Independent Test Criteria**:
- Metrics export to episode.schema.v1.json format
- Schema validation passes
- Integration with aggregation pipeline works

#### T029: [US3] Verify schema compatibility
**File**: `tests/test_schema_validation.py`
**Dependencies**: T003-T021
**Parallelizable**: Yes [P]

Test that new metrics conform to existing schema:

```python
def test_paper_metrics_schema_validation():
    """All new metrics produce schema-compliant output."""
    data = create_test_episode()
    metrics_dict = {
        "success": success_rate(data, horizon=100),
        "collision_count": collision_count(data),
        # ... all 22 metrics ...
    }
    # Validate against episode.schema.v1.json
    validate_episode_schema(metrics_dict)
```

**Acceptance**:
- [ ] All 22 metrics added to test dict
- [ ] Schema validation passes
- [ ] Metric names use lowercase_with_underscores
- [ ] All values are numeric (float/int)

---

#### T030: [US3] Test integration with benchmark runner
**File**: `tests/test_benchmark_integration.py`
**Dependencies**: T029
**Parallelizable**: Yes [P]

Verify metrics work with existing benchmark runner:

```python
def test_paper_metrics_in_runner():
    """Benchmark runner can call and store new metrics."""
    # Run minimal benchmark with new metrics
    # Verify JSONL output contains new metric keys
    # Verify values are valid
```

**Acceptance**:
- [ ] Runner successfully calls new metrics
- [ ] JSONL output includes new metric keys
- [ ] No errors during benchmark execution
- [ ] Output validates against schema

---

#### T031: [US3] Test integration with aggregation
**File**: `tests/test_aggregate.py`
**Dependencies**: T030
**Parallelizable**: Yes [P]

Verify new metrics work with aggregation pipeline:

```python
def test_paper_metrics_aggregation():
    """Aggregation computes statistics for new metrics."""
    # Create multiple episodes with new metrics
    # Run aggregation
    # Verify mean, median, CI computed correctly
```

**Acceptance**:
- [ ] Aggregation handles new metrics
- [ ] Summary statistics computed (mean, median, p95)
- [ ] Bootstrap CIs work for new metrics
- [ ] NaN values handled appropriately

---

#### T032: [US3] Add export format examples
**File**: `docs/benchmark.md`
**Dependencies**: T024, T029
**Parallelizable**: Yes [P]

Document JSON export format for new metrics:

**Content**:
- Example episode JSON with all 22 metrics
- Schema reference
- Handling of NaN/null values
- Integration with existing metrics

**Acceptance**:
- [ ] Complete JSON example provided
- [ ] Schema compliance noted
- [ ] NaN handling documented
- [ ] Example is valid JSON

---

## Phase 5: Polish & Cross-Cutting Concerns

**Goal**: Performance validation, final documentation, changelog

#### T033: [Polish] Add performance benchmark
**File**: `scripts/validation/performance_paper_metrics.py`
**Dependencies**: T003-T021
**Parallelizable**: Yes [P]

Create performance validation script:

```python
# Benchmark metric computation on various episode sizes
# Verify < 100ms per episode for 50 pedestrians
# Report throughput (steps/sec)
```

**Acceptance**:
- [ ] Script tests episodes with 10, 25, 50 pedestrians
- [ ] Verifies < 100ms target met
- [ ] Reports timing for each metric category
- [ ] Identifies any performance bottlenecks

---

#### T034: [Polish] Update CHANGELOG.md
**File**: `CHANGELOG.md`
**Dependencies**: All implementation tasks
**Parallelizable**: No (final step)

Add changelog entry:

```markdown
## [Unreleased]

### Added
- 22 social navigation metrics from paper 2306.16740v4 (Table 1)
  - 11 NHT (Navigation/Hard Task) metrics: success, collisions, time, path length, etc.
  - 11 SHT (Social/Human-aware Task) metrics: velocity, acceleration, jerk, compliance, etc.
- Extended EpisodeData with optional `obstacles` and `other_agents_pos` fields
- Comprehensive metric documentation in docs/benchmark.md
- Unit and integration tests with 100% coverage

### Changed
- None (backward compatible extension)

### Fixed
- None
```

**Acceptance**:
- [ ] Entry added to Unreleased section
- [ ] All new functionality documented
- [ ] Backward compatibility noted
- [ ] Follows keep-a-changelog format

---

#### T035: [Polish] Create quickstart example
**File**: `examples/demo_paper_metrics.py`
**Dependencies**: T003-T021
**Parallelizable**: Yes [P]

Create runnable example demonstrating new metrics:

```python
"""Demonstrate paper 2306.16740v4 metrics computation."""
# Create sample episode
# Compute all NHT metrics
# Compute all SHT metrics
# Print formatted results
```

**Acceptance**:
- [ ] Example runs without errors
- [ ] Computes and displays all 22 metrics
- [ ] Uses realistic episode data
- [ ] Output is human-readable
- [ ] Includes comments explaining metrics

---

## Task Dependencies

### Critical Path
```
T001 (EpisodeData) 
  → T002 (Helpers)
    → T003-T021 (All Metrics) [Parallel]
      → T022 (Unit Tests)
        → T023 (Integration Test)
          → T024 (Docs)
            → T025 (Index)
              → T034 (Changelog)
```

### User Story Completion Order
1. **US1 (P1)**: T001-T025 (Core metrics + tests + docs)
2. **US2 (P2)**: T026-T028 (Validation tests - can run parallel to US3)
3. **US3 (P3)**: T029-T032 (Export/integration - can run parallel to US2)
4. **Polish**: T033-T035 (Performance + changelog + example)

### Parallel Opportunities

**Within US1 (After T002)**:
- T003-T021 can be implemented in parallel (different functions)
- T024 (docs) can start once any metrics done

**Between US2 and US3**:
- T026-T028 and T029-T032 are independent, can run parallel

**Polish Tasks**:
- T033 and T035 can run parallel

---

## Parallel Execution Examples

### Sprint 1: Foundation + NHT Metrics
```
Developer A: T001 → T002 → T003, T008, T011 (success, timeout, time metrics)
Developer B: T005 → T006 → T007 → T004 (collision metrics)
Developer C: T009 → T010 → T012 → T013 (progress, path metrics)
```

### Sprint 2: SHT Metrics
```
Developer A: T014 (velocity stats) → T015 (accel stats) → T016 (jerk stats)
Developer B: T017 (clearing distance) → T018 (space compliance)
Developer C: T019 (distance to human) → T020 (TTC) → T021 (aggregated time)
```

### Sprint 3: Testing + Validation
```
Developer A: T022 (unit tests for all metrics)
Developer B: T026 → T027 (fixtures + validation tests)
Developer C: T029 → T030 → T031 (schema + integration tests)
```

### Sprint 4: Documentation + Polish
```
Developer A: T024 → T025 (docs update)
Developer B: T028 (edge case tests) → T032 (export examples)
Developer C: T033 (performance) → T035 (example) → T034 (changelog)
```

---

## Implementation Strategy

### MVP Scope (User Story 1 only)
**Deliverable**: All 22 metrics implemented, tested, and documented

**Tasks**: T001-T025 (25 tasks)

**Value**: Immediate ability to compute paper-standard metrics on episodes

**Timeline**: ~2-3 sprints depending on team size

### Incremental Delivery
1. **MVP**: US1 complete → Users can compute metrics
2. **Validation**: Add US2 → Users have confidence in correctness
3. **Integration**: Add US3 → Full benchmark pipeline integration
4. **Polish**: Performance + examples → Production-ready

### Success Criteria Mapping
- SC-001 (100% test coverage): T022, T027, T028
- SC-002 (< 100ms performance): T033
- SC-003 (batch aggregation): T031
- SC-004 (5% tolerance): T027
- SC-005 (edge case handling): T022, T028
- SC-006 (documentation): T024, T025, T032
- SC-007 (no breaking changes): T001, T029
- SC-008 (schema validation): T029, T030

---

## Notes

**Testing Philosophy**: Tests are essential for this feature (SC-001 requires 100% coverage). All metric implementations must be validated.

**Performance Budget**: < 100ms per episode for 50 pedestrians (T033 validates this).

**Backward Compatibility**: EpisodeData extensions use optional fields (T001) to maintain compatibility.

**Documentation Priority**: Each metric must have complete docstring with formula, units, edge cases, paper reference (enforced in T003-T021).

**Parallel Implementation**: Most metric functions are independent and can be implemented concurrently once foundation (T001-T002) is complete.
