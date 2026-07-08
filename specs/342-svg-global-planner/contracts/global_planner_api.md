# Global Planner API Contract

**Phase**: 1 (Design & Contracts)  
**Version**: 1.0.0  
**Status**: Draft

## Purpose

This document defines the public API contract for `robot_sf.planner.GlobalPlanner`. All public methods, parameters, return types, exceptions, and performance guarantees are specified here.

## Scope

**Public API** (stable interface):
- `GlobalPlanner` class
- `PlannerConfig` dataclass
- `PlanningFailedError` exception
- `POISampler` utility class

**Internal Implementation** (not part of contract):
- `VisibilityGraph` wrapper
- `_get_or_build_graph()` helper
- `_inflate_obstacles()` internal method
- `_smooth_path()` internal method

---

## 1. GlobalPlanner.plan()

### Signature

```python
def plan(
    self,
    start: Vec2D,
    goal: Vec2D,
    *,
    via_pois: Optional[list[str]] = None
) -> list[Vec2D]:
```

### Contract

**Preconditions**:
1. `start` must lie within map boundaries: `0 ≤ start.x ≤ map.width` and `0 ≤ start.y ≤ map.height`
2. `goal` must lie within map boundaries (same constraints as start)
3. If `via_pois` provided, all IDs must exist in `map_definition.poi_labels.keys()`
4. Planner must be initialized with valid `MapDefinition` (non-empty boundaries)

**Postconditions**:
1. Returned path is non-empty: `len(path) >= 2`
2. First waypoint equals `start`: `path[0] == start`
3. Last waypoint equals `goal`: `path[-1] == goal`
4. All intermediate waypoints lie outside inflated obstacles
5. All path segments are collision-free (line-of-sight through free space)
6. If `via_pois` specified, path visits each POI in order

**Performance Guarantees**:
- First call (cold cache): ≤ 500ms @ 50 obstacles on reference hardware
- Subsequent calls (warm cache): ≤ 100ms @ 50 obstacles
- Reference hardware: Apple M1 or equivalent (2020+ CPU)

**Exception Guarantees**:
- **Strong exception safety**: No state modification if exception raised
- `PlanningFailedError`: Raised when no valid path exists and `config.fallback_on_failure == False`
- `ValueError`: Raised for invalid inputs (out-of-bounds, invalid POI IDs)
- `TypeError`: Raised for incorrect parameter types

**Example**:
```python
planner = GlobalPlanner(map_def, config)

# Basic path
path = planner.plan(start=Vec2D(0, 0), goal=Vec2D(10, 5))
assert len(path) >= 2
assert path[0] == Vec2D(0, 0)
assert path[-1] == Vec2D(10, 5)

# Path via POIs
path = planner.plan(
    start=Vec2D(0, 0),
    goal=Vec2D(10, 5),
    via_pois=["poi_corridor_1", "poi_junction"]
)
```

---

## 2. GlobalPlanner.plan_multi_goal()

### Signature

```python
def plan_multi_goal(
    self,
    start: Vec2D,
    goals: list[Vec2D],
    *,
    optimize_order: bool = True
) -> list[Vec2D]:
```

### Contract

**Preconditions**:
1. `start` within map boundaries
2. All `goals` within map boundaries
3. `len(goals) >= 1` (at least one goal)

**Postconditions**:
1. Path visits all goals exactly once
2. First waypoint equals `start`
3. If `optimize_order == True`, goals visited in near-optimal order (nearest-neighbor heuristic)
4. If `optimize_order == False`, goals visited in list order
5. All path segments collision-free

**Performance**:
- ≤ 100ms per goal @ 50 obstacles (warm cache)
- TSP optimization overhead: O(k²) for k goals (negligible for k < 10)

**Exceptions**:
- `PlanningFailedError`: If any goal unreachable
- `ValueError`: If `goals` list empty

**Example**:
```python
goals = [Vec2D(5, 5), Vec2D(8, 2), Vec2D(3, 7)]
path = planner.plan_multi_goal(start=Vec2D(0, 0), goals=goals)

# Verify all goals visited
visited_positions = set(path)
assert all(g in visited_positions for g in goals)
```

---

## 3. PlannerConfig (Configuration Contract)

### Fields

```python
@dataclass
class PlannerConfig:
    robot_radius: float = 0.4
    min_safe_clearance: float = 0.3
    enable_smoothing: bool = True
    smoothing_epsilon: float = 0.1
    cache_graphs: bool = True
    fallback_on_failure: bool = True
```

### Constraints

| Field                | Type    | Constraint                           | Default |
|----------------------|---------|--------------------------------------|---------|
| `robot_radius`       | float   | > 0.0                                | 0.4     |
| `min_safe_clearance` | float   | >= 0.0                               | 0.3     |
| `enable_smoothing`   | bool    | —                                    | True    |
| `smoothing_epsilon`  | float   | > 0.0 if enable_smoothing else any   | 0.1     |
| `cache_graphs`       | bool    | —                                    | True    |
| `fallback_on_failure`| bool    | —                                    | True    |

### Validation

**When**: Validation occurs in `__post_init__` (immediately after construction)

**Exceptions**:
- `ValueError`: Raised for constraint violations with descriptive message

**Example**:
```python
# Valid config
config = PlannerConfig(robot_radius=0.5, min_safe_clearance=0.4)

# Invalid config (raises ValueError)
try:
    config = PlannerConfig(robot_radius=-0.1)
except ValueError as e:
    print(e)  # "robot_radius must be positive, got -0.1"
```

---

## 4. PlanningFailedError (Exception Contract)

### Attributes

```python
class PlanningFailedError(Exception):
    start: Vec2D
    goal: Vec2D
    reason: str
```

### Usage Contract

**When Raised**:
1. Start position inside inflated obstacle
2. Goal position inside inflated obstacle
3. No line-of-sight path exists (disconnected map regions)
4. pyvisgraph returns empty path (internal error)

**Suppression**:
- If `config.fallback_on_failure == True`, exception NOT raised
- Instead, warning logged and straight-line path `[start, goal]` returned

**Example Handling**:
```python
from robot_sf.planner import GlobalPlanner, PlanningFailedError

try:
    path = planner.plan(start, goal)
except PlanningFailedError as e:
    print(f"Planning failed: {e.reason}")
    print(f"  From: ({e.start.x}, {e.start.y})")
    print(f"  To: ({e.goal.x}, {e.goal.y})")
    # Handle failure (e.g., return to known safe position)
```

---

## 5. POISampler.sample()

### Signature

```python
def sample(
    self,
    count: int,
    strategy: Literal["random", "nearest", "farthest"] = "random"
) -> list[Vec2D]:
```

### Contract

**Preconditions**:
1. Sampler initialized with map containing ≥ 1 POI
2. `count >= 0`

**Postconditions**:
1. Returned list length: `min(count, len(available_pois))`
2. All positions are valid POIs from the map
3. No duplicate POIs in result (sampling without replacement)
4. If `strategy == "random"`, order is uniformly random
5. If `strategy == "nearest"`, POIs sorted by distance to implicit start (requires extension)
6. If `strategy == "farthest"`, POIs sorted by distance descending

**Reproducibility**:
- Seeded RNG passed to `__init__` ensures deterministic sampling
- Same seed + same map + same parameters → same result

**Example**:
```python
sampler = POISampler(map_def, seed=42)

# Sample 3 random POIs
pois = sampler.sample(count=3, strategy="random")
assert len(pois) == min(3, len(map_def.poi_positions))

# Reproducibility
sampler2 = POISampler(map_def, seed=42)
pois2 = sampler2.sample(count=3, strategy="random")
assert pois == pois2  # Same seed → same result
```

---

## 6. Integration with Existing APIs

### Compatibility with RouteNavigator

**Requirement**: Planner output must be compatible with `robot_sf.nav.navigation.RouteNavigator`

**Contract**:
```python
from robot_sf.planner import GlobalPlanner
from robot_sf.nav.navigation import RouteNavigator

# Generate path
planner = GlobalPlanner(map_def)
path = planner.plan(start, goal)

# Use with existing navigator (no adapter needed)
navigator = RouteNavigator(path)  # Accepts list[Vec2D]
```

**Verification**: Integration test in `tests/test_planner/test_navigation_integration.py`

---

### Compatibility with RobotSimulationConfig

**Requirement**: Planner configurable via unified config system

**Contract**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.gym_env.environment_factory import make_robot_env

config = RobotSimulationConfig()
config.use_planner = True
config.planner_clearance_margin = 0.4

env = make_robot_env(config=config)  # Planner auto-initialized inside
```

**Config Field Mapping**:
| RobotSimulationConfig Field     | PlannerConfig Field    |
|---------------------------------|------------------------|
| `planner_clearance_margin`      | `min_safe_clearance`   |
| `robot_radius`                  | `robot_radius`         |
| (implicit via robot config)     | `enable_smoothing`     |

---

## 7. Deprecation Policy

**Stable API** (v1.0.0):
- Public methods (`plan()`, `plan_multi_goal()`)
- `PlannerConfig` fields (robot_radius, min_safe_clearance, etc.)
- `PlanningFailedError` attributes

**Breaking Changes** (require major version bump):
- Removing public method
- Changing method signature (adding required parameter)
- Removing `PlannerConfig` field
- Changing exception type

**Non-Breaking Changes** (minor version bump):
- Adding optional parameter with default value
- Adding new public method
- Adding new `PlannerConfig` field with default

**Example Deprecation**:
```python
import warnings

def plan_legacy(self, start, goal):
    """Deprecated: Use plan() instead."""
    warnings.warn(
        "plan_legacy() is deprecated, use plan() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self.plan(start, goal)
```

---

## 8. Testing Requirements

### Unit Test Coverage

**Minimum**: 90% line coverage for `robot_sf/planner/` module

**Required Test Cases**:
1. Basic path planning (start → goal, no obstacles)
2. Path with obstacles (verify collision-free)
3. Path via POIs (verify waypoint order)
4. Multi-goal planning (verify all goals visited)
5. Configuration validation (invalid parameters → ValueError)
6. Planning failure handling (unreachable goal → PlanningFailedError)
7. Cache behavior (verify repeated calls faster)
8. Boundary conditions (start/goal on map edge)
9. Degenerate cases (start == goal → path length 2)

### Integration Test Coverage

**Required Scenarios**:
1. All example SVG maps generate valid paths
2. Generated paths work with `RouteNavigator`
3. `RobotSimulationConfig` integration
4. Performance validation (<100ms median)

---

## 9. Performance Benchmarks

### Acceptance Criteria

| Scenario                     | Metric           | Target         | Hard Limit    |
|------------------------------|------------------|----------------|---------------|
| Graph build (50 obstacles)   | Latency          | < 400ms        | < 800ms       |
| Path query (cold cache)      | Latency          | < 500ms        | < 1000ms      |
| Path query (warm cache)      | Latency          | < 100ms        | < 200ms       |
| Multi-goal (5 goals)         | Latency          | < 500ms        | < 1000ms      |
| Memory per map               | Peak RSS         | < 50MB         | < 100MB       |

**Measurement Conditions**:
- Hardware: Apple M1 or equivalent (2020+ CPU)
- Map: 50 polygonal obstacles, 20-30 vertices each
- Python 3.11+
- Warm interpreter (exclude import time)

**Benchmark Script**: `scripts/benchmark_planner.py`

---

## 10. Documentation Requirements

**Public API** (docstrings required):
- All public methods
- All parameters with type hints and descriptions
- All exceptions with "Raises:" section
- Performance characteristics with "Performance:" section
- Usage examples with "Example:" section

**Style**: Google-style docstrings (existing robot_sf convention)

**Example**:
```python
def plan(self, start: Vec2D, goal: Vec2D, *, via_pois: Optional[list[str]] = None) -> list[Vec2D]:
    """
    Compute collision-free path from start to goal.
    
    Args:
        start: Starting position in meters (must be within map bounds)
        goal: Goal position in meters (must be within map bounds)
        via_pois: Optional POI IDs to route through, in order
    
    Returns:
        List of waypoints forming collision-free path.
        Includes start and goal as first/last elements.
    
    Raises:
        PlanningFailedError: No valid path exists (if fallback disabled)
        ValueError: Start/goal out of bounds or invalid POI ID
    
    Performance:
        First call: 200-500ms (builds visibility graph)
        Cached: <100ms (queries existing graph)
    
    Example:
        >>> planner = GlobalPlanner(map_def)
        >>> path = planner.plan(Vec2D(0, 0), Vec2D(10, 5))
        >>> print(f"Path has {len(path)} waypoints")
    """
```

---

## 11. Versioning & Changelog

### Version Format

**Semantic Versioning**: `MAJOR.MINOR.PATCH`

### Version History

| Version | Date       | Changes                              |
|---------|------------|--------------------------------------|
| 1.0.0   | TBD        | Initial public release               |

### Changelog Location

Updates recorded in:
- `CHANGELOG.md` (project-wide)
- `specs/342-svg-global-planner/CHANGELOG.md` (feature-specific)

---

## 12. Open Questions

1. **Thread Safety**: Should `plan()` support concurrent calls? (Answer: No for MVP, document thread-unsafe)
2. **Batch Queries**: Support `plan_batch(starts: list, goals: list)` for parallel processing? (Defer to Phase 3)
3. **Dynamic Obstacles**: API for real-time obstacle updates? (Out of scope for static map MVP)
4. **Alternative Backends**: Pluggable graph builder (e.g., custom vs pyvisgraph)? (Defer to Phase 3)

---

## Sign-Off

**Author**: AI Agent (GitHub Copilot)  
**Reviewers**: TBD  
**Approval Date**: TBD

**Review Checklist**:
- [ ] All public methods documented with examples
- [ ] Exception contract clear (when/why raised)
- [ ] Performance guarantees measurable
- [ ] Backward compatibility considered
- [ ] Test coverage requirements defined
- [ ] Deprecation policy documented

---

## References

- **Global Planner Design**: `specs/342-svg-global-planner/global-planner-v2.md`
- **Data Model**: `specs/342-svg-global-planner/data-model.md`
- **Dev Guide**: `docs/dev_guide.md`
- **Constitution**: `.specify/memory/constitution.md`
