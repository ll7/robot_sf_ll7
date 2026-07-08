# Research & Technical Decisions: SVG-Based Global Planner

**Phase**: 0 (Outline & Research)  
**Date**: 2025-12-10  
**Status**: Complete

## Overview

This document consolidates research findings and technical decisions for implementing the SVG-based global planner. All unknowns from the Technical Context have been resolved through analysis of the existing codebase and evaluation of path planning libraries.

## Key Technical Decisions

### 1. Primary Algorithm: Visibility Graph via pyvisgraph

**Decision**: Use pyvisgraph library as primary path planning backend

**Rationale**:
- **Proven Implementation**: Mature, tested visibility graph library specifically designed for 2D path planning
- **Performance**: Meets targets (<50ms typical for path queries after graph build)
- **Optimal Paths**: Guarantees shortest path through visibility graph (obstacle vertices connected by line-of-sight)
- **Clean API**: Simple interface (`graph.build()`, `graph.shortest_path()`) reduces integration complexity
- **Python Native**: Pure Python with no complex C extensions, simplifies deployment

**Alternatives Considered**:
1. **Shapely + NetworkX (custom graph builder)**
   - Pros: Full control over edge costs, can implement width-aware costs
   - Cons: Higher implementation complexity, slower initial development
   - Rejected: Overkill for MVP; can add later as extension for advanced features

2. **RRT/PRM Sampling-Based Methods**
   - Pros: Handle non-polygonal obstacles, probabilistically complete
   - Cons: Suboptimal paths, slower (>100ms), parameter-sensitive, overkill for static 2D polygons
   - Rejected: Performance targets not met; unnecessary complexity for static maps

3. **A* on Discretized Grid**
   - Pros: Simple to implement, well-understood
   - Cons: Discretization artifacts, suboptimal paths, memory usage scales with resolution
   - Rejected: Visibility graph provides geometry-accurate optimal paths

**Implementation Note**: Reserve Shapely+NetworkX approach for Phase 3 "Advanced" features (custom edge costs, width-aware planning).

### 2. Obstacle Inflation Strategy

**Decision**: Robot-radius-aware buffering using Shapely's `.buffer()` method

**Rationale**:
- **Configuration Point Abstraction**: Inflating obstacles by (robot_radius + clearance_margin) converts kinematic robot to point robot problem
- **Existing Dependency**: Shapely already in use for `occupancy_grid.py`; no new dependency
- **Robust Implementation**: Shapely provides reliable polygon buffering with corner rounding
- **Clearance Guarantees**: Any path through inflated obstacles automatically maintains minimum clearance

**Trade-offs Accepted**:
- May close off narrow passages when robot cannot physically fit
- Intentional safety-first approach (better to reject path than risk collision)
- Configurable `min_safe_clearance` parameter allows user override

**Example**:
```python
from shapely.geometry import Polygon

inflated_obstacles = [
    Polygon(obstacle.vertices).buffer(robot_radius + clearance_margin)
    for obstacle in map_definition.obstacles
]
```

### 3. Path Smoothing Algorithm

**Decision**: Douglas-Peucker simplification + optional spline fitting

**Rationale**:
- **Waypoint Reduction**: Visibility graphs produce paths with many vertices (every obstacle corner); smoothing reduces count by 30%+
- **Obstacle Clearance Preservation**: Douglas-Peucker maintains path topology; filtered points stay within original path corridor
- **Performance**: Fast algorithm (O(n log n)), meets <10ms budget for typical paths
- **Optional Spline**: Can add Catmull-Rom or B-spline fitting for smoother trajectories in Phase 2

**Implementation Libraries**:
- Douglas-Peucker: `rdp` library (Ramer-Douglas-Peucker) or custom implementation
- Alternative: `simplification` library for more sophisticated algorithms

**Trade-offs**:
- Slight path elongation acceptable (<5% increase typical)
- Must validate smoothed path still collision-free (quick check via Shapely line intersections)

### 4. Graph Caching Strategy

**Decision**: Lazy construction with per-map-hash in-memory cache

**Rationale**:
- **Amortize Build Cost**: Graph construction (200-500ms) dominates first query; caching makes subsequent queries <100ms
- **Memory Efficiency**: Typical training session uses ~10 maps; caching all graphs consumes <100MB
- **Deterministic Hashing**: `hashlib.md5(repr(obstacles))` provides stable map identity
- **Invalidation**: Cache keyed by map content; obstacle changes automatically trigger rebuild

**Implementation Pattern**:
```python
_graph_cache: dict[str, VisibilityGraph] = {}

def _get_or_build_graph(map_def: MapDefinition) -> VisibilityGraph:
    map_hash = hashlib.md5(repr(map_def.obstacles).encode()).hexdigest()
    if map_hash not in _graph_cache:
        _graph_cache[map_hash] = build_visibility_graph(map_def)
    return _graph_cache[map_hash]
```

**Cache Management**:
- No explicit eviction policy for MVP (assume bounded map set)
- Future: Add LRU eviction if memory constraints arise

### 5. SVG POI Annotation Format

**Decision**: Extend SVG spec to support `<circle class="poi">` elements

**Rationale**:
- **Minimal Extension**: Circle is native SVG primitive; class-based selection follows existing spawn/goal pattern
- **Human Readable**: SVG editors can visualize POIs as visible markers
- **Backward Compatible**: Maps without POIs continue working; parser skips unknown classes

**Schema**:
```xml
<circle class="poi" id="poi_corridor_1" cx="50" cy="25" r="0.5" 
        label="corridor_junction" />
```

**Parser Modifications** (svg_map_parser.py):
```python
# Add to MapDefinition
@dataclass
class MapDefinition:
    # ... existing fields ...
    poi_positions: list[Vec2D] = field(default_factory=list)
    poi_labels: dict[str, str] = field(default_factory=dict)  # id -> label

# Parser logic
circles = svg_root.findall(".//circle[@class='poi']")
for circle in circles:
    cx, cy = float(circle.get('cx')), float(circle.get('cy'))
    poi_id = circle.get('id', f'poi_{len(map_def.poi_positions)}')
    label = circle.get('label', '')
    map_def.poi_positions.append(Vec2D(cx, cy))
    if label:
        map_def.poi_labels[poi_id] = label
```

### 6. Integration with Existing Navigation

**Decision**: Dual-mode integration - opt-in flag + fallback compatibility

**Migration Phases**:
1. **Phase 1 (Weeks 1-2)**: Add `use_planner: bool` flag to RobotSimulationConfig
   - Default: False (existing manual routes)
   - When True: `sample_route()` delegates to `GlobalPlanner.plan()`
   
2. **Phase 2 (Weeks 3-4)**: Auto-detect maps without manual paths
   - If `map_def.robot_routes` is empty → automatically use planner
   - Preserve backward compatibility for maps with existing routes

3. **Phase 3 (Week 5+)**: Deprecation notices
   - Mark manual `<path>` waypoints as deprecated in docs
   - Provide migration script to extract POIs from existing paths

**Configuration Example**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.gym_env.environment_factory import make_robot_env

config = RobotSimulationConfig()
config.use_planner = True
config.planner_clearance_margin = 0.4  # meters

env = make_robot_env(config=config)
```

### 7. Error Handling & Edge Cases

**Decision**: Explicit error types with diagnostic information

**Cases & Strategies**:

1. **No Valid Path** (`PlanningFailedError`):
   ```python
   class PlanningFailedError(Exception):
       def __init__(self, start, goal, reason: str):
           self.start, self.goal = start, goal
           super().__init__(f"No path from {start} to {goal}: {reason}")
   ```
   - Diagnostic: Report obstacle configuration, inflated geometry, reachability
   - Fallback (configurable): Return straight-line path with warning

2. **Start/Goal Inside Obstacles**:
   - Project to nearest free space using Shapely's `nearest_points()`
   - Log warning with original and projected positions
   - Continue planning from projected points

3. **Narrow Passages**:
   - Detect when path segment clearance < `min_safe_clearance` threshold
   - Emit warning: "Path traverses narrow gap at {position}: {clearance:.2f}m"
   - Allow path but flag in return metadata

4. **Degenerate Maps** (no obstacles):
   - Skip graph construction entirely
   - Return straight-line path: `[start, goal]`
   - Log info message about optimization

### 8. Testing Strategy

**Pyramid Structure**:

1. **Unit Tests** (tests/test_planner/):
   - `test_global_planner.py`: Core planning logic, clearance verification
   - `test_visibility_graph.py`: Graph construction, caching
   - `test_path_smoother.py`: Douglas-Peucker correctness
   - `test_poi_sampler.py`: POI generation strategies

2. **Integration Tests**:
   - `test_map_integration.py`: All example SVG maps generate valid paths
   - `test_navigation_integration.py`: RouteNavigator compatibility
   - Performance validation: <100ms median on reference hardware

3. **Smoke Tests**:
   - `scripts/validation/verify_planner.sh`: Quick sanity check
   - Maps: simple_corridor, narrow_passage, no_path, complex_warehouse

**Test Maps** (tests/fixtures/test_maps/):
```xml
<!-- simple_corridor.svg: Straight passage, no obstacles -->
<!-- narrow_passage.svg: 1m gap with 0.8m robot + 0.3m clearance (tight fit) -->
<!-- no_path.svg: Goal completely surrounded by obstacles -->
<!-- complex_warehouse.svg: 50+ obstacles, multiple valid paths -->
```

### 9. Performance Profiling

**Benchmark Script** (scripts/benchmark_planner.py):
```python
# Profile graph build + path queries
map_sizes = [10, 25, 50, 75, 100]  # obstacle counts
for num_obstacles in map_sizes:
    map_def = generate_test_map(num_obstacles)
    
    # Time graph build (one-time)
    start = time.perf_counter()
    planner = GlobalPlanner(map_def)
    build_time = time.perf_counter() - start
    
    # Time 100 path queries
    times = []
    for _ in range(100):
        start = time.perf_counter()
        path = planner.plan(sample_start(), sample_goal())
        times.append(time.perf_counter() - start)
    
    print(f"{num_obstacles} obstacles: "
          f"build={build_time*1000:.1f}ms, "
          f"query_median={np.median(times)*1000:.1f}ms")
```

**Expected Targets**:
- 50 obstacles: build ~400ms, query median ~50ms
- 100 obstacles: build ~800ms, query median ~90ms

### 10. Dependencies Summary

**New Dependencies**:
```toml
[project.dependencies]
pyvisgraph = ">=0.2.1"   # Visibility graph construction
networkx = ">=3.6"       # Graph algorithms (Dijkstra, A*)
```

**Existing Dependencies** (already satisfied):
```toml
shapely = ">=2.1.2"      # Polygon operations (already in use)
numpy = ">=1.24"         # Array operations (already in use)
```

**Optional (Future)**:
```toml
[project.optional-dependencies]
planning-viz = [
    "matplotlib>=3.5",   # Visualize paths and graphs
    "rdp>=0.8",         # Ramer-Douglas-Peucker simplification
]
```

## Resolved Unknowns

All "NEEDS CLARIFICATION" items from Technical Context have been resolved:

1. ✅ **Primary algorithm**: pyvisgraph visibility graph
2. ✅ **Obstacle handling**: Shapely buffering (robot radius + clearance)
3. ✅ **Path smoothing**: Douglas-Peucker simplification
4. ✅ **Caching strategy**: Per-map-hash in-memory cache
5. ✅ **SVG extension**: `<circle class="poi">` format
6. ✅ **Integration approach**: Opt-in flag + fallback compatibility
7. ✅ **Error handling**: Explicit exceptions with diagnostics
8. ✅ **Testing structure**: Unit + integration + smoke tests
9. ✅ **Performance validation**: Benchmark script profiling
10. ✅ **Dependencies**: pyvisgraph + networkx (minimal additions)

## References

### Academic Background
- Lee, D.T. (1978). "Proximity and reachability in the plane" - Visibility graphs
- Kavraki et al. (1996). "Probabilistic roadmaps" - Alternative sampling approach

### Libraries Evaluated
- **pyvisgraph**: https://github.com/TaipanRex/pyvisgraph
- **Shapely**: https://shapely.readthedocs.io/ (already in use)
- **NetworkX**: https://networkx.org/

### Existing Patterns
- **Factory pattern**: robot_sf.gym_env.environment_factory
- **Unified configs**: robot_sf.gym_env.unified_config
- **SVG parsing**: robot_sf.nav.svg_map_parser
- **Navigation tracking**: robot_sf.nav.navigation.RouteNavigator

## Next Steps

Phase 0 research complete. Ready to proceed to Phase 1: Design & Contracts.

**Artifacts to Generate in Phase 1**:
1. `data-model.md` - Entity relationships and validation rules
2. `contracts/global_planner_api.md` - Public API specification
3. `quickstart.md` - Developer onboarding guide
4. Update agent context files
