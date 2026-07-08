# Global Planner for SVG-Based Navigation

## Executive Summary

Implement an automated global planner that generates collision-free waypoint paths from spawn zones to goal zones around static obstacles. This replaces the current manual waypoint definition in SVG maps with geometry-based path planning, enabling faster map creation and more flexible robot navigation scenarios.

**Key benefit**: Reduce map authoring effort by auto-generating routes while maintaining compatibility with existing `RouteNavigator` and `GlobalRoute` infrastructure.

## Problem Statement

### Current Limitations
- **Manual effort**: Routes require hand-authored waypoint sequences in SVG files
- **Maintenance burden**: Obstacle changes invalidate manually defined routes
- **Limited flexibility**: Difficult to generate dynamic start/goal combinations

### Desired Capability
- **Automated path generation**: Compute collision-free paths from spawn→goal zones
- **Obstacle awareness**: Navigate around static obstacles with configurable clearance
- **Flexible start points**: Support both map-defined POIs and dynamic robot or pedestrian positions

## Requirements

### Functional Requirements
- [x] Generate waypoint paths avoiding static polygon obstacles
- [x] Support spawn/goal zone sampling (existing `MapDefinition` zones)
- [x] Optional POI-based waypoints via SVG annotations, that can also be used as start/goal points
- [x] Configurable obstacle clearance margin (robot radius + safety buffer)
- [x] Fallback behavior when no path exists (raise warning with detailed description of the problem, try to find approximate path)
- [x] Path smoothing/simplification to reduce waypoint count

### Non-Functional Requirements
- **Performance**: Path generation <100ms for typical maps (<50 obstacles, <20 POIs)
- **Accuracy**: Paths maintain minimum clearance of `robot_radius + 0.3m` from obstacles
- **Compatibility**: Produces `list[Vec2D]` compatible with existing `RouteNavigator`
- **Determinism**: Same inputs produce same paths (seeded random sampling)

### Explicitly Out of Scope (Non-Goals)
- ❌ Dynamic obstacle avoidance (pedestrians, other robots)
- ❌ Multi-robot path coordination
- ❌ 3D or multi-level navigation

### Optional goals with a low priority

- ⚪ Support multiple pathfinding algorithms (visibility graph, RRT, PRM)
- ⚪ Visualization tools for debugging generated paths
- ⚪ Integration with dynamic replanning frameworks in future
- ⚪ User-defined cost functions (prefer wider corridors, avoid certain areas)
- Real-time replanning during navigation
- Trajectory optimization (velocity profiles, acceleration limits)

## Current Architecture

**Validated data flow** (based on existing codebase):

```
SVG Map File (maps/svg_maps/*.svg)
    ↓
svg_map_parser.py → MapDefinition
    ├─ obstacles: list[Obstacle]
    ├─ robot_spawn_zones: list[Rect]
    ├─ robot_goal_zones: list[Rect]
    └─ robot_routes: list[GlobalRoute]  ← manually defined waypoints
         ↓
navigation.sample_route() → list[Vec2D]
    ↓
RouteNavigator.new_route() → tracks progress along waypoints
```

**Key existing components:**
- `robot_sf/nav/svg_map_parser.py` - Parses SVG elements into `MapDefinition`
- `robot_sf/nav/obstacle.py` - Polygon obstacles with vertex/line representations
- `robot_sf/nav/global_route.py` - Route container (spawn→waypoints→goal)
- `robot_sf/nav/navigation.py` - Route sampling and waypoint tracking
- `robot_sf/nav/occupancy_grid.py` - Grid-based collision checking (already uses Shapely)

## Design Proposal

### Module Structure

```
robot_sf/planner/
├── __init__.py
├── global_planner.py      # Main GlobalPlanner class
├── visibility_graph.py    # Visibility graph construction
├── poi_sampler.py         # POI generation strategies
└── path_smoother.py       # Post-processing utilities
```

### Public API

```python
from robot_sf.planner.global_planner import GlobalPlanner, PlannerConfig
from robot_sf.common.types import Vec2D
from robot_sf.nav.map_config import MapDefinition

# Initialize planner with map
config = PlannerConfig(
    clearance_margin=0.3,      # meters beyond robot radius
    max_iterations=1000,       # for sampling/search
    smoothing_enabled=True,    # reduce waypoint count
    poi_density=0.5,           # POIs per square meter (for auto-sampling)
)
planner = GlobalPlanner(map_def=map_definition, config=config)

# Generate path from spawn to goal
path = planner.plan(
    start=spawn_position,      # Vec2D or sample from spawn zone
    goal=goal_position,        # Vec2D or sample from goal zone
    use_pois=True,             # route through map-defined POIs
)
# Returns: list[Vec2D] - waypoints from start to goal

# Auto-generate POIs for a map (optional preprocessing)
poi_positions = planner.sample_pois(
    num_pois=10,               # or density-based
    strategy="visibility",     # "random", "visibility", "medial_axis"
    min_clearance=1.0,         # minimum obstacle distance
)
```

### SVG Map Extensions

#### POI Annotation Syntax

Extend SVG map specification to support `<circle>` elements with `poi` class:

```xml
<!-- Existing spawn/goal zones -->
<rect class="spawn" id="spawn_1" x="10" y="10" width="5" height="5" />
<rect class="goal" id="goal_1" x="90" y="90" width="5" height="5" />

<!-- NEW: Point-of-Interest markers -->
<circle class="poi" id="poi_corridor_1" cx="50" cy="25" r="0.5" 
        label="corridor_junction" />
<circle class="poi" id="poi_doorway" cx="30" cy="50" r="0.5" 
        label="room_entrance" />
```

**Parser modifications** (in `svg_map_parser.py`):
- Extract `<circle class="poi">` elements
- Add `poi_positions: list[Vec2D]` to `MapDefinition`
- Optional: `poi_graph: dict[str, list[str]]` for predefined connectivity

### Integration Points

**Replaces/Extends:**
- `robot_sf.nav.navigation.sample_route()` - new implementation delegates to planner
- `MapDefinition.robot_routes` - optionally auto-populated via planner

**Consumes:**
- `MapDefinition.obstacles` - polygon obstacles for visibility graph
- `MapDefinition.robot_spawn_zones` - sampling valid start positions
- `MapDefinition.robot_goal_zones` - sampling valid goal positions

**Produces:**
- `list[Vec2D]` - directly compatible with `RouteNavigator.new_route()`
- Optionally: populate `GlobalRoute` instances for backward compatibility

**Migration strategy:**
- Phase 1: Planner runs alongside existing manual routes (opt-in)
- Phase 2: Auto-generate routes for maps without `<path>` elements
- Phase 3: Deprecate manual `<path>` waypoints in favor of planner

## Implementation Options

### Comparison Table

| Approach | Setup Complexity | Runtime Performance | Path Quality | Maintenance |
|----------|-----------------|---------------------|--------------|-------------|
| **pyvisgraph** | Low (library handles it) | Fast (<50ms typical) | Optimal shortest path | External dependency |
| **Shapely + NetworkX** | Medium (custom graph build) | Medium (50-100ms) | Flexible (custom costs) | More control |
| **RRT/PRM sampling** | High (parameter tuning) | Slow (>100ms) | Suboptimal | Overkill for static maps |

### Recommended Approach: Hybrid Strategy

**Primary**: Use **pyvisgraph** for initial implementation
- Mature, tested visibility graph library
- Clean API: `graph.build()`, `graph.shortest_path()`
- Meets performance targets for typical maps

**Fallback/Extension**: Shapely + NetworkX for advanced features
- Custom edge costs (prefer wider corridors, penalize narrow gaps)
- Multi-criteria optimization (shortest + safest path)
- Graph caching and incremental updates

**Implementation phases:**
1. **MVP**: pyvisgraph-based planner with basic clearance inflation
2. **Enhanced**: Add path smoothing and POI support
3. **Advanced**: Custom graph builder with width-aware costs

### Example Usage Flow

```python
# Example: Generate route for training scenario
from robot_sf.planner.global_planner import GlobalPlanner
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.ped_npc.ped_zone import sample_zone

# Load map and initialize planner
map_def = load_map("maps/svg_maps/warehouse.svg")
planner = GlobalPlanner(map_def, clearance_margin=0.4)

# Sample spawn and goal from zones
spawn_pos = sample_zone(map_def.robot_spawn_zones[0], 1)[0]
goal_pos = sample_zone(map_def.robot_goal_zones[0], 1)[0]

# Generate path
waypoints = planner.plan(start=spawn_pos, goal=goal_pos)

# Use with existing navigator
navigator = RouteNavigator(waypoints=waypoints, proximity_threshold=1.0)
navigator.update_position(robot_position)
next_target = navigator.current_waypoint
```

## Technical Decisions

### Graph Representation
**Decision**: Visibility graph with lazy edge construction
- **Rationale**: Geometry-accurate, avoids discretization artifacts
- **Trade-off**: More edges than grid-based roadmap, but acceptable for <50 obstacles

### Obstacle Inflation
**Decision**: Robot-radius-aware buffering via Shapely
```python
inflated_obstacles = [
    obstacle.buffer(robot_radius + clearance_margin) 
    for obstacle in map_obstacles
]
```
- **Rationale**: Converts kinematic robot to point-robot problem
- **Trade-off**: May close off narrow passages (accept for safety)

### Path Smoothing
**Decision**: Douglas-Peucker simplification + optional spline fitting
- **Rationale**: Reduce waypoint count while maintaining obstacle clearance
- **Trade-off**: Slight path elongation acceptable for smoother trajectories

### Caching Strategy
**Decision**: Lazy graph construction, cache per map hash
```python
_graph_cache: dict[str, VisibilityGraph] = {}

def _get_or_build_graph(map_def: MapDefinition) -> VisibilityGraph:
    map_hash = hashlib.md5(repr(map_def.obstacles).encode()).hexdigest()
    if map_hash not in _graph_cache:
        _graph_cache[map_hash] = build_visibility_graph(map_def)
    return _graph_cache[map_hash]
```
- **Rationale**: Amortize graph build cost across multiple queries
- **Trade-off**: Memory usage (acceptable for ~10 maps in typical training)

## Success Criteria

### Functional Completeness
- [x] `GlobalPlanner.plan()` generates collision-free paths
- [x] Handles all existing SVG maps without errors
- [x] POI sampling with configurable strategies
- [x] SVG parser extended for `<circle class="poi">` elements
- [x] Backward compatibility: `navigation.sample_route()` delegates to planner

### Performance Benchmarks
- [x] Path generation: <100ms @ 50 obstacles (median case)
- [x] Graph build: <500ms @ 50 obstacles (one-time per map)
- [x] POI sampling: <50ms for 20 POIs

### Code Quality
- [x] Test coverage: >90% (unit tests for planner, graph builder, smoother)
- [x] Type hints: All public APIs fully annotated
- [x] Documentation: Google-style docstrings + usage examples
- [x] CI integration: Smoke tests on all example maps

### Edge Case Handling
- [x] No valid path exists → raise `PlanningFailedError` with diagnostics
- [x] Start/goal inside obstacles → project to nearest free space
- [x] Narrow passages → warn if clearance < `min_safe_clearance`
- [x] Degenerate maps (no obstacles) → return straight-line path

## Testing Strategy

### Unit Tests
```python
# tests/test_planner/test_global_planner.py
def test_simple_corridor_path():
    """Path through straight corridor avoids walls."""
    
def test_obstacle_clearance_maintained():
    """All waypoints maintain minimum clearance."""
    
def test_no_path_raises_exception():
    """Surrounded goal raises PlanningFailedError."""

def test_poi_routing():
    """Path passes through specified POIs in order."""
```

### Integration Tests
```python
# tests/test_planner/test_map_integration.py
def test_all_example_maps_plannable():
    """Every map in maps/svg_maps/ generates valid paths."""
    
def test_route_navigator_compatibility():
    """Planner output works with RouteNavigator."""
```

### Sample Maps for Testing
- `simple_corridor.svg` - Baseline (straight path)
- `narrow_passage.svg` - Stress test clearance constraints
- `no_path.svg` - Goal completely blocked
- `complex_warehouse.svg` - 50+ obstacles, multiple paths

### Performance Profiling
```bash
# Benchmark script
uv run python scripts/benchmark_planner.py \
    --map maps/svg_maps/warehouse.svg \
    --iterations 100 \
    --profile
```

**Expected output:**
```
Map: warehouse.svg (42 obstacles)
Graph build: 312ms (one-time)
Path planning (100 runs):
  - Median: 47ms
  - 95th percentile: 89ms
  - Max: 134ms
```

## Dependencies

### Required Packages
```toml
[project]
dependencies = [
    "shapely>=2.1.2",      # Polygon operations, obstacle buffering
    "networkx>=3.6",       # Graph search (Dijkstra, A*)
    "pyvisgraph>=0.2.1",   # Visibility graph construction
]
```

**Justification:**
- **Shapely**: Already in use for `occupancy_grid.py`; provides robust polygon union, buffering, and spatial queries
- **NetworkX**: Industry-standard graph library; minimal overhead for ~1000-node graphs
- **pyvisgraph**: Proven visibility graph implementation; faster than rolling our own

### Optional Packages (Future Extensions)
```toml
[project.optional-dependencies]
planning-viz = [
    "matplotlib>=3.5",     # Visualize paths and graphs
    "networkx[default]",   # Graph layout algorithms
]
```

## Migration Path

### Phase 1: Opt-In Planner (Weeks 1-2)
- Implement `GlobalPlanner` with pyvisgraph backend
- Add `use_planner: bool` flag to environment config
- Keep existing `sample_route()` as default

### Phase 2: Hybrid Mode (Weeks 3-4)
- Auto-detect maps with/without manual `<path>` elements
- Planner generates routes for maps lacking paths
- Add POI support to SVG parser

### Phase 3: Deprecation (Week 5+)
- Mark manual `<path>` waypoints as deprecated
- Provide migration script to extract POIs from existing paths
- Update all example maps to use planner-generated routes

## Open Questions

1. **Narrow passage handling**: Reject paths through gaps < `robot_diameter`, or allow with warning?
   - **Proposal**: Warn but allow; let `min_safe_clearance` be configurable
   
2. **POI ordering constraints**: Should planner enforce visiting POIs in sequence?
   - **Proposal**: Optional `ordered_pois: bool` parameter (default: `False`)
   
3. **Multi-goal support**: Plan tours visiting multiple goals?
   - **Proposal**: Out of scope for MVP; revisit in Phase 3

4. **Dynamic start points**: Cache graph, recompute only start/goal edges?
   - **Proposal**: Yes; implement lazy edge updates for robot position

## References

### Academic Background
- Lee, D.T. (1978). "Proximity and reachability in the plane" - Visibility graphs
- Kavraki et al. (1996). "Probabilistic roadmaps" - Alternative sampling approach

### Similar Implementations
- ROS `nav2_core::GlobalPlanner` - Plugin architecture reference
- OMPL (Open Motion Planning Library) - Geometric planners

### Existing Issues/Discussions
- Issue #254: Add noise to waypoint positions (affects planner output)
- Issue #342: SVG-based global planner (this document)
