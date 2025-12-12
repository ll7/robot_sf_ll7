# Data Model: SVG-Based Global Planner

**Phase**: 1 (Design & Contracts)  
**Date**: 2025-12-10  
**Status**: Complete

## Overview

This document defines the data structures, validation rules, and entity relationships for the global planner feature. All types integrate with existing robot_sf conventions (Vec2D, MapDefinition, unified configs).

## Core Entities

### 1. GlobalPlanner (Public API)

**Purpose**: Main entry point for path planning queries

```python
from dataclasses import dataclass
from typing import Optional
from robot_sf.common.types import Vec2D
from robot_sf.nav.map_config import MapDefinition

@dataclass
class PlannerConfig:
    """Configuration for global path planner."""
    
    robot_radius: float = 0.4
    """Robot body radius in meters (for obstacle inflation)."""
    
    min_safe_clearance: float = 0.3
    """Minimum clearance margin beyond robot radius (meters)."""
    
    enable_smoothing: bool = True
    """Apply Douglas-Peucker simplification to reduce waypoints."""
    
    smoothing_epsilon: float = 0.1
    """Douglas-Peucker tolerance parameter (meters)."""
    
    cache_graphs: bool = True
    """Enable in-memory caching of visibility graphs."""
    
    fallback_on_failure: bool = True
    """Return straight-line path when planning fails (with warning)."""
    
    def __post_init__(self):
        """Validate configuration constraints."""
        if self.robot_radius <= 0:
            raise ValueError(f"robot_radius must be positive, got {self.robot_radius}")
        if self.min_safe_clearance < 0:
            raise ValueError(f"min_safe_clearance cannot be negative, got {self.min_safe_clearance}")
        if self.smoothing_epsilon <= 0 and self.enable_smoothing:
            raise ValueError(f"smoothing_epsilon must be positive when smoothing enabled")


class GlobalPlanner:
    """
    Visibility-graph-based path planner for 2D environments with polygonal obstacles.
    
    Generates collision-free paths between arbitrary start/goal positions using
    pyvisgraph library. Caches visibility graphs per map for efficiency.
    
    Example:
        >>> planner = GlobalPlanner(map_definition, config)
        >>> path = planner.plan(start=Vec2D(0, 0), goal=Vec2D(10, 5))
        >>> print(f"Path has {len(path)} waypoints")
    """
    
    def __init__(
        self,
        map_definition: MapDefinition,
        config: Optional[PlannerConfig] = None
    ):
        """
        Initialize planner with map and configuration.
        
        Args:
            map_definition: Contains obstacles and POIs
            config: Planner configuration (uses defaults if None)
        
        Raises:
            ValueError: If map has zero boundaries or invalid obstacles
        """
        self.map_def = map_definition
        self.config = config or PlannerConfig()
        self._graph = None  # Lazy-built visibility graph
        
    def plan(
        self,
        start: Vec2D,
        goal: Vec2D,
        *,
        via_pois: Optional[list[str]] = None
    ) -> list[Vec2D]:
        """
        Compute collision-free path from start to goal.
        
        Args:
            start: Starting position (meters)
            goal: Goal position (meters)
            via_pois: Optional POI IDs to route through (ordered)
        
        Returns:
            List of waypoints forming collision-free path.
            Includes start and goal as first/last elements.
        
        Raises:
            PlanningFailedError: No valid path exists and fallback disabled
            
        Performance:
            First call: 200-500ms (graph build + query)
            Cached: <100ms (query only)
        """
        ...
        
    def plan_multi_goal(
        self,
        start: Vec2D,
        goals: list[Vec2D],
        *,
        optimize_order: bool = True
    ) -> list[Vec2D]:
        """
        Plan path visiting multiple goals (travelling salesman).
        
        Args:
            start: Starting position
            goals: List of goal positions to visit
            optimize_order: If True, reorder goals to minimize total path length
        
        Returns:
            Path visiting all goals, starting from start.
            Order respects optimize_order parameter.
            
        Note:
            Uses nearest-neighbor heuristic when optimize_order=True.
            Exact TSP solution deferred to Phase 3 (NP-hard).
        """
        ...
        
    def invalidate_cache(self) -> None:
        """Clear cached visibility graph (e.g., after map changes)."""
        self._graph = None
```

**Validation Rules**:
- `start` and `goal` must lie within map boundaries (`0 ≤ x ≤ width`, `0 ≤ y ≤ height`)
- `via_pois` IDs must exist in `map_definition.poi_labels`
- Raise `PlanningFailedError` if no path found and `fallback_on_failure=False`

---

### 2. VisibilityGraph (Internal)

**Purpose**: Encapsulates graph construction and query logic

```python
from typing import Protocol
from shapely.geometry import Polygon

class VisibilityGraph(Protocol):
    """
    Internal visibility graph abstraction (implementation detail).
    
    Note: This is a protocol matching pyvisgraph.VisGraph interface.
          Not directly exposed to users.
    """
    
    def build(
        self,
        polygons: list[Polygon],
        workers: int = 1
    ) -> None:
        """Construct visibility graph from polygons."""
        ...
        
    def shortest_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Query shortest path between two points."""
        ...
```

**Relationship to GlobalPlanner**:
- `GlobalPlanner` owns one `VisibilityGraph` instance (lazy-built)
- Graph built on first `plan()` call or when cache miss occurs
- Internal caching handled by `_get_or_build_graph()` helper

**Data Flow**:
```
MapDefinition.obstacles
    ↓ (inflate by robot_radius + clearance)
Shapely Polygons
    ↓ (build visibility graph)
VisibilityGraph
    ↓ (shortest_path query)
Raw waypoints
    ↓ (optional smoothing)
Final Path (list[Vec2D])
```

---

### 3. MapDefinition Extensions

**Purpose**: Add POI support to existing map structure

**Changes to robot_sf/nav/map_config.py**:
```python
from dataclasses import dataclass, field
from robot_sf.common.types import Vec2D

@dataclass
class MapDefinition:
    """
    Extended with POI positions for global planner integration.
    
    BACKWARD COMPATIBLE: Existing code without POIs continues to work.
    """
    
    # ... existing fields (width, height, obstacles, robot_spawn_zones, goals) ...
    
    poi_positions: list[Vec2D] = field(default_factory=list)
    """Point-of-interest waypoints parsed from SVG <circle class='poi'>."""
    
    poi_labels: dict[str, str] = field(default_factory=dict)
    """Mapping from POI ID to human-readable label."""
    
    def get_poi_by_label(self, label: str) -> Vec2D:
        """
        Retrieve POI position by label.
        
        Args:
            label: POI label (e.g., "corridor_junction")
            
        Returns:
            POI position as Vec2D
            
        Raises:
            KeyError: If label not found in map
        """
        poi_id = next((k for k, v in self.poi_labels.items() if v == label), None)
        if poi_id is None:
            raise KeyError(f"No POI with label '{label}' found in map")
        idx = list(self.poi_labels.keys()).index(poi_id)
        return self.poi_positions[idx]
```

**Validation Rules**:
- `len(poi_positions) == len(poi_labels)` (one-to-one correspondence)
- All POI positions must lie within map boundaries
- POI IDs must be unique within a map
- Empty `poi_positions` is valid (backward compatibility)

---

### 4. PlanningFailedError (Exception Type)

**Purpose**: Typed exception with diagnostic information

```python
from robot_sf.common.types import Vec2D

class PlanningFailedError(Exception):
    """
    Raised when no valid path exists between start and goal.
    
    Attributes:
        start: Starting position that was requested
        goal: Goal position that was requested
        reason: Human-readable explanation (e.g., "goal inside obstacle")
    """
    
    def __init__(self, start: Vec2D, goal: Vec2D, reason: str):
        self.start = start
        self.goal = goal
        self.reason = reason
        super().__init__(
            f"Planning failed: {reason}\n"
            f"  Start: ({start.x:.2f}, {start.y:.2f})\n"
            f"  Goal: ({goal.x:.2f}, {goal.y:.2f})"
        )
```

**When Raised**:
- Start or goal position inside inflated obstacle
- No line-of-sight edges connect start/goal to graph
- pyvisgraph returns empty path (disconnected components)

**Handling Strategy**:
```python
try:
    path = planner.plan(start, goal)
except PlanningFailedError as e:
    if planner.config.fallback_on_failure:
        logger.warning(f"Planning failed, using straight line: {e.reason}")
        path = [start, goal]  # Fallback path
    else:
        raise  # Re-raise for caller to handle
```

---

### 5. POISampler (Helper Utility)

**Purpose**: Generate intermediate waypoints from POIs

```python
from typing import Literal

class POISampler:
    """
    Utility for sampling POI-based waypoints.
    
    Used to create varied navigation scenarios by randomly selecting
    intermediate waypoints from map POIs.
    """
    
    def __init__(self, map_definition: MapDefinition, seed: Optional[int] = None):
        """
        Initialize sampler with map POIs.
        
        Args:
            map_definition: Must contain at least one POI
            seed: Random seed for reproducibility (None = non-deterministic)
            
        Raises:
            ValueError: If map has no POIs defined
        """
        if not map_definition.poi_positions:
            raise ValueError("Cannot sample POIs from map without POIs")
        self.pois = map_definition.poi_positions
        self.rng = np.random.default_rng(seed)
        
    def sample(
        self,
        count: int,
        strategy: Literal["random", "nearest", "farthest"] = "random"
    ) -> list[Vec2D]:
        """
        Sample intermediate POIs for path planning.
        
        Args:
            count: Number of POIs to sample (clamped to available POIs)
            strategy: Selection strategy
                - "random": Uniform sampling without replacement
                - "nearest": Select closest POIs to start (requires start param)
                - "farthest": Select farthest POIs from start
        
        Returns:
            List of POI positions (length min(count, len(pois)))
        """
        ...
```

**Use Case**:
```python
# Generate training episodes with varied routes
sampler = POISampler(map_def, seed=42)
via_waypoints = sampler.sample(count=3, strategy="random")
path = planner.plan(start, goal, via_pois=[...])
```

---

## Entity Relationships

```
┌─────────────────────┐
│ RobotSimulationConfig│
│  - use_planner: bool│
│  - clearance: float │
└──────────┬──────────┘
           │ uses
           ▼
┌─────────────────────┐       ┌──────────────────┐
│   GlobalPlanner     │──────▶│  PlannerConfig   │
│  - map_def          │ owns  │  - robot_radius  │
│  - config           │       │  - clearance     │
│  + plan()           │       └──────────────────┘
└──────────┬──────────┘
           │ lazy-builds
           ▼
┌─────────────────────┐       ┌──────────────────┐
│  VisibilityGraph    │──────▶│ pyvisgraph.VisGraph
│  (internal wrapper) │ wraps │  (3rd party)     │
└─────────────────────┘       └──────────────────┘
           ▲
           │ caches
┌──────────┴──────────┐
│  _graph_cache       │
│  dict[str, Graph]   │
└─────────────────────┘

┌─────────────────────┐       ┌──────────────────┐
│   MapDefinition     │──────▶│   POISampler     │
│  + poi_positions    │ feeds │  + sample()      │
│  + poi_labels       │       └──────────────────┘
└─────────────────────┘
```

---

## Validation Matrix

| Entity             | Constraint                              | Enforcement Point        | Error Type          |
|--------------------|-----------------------------------------|--------------------------|---------------------|
| PlannerConfig      | `robot_radius > 0`                      | `__post_init__`          | ValueError          |
| PlannerConfig      | `min_safe_clearance >= 0`               | `__post_init__`          | ValueError          |
| GlobalPlanner.plan | `start` within map bounds               | `plan()` entry           | ValueError          |
| GlobalPlanner.plan | `goal` within map bounds                | `plan()` entry           | ValueError          |
| GlobalPlanner.plan | `via_pois` IDs exist in map             | `plan()` validation      | KeyError            |
| MapDefinition      | `len(poi_positions) == len(poi_labels)` | `svg_map_parser.py`      | AssertionError      |
| MapDefinition      | All POIs within bounds                  | `svg_map_parser.py`      | ValueError          |
| POISampler         | Map has at least one POI                | `__init__`               | ValueError          |
| VisibilityGraph    | No degenerate polygons (area > ε)       | `_inflate_obstacles()`   | Warning (skip poly) |

---

## Type Aliases

```python
# robot_sf/common/types.py additions
from typing import TypeAlias

Path: TypeAlias = list[Vec2D]
"""Sequence of 2D waypoints forming a trajectory."""

POIID: TypeAlias = str
"""Unique identifier for point-of-interest (matches SVG id attribute)."""
```

---

## Serialization

### JSON Export (for debugging/visualization)

```python
def path_to_json(path: list[Vec2D]) -> str:
    """Serialize path for debugging/logging."""
    import json
    return json.dumps([{"x": p.x, "y": p.y} for p in path])

# Example output:
# [{"x": 0.0, "y": 0.0}, {"x": 5.2, "y": 3.1}, {"x": 10.0, "y": 5.0}]
```

### SVG Overlay (for visualization)

```python
def path_to_svg_polyline(path: list[Vec2D]) -> str:
    """Generate SVG <polyline> for path visualization."""
    points = " ".join(f"{p.x},{p.y}" for p in path)
    return f'<polyline points="{points}" stroke="blue" stroke-width="0.2" fill="none" />'
```

---

## Immutability & Thread Safety

**Design Principle**: `GlobalPlanner` is **thread-safe for reads after construction**.

- `PlannerConfig` is immutable (dataclass with no setters)
- `MapDefinition` assumed immutable after parsing
- `_graph_cache` uses map content hash as key (no mutation tracking needed)

**Not Thread-Safe**:
- Concurrent calls to `invalidate_cache()` during `plan()` execution
- Modifying `MapDefinition.obstacles` after planner creation

**Recommendation**: Create one `GlobalPlanner` instance per map, reuse across episodes.

---

## Performance Characteristics

| Operation                  | Complexity      | Typical Time (50 obstacles) |
|----------------------------|-----------------|-----------------------------|
| `GlobalPlanner.__init__`   | O(1)            | <1ms (lazy graph build)     |
| First `plan()` call        | O(n²)           | 200-500ms (graph + query)   |
| Cached `plan()` calls      | O(n log n)      | 50-100ms (query only)       |
| `plan_multi_goal()`        | O(k · n log n)  | k × single-goal time        |
| `invalidate_cache()`       | O(1)            | <1ms (clear reference)      |

*n = number of obstacle vertices, k = number of goals*

---

## Migration Notes

### Backward Compatibility

**Existing Code** (without planner):
```python
# Old: Manual route definition in SVG
map_def = parse_svg("map.svg")  # Contains <path> elements
navigator = RouteNavigator(map_def.robot_routes[0])
```

**New Code** (with planner):
```python
# New: Auto-generated routes
map_def = parse_svg("map.svg")  # Contains <circle class="poi">
planner = GlobalPlanner(map_def)
path = planner.plan(start, goal)
navigator = RouteNavigator(path)  # Same interface!
```

**Graceful Degradation**:
- Maps without POIs skip planner (use existing `robot_routes`)
- Config flag `use_planner=False` disables feature entirely
- Existing examples continue working unchanged

---

## Open Questions for Phase 2

1. **Spline Fitting**: Which library (scipy.interpolate vs custom Catmull-Rom)?
2. **Cache Eviction**: LRU policy or unbounded for MVP?
3. **Parallel Queries**: Support batch `plan()` calls with thread pool?
4. **Dynamic Obstacles**: Hook for runtime obstacle updates?

These defer to Phase 2 (Implementation) or Phase 3 (Advanced Features).

---

## Next Steps

Phase 1 data model complete. Ready to proceed to:
1. `contracts/global_planner_api.md` - Formal API contract
2. `quickstart.md` - Developer onboarding guide
3. Update agent context files
