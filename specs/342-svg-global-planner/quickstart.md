# Quickstart: SVG-Based Global Planner

**Audience**: Developers integrating the global planner into robot_sf environments  
**Prerequisites**: Familiarity with robot_sf factory pattern and unified configs  
**Time to Complete**: ~10 minutes

## Overview

The global planner generates collision-free paths between arbitrary positions using visibility graph algorithms. This guide covers:
1. Basic path planning (start → goal)
2. Integration with gym environments
3. Using POIs for waypoint routing
4. Multi-goal planning
5. Performance tuning

---

## Installation

### Dependencies

The planner requires two additional packages:

```bash
# Add to pyproject.toml dependencies
uv add "pyvisgraph>=0.2.1" "networkx>=3.6"

# Or install directly
uv pip install pyvisgraph>=0.2.1 networkx>=3.6
```

**Note**: `shapely>=2.1.2` already in use by robot_sf (no action needed).

### Verify Installation

```python
# Test import
from robot_sf.planner import GlobalPlanner, PlannerConfig

print("✓ Global planner available")
```

---

## 1. Basic Path Planning

### Standalone Usage

```python
from robot_sf.planner import GlobalPlanner, PlannerConfig
from robot_sf.nav.svg_map_parser import convert_map

# Load map
map_def = convert_map("maps/svg_maps/simple_corridor.svg")

# Configure planner
config = PlannerConfig(
    robot_radius=0.4,
    min_safe_clearance=0.3,
    enable_smoothing=True
)

# Create planner
planner = GlobalPlanner(map_def, config)

# Plan path
start = Vec2D(1.0, 2.0)
goal = Vec2D(18.0, 8.0)
path = planner.plan(start, goal)

print(f"Generated path with {len(path)} waypoints:")
for i, waypoint in enumerate(path):
    print(f"  {i}: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
```

**Expected Output**:
```
Generated path with 5 waypoints:
  0: (1.00, 2.00)   # Start
  1: (5.20, 3.10)   # Around obstacle
  2: (12.50, 5.80)  # Around obstacle
  3: (16.30, 7.20)  # Around obstacle
  4: (18.00, 8.00)  # Goal
```

---

## 2. Integration with Gym Environments

### Using Factory Functions

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

# Enable planner via config
config = RobotSimulationConfig()
config.use_planner = True
config.planner_clearance_margin = 0.4
config.map_pool = ["maps/svg_maps/simple_corridor.svg"]

# Create environment (planner auto-initialized)
env = make_robot_env(config=config)

# Environment now uses planner for route generation
obs, info = env.reset()
print(f"Robot spawned at: {info['robot_position']}")
print(f"Goal position: {info['goal_position']}")
print(f"Route waypoints: {len(info['route'])}")
```

### Manual Integration

```python
from robot_sf.planner import GlobalPlanner
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.nav.svg_map_parser import convert_map

# Load map and create planner
map_def = convert_map("maps/svg_maps/complex_warehouse.svg")
planner = GlobalPlanner(map_def)

# Generate path
start = (2.0, 3.0)
goal = (15.0, 12.0)
path = planner.plan(start, goal)

# Use with existing navigator
navigator = RouteNavigator(path)
next_waypoint = navigator.current_waypoint()
print(f"Heading to: ({next_waypoint[0]:.2f}, {next_waypoint[1]:.2f})")
```

---

## 3. POI-Based Waypoint Routing

### Annotating Maps with POIs

Edit your SVG map to include POI markers:

```xml
<!-- simple_corridor.svg -->
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
  <!-- Obstacles -->
  <rect class="obstacle" x="5" y="3" width="3" height="2" />
  
  <!-- POIs (point-of-interest waypoints) -->
  <circle class="poi" id="poi_corridor_mid" cx="10" cy="5" r="0.3" 
          label="corridor_midpoint" />
  <circle class="poi" id="poi_junction" cx="15" cy="7" r="0.3" 
          label="junction_A" />
</svg>
```

### Planning via POIs

```python
from robot_sf.planner import GlobalPlanner
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.common.types import Vec2D

# Load map with POIs
map_def = convert_map("maps/svg_maps/simple_corridor.svg")
print(f"Loaded {len(map_def.poi_positions)} POIs")

planner = GlobalPlanner(map_def)

# Plan path through specific POIs
path = planner.plan(
    start=Vec2D(1.0, 2.0),
    goal=Vec2D(18.0, 8.0),
    via_pois=["poi_corridor_mid", "poi_junction"]
)

print(f"Path visits {len(path)} waypoints (including 2 POIs)")
```

### Random POI Sampling

```python
from robot_sf.planner import POISampler

# Sample random intermediate waypoints
sampler = POISampler(map_def, seed=42)
random_pois = sampler.sample(count=3, strategy="random")

print(f"Sampled {len(random_pois)} random POIs:")
for poi in random_pois:
    print(f"  ({poi.x:.2f}, {poi.y:.2f})")
```

---

## 4. Multi-Goal Planning

### Visiting Multiple Targets

```python
from robot_sf.planner import GlobalPlanner
from robot_sf.common.types import Vec2D

planner = GlobalPlanner(map_def)

# Define multiple goals
goals = [
    Vec2D(10.0, 5.0),  # Goal 1
    Vec2D(15.0, 8.0),  # Goal 2
    Vec2D(8.0, 12.0),  # Goal 3
]

# Plan optimized tour (nearest-neighbor heuristic)
path = planner.plan_multi_goal(
    start=Vec2D(2.0, 3.0),
    goals=goals,
    optimize_order=True
)

print(f"Tour visits {len(goals)} goals in {len(path)} waypoints")
```

### Forced Goal Ordering

```python
# Visit goals in exact order (no optimization)
path = planner.plan_multi_goal(
    start=Vec2D(2.0, 3.0),
    goals=goals,
    optimize_order=False
)

# Verify order: path visits goals[0] before goals[1] before goals[2]
```

---

## 5. Error Handling

### Handling Planning Failures

```python
from robot_sf.planner import GlobalPlanner, PlanningFailedError

planner = GlobalPlanner(map_def)

try:
    # Attempt to plan (may fail if goal unreachable)
    path = planner.plan(start, goal)
except PlanningFailedError as e:
    print(f"Planning failed: {e.reason}")
    print(f"  Start: ({e.start.x}, {e.start.y})")
    print(f"  Goal: ({e.goal.x}, {e.goal.y})")
    # Handle failure (e.g., sample new goal)
```

### Fallback Behavior

```python
from robot_sf.planner import PlannerConfig

# Enable automatic fallback to straight line
config = PlannerConfig(fallback_on_failure=True)
planner = GlobalPlanner(map_def, config)

# Now plan() never raises PlanningFailedError
path = planner.plan(start, goal)  # Returns [start, goal] if no path found
```

---

## 6. Performance Tuning

### Caching for Repeated Queries

```python
from robot_sf.planner import PlannerConfig

# Enable caching (default behavior)
config = PlannerConfig(cache_graphs=True)
planner = GlobalPlanner(map_def, config)

# First call: ~400ms (builds graph)
import time
start_time = time.perf_counter()
path1 = planner.plan(Vec2D(1, 2), Vec2D(18, 8))
print(f"Cold cache: {(time.perf_counter() - start_time) * 1000:.1f}ms")

# Second call: ~50ms (uses cached graph)
start_time = time.perf_counter()
path2 = planner.plan(Vec2D(2, 3), Vec2D(17, 7))
print(f"Warm cache: {(time.perf_counter() - start_time) * 1000:.1f}ms")
```

### Adjusting Smoothing

```python
# More aggressive smoothing (fewer waypoints, slightly longer paths)
config = PlannerConfig(
    enable_smoothing=True,
    smoothing_epsilon=0.3  # Default: 0.1
)
planner = GlobalPlanner(map_def, config)

path = planner.plan(start, goal)
print(f"Smoothed path: {len(path)} waypoints")  # ~30% fewer than unsmoothed
```

### Clearance Trade-offs

```python
# Tighter clearance (allows narrow passages)
config = PlannerConfig(
    robot_radius=0.4,
    min_safe_clearance=0.1  # Reduced from 0.3
)
planner = GlobalPlanner(map_def, config)

# Safer clearance (avoids tight spaces)
config = PlannerConfig(
    robot_radius=0.4,
    min_safe_clearance=0.5  # Increased from 0.3
)
planner = GlobalPlanner(map_def, config)
```

---

## 7. Debugging & Visualization

### Export Path to JSON

```python
import json
from robot_sf.planner import GlobalPlanner

planner = GlobalPlanner(map_def)
path = planner.plan(start, goal)

# Serialize for debugging
path_json = json.dumps([{"x": p.x, "y": p.y} for p in path], indent=2)
print(path_json)
```

### Visualize in SVG

```python
# Generate SVG overlay for path
def path_to_svg(path: list[Vec2D]) -> str:
    points = " ".join(f"{p.x},{p.y}" for p in path)
    return f'<polyline points="{points}" stroke="blue" stroke-width="0.2" fill="none" />'

svg_overlay = path_to_svg(path)
print(svg_overlay)
# Copy into map SVG file for visualization
```

### Logging

```python
from loguru import logger

# Enable debug logging
logger.add("planner_debug.log", level="DEBUG")

# Planner emits diagnostic messages
planner = GlobalPlanner(map_def)
path = planner.plan(start, goal)
# Check planner_debug.log for:
#   - Graph build time
#   - Cache hits/misses
#   - Obstacle inflation details
#   - Smoothing statistics
```

---

## 8. Common Patterns

### Training Loop Integration

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

# Configure for training
config = RobotSimulationConfig()
config.use_planner = True
config.planner_clearance_margin = 0.4
config.map_pool = [
    "maps/svg_maps/corridor.svg",
    "maps/svg_maps/warehouse.svg",
    "maps/svg_maps/tight_spaces.svg",
]

env = make_robot_env(config=config)

# Training loop (planner generates new routes each reset)
for episode in range(100):
    obs, info = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
```

### Benchmark Scenario Generation

```python
from robot_sf.planner import GlobalPlanner, POISampler
from robot_sf.nav.svg_map_parser import convert_map
import numpy as np

map_def = convert_map("maps/svg_maps/benchmark_map.svg")
planner = GlobalPlanner(map_def)
sampler = POISampler(map_def, seed=42)

# Generate diverse test scenarios
scenarios = []
for i in range(50):
    # Random start/goal
    start = Vec2D(np.random.uniform(0, map_def.width), 
                  np.random.uniform(0, map_def.height))
    goal = Vec2D(np.random.uniform(0, map_def.width), 
                 np.random.uniform(0, map_def.height))
    
    # Optional: route via POIs
    via_pois = sampler.sample(count=2, strategy="random")
    
    path = planner.plan(start, goal)
    scenarios.append({"start": start, "goal": goal, "path": path})

print(f"Generated {len(scenarios)} test scenarios")
```

---

## 9. Migration from Manual Routes

### Before (Manual SVG Paths)

```xml
<!-- Old: Hand-crafted waypoints -->
<path class="route" d="M 2,3 L 10,5 L 18,8" />
```

```python
# Old code
map_def = convert_map("map.svg")
navigator = RouteNavigator(map_def.robot_routes[0])
```

### After (Auto-Generated Paths)

```xml
<!-- New: POI annotations only -->
<circle class="poi" id="poi_mid" cx="10" cy="5" r="0.3" />
```

```python
# New code
map_def = convert_map("map.svg")
planner = GlobalPlanner(map_def)
path = planner.plan(start, goal)
navigator = RouteNavigator(path)  # Same interface!
```

---

## 10. Next Steps

### Explore Advanced Features

- **Custom edge costs**: Weight graph edges by terrain difficulty (Phase 3)
- **Width-aware planning**: Consider robot width in narrow passages (Phase 3)
- **Dynamic replanning**: Handle moving obstacles (future work)

### Read Full Documentation

- **API Reference**: `specs/342-svg-global-planner/contracts/global_planner_api.md`
- **Design Decisions**: `specs/342-svg-global-planner/global-planner-v2.md`
- **Data Model**: `specs/342-svg-global-planner/data-model.md`

### Run Examples

```bash
# Example scripts (once implemented)
uv run python examples/advanced/global_planner_demo.py
uv run python examples/advanced/poi_routing_demo.py
```

### Ask for Help

- **GitHub Issues**: Report bugs or request features
- **Dev Guide**: `docs/dev_guide.md` for contribution guidelines
- **Constitution**: `.specify/memory/constitution.md` for design principles

---

## Troubleshooting

### "No module named 'pyvisgraph'"

**Solution**: Install missing dependency
```bash
uv add pyvisgraph>=0.2.1
```

### "PlanningFailedError: goal inside obstacle"

**Cause**: Goal position collides with inflated obstacle

**Solutions**:
1. Reduce `min_safe_clearance` if goal physically reachable
2. Enable fallback mode: `config.fallback_on_failure = True`
3. Adjust goal position to free space

### Slow first query (>1 second)

**Cause**: Visibility graph build dominates first call

**Solutions**:
1. Reduce obstacle count (simplify map)
2. Cache planner instance and reuse across episodes
3. Pre-build graphs during initialization phase

### "Path goes through obstacle" (collision detected)

**Cause**: Smoothing may introduce shortcuts through narrow gaps

**Solution**: Disable smoothing or reduce epsilon
```python
config = PlannerConfig(enable_smoothing=False)
# Or reduce tolerance
config = PlannerConfig(smoothing_epsilon=0.05)
```

---

## Summary

✓ **Installation**: Add `pyvisgraph` and `networkx` dependencies  
✓ **Basic Usage**: `GlobalPlanner(map_def).plan(start, goal)`  
✓ **Integration**: Set `config.use_planner = True` in gym environments  
✓ **POIs**: Annotate maps with `<circle class="poi">` for waypoints  
✓ **Performance**: Enable caching for <100ms repeated queries  
✓ **Error Handling**: Catch `PlanningFailedError` or enable fallback  

**Estimated time savings**: 70% reduction in map creation effort vs manual waypoints.

---

**Questions?** Open an issue or consult `specs/342-svg-global-planner/` documentation.
