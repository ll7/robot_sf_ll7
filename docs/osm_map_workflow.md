# OSM Map Workflow Guide (Phase 3, T042)

**Last Updated**: December 19, 2025  
**Document Status**: Complete reference guide for OSM-based map generation workflow

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Programmatic API](#programmatic-api)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)
7. [Examples](#examples)

---

## Overview

The OSM (OpenStreetMap) map extraction feature enables you to:

- **Extract driveable areas** from OpenStreetMap PBF files
- **Compute obstacle geometry** automatically
- **Render background maps** with affine transforms
- **Define zones and routes** via visual editor OR programmatic API
- **Serialize to YAML** for reproducibility and version control
- **Use with robot environments** for navigation training and evaluation

**Three-phase workflow**:

1. **Phase 1: Importer** - Load OSM PBF → extract driveable areas → compute obstacles
2. **Phase 2: Editor** - Visual zone/route editing → save to YAML
3. **Phase 3: Programmatic** - Define zones/routes in code → load YAML → use with environments

---

## Quick Start

### Option 1: Visual Editor (Phase 2)

```bash
# Run the interactive editor
uv run python examples/osm_map_editor_demo.py

# Output: YAML file with zones and routes
```

### Option 2: Programmatic API (Phase 3)

```python
from robot_sf.maps.osm_zones_config import (
    create_spawn_zone,
    create_goal_zone,
    create_route,
    create_config_with_zones_routes,
)
from robot_sf.maps.osm_zones_yaml import save_zones_yaml

# Create zones
spawn = create_spawn_zone(
    "spawn",
    polygon=[(0, 0), (10, 0), (10, 10)],
)
goal = create_goal_zone(
    "goal",
    polygon=[(90, 90), (100, 90), (100, 100)],
)

# Create route
route = create_route(
    "path",
    waypoints=[(0, 0), (100, 100)],
)

# Save to YAML
config = create_config_with_zones_routes(
    zones=[spawn, goal],
    routes=[route],
)
save_zones_yaml(config, "scenario.yaml")
```

### Option 3: Both Combined

```bash
# 1. Run editor to define zones visually
uv run python examples/osm_map_editor_demo.py

# 2. Load programmatically for validation/modification
python -c "
from robot_sf.maps.osm_zones_config import load_scenario_config
config = load_scenario_config('scenario.yaml')
print(f'Loaded {len(config.zones)} zones, {len(config.routes)} routes')
"
```

---

## Detailed Workflow

### Step 1: Acquire OpenStreetMap Data

**Download PBF file:**

```bash
# Option A: Using OSMnx (programmatic)
python -c "
import osmnx as ox
place = 'Oslo, Norway'
tags = {'building': True, 'highway': True}
gdf = ox.features_from_place(place, tags)
# Save bounds for next step
"

# Option B: Using Overpass API (web)
# Visit: https://overpass-turbo.eu/
# Query example:
[bbox:60.1,10.7,60.2,10.8];
(way[highway];node[amenity];);
out geom;

# Option C: Download from Geofabrik
# Visit: https://download.geofabrik.de/
# Select region → download .osm.pbf
```

**Store locally:**

```bash
mkdir -p maps/pbf
# Place downloaded file here: maps/pbf/region.pbf
```

### Step 2: Extract Driveable Areas

```python
from robot_sf.nav.osm_map_builder import osm_to_map_definition

# Load PBF and extract
map_def = osm_to_map_definition(
    pbf_file="maps/pbf/oslo.pbf",
    simplify_m=0.5,  # Simplify polygons to 0.5m
    buffer_m=2.0,     # Expand obstacles by 2m
)

print(f"Map bounds: {map_def.map_bounds}")
print(f"Obstacles: {len(map_def.obstacles)}")
print(f"Allowed areas: {len(map_def.allowed_areas)}")
```

### Step 3: Render Background Map

```python
from robot_sf.maps.osm_background_renderer import render_osm_background

# Render PNG with affine transform
output = render_osm_background(
    pbf_file="maps/pbf/oslo.pbf",
    output_dir="output/maps",
    dpi=100,
    simplify_m=0.5,
)

affine = output["affine_transform"]
print(f"PNG: {output['png_path']}")
print(f"Affine: {affine}")
```

### Step 4a: Define Zones Visually (Editor)

```bash
uv run python examples/osm_map_editor_demo.py
```

**In the editor:**

1. Click to create zone vertices
2. Press `Return` to close zone
3. Edit zone type (spawn/goal/crowded)
4. Add multiple zones
5. Define routes with waypoints
6. Save → YAML file created

### Step 4b: Define Zones Programmatically

```python
from robot_sf.maps.osm_zones_config import (
    create_spawn_zone,
    create_goal_zone,
    create_crowded_zone,
    create_route,
    create_config_with_zones_routes,
)

# Spawn zone
spawn = create_spawn_zone(
    "robot_start",
    polygon=[(10, 10), (20, 10), (20, 20), (10, 20)],
    priority=2,
)

# Goal zone
goal = create_goal_zone(
    "target_area",
    polygon=[(80, 80), (90, 80), (90, 90), (80, 90)],
)

# Crowded zone
crowd = create_crowded_zone(
    "busy_intersection",
    polygon=[(40, 40), (60, 40), (60, 60), (40, 60)],
    density=2.5,
)

# Navigation route
route = create_route(
    "main_corridor",
    waypoints=[(15, 15), (50, 50), (85, 85)],
    route_type="pedestrian",
)

# Create config
config = create_config_with_zones_routes(
    zones=[spawn, goal, crowd],
    routes=[route],
)
```

### Step 5: Save to YAML

```python
from robot_sf.maps.osm_zones_yaml import save_zones_yaml

save_zones_yaml(config, "scenarios/my_scenario.yaml")
```

**Generated YAML structure:**

```yaml
version: "1.0"
metadata:
  map_source: "oslo.pbf"
  description: "Downtown crossing scenario"
zones:
  robot_spawn:
    type: "spawn"
    polygon: [[10, 10], [20, 10], [20, 20]]
    priority: 2
  target_area:
    type: "goal"
    polygon: [[80, 80], [90, 80], [90, 90]]
  intersection:
    type: "crowded"
    polygon: [[40, 40], [60, 40], [60, 60]]
    metadata:
      density: 2.5
routes:
  corridor:
    name: "Main path"
    waypoints: [[15, 15], [50, 50], [85, 85]]
    route_type: "pedestrian"
```

### Step 6: Use with Robot Environment

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.maps.osm_zones_yaml import load_zones_yaml

# Load scenario YAML
zones_config = load_zones_yaml("scenarios/my_scenario.yaml")

# Create environment config
config = RobotSimulationConfig(
    map_pool=[...],  # Your OSM-derived map
    zones_config=zones_config,  # Add zones
)

# Create and use environment
env = make_robot_env(config=config)
obs, info = env.reset()

# Training loop
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

## Programmatic API

### Zone Creation Functions

#### `create_spawn_zone(name, polygon, priority=1, metadata=None) → Zone`

Create a robot spawn/initialization zone.

```python
spawn = create_spawn_zone(
    name="spawn_north",
    polygon=[(0, 0), (10, 0), (10, 10)],  # List of (x, y) tuples
    priority=2,  # Higher priority = preferred
    metadata={"description": "Northern start area"},
)
```

**Parameters:**
- `name` (str): Unique zone identifier
- `polygon` (list[tuple[float, float]]): Vertices in world coordinates
- `priority` (int): Selection priority (default 1)
- `metadata` (dict): Custom metadata

**Raises:**
- `ValueError`: If polygon has <3 points or is degenerate
- `TypeError`: If points aren't numeric

#### `create_goal_zone(name, polygon, metadata=None) → Zone`

Create a navigation goal/target zone.

```python
goal = create_goal_zone(
    name="goal_exit",
    polygon=[(90, 90), (100, 90), (100, 100)],
    metadata={"reward": 1.0},
)
```

#### `create_crowded_zone(name, polygon, density, metadata=None) → Zone`

Create a zone with pedestrian crowd annotation.

```python
crowd = create_crowded_zone(
    name="intersection",
    polygon=[(40, 40), (60, 40), (60, 60), (40, 60)],
    density=2.5,  # persons per m²
    metadata={"peak_hours": "8-18"},
)
```

**Density reference:**
- 0.1-0.5: Sparse (park-like)
- 0.5-2.0: Normal (urban street)
- 2.0-4.0: Dense (busy intersection)
- >4.0: Very dense (crowded shopping area)

#### `create_route(name, waypoints, route_type='pedestrian', metadata=None) → Route`

Create a navigation route with waypoints.

```python
route = create_route(
    name="main_path",
    waypoints=[(0, 0), (25, 25), (50, 50), (100, 100)],
    route_type="pedestrian",  # or 'wheelchair', 'vehicle'
    metadata={"speed": 1.2},
)
```

### Config Management

#### `create_config_with_zones_routes(zones=None, routes=None, version="1.0", metadata=None) → OSMZonesConfig`

Create a configuration container with zones and routes.

```python
config = create_config_with_zones_routes(
    zones=[spawn, goal, crowd],
    routes=[route],
    metadata={"map_source": "oslo.pbf"},
)
```

#### `load_scenario_config(yaml_file: str) → OSMZonesConfig`

Load a previously saved scenario from YAML.

```python
config = load_scenario_config("scenarios/crossing.yaml")

# Inspect loaded config
print(f"Zones: {list(config.zones.keys())}")
print(f"Routes: {list(config.routes.keys())}")
```

---

## Troubleshooting

### Issue: "Polygon points appear to be collinear"

**Cause**: Three or more points are on the same line.

**Fix**: Ensure zone corners form a proper polygon:

```python
# ❌ Wrong (collinear)
polygon = [(0, 0), (5, 5), (10, 10)]

# ✅ Correct (triangle)
polygon = [(0, 0), (10, 0), (5, 10)]
```

### Issue: "Polygon must have ≥3 points"

**Cause**: Zone has fewer than 3 vertices.

**Fix**: Add at least one more point:

```python
# ❌ Wrong (line, not polygon)
polygon = [(0, 0), (10, 10)]

# ✅ Correct (triangle)
polygon = [(0, 0), (10, 10), (5, 0)]
```

### Issue: "Density must be >0"

**Cause**: Crowded zone density is zero or negative.

**Fix**: Use positive density values:

```python
# ❌ Wrong
create_crowded_zone("crowd", polygon=..., density=0)

# ✅ Correct
create_crowded_zone("crowd", polygon=..., density=1.5)
```

### Issue: YAML file not loading

**Cause**: File path incorrect or YAML malformed.

**Fix**: Check file exists and is valid YAML:

```bash
# Verify file exists
ls -la scenarios/my_scenario.yaml

# Validate YAML syntax
uv run python -c "
import yaml
with open('scenarios/my_scenario.yaml') as f:
    data = yaml.safe_load(f)
    print('✓ Valid YAML')
"
```

### Issue: Environment creation fails with zones

**Cause**: Zone definition incompatible with map.

**Fix**: Ensure zones fit within map bounds:

```python
# Check map bounds
print(f"Map bounds: {map_def.map_bounds}")

# Verify zones are within bounds
for name, zone in config.zones.items():
    for x, y in zone.polygon:
        assert 0 <= x <= 100 and 0 <= y <= 100, f"Zone outside bounds: {name}"
```

---

## FAQ

### Q: Can I mix visual editor and programmatic zones?

**A**: Yes! Load the YAML from the editor, then add more zones programmatically:

```python
config = load_scenario_config("editor_output.yaml")

# Add new zone
new_zone = create_spawn_zone("spawn2", polygon=...)
config.zones["spawn2"] = new_zone

# Save
save_zones_yaml(config, "combined.yaml")
```

### Q: How do I ensure reproducibility?

**A**: Always save to YAML with metadata:

```python
config = create_config_with_zones_routes(
    zones=zones,
    routes=routes,
    metadata={
        "map_source": "oslo_v2.pbf",
        "scenario_name": "downtown_crossing",
        "created_by": "my_script.py",
    },
)
save_zones_yaml(config, "scenarios/v1.yaml")
```

### Q: What coordinate system is used?

**A**: World coordinates in meters. Origin (0, 0) is at map bottom-left, with x increasing right and y increasing up.

### Q: Can zones overlap?

**A**: Yes. Overlapping zones are valid and handled by the environment factory. Overlaps typically define transition areas or shared regions.

### Q: How many zones/routes can I define?

**A**: Theoretically unlimited, but performance degrades with >100 zones. Typical scenarios use 5-20 zones.

### Q: Can I edit zones after loading?

**A**: Yes, fully mutable:

```python
config = load_scenario_config("scenario.yaml")

# Modify existing zone
config.zones["spawn"].priority = 3

# Add new zone
new_goal = create_goal_zone("goal2", polygon=...)
config.zones["goal2"] = new_goal

# Remove zone
del config.zones["old_zone"]

# Save
save_zones_yaml(config, "modified.yaml")
```

### Q: How do I validate my scenario?

**A**: Use round-trip serialization:

```python
from robot_sf.maps.osm_zones_yaml import load_zones_yaml, save_zones_yaml

# Save original
save_zones_yaml(config, "original.yaml")

# Load back
config_reloaded = load_zones_yaml("original.yaml")

# Save again
save_zones_yaml(config_reloaded, "reloaded.yaml")

# Compare (should be identical modulo timestamps)
assert len(config_reloaded.zones) == len(config.zones)
print("✓ Scenario is valid and reproducible")
```

---

## Examples

### Example 1: Simple Navigation Task

```python
from robot_sf.maps.osm_zones_config import (
    create_spawn_zone,
    create_goal_zone,
    create_route,
    create_config_with_zones_routes,
)
from robot_sf.maps.osm_zones_yaml import save_zones_yaml

# Define simple A→B navigation
spawn = create_spawn_zone("start", [(0, 0), (10, 0), (10, 10)])
goal = create_goal_zone("end", [(90, 90), (100, 90), (100, 100)])
route = create_route("direct", [(5, 5), (100, 100)])

config = create_config_with_zones_routes(
    zones=[spawn, goal],
    routes=[route],
)

save_zones_yaml(config, "simple_navigation.yaml")
```

### Example 2: Multi-Agent Crossing

```python
# Multiple spawn and goal zones for diverse trajectories
zones = [
    create_spawn_zone("s_north", [(45, 0), (55, 0), (55, 10)]),
    create_spawn_zone("s_south", [(45, 90), (55, 90), (55, 100)]),
    create_goal_zone("g_north", [(45, 90), (55, 100)]),
    create_goal_zone("g_south", [(45, 0), (55, 10)]),
]

routes = [
    create_route("north_to_south", [(50, 5), (50, 95)]),
    create_route("south_to_north", [(50, 95), (50, 5)]),
]

config = create_config_with_zones_routes(zones=zones, routes=routes)
save_zones_yaml(config, "crossing.yaml")
```

### Example 3: Realistic Urban Scenario

See [examples/osm_programmatic_scenario.py](../examples/osm_programmatic_scenario.py) for complete reference with 3 realistic scenarios.

---

## See Also

- [OSM Map Editor Demo](../examples/osm_map_editor_demo.py) - Interactive visual editor
- [Programmatic Scenarios](../examples/osm_programmatic_scenario.py) - Code-first definition
- [OSM Zones YAML Module](../robot_sf/maps/osm_zones_yaml.py) - YAML I/O implementation
- [OSM Zones Config Module](../robot_sf/maps/osm_zones_config.py) - Programmatic API
- [SVG Map Editor Documentation](./SVG_MAP_EDITOR.md) - Visual editor guide

---

**Questions?** See [FAQ](#faq) or check example scripts.  
**Found a bug?** Please report on GitHub.  
**Want to contribute?** See CONTRIBUTING.md.

---

**Document Version**: 1.0  
**Last Updated**: December 19, 2025  
**Status**: Complete
