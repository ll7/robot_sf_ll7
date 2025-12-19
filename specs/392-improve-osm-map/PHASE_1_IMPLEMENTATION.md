# Phase 1 Implementation: Core Importer & Rendering

**Branch**: `392-Improve-osm-map-generation`  
**Phase**: 1 of 4  
**Duration**: Weeks 1–2  
**Tasks**: T001–T021 (21 total)  
**Status**: 🟢 **READY TO BEGIN**  
**Date Started**: 2025-12-19

---

## Quick Status Overview

| Subphase | Tasks | Status | Progress |
|----------|-------|--------|----------|
| 1a. Setup & Fixtures | T001–T004 | ⏳ Not started | 0/4 |
| 1b. Core Importer | T005–T013 | ⏳ Not started | 0/9 |
| 1c. MapDefinition | T014–T015 | ⏳ Not started | 0/2 |
| 1d. Background Rendering | T016–T017 | ⏳ Not started | 0/2 |
| 1e. Examples & Validation | T018–T021 | ⏳ Not started | 0/4 |
| **Phase 1 Total** | **T001–T021** | **⏳ Queued** | **0/21** |

---

## Critical Path (Sequential Dependencies)

```
T001 (fixture prep)
  ↓
T005 → T006 → T007,T008,T009,T010 → T011 → T012 → T013 (importer)
  ↓
T014,T015 (MapDefinition)
  ↓
T016,T017 (rendering)
  ↓
T019,T020,T021 (validation)
```

**Estimated sequential time**: ~8–10 days  
**With parallelization [P]**: ~5–7 days

---

## Subphase 1a: Project Setup & Fixture Preparation (4 tasks)

### T001: Create test PBF fixture

**Status**: ⏳ Ready to start  
**Acceptance Criteria**:
- [ ] PBF file loads without error
- [ ] Contains buildings, footways, and obstacles
- [ ] Size <1MB (single city block)
- [ ] Located at: `test_scenarios/osm_fixtures/sample_block.pbf`

**Implementation Notes**:
- Source: https://extract.bbbike.org/ or local OSM data
- Scope: Single urban block or campus quad
- Validation: `osmnx.graph_from_file(pbf_file)` succeeds

**Files Affected**: `test_scenarios/osm_fixtures/sample_block.pbf` (new)

---

### T002: Create expected output fixtures

**Status**: ⏳ Blocked by T001  
**Acceptance Criteria**:
- [ ] Reference PNG exists
- [ ] Metadata JSON with affine_transform field
- [ ] Both located in `test_scenarios/osm_fixtures/expected_outputs/`

**Implementation Notes**:
- Generated after T016–T017 complete
- Used for regression testing in Phase 2+

**Files Affected**: 
- `test_scenarios/osm_fixtures/expected_outputs/sample_block_bg.png` (new)
- `test_scenarios/osm_fixtures/expected_outputs/sample_block_meta.json` (new)

---

### T003: Add OSM dependencies to pyproject.toml

**Status**: ⏳ Parallelizable [P]  
**Acceptance Criteria**:
- [ ] `uv sync --all-extras` succeeds
- [ ] All imports available: `import osmnx, shapely, geopandas, pyproj, yaml`
- [ ] No version conflicts

**Dependencies to add**:
```toml
osmnx = "^1.9"
pyosmium = "^3.7"
geopandas = "^0.14"
shapely = "^2.0"
pyproj = "^3.6"
pyyaml = "^6.0"
```

**Files Affected**: `pyproject.toml`

**Command to execute**:
```bash
uv add osmnx pyosmium geopandas shapely pyproj pyyaml
```

---

### T004: Create module skeleton files

**Status**: ⏳ Parallelizable [P]  
**Acceptance Criteria**:
- [ ] Modules importable without error
- [ ] Contain docstrings and type hints
- [ ] Empty implementations with stub functions

**Files to create**:
1. `robot_sf/nav/osm_map_builder.py`
   ```python
   """OSM PBF to MapDefinition importer."""
   
   from dataclasses import dataclass, field
   from typing import Optional
   from robot_sf.nav.map_config import MapDefinition
   # ... imports
   
   @dataclass
   class OSMTagFilters:
       """Configuration for OSM tag filtering."""
       pass
   
   def osm_to_map_definition(...) -> MapDefinition:
       """Convert OSM PBF to MapDefinition."""
       pass
   ```

2. `robot_sf/maps/osm_background_renderer.py`
   ```python
   """Render OSM PBF to PNG background with affine transform."""
   
   from typing import Any, Dict
   # ... imports
   
   def render_osm_background(...) -> dict[str, Any]:
       """Render PBF as PNG background."""
       pass
   ```

**Files Affected**: 
- `robot_sf/nav/osm_map_builder.py` (new)
- `robot_sf/maps/osm_background_renderer.py` (new)

---

## Subphase 1b: Core Importer Implementation (9 tasks)

### T005: Implement OSMTagFilters dataclass

**Status**: ⏳ Blocked by T004  
**Acceptance Criteria**:
- [ ] Dataclass with default tag sets
- [ ] Driveable highways: footway, path, cycleway, etc.
- [ ] Obstacles: building, water, cliff
- [ ] Excluded: steps, motorway, private access

**Implementation Skeleton**:
```python
@dataclass
class OSMTagFilters:
    """Configuration for semantic OSM tag filtering."""
    
    driveable_highways: list[str] = field(
        default_factory=lambda: [
            "footway", "path", "cycleway", "bridleway", "pedestrian"
        ]
    )
    driveable_areas: list[str] = field(
        default_factory=lambda: [
            "pedestrian", "footway", "residential", "service"
        ]
    )
    obstacle_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("building", "*"),
            ("natural", "water"),
            ("natural", "cliff"),
        ]
    )
    excluded_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("highway", "steps"),
            ("access", "private"),
            ("access", "no"),
        ]
    )
```

**Test**: `tests/test_osm_map_builder.py::test_osm_tag_filters_defaults`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T006: Implement PBF loading via OSMnx

**Status**: ⏳ Blocked by T005  
**Acceptance Criteria**:
- [ ] Loads PBF fixture without error
- [ ] Returns GeoDataFrame with ways/areas
- [ ] Preserves OSM tags

**Implementation**:
```python
import osmnx as ox
import geopandas as gpd

def load_pbf(pbf_file: str, bbox: Optional[tuple] = None) -> gpd.GeoDataFrame:
    """Load OSM PBF file."""
    # Use OSMnx to read PBF
    gdf = ox.features_from_file(pbf_file, tags=...)
    return gdf
```

**Test**: `tests/test_osm_map_builder.py::test_pbf_loading`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T007: Implement tag filtering (driveable ways)

**Status**: ⏳ Parallelizable [P] with T008, T010, T011  
**Acceptance Criteria**:
- [ ] Returns only tagged highways
- [ ] Excludes steps, motorway, private access
- [ ] Result is valid GeoDataFrame

**Test**: `tests/test_osm_map_builder.py::test_tag_filtering_driveable`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T008: Implement obstacle extraction

**Status**: ⏳ Parallelizable [P] with T007, T010, T011  
**Acceptance Criteria**:
- [ ] Extracts buildings, water, cliffs
- [ ] Validates geometries
- [ ] Result is valid GeoDataFrame

**Test**: `tests/test_osm_map_builder.py::test_obstacle_extraction`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T009: Implement UTM projection

**Status**: ⏳ Blocked by T006  
**Acceptance Criteria**:
- [ ] Auto-detects UTM zone from centroid
- [ ] Projects to meter-based coordinates
- [ ] Round-trip error <0.1m

**Implementation**:
```python
import pyproj
from shapely.geometry import Point

def project_to_utm(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Project to local UTM zone."""
    # Get centroid
    centroid = Point(gdf.geometry.bounds.iloc[0][:2])
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone}"
    gdf_utm = gdf.to_crs(utm_crs)
    return gdf_utm, utm_zone
```

**Test**: `tests/test_osm_map_builder.py::test_utm_projection`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T010: Implement line buffering

**Status**: ⏳ Parallelizable [P] with T007, T008, T011  
**Acceptance Criteria**:
- [ ] Buffers lines to polygons
- [ ] Uses round cap and join styles
- [ ] Result is valid (no self-intersections)

**Implementation**:
```python
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

def buffer_ways(gdf: gpd.GeoDataFrame, half_width_m: float = 1.5) -> list[Polygon]:
    """Buffer line ways to polygons."""
    buffered = []
    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            buffered.append(geom.buffer(half_width_m, cap_style="round", join_style="round"))
    return buffered
```

**Test**: `tests/test_osm_map_builder.py::test_line_buffering`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T011: Implement polygon cleanup

**Status**: ⏳ Parallelizable [P] with T007, T008, T010  
**Acceptance Criteria**:
- [ ] Repairs self-intersections with buffer(0)
- [ ] Simplifies with 0.1m tolerance
- [ ] Skips invalid/empty polygons

**Implementation**:
```python
def cleanup_polygons(polys: list[Polygon]) -> list[Polygon]:
    """Clean up polygons: repair, simplify, validate."""
    cleaned = []
    for poly in polys:
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)  # Repair
        if poly.is_empty:
            continue
        poly = poly.simplify(0.1)  # Simplify
        if poly.area > 0.1:  # Skip small artifacts
            cleaned.append(poly)
    return cleaned
```

**Test**: `tests/test_osm_map_builder.py::test_polygon_cleanup`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T012: Implement obstacle derivation

**Status**: ⏳ Blocked by T011  
**Acceptance Criteria**:
- [ ] Computes bounds - walkable_union = obstacles
- [ ] Validates result (no gaps)
- [ ] Merges explicit obstacles

**Implementation**:
```python
from shapely.geometry import box
from shapely.ops import unary_union

def compute_obstacles(bounds_box: tuple, walkable_union: Polygon) -> list[Polygon]:
    """Compute obstacles as complement of walkable areas."""
    bounds_poly = box(*bounds_box)
    obstacles_union = bounds_poly.difference(walkable_union)
    if obstacles_union.is_empty:
        return []
    if obstacles_union.geom_type == "Polygon":
        return [obstacles_union]
    else:  # MultiPolygon
        return list(obstacles_union.geoms)
```

**Test**: `tests/test_osm_map_builder.py::test_obstacle_derivation`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

### T013: Implement core entry point

**Status**: ⏳ Blocked by T012  
**Acceptance Criteria**:
- [ ] End-to-end PBF → MapDefinition works
- [ ] MapDefinition has bounds, obstacles, allowed_areas
- [ ] Integration test passes

**Implementation**:
```python
def osm_to_map_definition(
    pbf_file: str,
    bbox: Optional[tuple] = None,
    line_buffer_m: float = 1.5,
    tag_filters: OSMTagFilters = None,
) -> MapDefinition:
    """Convert OSM PBF to MapDefinition."""
    if tag_filters is None:
        tag_filters = OSMTagFilters()
    
    # Load
    gdf = load_pbf(pbf_file, bbox)
    
    # Filter & extract
    driveable_ways = filter_driveable_ways(gdf, tag_filters)
    obstacles_gdf = extract_obstacles(gdf, tag_filters)
    
    # Project
    gdf_utm, utm_zone = project_to_utm(gdf)
    
    # Buffer & cleanup
    buffered = buffer_ways(driveable_ways, line_buffer_m / 2)
    buffered = cleanup_polygons(buffered)
    walkable_union = unary_union(buffered)
    
    # Compute obstacles
    bounds = gdf_utm.total_bounds
    obstacles_polys = compute_obstacles(bounds, walkable_union)
    obstacles_list = [Obstacle(vertices) for vertices in obstacles_polys]
    
    # Build MapDefinition
    return MapDefinition(
        bounds=box(*bounds),
        obstacles=obstacles_list,
        allowed_areas=buffered,  # Explicit walkable areas
    )
```

**Test**: `tests/test_osm_map_builder.py::test_osm_to_map_definition_integration`

**Files Affected**: `robot_sf/nav/osm_map_builder.py`

---

## Subphase 1c: MapDefinition & Backward-Compat Setup (2 tasks)

### T014: Modify MapDefinition dataclass

**Status**: ⏳ Parallelizable [P] with importer  
**Acceptance Criteria**:
- [ ] Field `allowed_areas: list[Polygon] | None = None` added
- [ ] Type checks pass
- [ ] Existing deserialization unchanged

**Implementation** (in `robot_sf/nav/map_config.py`):
```python
from dataclasses import dataclass, field
from shapely.geometry import Polygon
from typing import Optional

@dataclass
class MapDefinition:
    bounds: Polygon
    obstacles: list[Obstacle]
    spawn_zones: list[SpawnZone] = field(default_factory=list)
    goal_zones: list[GoalZone] = field(default_factory=list)
    crowded_zones: list[CrowdedZone] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    
    # NEW FIELD:
    allowed_areas: Optional[list[Polygon]] = None
```

**Test**: `tests/test_osm_map_builder.py::test_map_definition_backward_compat`

**Files Affected**: `robot_sf/nav/map_config.py`

---

### T015: Add helper method is_point_in_driveable_area

**Status**: ⏳ Blocked by T014  
**Acceptance Criteria**:
- [ ] Method uses allowed_areas if present
- [ ] Falls back to obstacle complement
- [ ] Works for both cases

**Implementation** (in `robot_sf/nav/map_config.py`):
```python
def is_point_in_driveable_area(self, point: tuple[float, float]) -> bool:
    """Check if point is in driveable area."""
    from shapely.geometry import Point
    p = Point(point)
    
    if self.allowed_areas is not None:
        # Use explicit walkable areas
        return any(poly.contains(p) for poly in self.allowed_areas)
    else:
        # Fallback: NOT in obstacle
        return not any(obs.contains_point(point) for obs in self.obstacles)
```

**Test**: `tests/test_osm_map_builder.py::test_is_point_in_driveable_area`

**Files Affected**: `robot_sf/nav/map_config.py`

---

## Subphase 1d: Background Rendering (2 tasks)

### T016: Implement render_osm_background

**Status**: ⏳ Blocked by T004  
**Acceptance Criteria**:
- [ ] PNG file created
- [ ] Metadata dict returned with affine_transform
- [ ] PNG shows driveable areas, obstacles, streets

**Implementation** (in `robot_sf/maps/osm_background_renderer.py`):
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any, Dict
import json
from pathlib import Path

def render_osm_background(
    pbf_file: str,
    output_dir: str = "output/maps/",
    pixels_per_meter: float = 2.0,
) -> dict[str, Any]:
    """Render OSM PBF to PNG background."""
    # Load PBF
    import osmnx as ox
    gdf = ox.features_from_file(pbf_file)
    
    # Get bounds
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    # Plot layers
    # ... (buildings, water, streets)
    
    # Compute affine transform
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]
    pixel_width = int(width_m * pixels_per_meter)
    pixel_height = int(height_m * pixels_per_meter)
    
    affine = {
        "pixel_origin": [0, 0],
        "pixel_per_meter": pixels_per_meter,
        "bounds_meters": [bounds[0], bounds[1], bounds[2], bounds[3]],
    }
    
    # Save PNG
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    png_path = f"{output_dir}/background.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    
    return {
        "png_path": png_path,
        "affine_transform": affine,
    }
```

**Test**: `tests/test_osm_background_renderer.py::test_render_osm_background`

**Files Affected**: `robot_sf/maps/osm_background_renderer.py`

---

### T017: Implement affine transform validation

**Status**: ⏳ Blocked by T016  
**Acceptance Criteria**:
- [ ] Round-trip pixel↔world transformation works
- [ ] Error <±1 pixel, ±0.1m
- [ ] Deterministic

**Implementation** (in `robot_sf/maps/osm_background_renderer.py`):
```python
def validate_affine_transform(transform: dict[str, Any]) -> bool:
    """Validate affine transform round-trip accuracy."""
    pixel_origin = transform["pixel_origin"]
    scale = transform["pixel_per_meter"]
    bounds = transform["bounds_meters"]
    
    # Test: pixel (0, 0) → world → pixel
    world_x = bounds[0] + (pixel_origin[0] / scale)
    world_y = bounds[1] + (pixel_origin[1] / scale)
    
    pixel_x_back = (world_x - bounds[0]) * scale
    pixel_y_back = (world_y - bounds[1]) * scale
    
    error = max(abs(pixel_x_back - pixel_origin[0]), abs(pixel_y_back - pixel_origin[1]))
    return error < 1.0  # ±1 pixel tolerance
```

**Test**: `tests/test_osm_background_renderer.py::test_affine_transform_round_trip`

**Files Affected**: `robot_sf/maps/osm_background_renderer.py`

---

## Subphase 1e: Examples & Validation (4 tasks)

### T018: Create example script

**Status**: ⏳ Blocked by T013, T016  
**Acceptance Criteria**:
- [ ] Script runs headless
- [ ] Produces MapDefinition + PNG
- [ ] No errors

**File**: `examples/osm_map_quickstart.py`

---

### T019–T021: Backward-compat validation (3 tasks)

**Status**: ⏳ Blocked by T013  

These three tasks validate that existing code still works:
- T019: pygame visualization unchanged
- T020: sensor suite unchanged
- T021: planners unchanged

---

## How to Track Progress

**Template for each task completion**:
```markdown
## Task T00X: [Name]

**Status**: ✅ COMPLETE

**Changes**:
- File: robot_sf/nav/osm_map_builder.py — Added `function_name()` with implementation
- File: tests/test_osm_map_builder.py — Added `test_function_name()` test

**Test Result**: ✅ PASS (1 passed in 0.23s)

**Notes**: [Any implementation notes or decisions]
```

---

## Implementation Best Practices

1. **Test-driven**: Write test first (red), then implementation (green)
2. **Small commits**: Commit after each task completion
3. **Lint before commit**: `uv run ruff check --fix .` + `uv run ruff format .`
4. **Type check**: `uvx ty check .`
5. **Parallel execution**: Assign [P] tasks to different developers where possible
6. **Documentation**: Every function gets a docstring; every test is commented

---

**Ready to begin Phase 1? Start with T001.**
