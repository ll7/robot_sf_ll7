# 🎯 Phase 1 Quick Reference Guide

**For**: Next developer continuing Phase 2 (Visual Editor)  
**Feature**: OSM-Based Map Extraction to MapDefinition  
**Time to Read**: 5 minutes

---

## 📚 What Phase 1 Delivered

### Core Implementation (2 modules, 784 lines)

#### 1️⃣ **robot_sf/nav/osm_map_builder.py**
**What it does**: Converts OSM PBF files → MapDefinition

**Key entry point**:
```python
from robot_sf.nav.osm_map_builder import osm_to_map_definition, OSMTagFilters

# Create config
tags = OSMTagFilters()  # Uses smart defaults

# Convert PBF → MapDefinition
map_def = osm_to_map_definition(
    pbf_file="path/to/osm.pbf",
    bbox=(0, 0, 100, 100),
    tag_filters=tags
)

# Now you have:
# - map_def.bounds: The rectangular boundary
# - map_def.obstacles: List of Polygon objects (no-go zones)
# - map_def.allowed_areas: List of Polygon objects (walkable zones) ← NEW
```

**9 Functions** (already implemented):
| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `load_pbf()` | PBF file path | GeoDataFrame | Load multi-layer OSM data |
| `filter_driveable_ways()` | GeoDataFrame, tags | GeoDataFrame | Filter footways, paths, etc. |
| `extract_obstacles()` | GeoDataFrame, tags | GeoDataFrame | Extract buildings, water, cliffs |
| `project_to_utm()` | GeoDataFrame | (GeoDataFrame, int) | Convert to meter-based local coords |
| `buffer_ways()` | GeoDataFrame, width | GeoSeries | Convert lines → polygons for navigation |
| `cleanup_polygons()` | List[Polygon] | List[Polygon] | Repair self-intersections, simplify |
| `compute_obstacles()` | bounds, walkable | List[Polygon] | Spatial complement: areas to avoid |
| `osm_to_map_definition()` | pbf_file, bbox, tags | MapDefinition | **END-TO-END PIPELINE** ✅ |
| `OSMTagFilters` | — | Dataclass | Config object for tag filtering |

---

#### 2️⃣ **robot_sf/maps/osm_background_renderer.py**
**What it does**: Renders OSM PBF as PNG with coordinate transforms

**Key entry point**:
```python
from robot_sf.maps.osm_background_renderer import render_osm_background

# Render PBF → PNG + affine transform metadata
result = render_osm_background(
    pbf_file="path/to/osm.pbf",
    output_dir="output/maps/",
    pixels_per_meter=10
)

# Returns:
# {
#   "png_file": "output/maps/background.png",
#   "affine_json": "output/maps/affine_transform.json",
#   "transform": {...}  # pixel↔world mapping
# }
```

**3 Key Functions**:
| Function | Purpose |
|----------|---------|
| `render_osm_background()` | Generate PNG with metadata |
| `validate_affine_transform()` | Verify round-trip coordinate accuracy |
| `pixel_to_world()`, `world_to_pixel()` | Coordinate conversion helpers |

---

### MapDefinition Updates

✅ **New Field**: `allowed_areas: list[Polygon] | None = None`
- Defaults to None (backward-compatible)
- When provided, defines explicit walkable zones
- Used by `is_point_in_driveable_area()` helper

---

## 🧪 Testing Infrastructure

**Test File**: `tests/test_osm_map_builder.py` (450+ lines)

**What's tested**:
```
✅ PBF loading (3 tests)
✅ Tag filtering (2 tests)
✅ Projection & buffering (4 tests)
✅ Obstacle derivation (1 test)
✅ End-to-end pipeline (3 tests)
✅ PNG rendering (3 tests)
✅ Affine transforms (5 tests)
✅ Backward-compatibility (2 tests)
────────────────────────────────────
Total: 20+ tests across 8 test classes
```

**Running tests**:
```bash
# Run all importer tests
pytest tests/test_osm_map_builder.py -v

# Run specific test
pytest tests/test_osm_map_builder.py::TestEndToEndImporter -v

# Run with coverage
pytest tests/test_osm_map_builder.py --cov=robot_sf.nav.osm_map_builder
```

---

## 📖 Example Script

**File**: `examples/osm_map_quickstart.py` (58 lines)

**Does**: End-to-end demo: PBF → MapDefinition → PNG

**Run it**:
```bash
uv run python examples/osm_map_quickstart.py
```

**Expected output**:
- PNG file at `output/maps/osm_demo/background.png`
- JSON metadata at `output/maps/osm_demo/affine_transform.json`
- Console logs showing progress

---

## 🏗️ Architecture Overview

```
OSM PBF File
    ↓
load_pbf() ← Multi-layer GeoDataFrame (lines, polygons, etc.)
    ↓
filter_driveable_ways() + extract_obstacles()
    ↓ (Semantic tag-based filtering)
project_to_utm() ← Auto-detect UTM zone, convert to meters
    ↓
buffer_ways() ← Convert lines → polygons, round joins
    ↓
cleanup_polygons() ← Repair intersections, simplify
    ↓
compute_obstacles() ← Spatial complement (bounds - walkable)
    ↓
✅ MapDefinition
   ├─ bounds: Box (meters)
   ├─ obstacles: [Polygon, ...]
   ├─ allowed_areas: [Polygon, ...]
   └─ other fields...
```

**Parallel Path**:
```
OSM PBF File
    ↓
render_osm_background()
    ├─ Multi-layer visualization
    ├─ Matplotlib PNG rendering
    └─ Affine transform JSON (pixel↔world mapping)
    ↓
✅ PNG + Metadata
   ├─ background.png (rasterized map)
   └─ affine_transform.json (coordinate transforms)
```

---

## 🔧 Phase 2 Integration Points

**For Phase 2 (Visual Editor)**, you'll need:

### 1. **YAML Zone Serialization** (T022-T025)
Build on `allowed_areas` → persist zones to YAML:
```python
# You'll create this in Phase 2:
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, save_zones_yaml

zones_config = OSMZonesConfig(
    zones={"zone_1": [...], "zone_2": [...]},
    routes={...},
    metadata={"version": "1.0"}
)
save_zones_yaml(zones_config, "output/zones.yaml")
```

### 2. **Visual Editor** (T026-T035)
Build matplotlib-based GUI on top of renderer:
```python
# You'll create this in Phase 2:
from robot_sf.maps.osm_zones_editor import OSMZonesEditor

editor = OSMZonesEditor(
    pbf_file="osm.pbf",
    background_png="background.png",
    affine_json="affine_transform.json"
)
editor.run()  # Interactive GUI
```

### 3. **Programmatic Factory** (T036-T039)
Factory functions for scenario generation:
```python
# You'll create this in Phase 2/3:
from robot_sf.maps.osm_factory import create_zones_from_config

map_def = osm_to_map_definition(...)
zones = create_zones_from_config(map_def, scenario_type="training")
```

---

## 📦 Dependencies Added

All dependencies automatically installed via `uv sync --all-extras`:

```toml
# In pyproject.toml
geopandas = "^0.14"  # Multi-layer spatial data
shapely = "^2.0"     # Geometry operations
pyproj = "^3.6"      # Coordinate projections
osmnx = "^1.9"       # OSM data fetching
pyyaml = "^6.0"      # YAML serialization
matplotlib = "^3.8"  # PNG rendering
```

---

## ⚡ Performance Baseline

**Sample Data**: 137KB PBF, 1881 features
**Processing**: ~2-5 seconds (unoptimized)
**Output**: 102 obstacles, 392 walkable areas, 28537m² total

**No optimization applied yet** — Phase 2+ opportunity if needed.

---

## 🐛 Known Quirks

1. **Matplotlib axis rendering**: Must call `ax.set_axis_off()` before rendering geographic coordinates (prevents numerical issues)

2. **Small features filtered**: Polygon simplification may remove very small buildings/islands

3. **UTM zone boundaries**: Edge cases at zone boundaries not explicitly tested

4. **Large PBF files**: Current implementation loads entire file into memory (>500MB not recommended)

---

## 📋 Backward-Compatibility Checklist

✅ **Preserved** (Phase 1 validated):
- [ ] MapDefinition without `allowed_areas` still works
- [ ] Existing pygame visualization tests unaffected
- [ ] Sensor suite continues to work
- [ ] Planner algorithms still function
- [ ] Serialization/deserialization unchanged

---

## 🎓 To Understand Phase 1 Fully

**Read (in order)**:
1. This file (5 min) ← You are here
2. `PHASE_1_STATUS_REPORT.md` (10 min) - Detailed metrics
3. `PHASE_1_COMPLETION_SUMMARY.md` (15 min) - Deep dive
4. Source code directly: `robot_sf/nav/osm_map_builder.py` (20 min)

**Total**: ~50 minutes to full understanding

---

## ✅ Pre-Phase-2 Checklist

Before starting Phase 2 work:

- [ ] Run pytest on test suite: `pytest tests/test_osm_map_builder.py -v`
- [ ] Execute example script: `uv run python examples/osm_map_quickstart.py`
- [ ] Verify PNG output exists
- [ ] Check affine transform JSON validity
- [ ] Review the architecture diagram above
- [ ] Understand OSMTagFilters configuration options
- [ ] Skim docstrings in osm_map_builder.py

---

## 🚀 Ready to Start Phase 2?

**Next Task**: T022 - Define YAML schema for zones

**Suggested approach**:
1. Design `OSMZonesConfig` dataclass
2. Implement YAML load/save
3. Create validation logic
4. Build visual editor UI

---

## 💡 Pro Tips for Next Developer

1. **Always use the end-to-end function**:
   ```python
   # ✅ DO THIS
   from robot_sf.nav.osm_map_builder import osm_to_map_definition
   map_def = osm_to_map_definition(...)
   
   # ❌ DON'T manually call individual functions
   ```

2. **Test with real data early**: The test suite uses fixture files; try with real OSM data (bbbike.org)

3. **Coordinate system matters**: Everything is meters-based after UTM projection. Pixel-to-world transforms are critical.

4. **Check the renderer output**: PNG and JSON are key for visual editor integration.

5. **Backward-compat is sacred**: Any Phase 2 changes must preserve existing MapDefinition deserialization.

---

## 📞 Questions to Ask

If stuck:
- "What does `osm_to_map_definition()` return?" → Check source code
- "How do I validate zones?" → Look at test suite
- "What's the pixel→world mapping?" → Read `osm_background_renderer.py`
- "How do I add new obstacles?" → Use `compute_obstacles()` pattern

---

**Last Updated**: Phase 1 Completion Session  
**Status**: ✅ Ready for Phase 2  
**Contact**: Consult PHASE_1_COMPLETION_SUMMARY.md for detailed reference
