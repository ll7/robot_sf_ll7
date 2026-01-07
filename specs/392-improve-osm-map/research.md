# Research & Analysis: OSM-Based Map Extraction

**Phase**: 0 (Research & Clarification)  
**Date**: 2025-12-19  
**Status**: Complete  
**Research Artifacts**: See `/research/` folder (5 AI-generated proposals)

---

## Problem Statement

Currently, map creation from OpenStreetMap uses a lossy, manual workflow:
1. Export OSM to SVG (colors represent categories)
2. Manually edit in Inkscape (split shapes, fix colors, adjust for scale)
3. Parse via `svg_map_parser.py` (color-based classification)

**Limitations**:
- Information loss: OSM semantic tags (highway type, access restrictions, sidewalk presence) are stripped during export.
- Scale imprecision: Scale factors inferred from filenames (e.g., `_1350`); no meter-based ground truth.
- Polygon validity: Self-intersecting SVG paths cause parsing failures.
- Driveable area implicit: Free space defined as "obstacles' complement" rather than semantic walkable union.
- High friction: Requires Inkscape round-trip for any map modification.

---

## Proposed Solution: Semantic OSM PBF Pipeline

**Core insight**: Keep PBF as ground truth; derive `MapDefinition` programmatically without Inkscape.

### Architecture Overview

```
Local PBF File (ground truth)
    ↓
[1] Parse & filter by semantic tags
    (highway=footway/path/cycleway, exclude steps, etc.)
    ↓
[2] Project to UTM (meter-based)
    ↓
[3] Buffer lines to driveable polygons (default 3m width)
    ↓
[4] Compute obstacles as bounds complement
    (removes guesswork, ensures no gaps)
    ↓
[5] Render visual background (PNG)
    + store affine transform (pixel↔world mapping)
    ↓
[6a] Launch visual editor (optional)
    → Draw spawn/goal/crowded zones
    → Save YAML (world coordinates, deterministic)
    ↓
[6b] Programmatic annotation (alternative)
    → Code-based zone definition
    ↓
[7] Produce MapDefinition
    (with optional `allowed_areas` field for explicit bounds)
```

---

## Technical Decisions: Consensus Across Research

All five AI-generated proposals converge on similar recommendations:

### 1. **Data Source: Local PBF Files**

**Decision**: Use local PBF files as ground truth (not live Overpass API).

**Rationale**:
- ✅ Reproducible: No time-dependent API changes
- ✅ Offline: No rate limits, no network failures
- ✅ Consistent scale: Meter-based projections eliminate guesswork
- ✅ Version control: Can be committed/archived for long-term reproducibility

**Source**: https://extract.bbbike.org/ (curated, easy-to-download OSM subsets)

### 2. **Semantic Tag Filtering**

**Decision**: Filter ways/areas by explicit OSM highway tags.

**Tag Classification**:

**Driveable (walkable) highways**:
- `highway=footway` (dedicated pedestrian path)
- `highway=path` (generic path, often walkable)
- `highway=cycleway` (bike path, inclusive for pedestrians)
- `highway=bridleway` (horse path, pedestrian-accessible)
- `highway=pedestrian` (pedestrian street/zone)
- `highway=residential` (if area=yes: pedestrian square/plaza)
- `highway=unclassified` (minor road, some accessible as shared space)
- `highway=service` (if area=yes: service plaza)

**Excluded (non-driveable)**:
- `highway=steps` (stairs, impassable)
- `highway=motorway`, `trunk`, `primary`, `secondary` (vehicle-only unless sidewalk present)
- Features with `access=private` or `access=no` (explicit prohibition)

**Obstacles** (barriers):
- `building=*` (all building types)
- `natural=water` (ponds, lakes; optional in MVP)
- `waterway=river` (major water features; optional)
- `natural=cliff` (cliffs, unpassable)

**Reference**: 
- [OpenStreetMap Highways Wiki](https://wiki.openstreetmap.org/wiki/Key:highway)
- [OSMnx tag filters](https://osmnx.readthedocs.io/en/stable/#tag-filters)

### 3. **Geometry Pipeline: Buffering & Projection**

**Decision**: Buffer lines to polygons in UTM (meter-based) coordinate system.

**Workflow**:
1. Parse PBF in lat/lon (EPSG:4326, geographic)
2. Reproject to local UTM zone (e.g., EPSG:32632 for UTM zone 32N)
3. Buffer lines by half-width `r = width / 2` (default width 3m → r=1.5m)
4. Union buffered lines + area polygons → walkable union
5. Reproject back to lat/lon for storage in MapDefinition (local frame: meters from origin)

**Why UTM**:
- Meter-based coordinates (not degrees)
- Minimal distortion for local regions (<100 km²)
- Automatic zone selection based on region center

**Buffering parameters**:
- Default line width: 3m (reasonable pedestrian path width)
- Respects OSM `width` tag if present
- Cap style: round (smooth corners, no sharp artifacts)
- Join style: round (smooth intersections)

**Reference**: 
- Shapely BufferCapStyle documentation
- PyProj for projection transforms

### 4. **Obstacle Derivation: Complement Strategy**

**Decision**: Derive obstacles as the geometric complement of walkable union within map bounds.

**Algorithm**:
```
bounds = bbox(all_features)
walkable_union = union(buffered_lines, area_polygons)
obstacles = bounds_polygon - walkable_union

// Add explicit obstacles
obstacles.union(buildings, water, cliffs)
```

**Advantages**:
- ✅ Deterministic: No manual classification needed
- ✅ Complete: No accidental gaps (everything outside walkable is obstacle)
- ✅ Backward-compatible: Works with existing Obstacle class

**Polygon cleanup**:
- Use `buffer(0)` to repair self-intersections and invalid topologies
- Simplify with `simplify(tolerance=0.1)` to reduce vertex count
- Skip empty or degenerate (<0.1 m²) polygons

### 5. **Visual Context: Background Rendering**

**Decision**: Render PBF as PNG/SVG background with optional affine transform.

**Output**:
- **PNG**: Raster background for visual verification (fast overlay)
- **Metadata**: Affine transform (pixel ↔ world coordinate mapping)
  - Used by visual editor to map mouse clicks back to world coordinates
  - Format: `{"pixel_origin": [x, y], "pixel_per_meter": scale}`

**Rendering approach**:
- Matplotlib with basemap or Folium (lightweight, no external services)
- Layers: OSM streets (light gray), buildings (darker gray), water (light blue)
- Scale: Configurable; default resolution 2 pixels/meter

**Alternative (future)**: Render using Mapnik/Tilelive for higher fidelity, but Matplotlib MVP sufficient.

### 6. **Deterministic Zone & Route Serialization**

**Decision**: Store zones/routes in YAML (world coordinates, deterministic ordering).

**Why YAML** (not SVG or JSON):
- Human-readable, git-diff friendly
- Preserves ordering (lists, dicts with ordered keys)
- Supports comments (important for zone purposes)
- Minimal: Only essential vertices and metadata

**Schema** (v1.0):
```yaml
# Zone and route annotations for map
version: "1.0"
metadata:
  source_pbf: "uni_campus.pbf"
  map_bounds: [48.3, 11.9, 48.32, 11.92]  # (minlat, minlon, maxlat, maxlon)
  created_at: "2025-12-19T14:23:45Z"
  
zones:
  spawn_a:
    type: "spawn"
    polygon: [[100.0, 200.0], [110.0, 200.0], [110.0, 210.0], [100.0, 210.0]]  # meters
    priority: 1
    
  goal_zone:
    type: "goal"
    polygon: [[500.0, 600.0], [520.0, 600.0], [520.0, 620.0], [500.0, 620.0]]
    
  crowded_area:
    type: "crowded"
    polygon: [[300.0, 400.0], [350.0, 400.0], [350.0, 450.0], [300.0, 450.0]]
    density: 0.5  # pedestrian density hint

routes:
  main_route:
    type: "pedestrian_path"
    waypoints: [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0], [500.0, 600.0]]
```

**Determinism guarantees**:
- Fixed numeric precision: 3 decimal places for meters (≈1mm accuracy)
- Sorted zone/route keys (alphabetical)
- No floating-point accumulation errors (round before write)

### 7. **Visual Editor: Lightweight Matplotlib Integration**

**Decision**: Render background + launch simple click-based editor.

**Features**:
- Display PNG background with grid overlay
- Click to add polygon vertices (spawns, goals, crowded zones)
- Click to draw polylines (routes)
- Vertex-level editing: move/add/delete points
- Undo/redo stack
- Snapping: Optionally snap vertices to nearest driveable area edge (tolerance: 0.5m)
- Validation: Warn if polygon crosses obstacle or extends outside bounds

**Key UX decisions**:
- Affine transform maps pixel coordinates back to world coordinates
- Real-time visual feedback (vertex markers, polyline preview)
- Save to YAML (stores in world coordinates)
- No intermediate file writes; save only on user request

**Implementation**: Matplotlib event loop (mouse click handlers) + lightweight state machine

---

## Data Model: Hybrid `allowed_areas` Field

**Decision**: Add optional `allowed_areas: list[Polygon] | None` field to `MapDefinition`.

**Rationale** (Option C from spec clarification):
- Explicit driveable bounds populated only by OSM importer
- Legacy workflows (`allowed_areas = None`) remain unchanged
- Zero breakage to pygame visualization, sensor suite, planners
- Backward-compatible evolution path

**Implementation**:
```python
@dataclass
class MapDefinition:
    bounds: Polygon
    obstacles: list[Obstacle]
    spawn_zones: list[SpawnZone] = field(default_factory=list)
    goal_zones: list[GoalZone] = field(default_factory=list)
    crowded_zones: list[CrowdedZone] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    
    # NEW FIELD:
    allowed_areas: list[Polygon] | None = None  # Populated by OSM importer
```

**Usage**:
- Planners can opt-in: `if map_def.allowed_areas: use_for_snapping(map_def.allowed_areas)`
- Editor validation: Check zones against `allowed_areas` if present
- Sensor suite: Unchanged (uses obstacles as before)

---

## Technology Stack: Recommended Libraries

| Component | Library | Version | Why |
|-----------|---------|---------|-----|
| PBF parsing | PyOsmium or PyRosm | latest | Fast, idiomatic OSM access |
| Semantic queries | OSMnx | ~1.9+ | High-level tag filtering, GeoDataFrame integration |
| Geometry | Shapely | ~2.0+ | Buffering, union, difference, repair |
| Spatial ops | GeoPandas | ~0.14+ | Vectorized filtering, CRS transforms |
| Projection | PyProj | ~3.6+ | UTM conversion, coordinate systems |
| Visualization | Matplotlib | ~3.8+ | Background rendering, editor overlay |
| Data serialization | PyYAML | ~6.0+ | Deterministic, human-readable zone storage |
| Validation | Pydantic | ~2.0+ | Config/schema validation (optional, may reuse existing) |

**New dependencies to add to `pyproject.toml`**:
```toml
osmnx = "^1.9"
pyosmium = "^3.7"
geopandas = "^0.14"
shapely = "^2.0"
pyproj = "^3.6"
pyyaml = "^6.0"
```

**Note**: All are well-maintained, production-ready libraries with active communities.

---

## MVP Scope & Phases

### Phase 1 (Weeks 1–2): Core Importer & Rendering
**Goals**: 
- Parse local PBF → filter by tags → buffer → compute obstacles
- Populate MapDefinition with `allowed_areas`
- Render PNG background + affine transform metadata

**Deliverables**:
- `robot_sf/nav/osm_map_builder.py` (core importer)
- `robot_sf/maps/osm_background_renderer.py` (background generation)
- Example: `examples/osm_map_quickstart.py` (demo)
- Tests: Smoke test (PBF → MapDefinition), bounds validation, polygon quality
- Backward-compat check: pygame viz, sensor suite still work

### Phase 2 (Weeks 3–4): Visual Editor
**Goals**:
- Lightweight matplotlib editor over background PNG
- Draw/edit spawn/goal/crowded zones and routes
- Save to YAML in world coordinates

**Deliverables**:
- `robot_sf/maps/osm_zones_editor.py` (editor UI)
- `robot_sf/maps/osm_zones_yaml.py` (YAML schema, load/save)
- Tests: YAML round-trip determinism, coordinate transforms, bounds validation

### Phase 3 (Weeks 5–6): Programmatic Configuration
**Goals**:
- Code-based zone and route definition (alternative to visual editor)
- Config-driven map generation (YAML scenario files)

**Deliverables**:
- Programmatic zone creation API
- Scenario config parser
- Tests: Config → MapDefinition equivalence

### Phase 4 (Weeks 7–8): Documentation & Cleanup
**Goals**:
- User guide: PBF acquisition, import workflow, editor usage
- Integration guide: Using OSM-derived maps in training/evaluation
- Examples: End-to-end demos

**Deliverables**:
- `docs/osm_map_workflow.md`
- Updated `docs/SVG_MAP_EDITOR.md` (new section: OSM pipeline)
- Archived legacy SVG docs or cross-references

---

## Clarifications Resolved

### Q1: Should driveable areas be explicit (stored) or implicit (computed)?
**A**: Option C (Hybrid). Add optional `allowed_areas` field; populated by OSM importer only; backward-compatible.

### Q2: Which OSM tags should be included in the MVP?
**A**: Consensus from research:
- **Driveable**: footway, path, cycleway, bridleway, pedestrian, (residential/service if area=yes)
- **Obstacles**: building, water, cliff
- **Exclusions**: steps, motorway, access=private/no

### Q3: Should we use pyosmium, osmnx, or pyrosm?
**A**: **Recommendation**: Use OSMnx for MVP (higher-level, easier filtering). Optionally switch to PyOsmium for performance if needed later.

### Q4: What projection should we use?
**A**: Local UTM zone based on region center. Automatic via PyProj.

### Q5: How do we ensure deterministic polygon simplification?
**A**: Use fixed tolerance (0.1m), fixed random seed if randomization needed, store precision as metadata.

### Q6: Should zones be stored in pixels or world coordinates?
**A**: **World coordinates (meters)**. Ensures reproducibility across machines/sessions, independent of rendering resolution.

### Q7: What about existing SVG workflows?
**A**: Keep `svg_map_parser.py` unchanged. OSM importer produces `MapDefinition` directly. Users can coexist; no forced migration.

---

## Edge Cases & Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Large PBF performance (>100MB) | Medium | Use bounding box to query subset; lazy load; profile early |
| OSM data inconsistency (e.g., missing tags) | Medium | Document tag assumptions; provide fallback tag sets; validate output visually |
| Polygon validity issues (self-intersections) | Medium | Use Shapely `buffer(0)` cleanup; test on real data early; archive problematic PBF samples |
| Affine transform precision loss | Low | Store as floats; document numerical tolerance; test round-trip coordinate transforms |
| Zone drawing mistakes (crosses obstacles) | Low | Real-time validation warnings in editor; undo/redo support |
| Map bounds edge cases (island regions, disconnected areas) | Low | Use convex hull or bounding box; document assumptions |

---

## Success Criteria (Measurable)

1. ✅ OSM PBF import produces valid MapDefinition in <2 seconds (for 10km² regions)
2. ✅ Polygon quality: No self-intersections; all valid per Shapely validation
3. ✅ Scaling accuracy: Buffered line width matches expected meters (±0.1m tolerance)
4. ✅ Obstacle completeness: No gaps between obstacles and bounds
5. ✅ Visual rendering: PNG background aligns with computed obstacles (visual spot-check)
6. ✅ Affine transform: Round-trip pixel↔world coordinate transforms accurate (±1 pixel, ±0.1m)
7. ✅ YAML determinism: Re-running save→load yields byte-identical output
8. ✅ Zone coverage: Spawn/goal/crowded zones reproducible in editor across runs
9. ✅ Backward compatibility: Existing pygame visualizations, sensor suite, planners unaffected
10. ✅ Documentation: User guide complete; examples runnable end-to-end

---

## Conclusion

The research converges on a **semantic OSM PBF pipeline** as the optimal path forward:
- **Ground truth**: Local PBF files (reproducible, meter-based)
- **Processing**: Tag-based filtering, buffering, obstacle derivation (deterministic)
- **Output**: `MapDefinition` with optional explicit `allowed_areas` (backward-compatible)
- **Visual**: PNG background + lightweight editor (no Inkscape)
- **Serialization**: YAML zones/routes in world coordinates (git-friendly, deterministic)

This approach **eliminates** the lossy SVG export workflow while **preserving** backward compatibility. Ready for Phase 1 implementation design.

---

**Next Step**: Phase 1 design document (plan.md) detailing module architecture, APIs, workflows, and implementation tactics.
