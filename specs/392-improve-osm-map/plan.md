# Implementation Plan: OSM-Based Map Extraction to MapDefinition

**Branch**: `392-Improve-osm-map-generation` | **Date**: 2025-12-19 | **Spec**: [spec.md](./spec.md) | **Research**: [research.md](./research.md)  
**Input**: Feature specification + 5 AI research proposals + dev_guide.md constitution

**Purpose**: Execute Phase 1–4 implementation plan for semantic OSM PBF pipeline with visual editor, replacing lossy SVG export workflow. Deliverables: 4 new modules, updated MapDefinition, end-to-end examples, comprehensive tests, backward-compat validation.

---

## Summary

**Problem**: Current map creation workflow (OSM → SVG export → Inkscape → parse) is lossy, scale-imprecise, and manual.

**Solution**: Semantic OSM PBF pipeline:
1. Parse local PBF files (ground truth, reproducible)
2. Filter ways/areas by semantic tags (highway=footway/path/cycleway, exclude steps)
3. Buffer lines to driveable polygons (default 3m width, UTM projection)
4. Compute obstacles as geometric complement within bounds (deterministic)
5. Render visual background (PNG + affine transform for world↔pixel mapping)
6. Visual editor for zones and routes (lightweight Matplotlib, save to YAML in world coordinates)
7. Produce MapDefinition with optional `allowed_areas` field (backward-compatible)

**Key architectural decision**: Add optional `allowed_areas: list[Polygon] | None` field to MapDefinition. Populated by OSM importer; `None` for existing workflows. Zero breakage to pygame, sensors, planners.

**Technology stack**: OSMnx, Shapely, GeoPandas, PyProj, Matplotlib, PyYAML.

---

## Technical Context

**Language/Version**: Python 3.11 (existing project standard)

**Primary Dependencies**:
- OSMnx (~1.9+): High-level OSM tag filtering and GeoDataFrame integration
- PyOsmium (~3.7+): Optional fast PBF parsing (fallback if OSMnx performance insufficient)
- Shapely (~2.0+): Geometry buffering, union, difference, cleanup
- GeoPandas (~0.14+): Vectorized spatial operations, CRS transforms
- PyProj (~3.6+): Projection transforms (WGS84 ↔ UTM)
- Matplotlib (~3.8+): Background PNG rendering, editor UI (click handlers)
- PyYAML (~6.0+): Deterministic zone/route serialization

**Storage**: 
- Input: Local PBF files (OSM extracts, e.g., from bbbike.org)
- Output: MapDefinition objects (in-memory), PNG backgrounds, YAML zone/route files (under `output/maps/`)

**Testing**: pytest (existing project standard)
- Unit tests: PBF parsing, tag filtering, buffering, polygon cleanup, coordinate transforms
- Integration tests: PBF → MapDefinition round-trip, backward-compat validation (pygame, sensors)
- Visual regression: Background rendering alignment with computed obstacles (manual spot-checks + automated bounds validation)
- Editor tests: YAML round-trip determinism, coordinate transforms, snapping logic

**Target Platform**: Linux (CI/CD), macOS (development), Docker (training)

**Project Type**: Single Python package (robot_sf) with new submodules

**Performance Goals**:
- PBF import: <2s for 10km² regions (soft target: <5s for 100km²)
- Background rendering: <1s for PNG output
- Editor responsiveness: <100ms click→vertex response

**Constraints**:
- Backward compatibility: Zero breaking changes to MapDefinition (optional field only)
- Determinism: YAML serialization must be bit-identical across runs
- Offline: No live API calls; local PBF only
- Projection accuracy: <0.1m error margin after UTM round-trip

**Scale/Scope**:
- MVP: Urban campus or district (10–100 km²)
- ~2000 lines new code (4 modules + helpers + tests)
- 4 implementation phases over 8 weeks

---

## Constitution Check

**Principles assessed** (Robot SF Constitution v1.4.0):

| Principle | Requirement | Status | Mitigation |
|-----------|-------------|--------|-----------|
| I. Reproducibility | All outputs derivable from versioned inputs (code + config + seeds) | ✅ PASS | PBF files versioned; buffering deterministic; YAML schema versioned |
| II. Factory Abstraction | Expose via factory functions; no direct class instantiation | ✅ PASS | New `make_osm_env()` factory (or `osm_to_map_definition()` function) |
| III. Benchmark First | Outputs support aggregation & metrics; schemas validated | ✅ PASS | MapDefinition schema stable; no breaking changes; `allowed_areas` optional |
| IV. Unified Config | All parameters in unified config layer | ✅ PASS | OSM importer config: tag filters, line_width, buffer_params (dataclass) |
| V. Minimal Baselines | New baseline required? | N/A | OSM import is data prep, not a baseline algorithm |
| VI. Metrics Transparency | Any new metrics documented with weights/provenance | N/A | No new metrics; existing MapDefinition metrics unchanged |
| VII. Backward Compat | Public contracts unchanged or explicitly versioned | ✅ PASS | MapDefinition optional field; svg_map_parser unchanged; zero breakage |
| VIII. Docs as API | Every new surface documented in central index | ✅ PLAN | Will add entry to docs/README.md; create osm_map_workflow.md |
| IX. Test Coverage | Public behavior changes require smoke + assertion tests | ✅ PLAN | PBF→MapDefinition smoke; polygon quality assertions; backward-compat tests |
| X. Scope Discipline | Stay within social navigation simulation focus | ✅ PASS | OSM import is map preparation; directly supports robot navigation eval |
| XI. Library Reuse | Reusable code in robot_sf/; examples orchestrate only | ✅ PLAN | Core logic in robot_sf/nav/, robot_sf/maps/; example calls factory |
| XII. Preferred Logging | Use Loguru for non-trivial runtime messages (no print) | ✅ PLAN | Import errors, buffer progress, polygon warnings → logger.info/warning |
| XIII. Test Value | Verify test significance before fix effort | ✅ PLAN | New tests explicitly document what they verify (contract, regression, edge case) |

**Gate Status**: ✅ PASS (no violations; all principles either satisfied or mitigated in design)

**Re-check required**: After Phase 1 design finalization (before implementation)

---

## Project Structure

### Documentation (this feature)

```text
specs/392-improve-osm-map/
├── spec.md              # Feature specification (5 user stories, 23 FRs, 14 SCs, 4 phases)
├── plan.md              # This file (implementation design)
├── research.md          # Phase 0 research consolidation (technology stack, tag filters, design decisions)
├── data-model.md        # Phase 1 output: entity definitions, MapDefinition changes, YAML schema
├── quickstart.md        # Phase 1 output: user guide, demo workflow
├── contracts/
│   ├── map-definition-api.md    # MapDefinition API contract (fields, types, invariants)
│   └── osm-zones-schema.json    # YAML zone/route schema JSON
├── tasks.md             # Phase 2 output: atomic, testable tasks (JIRA-style)
└── research/            # Supporting AI-generated proposals (5 documents)
```

### Source Code (repository root)

**Structure Decision**: Single Python package (Option 1) — extend existing `robot_sf/` with new submodules.

```text
robot_sf/
├── nav/
│   ├── map_config.py          # MODIFIED: Add optional allowed_areas field to MapDefinition
│   ├── obstacle.py            # Unchanged; used by OSM importer
│   ├── svg_map_parser.py       # Unchanged; kept for backward compat
│   └── osm_map_builder.py      # NEW: Core PBF importer (parse, filter, buffer, compute obstacles)
│
├── maps/
│   ├── __init__.py
│   ├── osm_background_renderer.py   # NEW: Render PBF → PNG + affine transform
│   ├── osm_zones_editor.py          # NEW: Visual editor (Matplotlib click handlers)
│   ├── osm_zones_yaml.py            # NEW: YAML schema, load/save (deterministic)
│   └── map_visualizer.py            # Existing (may extend for OSM background overlay)
│
└── common/
    ├── types.py          # Existing (may add OSMCoord type if needed)
    └── errors.py         # Existing (reuse for OSM import errors)

examples/
├── osm_map_quickstart.py         # NEW: End-to-end demo (PBF → render → edit → save)
└── osm_map_advanced/             # NEW: Advanced examples (programmatic config, large PBFs)

tests/
├── test_osm_map_builder.py       # NEW: Unit tests for importer (PBF→MapDef, tag filters, buffering)
├── test_osm_zones_yaml.py        # NEW: YAML round-trip determinism, coordinate transforms
├── test_osm_background_renderer.py # NEW: PNG rendering, affine transforms, bounds validation
├── test_osm_zones_editor.py      # NEW: Editor click handlers, undo/redo, snapping logic
└── test_osm_backward_compat.py   # NEW: Backward-compat validation (pygame, sensors, planners)

output/
├── maps/                    # OSM-generated map artifacts (PNG, YAML)
│   ├── uni_campus_bg.png    # Background rendering
│   ├── uni_campus_zones.yaml # Zone/route annotations
│   └── ...
└── ...

docs/
├── osm_map_workflow.md       # NEW: User guide, workflow, troubleshooting
├── SVG_MAP_EDITOR.md         # MODIFIED: Add section "OSM-Based Extraction" (cross-reference to osm_map_workflow.md)
└── README.md                 # MODIFIED: Add entry to "Map Generation" section
```

---

## Module Design & APIs

### Module 1: `robot_sf/nav/osm_map_builder.py` (Core Importer)

**Responsibility**: Parse local PBF → filter by semantic tags → buffer lines → compute obstacles → populate MapDefinition

**Key Functions**:

```python
def osm_to_map_definition(
    pbf_file: str,
    bbox: tuple[float, float, float, float] | None = None,  # (minlat, minlon, maxlat, maxlon)
    line_buffer_m: float = 1.5,  # Half-width radius (default 3m total)
    tag_filters: OSMTagFilters = OSMTagFilters(),  # Configurable tag set
    project_to_utm: bool = True,
) -> MapDefinition:
    """
    Convert OSM PBF to MapDefinition.
    
    Returns:
        MapDefinition with:
        - bounds: Bounding box polygon
        - obstacles: List[Obstacle] (from complement + explicit obstacles)
        - allowed_areas: List[Polygon] (explicit walkable union)
    """
```

**Algorithm**:
1. Load PBF using OSMnx
2. Filter ways by tag_filters (driveable highways)
3. Filter areas by tag_filters (buildings, water, obstacles)
4. Project geometries to UTM zone (auto-detect from region center)
5. Buffer driveable ways to polygons (Shapely buffer with round caps/joins)
6. Union buffered ways + area polygons → walkable_union
7. Compute obstacles = bounds_polygon - walkable_union
8. Cleanup: buffer(0) invalid polygons, simplify, validate
9. Return MapDefinition(bounds, obstacles, allowed_areas=walkable_union)

**Configuration** (dataclass):
```python
@dataclass
class OSMTagFilters:
    driveable_highways: list[str] = field(
        default_factory=lambda: ["footway", "path", "cycleway", "bridleway", "pedestrian"]
    )
    driveable_areas: list[str] = field(
        default_factory=lambda: ["pedestrian", "footway", "residential", "service"]
    )
    obstacle_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [("building", "*"), ("natural", "water"), ("natural", "cliff")]
    )
    excluded_tags: list[tuple[str, str]] = field(
        default_factory=lambda: [("highway", "steps"), ("access", "private"), ("access", "no")]
    )
```

---

### Module 2: `robot_sf/maps/osm_background_renderer.py` (Visual Context)

**Responsibility**: Render PBF as PNG background with affine transform (pixel ↔ world mapping)

**Key Functions**:

```python
def render_osm_background(
    pbf_file: str,
    bbox: tuple[float, float, float, float],  # (minlat, minlon, maxlat, maxlon)
    output_dir: str = "output/maps/",
    pixels_per_meter: float = 2.0,
    show_buildings: bool = True,
    show_water: bool = True,
) -> dict[str, Any]:
    """
    Render PBF as PNG background image.
    
    Returns:
        {
            "png_path": str,
            "affine_transform": {
                "pixel_origin": [x, y],  # pixel coords of bounds origin
                "pixel_per_meter": float,  # scale
                "bounds_meters": Polygon,  # world-space bounds
            }
        }
    """
```

**Implementation**:
1. Load PBF geometries (ways, buildings, water)
2. Project to local UTM zone
3. Create Matplotlib figure with computed bounds
4. Render layers:
   - Base: light gray background (driveable areas as light overlay)
   - Buildings: dark gray polygons
   - Water: light blue polygons
   - Streets: thin black lines (optional)
5. Compute affine transform: pixel origin + scale
6. Save PNG to output_dir
7. Return metadata dict with transform + bounds

**Metadata file** (JSON sidecar):
```json
{
  "pbf_file": "uni_campus.pbf",
  "bbox": [48.3, 11.9, 48.32, 11.92],
  "created_at": "2025-12-19T14:23:45Z",
  "affine_transform": {
    "pixel_origin": [0, 0],
    "pixel_per_meter": 2.0,
    "bounds_latlon": [48.3, 11.9, 48.32, 11.92]
  }
}
```

---

### Module 3: `robot_sf/maps/osm_zones_editor.py` (Visual Editing)

**Responsibility**: Lightweight Matplotlib-based click editor for zones and routes

**Key Functions**:

```python
class OSMZonesEditor:
    def __init__(
        self,
        background_png: str,
        affine_transform: dict,
        allowed_areas: list[Polygon],
        obstacles: list[Obstacle],
    ):
        """Interactive editor for drawing zones and routes over background."""
        
    def launch(self) -> dict[str, Any]:
        """
        Open interactive editor.
        
        Returns:
            {
                "zones": {"spawn_a": Polygon, "goal": Polygon, ...},
                "routes": {"main_route": list[tuple], ...},
            }
        """
```

**Editor Features**:
- Display PNG background with grid overlay
- Click to place polygon vertices (spawn, goal, crowded zones)
- Drag to move vertices
- Right-click to delete vertices
- Draw polylines for routes
- Undo/redo stack (Ctrl+Z, Ctrl+Y)
- Snapping: Optionally snap to nearest driveable area boundary (tolerance: 0.5m, toggle via Shift)
- Real-time validation:
  - Warn if vertex outside driveable areas
  - Warn if polygon crosses obstacle
  - Highlight out-of-bounds zones in red
- Save to YAML (Ctrl+S)

**Keyboard shortcuts**:
- `p`: Switch to polygon mode
- `r`: Switch to route mode
- `s`: Toggle snapping
- `z`: Undo
- `y`: Redo
- `Ctrl+S`: Save to YAML
- `Esc`: Exit (prompt to save if modified)

---

### Module 4: `robot_sf/maps/osm_zones_yaml.py` (Deterministic Serialization)

**Responsibility**: YAML schema definition, load/save with deterministic ordering and precision

**Key Functions**:

```python
@dataclass
class OSMZonesConfig:
    """Zones and routes for OSM-derived map (YAML v1.0 schema)."""
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    zones: dict[str, Zone] = field(default_factory=dict)
    routes: dict[str, Route] = field(default_factory=dict)

def load_zones_yaml(yaml_file: str) -> OSMZonesConfig:
    """Load YAML file; validate schema; round-trip determinism guaranteed."""
    
def save_zones_yaml(config: OSMZonesConfig, yaml_file: str) -> None:
    """Save to YAML with deterministic ordering and precision (3 decimal places)."""
```

**YAML Schema (v1.0)**:
```yaml
version: "1.0"
metadata:
  source_pbf: "uni_campus.pbf"
  map_bounds: [48.3, 11.9, 48.32, 11.92]  # (minlat, minlon, maxlat, maxlon)
  created_at: "2025-12-19T14:23:45Z"
  utm_zone: 32  # UTM zone used for buffering

zones:
  spawn_zone_a:
    type: "spawn"  # One of: spawn, goal, crowded
    polygon: [[100.0, 200.0], [110.0, 200.0], [110.0, 210.0], [100.0, 210.0]]  # meters
    priority: 1
    metadata: {}  # Optional user notes
    
  goal_zone:
    type: "goal"
    polygon: [[500.0, 600.0], [520.0, 600.0], [520.0, 620.0], [500.0, 620.0]]
    
  crowded_zone:
    type: "crowded"
    polygon: [[300.0, 400.0], [350.0, 400.0], [350.0, 450.0], [300.0, 450.0]]
    density: 0.5

routes:
  main_path:
    type: "pedestrian"  # One of: pedestrian, robot, obstacle_course
    waypoints: [[100.0, 200.0], [150.0, 250.0], [500.0, 600.0]]
    width: 1.0  # Optional: corridor width in meters
```

**Determinism Guarantees**:
- Fixed numeric precision: 3 decimal places (≈1mm accuracy in meters)
- Sorted zone/route keys (alphabetical)
- Consistent YAML structure (no floating-point accumulation errors)
- Version tag in header (enables future schema migrations)

---

### Updated Module: `robot_sf/nav/map_config.py` (MapDefinition)

**Change**: Add optional `allowed_areas` field

```python
@dataclass
class MapDefinition:
    bounds: Polygon
    obstacles: list[Obstacle]
    spawn_zones: list[SpawnZone] = field(default_factory=list)
    goal_zones: list[GoalZone] = field(default_factory=list)
    crowded_zones: list[CrowdedZone] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    
    # NEW FIELD (Optional, populated by OSM importer only):
    allowed_areas: list[Polygon] | None = None
    
    def is_point_in_driveable_area(self, point: tuple[float, float]) -> bool:
        """Check if point is in driveable area (uses allowed_areas if present)."""
        if self.allowed_areas is not None:
            return any(poly.contains(Point(point)) for poly in self.allowed_areas)
        # Fallback: check if NOT in obstacle
        return not any(obs.contains_point(point) for obs in self.obstacles)
```

**Backward Compatibility**:
- Field defaults to `None` (no breakage)
- Existing code ignores field (pygame, sensors, planners unchanged)
- Opt-in usage: Planners/editor can check `if map_def.allowed_areas: use_it()`

---

## Implementation Tactics

### Phase 1 Approach (Weeks 1–2)

**Goal**: Get core importer + rendering working end-to-end

**Tactics**:
1. **Start small**: Begin with small PBF fixture (single city block, <1MB)
2. **Stub out dependencies**: Create empty modules with documented API signatures; fill in one at a time
3. **Test early**: Every small piece has a test (PBF load, tag filter, buffer, polygon cleanup)
4. **Validate visually**: After rendering, manually inspect PNG vs ground truth; take screenshots
5. **Backward-compat check**: Run existing tests (pygame, sensors) after each major piece

**Key Milestones**:
- [ ] OSM tag filters working (unit test: load PBF → filter ways/areas)
- [ ] Buffering pipeline working (unit test: buffer geodf → valid polygons)
- [ ] Obstacle derivation working (unit test: complement → bounds check)
- [ ] PNG rendering working (visual spot-check: output/*.png exists and looks reasonable)
- [ ] MapDefinition integration (integration test: PBF → MapDef → pygame viz unchanged)

### Phase 2 Approach (Weeks 3–4)

**Goal**: Visual editor working end-to-end

**Tactics**:
1. **Matplotlib click handler first**: Get basic polygon drawing working
2. **Affine transform correctness**: Verify pixel→world coordinate transforms (unit tests)
3. **YAML determinism**: Test save→load round-trip yields identical bytes
4. **Editor validation**: Add real-time warnings (out-of-bounds, obstacle crossing)
5. **UX polish**: Keyboard shortcuts, snapping, undo/redo

**Key Milestones**:
- [ ] Click handlers respond (manual test: click on PNG → vertex appears)
- [ ] YAML save/load working (unit test: create zones → save → load → identical)
- [ ] Coordinate transforms correct (unit test: pixel↔world round-trip ±1px)
- [ ] Validation warnings working (manual test: draw across obstacle → warning displayed)

### Phase 3 Approach (Weeks 5–6)

**Goal**: Programmatic configuration alternative

**Tactics**:
1. **Config schema first**: Define YAML scenario structure (zones + routes in code)
2. **Parser**: Load scenario YAML → create zones programmatically
3. **Equivalence test**: Programmatic config should produce identical MapDef as editor

**Key Milestones**:
- [ ] Scenario config parser working
- [ ] Programmatic zone creation API
- [ ] Equivalence tests passing

### Phase 4 Approach (Weeks 7–8)

**Goal**: Documentation + cleanup

**Tactics**:
1. **Write user guide**: PBF acquisition, import workflow, editor tutorial
2. **Update central docs**: Add entry to docs/README.md, SVG_MAP_EDITOR.md
3. **Examples**: End-to-end demo in examples/
4. **Cleanup**: Remove TODOs, add docstrings, ensure all tests pass

---

## Testing Strategy

**Test Categories** (all mandatory for Phase 1 completion):

| Test | File | Purpose | Type |
|------|------|---------|------|
| PBF load | test_osm_map_builder.py | Can read small PBF fixture | Unit |
| Tag filter | test_osm_map_builder.py | Filters ways/areas correctly | Unit |
| Buffering | test_osm_map_builder.py | Buffers lines to valid polygons | Unit |
| Polygon cleanup | test_osm_map_builder.py | buffer(0) repairs self-intersections | Unit |
| Obstacle derivation | test_osm_map_builder.py | Complement computes correct obstacles | Unit |
| MapDef integration | test_osm_map_builder.py | OSM output is valid MapDefinition | Integration |
| Backward-compat pygame | test_osm_backward_compat.py | Existing pygame viz works unchanged | Integration |
| Backward-compat sensors | test_osm_backward_compat.py | Sensor suite ignores allowed_areas | Integration |
| PNG rendering | test_osm_background_renderer.py | Produces valid PNG file | Integration |
| Affine transform | test_osm_background_renderer.py | pixel↔world coordinates round-trip ±1px | Unit |
| YAML schema | test_osm_zones_yaml.py | Loads/validates YAML correctly | Unit |
| YAML determinism | test_osm_zones_yaml.py | save→load→save yields byte-identical output | Unit |
| Editor click handlers | test_osm_zones_editor.py | Click on background adds vertex | Integration |
| Editor snapping | test_osm_zones_editor.py | Vertices snap to boundary within tolerance | Unit |
| Backward-compat all | test_osm_backward_compat.py | Full train/eval cycle unchanged | Smoke |

**Minimum Coverage Target**: 85% for new modules (osm_map_builder, osm_zones_yaml, osm_background_renderer)

**Fixtures**:
- Small PBF (single city block, <1MB) checked into `test_scenarios/osm_fixtures/`
- Expected outputs (PNG, YAML) for regression testing
- Backward-compat baseline (existing environment creation + step loop)

---

## Risk Assessment & Mitigation

| Risk | Severity | Mitigation | Owner |
|------|----------|-----------|-------|
| Large PBF performance | Medium | Use bbox to query subset; profile early; lazy-load | Phase 1 |
| OSM tag inconsistency | Medium | Document assumptions; validate on real data; archive problematic PBFs | Phase 1 |
| Polygon validity issues | Medium | Extensive buffer(0) testing; simplify safety; fail gracefully with logging | Phase 1 |
| Affine transform precision | Low | Store as floats; document ±1px tolerance; round-trip tests | Phase 2 |
| Editor UX friction | Medium | Early manual testing; iterate on shortcuts/validation; user feedback | Phase 2 |
| Backward-compat breakage | High | Run full test suite after each major change; test on CI | All phases |
| Schedule slip | Medium | Prioritize Phase 1; defer Phase 3 if necessary; ship Phase 2b with Phase 1 | Mgmt |

---

## Success Criteria & DoD

**Phase 1 Done**:
- [ ] `osm_map_builder.py` functional; core importer working
- [ ] `osm_background_renderer.py` functional; PNG + transform metadata correct
- [ ] MapDefinition.allowed_areas field added (optional, backward-compat validated)
- [ ] Core tests passing (PBF→MapDef, buffering, polygon cleanup, backward-compat)
- [ ] Example script runnable end-to-end
- [ ] Existing pygame/sensor tests still pass
- [ ] Documentation: osm_map_workflow.md drafted

**Phase 2 Done**:
- [ ] `osm_zones_editor.py` functional; visual editing working
- [ ] `osm_zones_yaml.py` functional; YAML round-trip deterministic
- [ ] Editor tests passing (click handlers, snapping, validation)
- [ ] Example demonstrating editor workflow
- [ ] Full backward-compat suite passing (train/eval cycle unchanged)

**Phase 3 Done**:
- [ ] Programmatic config API finalized
- [ ] Scenario parser working
- [ ] Equivalence tests (programmatic ≡ editor output)

**Phase 4 Done**:
- [ ] User guide complete (docs/osm_map_workflow.md)
- [ ] docs/SVG_MAP_EDITOR.md updated with OSM section
- [ ] docs/README.md updated with new entry
- [ ] All examples runnable
- [ ] All tests passing
- [ ] Type checking clean (ty check)
- [ ] Lint clean (ruff)

---

## Next Steps

1. **Immediate**: Prepare data-model.md (entity definitions, MapDefinition schema change, YAML contract)
2. **Immediate**: Prepare quickstart.md (user guide, demo workflow end-to-end)
3. **Follow-up**: Generate tasks.md (atomic, testable implementation tasks via /speckit.tasks)
4. **Follow-up**: Begin Phase 1 implementation (start with osm_map_builder.py)

---

## Complexity Tracking

**Complexity Assessment**: MEDIUM

**Justification**:
- Introduces 4 new modules (~2000 LOC total)
- Depends on mature external libraries (OSMnx, Shapely, GeoPandas)
- API design straightforward (no novel algorithms)
- Backward-compatible (optional field; zero breakage)
- Constitution compliance clear (all principles satisfied)

**Complexity Mitigations**:
- Start with small PBF fixtures (reduce scope for early testing)
- Stub out empty modules first (design before implementation)
- Test each component independently (avoid integration surprises)
- Backward-compat validation at every phase (catch breakage early)

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
