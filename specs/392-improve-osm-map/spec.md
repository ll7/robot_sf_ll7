# Feature Specification: Robust OSM-Based Map Extraction to MapDefinition

**Feature Branch**: `392-Improve-osm-map-generation`  
**Created**: 2025-12-19  
**Status**: Draft  
**Input**: User description: Improve OSM map extraction by moving from color-based SVG parsing to semantic OSM tag-based pipeline with local PBF ingestion, driveable area computation, and visual background generation.

## Overview

Currently, map creation from OpenStreetMap uses a lossy, color-based SVG export workflow: export SVG → manually edit colors/split shapes in Inkscape → parse via `svg_map_parser.py`. This approach has fundamental limitations:

- **Information loss**: OSM tags (highway, sidewalk, access) are stripped during SVG export
- **Scale imprecision**: Scale factors must be inferred from filenames (e.g., `_1350`)
- **Polygon validity**: Self-intersecting SVG paths cause parsing failures
- **Driveable area implicit**: Free space is defined as "obstacles' complement" rather than semantic walkable areas

**Solution**: Replace the SVG export with a **PBF-based pipeline** that:

1. Ingests local OSM PBF files (ground truth)
2. Filters ways/areas by semantic tags (highway=footway/path/pedestrian + area=yes, etc.)
3. Buffers lines to driveable polygons (default 3 m width)
4. Derives obstacles as the geometric complement within map bounds (removes guesswork)
5. Outputs a `MapDefinition` directly (programmatic, reproducible)
6. Renders a visual background (PNG/SVG-like) from the same PBF for verification

**Key architectural decisions**:

- **Data source**: Local PBF files (reproducible, offline, no rate limits)
- **MVP scope**: Hybrid approach — driveable areas (footway/path/cycleway/pedestrian) + building obstacles (most impactful); water/landuse optional
- **Output format**: `MapDefinition` objects (code-first, no Inkscape round-tripping)
- **Visual context**: Render map background from PBF for verification; spawn/goal zones annotated in code/config

---

## User Scenarios & Testing

### User Story 1 — Data Engineer: Convert OSM PBF to MapDefinition (Priority: P1)

**Scenario**: A data engineer has downloaded a PBF extract (e.g., `uni_campus.pbf`) and wants to generate a `MapDefinition` suitable for robot navigation in seconds, without manual SVG editing.

**Why this priority**: Core MVP — directly replaces the broken SVG workflow. Unblocks all downstream users.

**Independent Test**: Can be fully tested by loading a small PBF fixture, calling `osm_to_map_definition(pbf_file=...)`, and verifying the output is a valid `MapDefinition` with non-empty obstacles and bounds matching the query region.

**Acceptance Scenarios**:

1. **Given** a valid local PBF file (e.g., `uni_campus.pbf`), **When** calling `osm_to_map_definition(pbf_file="uni_campus.pbf")`, **Then** the function returns a `MapDefinition` with obstacles and bounds computed from walkable/non-walkable areas.

2. **Given** a PBF with buildings and footways, **When** processing with default tag filters, **Then** buildings are treated as obstacles and footways are buffered into driveable polygons.

3. **Given** a PBF with ways tagged `highway=steps`, **When** filtering, **Then** steps are explicitly excluded from driveable areas.

4. **Given** a PBF with multiple disconnected regions (e.g., a campus with multiple quads), **When** processing, **Then** all walkable areas and obstacles are included in a single `MapDefinition`.

---

### User Story 2 — Visualization Specialist: Render Map Background from PBF (Priority: P1)

**Scenario**: A specialist wants to visually verify that the extracted driveable areas match the real-world layout. They render a background image (PNG or SVG) overlaid with the computed obstacles and driveable zones.

**Why this priority**: Essential for trust and debugging. Users cannot iterate confidently without seeing what the map looks like.

**Independent Test**: Can be fully tested by rendering a PBF as an image, overlaying computed obstacles, and verifying the visual layout matches the real terrain (manual visual inspection + automated bounds checks).

**Acceptance Scenarios**:

1. **Given** a `MapDefinition` generated from a PBF, **When** rendering a background PNG/SVG from the same PBF, **Then** the background visually matches the obstacle/driveable area positions.

2. **Given** a background image at a known scale/projection, **When** overlaying the `MapDefinition` obstacles, **Then** geospatial alignment is accurate (tested by visual spot-checks on known landmarks).

3. **Given** a small test PBF (e.g., a single building block), **When** rendering and inspecting, **Then** the output clearly shows which areas are walkable and which are obstacles.

---

### User Story 3 — Robot Sim User: Annotate Spawn/Goal Zones Programmatically (Priority: P2)

**Scenario**: After MapDefinition generation, a user wants to specify robot spawn zones, goal zones, and crowded pedestrian areas without touching Inkscape. They provide these as code configuration (YAML or Python).

**Why this priority**: Completes the workflow. Allows users to fully define maps in code/config, aligning with your programmatic-first philosophy.

**Independent Test**: Can be fully tested by generating a `MapDefinition` from PBF, adding spawn/goal zones via config, and verifying zones are valid polygons inside driveable areas.

**Acceptance Scenarios**:

1. **Given** a base `MapDefinition`, **When** adding spawn zones (as lat/lon or local coordinates), **Then** zones are converted to local frame and stored in the `MapDefinition`.

2. **Given** multiple spawn/goal zone candidates (e.g., sample points on driveable areas), **When** selecting a subset via config, **Then** only selected zones are retained.

3. **Given** a config file with zone definitions, **When** loading it alongside a PBF-generated `MapDefinition`, **Then** zones are validated against driveable boundaries (warning if outside).

---

### User Story 4 — Maintenance: Update Map When Real-World OSM Data Changes (Priority: P3)

**Scenario**: OSM editors update the campus map (e.g., new building, sidewalk widening). A maintainer re-runs the PBF extraction and regenerates the `MapDefinition` automatically. Existing spawn/goal zone configs are reapplied without modification.

**Why this priority**: Long-term sustainability. Decouples map data from OSM updates.

**Independent Test**: Can be fully tested by simulating an OSM update (e.g., adding a building tag), re-running the pipeline, and verifying the new `MapDefinition` reflects the change while zone configs remain valid.

**Acceptance Scenarios**:

1. **Given** an updated PBF reflecting real-world changes, **When** re-running the pipeline with the same config, **Then** the new `MapDefinition` reflects OSM updates while zones remain in place.

2. **Given** zone configs that reference specific regions (e.g., "north quadrant spawn zone"), **When** zones are validated, **Then** out-of-bounds zones are flagged for review before application.

---

### Edge Cases

- What happens when a PBF has no driveable areas (e.g., a dense forest with no paths)? → System emits warning and produces a minimal `MapDefinition` with empty obstacles (or full bounds as single obstacle).
- How does the system handle ways tagged with conflicting attributes (e.g., `highway=footway` but `access=no`)? → Conservative approach: exclude such ways from driveable areas; log warnings.
- What if a PBF has extremely large multipolygons (e.g., a sprawling park boundary)? → Polygon validity checks and simplification (Shapely `buffer(0)`, `simplify`) prevent crashes.
- How are self-intersecting obstacle polygons handled? → Automatically repaired with `Shapely.buffer(0)` and logged.

---

## Requirements

### Functional Requirements

- **FR-001**: System MUST accept a local OSM PBF file path and extract driveable ways/areas via OSM tag filters.
- **FR-002**: System MUST support the following driveable line tags: `highway=footway`, `highway=path`, `highway=cycleway`, `highway=bridleway`, `highway=pedestrian`.
- **FR-003**: System MUST support driveable area tags: `highway=[pedestrian|footway|path|service|residential|unclassified]` + `area=yes`.
- **FR-004**: System MUST exclude `highway=steps` and any ways tagged `access=no` or `access=private` from driveable areas.
- **FR-005**: System MUST buffer driveable line ways to polygons using a configurable width (default 3 m total; half-width 1.5 m on each side).
- **FR-006**: System MUST use OSM `width` tag if available; otherwise apply default width.
- **FR-007**: System MUST include `building=*` features as obstacles.
- **FR-008**: System MUST compute obstacles as the geometric complement: bounds polygon minus walkable union.
- **FR-009**: System MUST project geometries to a meter-based CRS (e.g., UTM) for accurate buffering before reprojecting to local frame.
- **FR-010**: System MUST validate and repair polygon geometries (self-intersections, invalid rings) using Shapely `buffer(0)` and `is_valid` checks.
- **FR-011**: System MUST output a `MapDefinition` object with computed obstacles, bounds, width, height, and empty (optional) spawn/goal zones.
- **FR-012**: System MUST render a visual background (PNG or SVG-like) from the same PBF showing the computed driveable/obstacle layout for verification.
- **FR-013**: System MUST support programmatic annotation of spawn zones, goal zones, and crowded pedestrian areas via Python config (dict/dataclass) or YAML.
- **FR-014**: System MUST provide a demo script (under `examples/`) showing end-to-end PBF-to-MapDefinition pipeline with visualization.
- **FR-015**: System MUST maintain backward compatibility with existing `svg_map_parser.py` (no breaking changes).

### Key Entities

- **MapDefinition**: Represents a navigable map with obstacles, bounds, spawn/goal zones, and optional routes. Generated directly from OSM PBF.
- **Obstacle**: A polygon representing non-traversable space. Derived from OSM buildings, water, landuse, or complement geometry.
- **DrivableArea** (implicit): Union of buffered line ways and area polygons matching tag filters. Not stored; computed intermediate.
- **BoundingBox**: Geographic region (lat/lon bbox) defining the extent of the map. Converted to local meter-based frame.
- **ZoneConfig**: Programmatic specification of spawn zones, goal zones, and crowded areas. Validated against driveable boundaries.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: **Workflow speed** — Users can convert a PBF to a `MapDefinition` with obstacles in < 5 seconds (for typical campus-sized regions ~1 km²).
- **SC-002**: **Data precision** — Driveable areas computed from OSM semantics match manually validated real-world walkable paths (e.g., sidewalks, plazas) with > 90% spatial accuracy.
- **SC-003**: **Polygon quality** — All generated obstacles are valid Shapely polygons with no self-intersections (100% validity rate in test suite).
- **SC-004**: **Scale accuracy** — Computed map bounds (width/height) match PBF geography to within ±5 m for regions under 2 km².
- **SC-005**: **Visual verification** — Rendered background image accurately aligns with computed obstacles (manual visual inspection on 3+ test cases; automated affine transform validation).
- **SC-006**: **Reproduction** — Given the same PBF and config, running the pipeline twice produces bitwise-identical `MapDefinition` outputs (deterministic).
- **SC-007**: **Test coverage** — Core pipeline functions (tag filtering, buffering, obstacle generation) achieve ≥ 85% code coverage in `robot_sf/nav/osm_map_builder.py`.
- **SC-008**: **Documentation** — Workflow documented in `docs/` with examples; section added to `docs/SVG_MAP_EDITOR.md` explaining OSM-based alternative.
- **SC-009**: **No breakage** — Existing tests for `svg_map_parser.py` and `MapDefinition` remain passing (backward compatibility).
- **SC-010**: **User feedback** — At least one successful end-to-end test on a real campus PBF (e.g., `uni_campus_1350.pbf`).

---

## Assumptions

- Users have access to PBF files (e.g., from Geofabrik or OSM Planet exports) or know how to download them via Overpass.
- OSM data quality is reasonably good (tags are mostly consistent); the system is tolerant of minor inconsistencies.
- Default line width of 3 m is acceptable as a baseline; users can override in config if needed.
- Meter-based CRS (UTM) is appropriate for accuracy; no support for very-long-distance maps where Earth curvature matters.
- Spawn/goal zone placement is driven by user config; no automatic inference (e.g., sampling random walkable points) is required for MVP.
- Visual background rendering is secondary (PNG export is sufficient; fancy SVG with labels is nice-to-have).

---

## Constraints & Trade-offs

### Constraints

- **Dependency footprint**: OSMnx, GeoPandas, Shapely, PyProj (totaling ~50 MB installed). Acceptable given the advanced GIS nature of the feature.
- **Performance**: Large PBFs (> 100 MB) may take 10+ seconds; acceptable for one-time map generation, not for real-time queries.
- **Offline**: Requires pre-downloaded PBF; no live Overpass API queries in MVP (avoids rate limits, keeps deterministic).

### Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **Local PBF only** | Requires user to download/manage files; eliminates live API dependency; reproducible. |
| **Hybrid MVP scope** (driveable + buildings) | Covers 80% of use cases; adds buildings/water later; keeps initial complexity manageable. |
| **MapDefinition code output** | No Inkscape GUI; faster, reproducible; zone annotation in code/config (slight learning curve for non-programmers). |
| **Rendered background** | PNG/basic SVG; not a full interactive editor; acceptable for verification. |

---

## Implementation Plan (Incremental)

### Phase 1: Core Importer (Weeks 1–2)

**Deliverable**: Minimal, working `osm_to_map_definition(pbf_file)` function.

- [ ] Create `robot_sf/nav/osm_map_builder.py` module with:
  - `osm_to_map_definition(pbf_file: str, bbox: tuple[float, float, float, float] | None = None) -> MapDefinition`
  - Tag filter lists (driveable lines, driveable areas, obstacles)
  - Geometry buffering, projection, and cleanup logic
- [ ] Add unit tests (tag filtering, buffering, polygon repair, bounds calculation)
- [ ] Validate on `uni_campus_1350.pbf` (if available) or small test fixture

**Definition of Done**: Function produces valid `MapDefinition` objects; unit tests pass; manual spot-checks on 1–2 real PBFs.

---

### Phase 2: Visual Background Rendering (Weeks 2–3)

**Deliverable**: Render PBF as PNG/SVG background with overlay visualization.

- [ ] Add rendering function: `render_osm_background(pbf_file, output_path) -> None`
- [ ] Overlay computed obstacles and driveable areas as colored layers
- [ ] Store affine transform (scale, offset) alongside rendered image for future alignment
- [ ] Example: overlay in `examples/` showing background + `MapDefinition`

**Definition of Done**: Visual inspection confirms layout matches real terrain; affine transform is accurate; demo runs without errors.

---

### Phase 3: Zone Annotation & Config (Weeks 3–4)

**Deliverable**: Programmatic spawn/goal zone annotation via Python/YAML config.

- [ ] Add `ZoneConfig` dataclass (spawn_zones, goal_zones, crowded_zones)
- [ ] Function: `apply_zones_to_map_definition(map_def: MapDefinition, config: ZoneConfig) -> MapDefinition`
- [ ] Validation: ensure zones are inside driveable areas (or log warnings)
- [ ] Config loading: YAML parser for zone definitions
- [ ] Example config file and demo

**Definition of Done**: Zone config loads correctly; zones are validated; demo produces annotated `MapDefinition`.

---

### Phase 4: Documentation & Integration (Week 4)

**Deliverable**: Docs, examples, and backward-compatibility validation.

- [ ] Add section to `docs/SVG_MAP_EDITOR.md` explaining OSM-based workflow
- [ ] Create `examples/osm_map_quickstart.py` (minimal end-to-end example)
- [ ] Ensure existing `svg_map_parser.py` tests still pass
- [ ] Update `docs/README.md` with links to new docs
- [ ] Add docstrings following pydocstyle (robot_sf convention)

**Definition of Done**: Docs are clear; examples run without errors; all existing tests pass; CI green.

---

## Testing Strategy

### Unit Tests

- **Tag filtering**: Verify `highway=footway` included, `highway=steps` excluded, `access=no` excluded
- **Buffering**: Check line buffer widths (default 3 m), OSM `width` tag override
- **Polygon repair**: Invalid geometries (self-intersections) are repaired and marked valid
- **Bounds calculation**: Width/height computed correctly from bbox
- **Projection**: Geometries correctly project to UTM and back to local frame

### Integration Tests

- **End-to-end PBF pipeline**: Load small test fixture → produce `MapDefinition` → verify bounds, obstacles, driveable areas
- **Visual rendering**: Render background, overlay obstacles, manually inspect alignment
- **Zone annotation**: Load config, apply zones, verify zones stored correctly and marked valid/invalid as needed

### Validation (Smoke Tests)

- **Real-world PBF**: Process `uni_campus_1350` (or similar); visually inspect output matches expectation
- **Performance**: < 5 seconds for typical region
- **Backward compatibility**: Existing `svg_map_parser.py` tests pass; no breaking changes to `MapDefinition`

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| PBF file large / processing slow | Users frustrated if > 10 sec | Implement progress logging; optimize buffering with spatial indexing if needed |
| OSM data quality issues (missing tags, inconsistencies) | Driveable areas incomplete/wrong | Conservative tag filters; clear logging of excluded ways; manual override via config |
| Polygon validity bugs (edge cases) | Crashes or invalid geometries | Extensive Shapely `buffer(0)` and validation; unit tests for known pathological cases |
| Downstream planners fail on unexpected obstacle geometry | Silent failures | MapDefinition validation in sim before use; clear error messages |

---

## Links & References

- **OSM Tag Wiki**: https://wiki.openstreetmap.org/wiki/Map_features
- **Existing parser**: `robot_sf/nav/svg_map_parser.py`
- **Example OSM map**: `maps/osm_svg_maps/uni_campus_1350.svg`
- **Research notes**: `specs/392-improve-osm-map/research/`
- **MapDefinition class**: `robot_sf/nav/map_config.py`
- **Obstacle class**: `robot_sf/nav/obstacle.py`
