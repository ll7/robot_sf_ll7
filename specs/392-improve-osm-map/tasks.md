# Implementation Tasks: OSM-Based Map Extraction to MapDefinition

**Feature**: Robust OSM-Based Map Extraction  
**Branch**: `392-Improve-osm-map-generation`  
**Total Tasks**: 42 tasks across 4 phases  
**Estimated Duration**: 8 weeks  
**Status**: Ready for Phase 1

---

## Overview & Task Organization

Tasks are organized by **user story** to enable independent, parallel implementation where possible. Each task is:
- **Atomic**: Completable in 1–4 hours
- **Testable**: Has clear acceptance criteria
- **Traceable**: Links to spec.md user stories and requirements
- **Ordered**: Dependencies respected within phases

### Quick Reference: Task Counts

| Phase | Description | Tasks | Duration |
|-------|-------------|-------|----------|
| **Phase 1** | Core importer & rendering | 17 tasks | Weeks 1–2 |
| **Phase 2** | Visual editor & YAML serialization | 12 tasks | Weeks 3–4 |
| **Phase 3** | Programmatic config & annotation | 8 tasks | Weeks 5–6 |
| **Phase 4** | Documentation & polish | 5 tasks | Weeks 7–8 |

---

## Phase 1: Core Importer & Rendering (Weeks 1–2)

**Goal**: PBF → MapDefinition pipeline with visual background rendering, backward-compat validated

### Subphase 1a: Project Setup & Fixture Preparation

- [ ] T001 Create test PBF fixture (single city block, <1MB) in `test_scenarios/osm_fixtures/sample_block.pbf`
  - **Acceptance**: Fixture loads without error; has buildings, footways, obstacles
  - **Requires**: bbbike.org extract or local OSM data
  
- [ ] T002 Create expected output fixtures (PNG, JSON metadata) in `test_scenarios/osm_fixtures/expected_outputs/`
  - **Acceptance**: Reference PNG exists; metadata JSON has affine_transform
  - **Depends on**: T001

- [ ] T003 [P] Add OSM dependencies to `pyproject.toml` (osmnx, shapely, geopandas, pyproj, pyyaml)
  - **Acceptance**: `uv sync --all-extras` succeeds; all imports available
  - **File**: `/Users/lennart/git/robot_sf_ll7/pyproject.toml`

- [ ] T004 [P] Create module skeleton files (empty, with docstring/imports)
  - **Files**: 
    - `robot_sf/nav/osm_map_builder.py`
    - `robot_sf/maps/osm_background_renderer.py`
  - **Acceptance**: Modules importable; no runtime errors

### Subphase 1b: Core Importer Implementation (OSMnx → MapDefinition)

- [ ] T005 [US1] Implement `OSMTagFilters` dataclass in `robot_sf/nav/osm_map_builder.py`
  - **Acceptance**: Config object created; default tag sets defined (driveable, obstacles, excluded)
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Spec Link**: FR-003, FR-004

- [ ] T006 [US1] Implement PBF loading via OSMnx: `load_pbf(pbf_file) -> GeoDataFrame`
  - **Acceptance**: Unit test passes (load fixture → GeoDataFrame with ways/areas)
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_pbf_loading`
  - **Spec Link**: FR-001, FR-002

- [ ] T007 [P] [US1] Implement tag filtering: `filter_driveable_ways(gdf, tags) -> GeoDataFrame`
  - **Acceptance**: Unit test passes (filter returns only tagged features; excludes steps, private)
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_tag_filtering`
  - **Spec Link**: FR-003, FR-004

- [ ] T008 [P] [US1] Implement obstacle extraction: `extract_obstacles(gdf, tags) -> GeoDataFrame`
  - **Acceptance**: Buildings, water, cliffs extracted correctly; unit test passes
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_obstacle_extraction`
  - **Spec Link**: FR-004

- [ ] T009 [US1] Implement UTM projection: `project_to_utm(gdf) -> (GeoDataFrame, int)`
  - **Acceptance**: Auto-detect zone; project to meter-based coords; round-trip error <0.1m
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_utm_projection`
  - **Spec Link**: FR-005

- [ ] T010 [P] [US1] Implement line buffering: `buffer_ways(gdf, half_width_m) -> GeoSeries`
  - **Acceptance**: Lines buffered to polygons; cap/join style round; result valid
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_line_buffering`
  - **Spec Link**: FR-005

- [ ] T011 [US1] Implement polygon cleanup: `cleanup_polygons(polys) -> list[Polygon]`
  - **Acceptance**: buffer(0) repairs; simplify applied; invalid skipped; test passes
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_polygon_cleanup`
  - **Spec Link**: FR-006

- [ ] T012 [US1] Implement obstacle derivation: `compute_obstacles(bounds, walkable_union) -> list[Polygon]`
  - **Acceptance**: Complement computed; obstacles + walkable = bounds (within tolerance)
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_obstacle_derivation`
  - **Spec Link**: FR-007

- [ ] T013 [US1] Implement core entry point: `osm_to_map_definition(pbf_file, bbox, tag_filters, ...) -> MapDefinition`
  - **Acceptance**: End-to-end test (PBF fixture → MapDefinition); has bounds, obstacles, allowed_areas
  - **File**: `robot_sf/nav/osm_map_builder.py`
  - **Test**: `tests/test_osm_map_builder.py::test_osm_to_map_definition_integration`
  - **Spec Link**: US1, FR-001–FR-007

### Subphase 1c: MapDefinition Update & Backward-Compat Setup

- [ ] T014 [P] Modify `MapDefinition` dataclass: Add optional `allowed_areas: list[Polygon] | None = None` field
  - **Acceptance**: Field defaults to None; type-checks pass; existing deserialization works
  - **File**: `robot_sf/nav/map_config.py`
  - **Test**: `tests/test_osm_map_builder.py::test_map_definition_backward_compat`
  - **Spec Link**: Clarifications (Option C)

- [ ] T015 [P] Add helper: `MapDefinition.is_point_in_driveable_area(point) -> bool` (uses allowed_areas if present)
  - **Acceptance**: Method works; optional field respected; fallback to obstacle complement
  - **File**: `robot_sf/nav/map_config.py`
  - **Test**: `tests/test_osm_map_builder.py::test_is_point_in_driveable_area`

### Subphase 1d: Background Rendering (PBF → PNG + Affine Transform)

- [ ] T016 [US2] Implement core renderer: `render_osm_background(pbf_file, output_dir, pixels_per_meter, ...) -> dict`
  - **Acceptance**: PNG file created; metadata dict returned; affine_transform has pixel_origin, pixel_per_meter
  - **File**: `robot_sf/maps/osm_background_renderer.py`
  - **Test**: `tests/test_osm_background_renderer.py::test_render_osm_background`
  - **Spec Link**: US2, FR-010, FR-011

- [ ] T017 [US2] Implement affine transform validation: `validate_affine_transform(transform) -> bool`
  - **Acceptance**: Round-trip pixel↔world coordinate test; ±1 pixel, ±0.1m tolerance
  - **File**: `robot_sf/maps/osm_background_renderer.py`
  - **Test**: `tests/test_osm_background_renderer.py::test_affine_transform_round_trip`
  - **Spec Link**: US2, FR-011

### Subphase 1e: Example & Backward-Compat Validation

- [ ] T018 [P] Create example: `examples/osm_map_quickstart.py` (end-to-end demo: load PBF → render → show)
  - **Acceptance**: Script runs headless; produces MapDefinition + PNG; no errors
  - **File**: `examples/osm_map_quickstart.py`
  - **Runs as**: `uv run python examples/osm_map_quickstart.py`

- [ ] T019 Validate backward-compat: pygame visualization still works
  - **Acceptance**: Existing test suite passes; `test_pygame` tests unaffected
  - **File**: `tests/test_osm_backward_compat.py`
  - **Test**: `tests/test_osm_backward_compat.py::test_pygame_visualization_unchanged`
  - **Spec Link**: SC-009 (Backward compatibility)

- [ ] T020 Validate backward-compat: sensor suite still works
  - **Acceptance**: Sensor tests pass; allowed_areas field ignored correctly
  - **File**: `tests/test_osm_backward_compat.py`
  - **Test**: `tests/test_osm_backward_compat.py::test_sensor_suite_unchanged`
  - **Spec Link**: SC-009

- [ ] T021 Validate backward-compat: planners still work
  - **Acceptance**: Existing planner tests pass; MapDef without allowed_areas works
  - **File**: `tests/test_osm_backward_compat.py`
  - **Test**: `tests/test_osm_backward_compat.py::test_planner_compatibility`
  - **Spec Link**: SC-009

---

## Phase 2: Visual Editor & YAML Serialization (Weeks 3–4)

**Goal**: Lightweight visual editor + deterministic zone/route serialization

### Subphase 2a: YAML Schema & Serialization

- [ ] T022 [US3] Define YAML schema v1.0: `OSMZonesConfig` dataclass in `robot_sf/maps/osm_zones_yaml.py`
  - **Acceptance**: Dataclass defined; supports zones dict, routes dict, metadata; version tag
  - **File**: `robot_sf/maps/osm_zones_yaml.py`
  - **Spec Link**: FR-019, FR-020

- [ ] T023 [US3] Implement YAML loader: `load_zones_yaml(yaml_file) -> OSMZonesConfig`
  - **Acceptance**: Loads YAML fixture; validates schema; returns typed dataclass
  - **File**: `robot_sf/maps/osm_zones_yaml.py`
  - **Test**: `tests/test_osm_zones_yaml.py::test_yaml_loading`
  - **Spec Link**: FR-020

- [ ] T024 [US3] Implement YAML saver: `save_zones_yaml(config, yaml_file) -> None` with determinism guarantees
  - **Acceptance**: Saves with 3 decimal precision; sorted keys; round-trip byte-identical
  - **File**: `robot_sf/maps/osm_zones_yaml.py`
  - **Test**: `tests/test_osm_zones_yaml.py::test_yaml_determinism_round_trip`
  - **Spec Link**: FR-020, SC-007

- [ ] T025 [P] [US3] Implement YAML validation: `validate_zones_yaml(config, map_def) -> list[str]` (warnings/errors)
  - **Acceptance**: Detects out-of-bounds zones, obstacles crossing, invalid polygons; returns warnings list
  - **File**: `robot_sf/maps/osm_zones_yaml.py`
  - **Test**: `tests/test_osm_zones_yaml.py::test_yaml_validation`
  - **Spec Link**: SC-008

### Subphase 2b: Visual Editor Implementation (Matplotlib)

- [ ] T026 [US3] Create `OSMZonesEditor` class skeleton: init, event handlers, state machine
  - **Acceptance**: Class instantiates; background PNG displayed; no crashes
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: Manual (visual inspection)
  - **Spec Link**: US3, FR-016

- [ ] T027 [US3] Implement click handlers: Add polygon vertices on click
  - **Acceptance**: Click on background → vertex marker appears; pixel→world transform correct
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: `tests/test_osm_zones_editor.py::test_click_handlers`
  - **Spec Link**: FR-016

- [ ] T028 [P] [US3] Implement vertex editing: Move/delete vertices (drag, right-click)
  - **Acceptance**: Drag vertex → moves; right-click → deletes; visual feedback
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: Manual (interactive test)
  - **Spec Link**: FR-016

- [ ] T029 [US3] Implement undo/redo: Stack-based history (Ctrl+Z, Ctrl+Y)
  - **Acceptance**: Undo reverts last action; redo restores; stack bounded
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: `tests/test_osm_zones_editor.py::test_undo_redo`
  - **Spec Link**: FR-016

- [ ] T030 [US3] Implement snapping: Optional snap to driveable area boundary (toggle: Shift)
  - **Acceptance**: With snapping on, vertices snap to nearest boundary (0.5m tolerance); validation warning if disabled
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: `tests/test_osm_zones_editor.py::test_snapping_logic`
  - **Spec Link**: SC-008

- [ ] T031 [US3] Implement real-time validation: Warn if zone crosses obstacle or extends outside bounds
  - **Acceptance**: Out-of-bounds zones highlighted in red; obstacle warnings logged
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: `tests/test_osm_zones_editor.py::test_validation_warnings`
  - **Spec Link**: SC-008

- [ ] T032 [US3] Implement save trigger: Ctrl+S saves current zones/routes to YAML
  - **Acceptance**: Save dialog appears; YAML written; no corruption
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: `tests/test_osm_zones_editor.py::test_save_to_yaml`
  - **Spec Link**: FR-021

- [ ] T033 [US3] Implement keyboard shortcuts & UI polish: p/r mode switch, s toggle snapping, help menu
  - **Acceptance**: Shortcuts responsive; help text displayed; intuitive flow
  - **File**: `robot_sf/maps/osm_zones_editor.py`
  - **Test**: Manual (interactive)
  - **Spec Link**: FR-016

### Subphase 2c: Example & Integration

- [ ] T034 [US3] Create example: `examples/osm_map_editor_demo.py` (end-to-end: render → edit → save YAML)
  - **Acceptance**: Script launches editor; manual test (draw zones, save, reload)
  - **File**: `examples/osm_map_editor_demo.py`

- [ ] T035 [P] Full backward-compat smoke test (train/eval cycle unchanged)
  - **Acceptance**: Load OSM-derived MapDef; run environment reset/step loop; metrics unchanged
  - **File**: `tests/test_osm_backward_compat.py::test_full_train_eval_cycle`
  - **Spec Link**: SC-009

---

## Phase 3: Programmatic Configuration & Annotation (Weeks 5–6)

**Goal**: Code-first zone/route definition alternative to visual editor

### Subphase 3a: Programmatic Zone Creation API

- [ ] T036 [US4] Implement `create_spawn_zone(name, polygon, priority=1) -> Zone` helper
  - **Acceptance**: Zone object created; validated against allowed_areas; stored in config
  - **File**: `robot_sf/maps/osm_zones_yaml.py` or new `robot_sf/maps/osm_zones_config.py`
  - **Test**: `tests/test_osm_zones_config.py::test_create_spawn_zone`
  - **Spec Link**: US4, FR-014

- [ ] T037 [P] [US4] Implement `create_goal_zone(name, polygon) -> Zone` helper
  - **Acceptance**: Zone object created; type='goal'; validation passes
  - **File**: `robot_sf/maps/osm_zones_config.py`
  - **Test**: `tests/test_osm_zones_config.py::test_create_goal_zone`
  - **Spec Link**: US4, FR-014

- [ ] T038 [P] [US4] Implement `create_crowded_zone(name, polygon, density) -> Zone` helper
  - **Acceptance**: Zone with density metadata stored; validation passes
  - **File**: `robot_sf/maps/osm_zones_config.py`
  - **Test**: `tests/test_osm_zones_config.py::test_create_crowded_zone`
  - **Spec Link**: US4, FR-014

- [ ] T039 [US4] Implement `create_route(name, waypoints, route_type='pedestrian') -> Route` helper
  - **Acceptance**: Route object created; waypoints in world coords; stored in config
  - **File**: `robot_sf/maps/osm_zones_config.py`
  - **Test**: `tests/test_osm_zones_config.py::test_create_route`
  - **Spec Link**: US4, FR-015

### Subphase 3b: Scenario Config Parser

- [ ] T040 [US4] Implement scenario config loader: `load_scenario_config(yaml_file) -> OSMZonesConfig`
  - **Acceptance**: Parses scenario YAML (zones + routes defined in code); returns typed config
  - **File**: `robot_sf/maps/osm_zones_config.py`
  - **Test**: `tests/test_osm_zones_config.py::test_scenario_config_loading`
  - **Spec Link**: FR-017, FR-018

- [ ] T041 [US4] Implement equivalence test: Programmatic config ≡ editor output
  - **Acceptance**: Create zones via API → save YAML; create via editor → save YAML; content identical (modulo timestamp)
  - **File**: `tests/test_osm_zones_config.py::test_programmatic_editor_equivalence`
  - **Spec Link**: US4, SC-010

---

## Phase 4: Documentation & Polish (Weeks 7–8)

**Goal**: Complete user guide, documentation, cleanup

### Subphase 4a: Documentation

- [ ] T042 Create user guide: `docs/osm_map_workflow.md`
  - **Sections**: Problem statement, quick start, detailed workflow, troubleshooting, FAQ
  - **Acceptance**: Guide complete; includes PBF acquisition, import, rendering, editor usage
  - **File**: `docs/osm_map_workflow.md`
  - **Spec Link**: SC-012 (Documentation)

- [ ] T043 [P] Update `docs/SVG_MAP_EDITOR.md`: Add new section "OSM-Based Extraction"
  - **Acceptance**: Cross-reference to osm_map_workflow.md; highlight advantages (semantic, reproducible, offline)
  - **File**: `docs/SVG_MAP_EDITOR.md`
  - **Spec Link**: SC-012

- [ ] T044 [P] Update `docs/README.md`: Add entry to map generation section
  - **Acceptance**: New section added with links to osm_map_workflow.md and examples
  - **File**: `docs/README.md`
  - **Spec Link**: SC-012

### Subphase 4b: Final Validation & Cleanup

- [ ] T045 Run full test suite: All Phase 1–3 tests passing; coverage ≥85% for new modules
  - **Acceptance**: `uv run pytest tests -n auto` passes; coverage report confirms targets
  - **File**: CI pipeline
  - **Spec Link**: SC-001 (Test coverage)

- [ ] T046 [P] Lint & type check: `uv run ruff check .` and `uvx ty check . --exit-zero` pass
  - **Acceptance**: No new linting or type errors introduced
  - **File**: All new modules
  - **Spec Link**: Constitution (Principle XI, XII)

- [ ] T047 Run example scripts end-to-end: All examples runnable, produce expected outputs
  - **Acceptance**: `examples/osm_map_quickstart.py`, `examples/osm_map_editor_demo.py` run without errors
  - **Spec Link**: SC-002 (Demo availability)

- [ ] T048 Performance validation: PBF import <2s (small fixture), rendering <1s, editor responsive
  - **Acceptance**: Benchmarks meet targets; documented in performance notes if applicable
  - **File**: Tests or `scripts/validation/`
  - **Spec Link**: SC-003 (Performance)

- [ ] T049 Final backward-compat check: Full train/eval cycle with both OSM-derived and legacy maps
  - **Acceptance**: Existing workflows completely unaffected; no breakage
  - **File**: `tests/test_osm_backward_compat.py`
  - **Spec Link**: SC-009, Constitution (Principle VII)

---

## Dependency Graph & Parallel Opportunities

### Critical Path (Sequential)

```
T001 (fixture) 
  ↓
T005 → T006 → T007,T008,T009,T010,T011 → T012 → T013 (core importer)
  ↓
T014,T015 (MapDefinition)
  ↓
T016,T017 (rendering)
  ↓
T019,T020,T021,T022 (validation)
```

### Parallel Opportunities

**Phase 1 Parallelization**:
- T003 (deps), T002 (fixtures) — parallel with T005–T012
- T007, T008, T010, T011 — all can run in parallel (independent functions)
- T016, T017 — parallel with T012 (different modules)
- T019, T020, T021 — all parallel (independent validation)

**Phase 2 Parallelization**:
- T022–T025 (YAML) — parallel with T026–T033 (editor)
- T027–T033 — can be parallelized (different features)

**Phase 3 Parallelization**:
- T036–T039 (API) — parallel implementation
- T040–T041 — can start after API basics done

**Estimated Parallel Speedup**: ~2–2.5x with aggressive parallelization

---

## Acceptance Criteria Summary

| Task Range | Focus | Key Criteria |
|-----------|-------|--------------|
| T001–T004 | Setup | Fixtures exist, deps installed, modules importable |
| T005–T013 | Importer | PBF → MapDef works; obstacles correct; allowed_areas populated |
| T014–T015 | MapDef update | Field added; backward-compat verified |
| T016–T017 | Rendering | PNG created; affine transform round-trip ±1px |
| T018–T021 | Examples & validation | Scripts run; existing tests pass; no breakage |
| T022–T025 | YAML | Schema defined; deterministic round-trip; validation works |
| T026–T033 | Editor | Click handlers work; snapping/validation/undo/redo functional |
| T034–T035 | Integration | Example demo works; full cycle backward-compat |
| T036–T041 | Programmatic | API helpers work; scenario parser works; equivalence verified |
| T042–T049 | Documentation & QA | Docs complete; tests pass; lint/type clean; perf validated |

---

## Per-Phase Done Criteria

### Phase 1 Done
- [ ] T001–T021 all complete
- [ ] Core importer fully functional (PBF → MapDef)
- [ ] Background rendering working (PNG + affine transform)
- [ ] MapDefinition.allowed_areas field added & backward-compatible
- [ ] All new tests passing (85%+ coverage)
- [ ] Backward-compat smoke tests passing (pygame, sensors, planners)
- [ ] Example script runnable end-to-end
- [ ] No lint/type errors introduced

### Phase 2 Done
- [ ] T022–T035 all complete
- [ ] YAML schema v1.0 finalized (deterministic)
- [ ] Visual editor fully functional (draw, edit, save)
- [ ] Full backward-compat cycle validated (train/eval unchanged)
- [ ] Editor example demo working
- [ ] All tests passing

### Phase 3 Done
- [ ] T036–T041 all complete
- [ ] Programmatic API finalized
- [ ] Scenario config parser working
- [ ] Equivalence tests passing (programmatic ≡ editor)

### Phase 4 Done
- [ ] T042–T049 all complete
- [ ] User guide published (osm_map_workflow.md)
- [ ] Central docs updated (README.md, SVG_MAP_EDITOR.md)
- [ ] Full test suite passing (85%+ coverage)
- [ ] Lint/type checks clean
- [ ] Performance targets met
- [ ] All examples runnable

---

## Task Status Tracking

Use this format to track progress:

```markdown
### Phase 1 Progress
- [x] T001 — Fixture prep
- [x] T002 — Expected outputs
- [ ] T003 — Dependencies
- [ ] T004 — Module skeleton
- [ ] T005 — TagFilters
...
```

---

## Notes for Implementers

1. **Start with T001–T004**: Get fixtures and infrastructure ready before diving into code.
2. **Test early and often**: Each task should have at least a unit test before moving to next task.
3. **Backward-compat priority**: T019–T021 validation is critical; don't defer.
4. **Parallel where possible**: Use parallel markers [P] to parallelize within teams.
5. **Visual validation**: T016–T017 and T026–T033 benefit from manual inspection; take screenshots.
6. **Documentation**: Write docstrings as you code; Phase 4 is about aggregating, not discovering.
7. **Performance**: Profile early (T009, T016) to catch bottlenecks before scaling up.

---

**Ready to implement?** Begin with Phase 1, Task T001. Good luck! 🚀
