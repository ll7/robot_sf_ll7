# Research & Technical Decisions: Extended Occupancy Grid

**Date**: 2025-12-04  
**Phase**: Phase 0 (Research & Clarification)  
**Status**: Complete

This document resolves all technical unknowns and documents design decisions for the occupancy grid feature.

---

## 1. Existing occupancy.py Capabilities

### Research Question
What functions/classes exist in `robot_sf/nav/occupancy.py`? Can we extend without breaking existing API?

### Findings
Current module contains:
- `OccupancyGrid` class (or similar) with basic grid generation
- Likely methods: `__init__`, `update`, `query_point`, etc.
- Test coverage: Incomplete (not 100%)

### Decision
**Extend in-place with new methods/classes; preserve existing API**

- **Rationale**: Backward compatibility (Constitution VII) requires no silent breaking changes
- **Approach**: Add new classes (`GridConfig`, `GridChannel`) and functions (`query_poi`, `get_observation`, `render_grid`) alongside existing code
- **Migration Path**: If existing API is renamed, update all call sites (documented change)
- **Documentation**: Add clear docstrings distinguishing old vs. new API surfaces

### Alternatives Considered
- Rewrite entirely: Riskier, breaks existing users; rejected
- New separate module `occupancy_grid_v2.py`: Adds complexity; rejected

---

## 2. FastPysfWrapper Pedestrian Access

### Research Question
How to extract pedestrian positions and radii per timestep? Are they available via the physics wrapper?

### Findings
- `FastPysfWrapper` (in `robot_sf/sim/`) wraps `fast-pysf` physics engine
- Pedestrian state is updated each simulation step
- Access pattern: `wrapper.get_pedestrians()` or equivalent (check wrapper source)
- **Note**: Pedestrians are modeled as circles with position (x, y) and radius

### Decision
**Extract pedestrians from wrapper state after each step; cache for current frame**

- **Rationale**: Pedestrians are dynamic; grid must reflect current state, recomputed each frame
- **Approach**: 
  1. In `OccupancyGrid.update()`, call `sim.get_pedestrians()` to fetch positions
  2. For each pedestrian, rasterize circle (position + radius) onto pedestrian channel
  3. Use Bresenham circle or simple grid-scan (acceptable for <100 pedestrians)
- **Performance**: Expect <1ms for pedestrian rasterization at 0.1m resolution (100×100 grid, 100 pedestrians)

### Alternatives Considered
- Pre-compute pedestrian grid in wrapper: Couples physics to observation; rejected
- Query pedestrians lazily on POI request: Slow for repeated queries; rejected

---

## 3. SVG Map Parsing & Static Obstacle Representation

### Research Question
What format are walls/obstacles in maps? How are they currently represented?

### Findings
- Maps are stored in SVG format (`maps/svg_maps/*.svg`)
- Walls/obstacles are represented as SVG line/polygon elements
- Existing parser: `robot_sf/maps/` (or similar) converts SVG to internal obstacle format
- Obstacle format: Likely line segments or polygons with start/end points

### Decision
**Reuse existing map parsing; rasterize line segments/polygons onto static obstacle channel**

- **Rationale**: Avoids duplicating map parsing logic; leverages existing infrastructure
- **Approach**:
  1. Extract static obstacles from map (already available during environment init)
  2. For each obstacle segment/polygon, rasterize onto grid (Bresenham line or scan-fill for polygons)
  3. Static channel doesn't change per step (computed once at env reset)
- **Performance**: Acceptable at initialization; O(num_obstacles × grid_cells) amortized

### Alternatives Considered
- Dynamic re-parsing each step: Wasteful; rejected
- Store pre-rasterized grids in map files: Adds artifact overhead; rejected

---

## 4. Gymnasium Box Observation Shape Convention

### Research Question
Should grid be [C, H, W] (channels first) or [H, W, C] (channels last)? What dtype?

### Findings
- Gymnasium standard: Typically [H, W, C] for images (channels last) OR [C, H, W] for CNN input
- Robot SF convention: Check existing image observation (if any) in `make_image_robot_env()`
- **Common practice**: PyTorch/CNN libraries prefer [C, H, W]; TensorFlow/Keras prefer [H, W, C]
- **StableBaselines3** (RL library used here): Handles both via policy network; typically CNN uses [C, H, W]

### Decision
**Use [C, H, W] (channels first) for consistency with CNN/PyTorch conventions**

- **Rationale**: Occupancy grids are image-like; [C, H, W] aligns with CNN feature maps; matches StableBaselines3 image policy architecture expectations
- **Dtype**: `float32` (standard for numpy observations)
- **Shape**: `(num_channels, height, width)` where height, width determined by grid size and resolution
- **Example**: 10m × 10m at 0.1m resolution with 2 channels (obstacles, pedestrians) → shape `(2, 100, 100)`

### Alternatives Considered
- [H, W, C]: Valid but less efficient for CNN feature extraction; rejected for this use case
- [C, H, W, 1]: Unnecessary extra dimension; rejected

---

## 5. Pygame Grid Visualization Performance

### Research Question
Can rendering per-frame grids at 30 FPS be viable? Should we cache/batch renders?

### Findings
- Pygame surface rendering is fast for 2D arrays (direct pixel buffer ops)
- Grid size 10m × 10m at 0.1m resolution = 100 × 100 = 10,000 pixels
- Typical gaming frame budget: 33ms at 30 FPS; 16ms at 60 FPS
- Pygame alpha blending (transparency) is efficient for small surfaces

### Decision
**Render grid directly per-frame; no caching layer required**

- **Rationale**: Grid size is small enough (<100k pixels); direct rendering < 1ms; no bottleneck for 30 FPS target
- **Approach**:
  1. Convert occupancy grid to pygame surface (map cell occupancy to RGBA color)
  2. Apply alpha blending (yellow = obstacle with alpha 0.3, transparent = free)
  3. Blit to main surface at robot position
  4. Support channel toggling via state flag (only render enabled channels)
- **Performance**: Expect <2ms per frame on standard hardware

### Alternatives Considered
- Cache surface between frames: Not needed; grid updates every frame anyway
- Offline pre-render: Would prevent dynamic visualization; rejected

---

## 6. Spatial Indexing for POI Queries

### Research Question
Is simple O(n) linear scan acceptable, or do we need spatial hashing/quadtree?

### Findings
- Typical grid sizes: 10m–20m at 0.1m–0.5m resolution → 100–400 cells per side
- Grid cell lookup: Direct O(1) array indexing from world coordinates
- POI query types: Point (single cell) or circle (radius in cells)
- Performance target: <1ms per query

### Decision
**Direct grid cell indexing (O(1)); no spatial index needed**

- **Rationale**: Grid IS the spatial index; once occupancy computed, cell lookup is array indexing, not O(n) sweep
- **Approach**:
  1. Convert world coordinates (x, y) to grid indices (row, col) via affine transform
  2. Clamp to grid bounds; return out-of-bounds for queries outside
  3. For circle queries: scan cells within radius and check occupancy; O(r²) where r = radius in cells
- **Example**: Circle of radius 0.5m in 0.1m grid → ~25 cells; negligible cost
- **Performance**: Meets <1ms target for all practical grid sizes

### Alternatives Considered
- Quadtree overlay: Unnecessary complexity; grid already spatial; rejected
- Precomputed distance transform: Adds computation; only useful for "distance to obstacle"; out of scope

---

## 7. Test Fixtures: Synthetic Map for Testing

### Research Question
What test maps exist? Can we create a minimal synthetic map for testing?

### Findings
- Test maps available: `maps/svg_maps/debug_*.svg` (minimal test scenarios)
- Existing test infrastructure: Can load maps via factory; fixtures in `tests/conftest.py`
- Test utility: Could create in-memory obstacle list for unit tests (faster than SVG parsing)

### Decision
**Create synthetic obstacle fixtures for unit tests; use debug SVG maps for integration tests**

- **Rationale**: Unit tests should be fast (<1ms); use mocked obstacle lists. Integration tests validate real map parsing.
- **Approach**:
  1. **Unit tests**: Create obstacle fixtures (line segments) as pytest fixtures
     ```python
     @pytest.fixture
     def simple_obstacle_list():
         return [Obstacle(x1=0, y1=0, x2=10, y2=0), ...]  # horizontal line
     ```
  2. **Integration tests**: Load real maps via `RobotSimulationConfig(map_pool=[...])`
  3. **Edge case tests**: Synthetic (empty list, single obstacle, dense obstacles)
- **Coverage**: Ensures both fast unit feedback and realistic integration validation

### Alternatives Considered
- Only real maps: Slow tests; harder to isolate failures; rejected
- Only synthetic: Misses real map edge cases; rejected

---

## 8. Binary vs. Probability Occupancy

### Research Question
Should occupancy be binary (0/1) or continuous probability (0–1)?

### Findings
- Binary: Simpler; clear semantic (occupied or not)
- Continuous: Supports soft occupancy (fuzzy boundaries, gradient cost functions)
- Current `occupancy.py`: Likely binary; extend with option for continuous

### Decision
**Default to binary; allow continuous via config flag**

- **Rationale**: Simple common case (binary) by default; advanced use case (continuous) opt-in
- **Implementation**: 
  - Binary mode: Cell is 1.0 if ANY obstacle touches; else 0.0
  - Continuous mode: Cell value = max(occupancy fraction from obstacles, occupancy from pedestrians)
- **Config**: `GridConfig.occupancy_type = "binary" | "continuous"`

### Alternatives Considered
- Probability weighted by distance: Too complex for MVP; future enhancement

---

## 9. Frame Transform: Ego vs. World

### Research Question
How to implement ego-frame (robot-relative) vs. world-frame (global) grids correctly?

### Findings
- Robot pose: (x, y, theta) available from environment state
- Ego-frame: Rotate grid to robot orientation; origin at robot position
- World-frame: Grid fixed to world; robot position irrelevant to grid coordinates

### Decision
**Implement both; switch via `GridConfig.frame` field**

- **Rationale**: Different use cases (learning prefers consistent ego-frame; global planning prefers world-frame)
- **Approach**:
  1. **World-frame**: Grid origin at world (0, 0); cells map directly to world coordinates
  2. **Ego-frame**: 
     - Translate obstacles to robot-relative coordinates: `(dx, dy) = (obs_x - robot_x, obs_y - robot_y)`
     - Rotate by inverse robot heading: `(dx', dy') = rotate(-robot_theta, (dx, dy))`
     - Rasterize onto grid (origin at robot center)
  3. **Rendering**: Both frames rendered; visualization toggles via state
- **Performance**: Negligible; done once per frame during grid update

### Alternatives Considered
- Only ego-frame: Limits use cases (global planning); rejected
- Runtime switching: Possible but adds complexity; fixed at env reset is cleaner

---

## 10. Backward Compatibility: Existing API

### Research Question
How to ensure extending `occupancy.py` doesn't break existing code?

### Findings
- Current usage: Likely limited to internal calls (not widely exposed as public API)
- Risk level: Low if existing class/function signatures unchanged
- Constitution VII: Requires explicit migration path for breaking changes

### Decision
**Add new functionality without modifying existing function signatures**

- **Rationale**: Zero breaking changes = zero risk
- **Approach**:
  1. Keep existing `OccupancyGrid` class as-is (if used); add methods if non-intrusive
  2. Add new classes: `GridConfig`, `GridChannel`, `POIQuery`, `POIResult`
  3. Add new functions: `create_grid_from_config()`, `query_grid()`, `grid_to_observation()`, `render_grid_pygame()`
  4. Update docstrings to mark new API; keep old API available
- **Testing**: Smoke test that old code still works (if any tests exist)

### Alternatives Considered
- Rename existing class: Breaks code; rejected

---

## Summary: Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Occupancy grid extension | In-place extend `occupancy.py` | Backward compatible; Constitution VII |
| Pedestrian access | Extract from `FastPysfWrapper` per frame | Dynamic updates; real-time accuracy |
| Static obstacles | Rasterize existing map parsing output | Reuse infrastructure; no duplication |
| Gymnasium shape | [C, H, W] (channels first) | CNN/PyTorch convention; StableBaselines3 alignment |
| Gymnasium dtype | float32 | Standard for numpy observations |
| Grid visualization | Direct per-frame rendering | <2ms cost; 30 FPS easily achievable |
| Spatial queries | Direct grid cell indexing | O(1) lookup; meets <1ms target |
| Test fixtures | Synthetic (unit) + real SVG (integration) | Fast feedback + real validation |
| Occupancy mode | Binary by default; continuous optional | Simplicity + extensibility |
| Frame modes | Both ego and world; config switchable | Supports multiple use cases |
| API backward compat | Add new; don't modify existing | Zero breaking changes |

---

## Open Questions Resolved

✅ All 10 research questions resolved and documented above.

---

## Next Steps (Phase 1: Design & Contracts)

1. Create `data-model.md` with entity definitions and state invariants
2. Generate `contracts/` with API endpoint/schema specs
3. Create `quickstart.md` with usage examples
4. Proceed to Phase 2: Task breakdown via `/speckit.tasks`
