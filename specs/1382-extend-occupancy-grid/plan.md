# Implementation Plan: Extended Occupancy Grid with Multi-Channel Support

**Branch**: `1382-extend-occupancy-grid` | **Date**: 2025-12-04 | **Spec**: [specs/1382-extend-occupancy-grid/spec.md](../spec.md)  
**Input**: Feature specification from `/specs/1382-extend-occupancy-grid/spec.md`

## Summary

Extend the occupancy grid module in `robot_sf/nav/occupancy.py` to support configurable multi-channel grids (static obstacles, pedestrians) with both ego-rotated and world-aligned frame modes. Integrate grids as gymnasium observation layers, provide point-of-interest query APIs for spawn validation, visualize grids in pygame, and achieve 100% test coverage. This enables RL agents to learn from grid-based representations and supports automated scenario validation.

## Technical Context

**Language/Version**: Python 3.11 (uv-managed environment; see `.venv`)  
**Primary Dependencies**: 
- `gymnasium` (environment API)
- `robot_sf.gym_env` (factory functions, unified config)
- `robot_sf.sim.FastPysfWrapper` (pedestrian physics)
- `pygame` (visualization; optional for render)
- `numpy` (grid computation)

**Storage**: File-based (configs in YAML, maps in SVG, grids computed in-memory, no persistence required)  
**Testing**: 
- Unit/integration: `pytest` (tests/ suite)
- Visual/GUI: `pytest` headless mode with `DISPLAY=`, `MPLBACKEND=Agg`, `SDL_VIDEODRIVER=dummy` (test_pygame/)
- Coverage: `coverage.py` (automatic collection)

**Target Platform**: Linux/macOS headless + interactive pygame desktop  
**Project Type**: Single library (`robot_sf/` package) with examples/scripts  
**Performance Goals**: 
- Grid generation: <5ms for 10m×10m at 0.1m resolution
- POI queries: <1ms per query
- Visualization: 30+ FPS with grid overlay
- Test coverage: 100% of `occupancy.py`

**Constraints**: 
- Planar (2D) motion only; no z-axis/height considerations
- Deterministic seed propagation for reproducibility
- O(1) or O(log N) query performance (spatial index required for large obstacle sets)
- Backward compatible with existing `occupancy.py` API (extend, don't break)

**Scale/Scope**: 
- Single module extension (~500–1000 LOC for core grid + channels)
- Up to 100 pedestrians per scene
- Grids up to 20m × 20m at variable resolution
- Integration with 2 existing subsystems (gym_env factories, sim physics wrapper)

## Constitution Check

*Gate Status: ✅ PASS*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| **I. Reproducible Core** | Grids computed deterministically from seed + config | ✅ PASS | Seed propagation through env factory; config-driven grid parameters |
| **II. Factory Abstraction** | Grid observation exposed via factory functions, not direct instantiation | ✅ PASS | Gymnasium integration via `make_robot_env(config=GridConfig(...))` |
| **III. Benchmark & Metrics** | Grids support episode schema; metrics traceability | ✅ PASS | POI queries enable spawn validation; grid data part of observation layer |
| **IV. Unified Config** | Grid parameters in unified config layer, not ad-hoc kwargs | ✅ PASS | `GridConfig` (or inline in `RobotSimulationConfig`) specifies all grid tuning |
| **V. Minimal Baselines** | Not applicable (grids are observation layer, not baseline algorithm) | ✅ N/A | Feature supports baselines via gym API |
| **VI. Metrics Transparency** | Grid generation and query results traceable; no opaque magic values | ✅ PASS | POI query returns explicit occupied/free status; grid metadata (size, resolution) retained |
| **VII. Backward Compatibility** | Extend `occupancy.py` without breaking existing API | ✅ PASS | New functions/classes added; old API paths preserved if in use |
| **VIII. Documentation as API** | Public grid factory, POI query API, visualization must have README entries | ✅ PASS | Success criteria SC-007 mandates update to `docs/dev/occupancy/` with usage examples |
| **IX. Test Coverage** | Public grid behavior covered by smoke + unit tests | ✅ PASS | User Story 5 + SC-001 require 100% coverage of `occupancy.py` |
| **X. Scope Discipline** | Grid-focused; no general image processing, arbitrary geometry libraries | ✅ PASS | Out-of-scope excludes 3D grids, serialization, external mapping systems |
| **XI. Library Reuse** | Core grid logic in `robot_sf/nav/occupancy.py`, not duplicated in scripts | ✅ PASS | Examples/scripts orchestrate factory functions; no bespoke grid code |
| **XII. Loguru Logging** | All library code using Loguru, not print() (except justified CLI) | ✅ PASS | Grid module uses structured logging; visualization debug messages via Loguru |
| **XIII. Test Value** | All new tests documented with purpose; low-value tests not accumulate | ✅ PASS | 100% coverage + edge case tests; docstrings explain each test's role |

**Conclusion**: Feature aligns with all constitution principles. Proceed to Phase 0 research.

## Project Structure

### Documentation (this feature)

```text
specs/1382-extend-occupancy-grid/
├── spec.md                      # Feature specification ✅ (complete)
├── plan.md                      # This file (Phase 0/1 output)
├── research.md                  # Phase 0 output (resolve unknowns)
├── data-model.md                # Phase 1 output (entity/contract design)
├── quickstart.md                # Phase 1 output (getting started guide)
├── contracts/                   # Phase 1 output (API schemas if applicable)
└── checklists/
    └── requirements.md          # Quality validation ✅ (passed)
```

### Source Code (repository)

```text
robot_sf/
├── nav/
│   └── occupancy.py             # EXTEND THIS: Core grid logic, POI queries
├── gym_env/
│   ├── environment_factory.py   # ADD: GridObs support in make_robot_env()
│   └── unified_config.py        # ADD: GridConfig class or grid fields
├── render/
│   └── sim_view.py              # EXTEND: Grid visualization overlay + channel toggle
└── common/
    └── types.py                 # MAY ADD: Grid-specific type aliases (GridPoint, GridCell, etc.)

tests/
├── test_occupancy_grid.py       # ADD: Core grid generation, channels, frame modes
├── test_occupancy_queries.py    # ADD: POI/AOI query tests, spawn validation
├── test_occupancy_edge_cases.py # ADD: Empty grids, boundary conditions, extreme resolutions
└── test_occupancy_gymnasium.py  # ADD: Observation space integration tests

test_pygame/
└── test_occupancy_visualization.py  # ADD: Visual tests for grid overlay, channel toggling

examples/
└── (existing)                   # MAY ADD: Example script showing grid visualization

docs/dev/occupancy/
├── Update_or_extend_occupancy.md # REWRITE/EXTEND: Updated with grid usage, integration, extension guide
└── (other occupancy docs)
```

**Structure Decision**: Extend existing `robot_sf/nav/occupancy.py` with new classes and functions for multi-channel grids. Add configuration in `unified_config.py` or as a standalone `GridConfig` class. Integrate visualization in `sim_view.py` with a conditional grid render layer. Tests split by concern (generation, queries, edge cases, gymnasium integration, visualization).

## Phase 0: Research & Unknowns

### Known Technical Decisions
1. **Grid Representation**: Binary/probability occupancy (0.0 = free, 1.0 = occupied; or continuous 0–1)
2. **Frame Transform**: Standard 2D rotation matrix (ego-frame via cos/sin, world-frame direct mapping)
3. **Pedestrian Model**: Circle-based occupancy (position + radius from FastPysfWrapper)
4. **Static Obstacles**: Line segments or polygon boundaries (from map SVG parsing, already in codebase)
5. **Gymnasium Integration**: Box observation space for grid arrays (shape: [num_channels, height, width], dtype: float32)
6. **Visualization**: Pygame surface with alpha blending (yellow tint for obstacles, transparent for free)
7. **POI Query Semantics**: Conservative (cell occupied if ANY obstacle/pedestrian touches it)

### Research Tasks (Phase 0 Output: research.md)

1. **Existing occupancy.py capabilities**: What functions/classes exist? Can we extend without breaking?
2. **FastPysfWrapper pedestrian access**: How to extract pedestrian positions and radii? Are they available per-timestep?
3. **SVG map parsing for static obstacles**: What format are walls/obstacles in? How are they currently represented?
4. **Gymnasium Box observation shape conventions**: Should grid be [C, H, W] or [H, W, C]? What dtype?
5. **Pygame alpha blending performance**: Is renderingper-frame grids at 30 FPS viable? Should we cache/batch?
6. **Spatial indexing for queries**: Is a simple O(n) linear scan acceptable, or do we need spatial hashing/quadtree?
7. **Test fixture: map with known obstacles**: What test maps exist? Can we create a minimal synthetic map for testing?

**Deliverable**: `research.md` documenting each question, decision rationale, and alternatives considered.

---

## Phase 1: Design & Contracts

### 1. Data Model (`data-model.md`)

**Entities**:
- **OccupancyGrid**: Main grid (2D numpy array or dict of channels)
  - Fields: size_m (tuple), resolution_m (float), frame (str: "ego" or "world"), timestamp (float), channels (dict)
  - Invariants: All channels same shape (height, width); size > 0; resolution > 0
  
- **GridChannel**: Named occupancy layer
  - Fields: name (str), data (2D numpy array float32), occupancy_type (str: "binary" or "continuous")
  - Invariants: 0 ≤ data ≤ 1.0
  
- **GridConfig**: Configuration
  - Fields: size_m, resolution_m, frame, enabled_channels (list), include_pedestrians (bool), include_obstacles (bool)
  
- **POIQuery**: Query input
  - Fields: world_x, world_y, region_type ("point" or "circle"), radius_m (if circle)
  
- **POIResult**: Query output
  - Fields: is_occupied (bool), occupancy_value (float), channel_values (dict), is_within_bounds (bool)

### 2. API Contracts (`contracts/`)

**Core Functions** (pseudocode):

```python
def create_occupancy_grid(
    robot_pose: RobotPose,
    config: GridConfig,
    obstacles: List[Obstacle],  # from map
    pedestrians: List[Pedestrian]  # from sim
) -> OccupancyGrid

def query_occupancy(grid: OccupancyGrid, query: POIQuery) -> POIResult

def get_observation(grid: OccupancyGrid) -> np.ndarray  # for gymnasium

def render_grid(grid: OccupancyGrid, surface: pygame.Surface, robot_pose: RobotPose)

def toggle_grid_channel(enabled_channels: List[str]) -> None  # pygame visualization state
```

**Gymnasium Integration**:
```python
# In unified_config.py or environment_factory.py:
class GridObservation(dict):
    """Occupancy grid as gymnasium observation component"""
    grid: OccupancyGrid
    array: np.ndarray  # flattened for Box space

# Usage:
env = make_robot_env(config=RobotSimulationConfig(use_occupancy_grid=True, grid_config=GridConfig(...)))
obs = env.reset()  # obs["occupancy_grid"] is np.ndarray
```

### 3. Quickstart (`quickstart.md`)

Example snippet:
```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig, GridConfig

config = RobotSimulationConfig(
    use_occupancy_grid=True,
    grid_config=GridConfig(size_m=10.0, resolution_m=0.1, frame="ego", include_pedestrians=True)
)
env = make_robot_env(config=config, debug=True)
obs = env.reset(seed=42)
print(obs["occupancy_grid"].shape)  # (2, 100, 100) -> [channels, height, width]
```

### 4. Agent Context Update

Run update-agent-context script (if applicable) to register new symbols and grid types in Copilot context.

---

## Complexity Tracking

**Complexity Level**: MODERATE

- Core logic: grid rasterization (standard 2D scan), frame transforms (2D rotation matrix)
- Integration points: gymnasium API, pygame rendering, unified_config schema
- Test volume: ~100 test cases across 4 test files (generation, queries, edge cases, visualization)
- Performance-sensitive code: grid generation loop (should be <5ms)

**Justification**: Feature is well-scoped (single module extension), no algorithm research required, and existing codebase provides most dependencies (pygame, gymnasium, numpy). Constitution gates all pass.

---

## Next Steps

1. **Phase 0** (this command continuation): Generate `research.md` with all unknowns resolved
2. **Phase 1** (continuation): Generate `data-model.md`, `quickstart.md`, and populate `contracts/`
3. **Phase 2** (separate command `/speckit.tasks`): Generate detailed task breakdown with story mapping, dependencies, and sprint-size estimates
4. **Implementation**: Teams pick up tasks from Phase 2 output; follow dev_guide.md quality gates and Constitution principles

---

**Status**: ✅ Plan Phase 0–1 ready to proceed. No blockers identified.
