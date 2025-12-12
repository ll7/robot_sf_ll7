# Implementation Plan: SVG-Based Global Planner

**Branch**: `342-svg-global-planner` | **Date**: 2025-12-10 | **Spec**: [spec.md](./spec.md) | **Design**: [global-planner-v2.md](./global-planner-v2.md)
**Input**: Feature specification from `/specs/342-svg-global-planner/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement an automated global planner that generates collision-free waypoint paths from spawn zones to goal zones around static obstacles. Uses visibility graph approach (pyvisgraph library) for optimal shortest paths, with configurable clearance margins and path smoothing. Replaces manual waypoint definition in SVG maps while maintaining compatibility with existing RouteNavigator and GlobalRoute infrastructure. Target performance: <100ms path generation, <500ms graph build for typical maps (≤50 obstacles).

## Technical Context

**Language/Version**: Python 3.11+ (matches existing robot_sf codebase)  
**Primary Dependencies**: 
  - shapely>=2.1.2 (polygon operations, obstacle buffering - already in use)
  - networkx>=3.6 (graph search algorithms - Dijkstra, A*)
  - pyvisgraph>=0.2.1 (visibility graph construction)
  - Existing: robot_sf.nav (svg_map_parser, obstacle, navigation, occupancy_grid)
  
**Storage**: In-memory graph caching (per map hash), no persistent storage required  
**Testing**: pytest (unit tests for planner, integration tests for map compatibility)  
**Target Platform**: Linux/macOS (research workstations, headless CI)  
**Project Type**: Single library extension (new robot_sf/planner/ module)  
**Performance Goals**: 
  - Path generation: <100ms median @ 50 obstacles
  - Graph build: <500ms one-time @ 50 obstacles
  - POI sampling: <50ms for 20 POIs
  
**Constraints**: 
  - Maintains backward compatibility with existing RouteNavigator API
  - Must produce list[Vec2D] output format
  - Deterministic paths (seeded randomness)
  - Clearance margin: robot_radius + 0.3m default
  
**Scale/Scope**: 
  - Target maps: <50 obstacles typical, <100 max tested
  - Graph nodes: ~200-300 vertices typical (obstacle corners + start/goal)
  - Cache size: ~10 maps concurrent in typical training session

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Reproducible Social Navigation Research Core
✅ **PASS** - Planner generates deterministic paths given seeded random sampling. Supports factory-based environment creation pattern. Output (waypoints) is reproducible from versioned configs + seeds.

### Principle II: Factory-Based Environment Abstraction
✅ **PASS** - Integrates with existing factory functions (make_robot_env, etc.). Does not require direct environment instantiation. Planner invoked via navigation.sample_route() or new GlobalPlanner API.

### Principle III: Benchmark & Metrics First
✅ **PASS** - Planner output (waypoint paths) compatible with existing benchmark runner via RouteNavigator. No new metric schema changes. Enables automated route generation for benchmark scenarios.

### Principle IV: Unified Configuration & Deterministic Seeds
✅ **PASS** - Introduces PlannerConfig dataclass with explicit defaults. Seeds propagate for POI sampling and zone sampling. No ad-hoc kwargs.

### Principle V: Minimal, Documented Baselines
✅ **PASS** - Planner is not a baseline itself but enables existing baselines (e.g., SocialForce in navigation.py) to use auto-generated routes. Does not introduce new baseline planners.

### Principle VI: Metrics Transparency & Statistical Rigor
✅ **PASS** - No new metrics introduced. Uses existing episode metrics. Path quality implicitly measured through existing collision/comfort metrics.

### Principle VII: Backward Compatibility & Evolution Gates
✅ **PASS** - Maintains list[Vec2D] output format. Extends MapDefinition with optional poi_positions field (additive change). Existing manual routes continue working (opt-in planner via config flag).

### Principle VIII: Documentation as an API Surface
✅ **PASS** - Will add documentation to docs/README.md. Public API (GlobalPlanner, PlannerConfig) will have Google-style docstrings. Migration guide in global-planner-v2.md Phase 1-3.

### Principle IX: Test Coverage for Public Behavior
✅ **PASS** - Test plan includes smoke tests (simple corridor, clearance verification), unit tests (planner, graph builder, smoother), integration tests (all example maps, RouteNavigator compatibility).

### Principle X: Scope Discipline
✅ **PASS** - Strictly focused on static obstacle path planning for social navigation. No general robotics, perception models, or unrelated RL algorithms. Excludes dynamic obstacle avoidance, multi-robot coordination, 3D navigation (explicitly out-of-scope).

### Principle XI: Library Reuse & Helper Documentation
✅ **PASS** - New functionality in robot_sf/planner/ library module. Reuses existing robot_sf.nav components (svg_map_parser, obstacle, navigation). Will include docstrings for non-trivial helpers (graph caching, obstacle inflation, path smoothing).

### Principle XII: Preferred Logging & Observability
✅ **PASS** - Will use Loguru for all library logging. No print() in robot_sf/planner/ modules except for potential CLI debug scripts. Appropriate log levels: WARNING for narrow passage detection, ERROR for planning failures.

### Principle XIII: Test Value Verification & Maintenance Discipline
✅ **PASS** - Test plan follows core coverage priorities: (1) public API contracts (GlobalPlanner.plan()), (2) schema compliance (list[Vec2D] output), (3) integration with RouteNavigator. Tests verify actual functional requirements, not implementation details.

**Overall Status**: ✅ ALL GATES PASS - No violations. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
robot_sf/
├── planner/                    # NEW: Global planning module
│   ├── __init__.py            # Public exports: GlobalPlanner, PlannerConfig
│   ├── global_planner.py      # Main GlobalPlanner class with plan() method
│   ├── visibility_graph.py    # Visibility graph construction & caching
│   ├── poi_sampler.py         # POI generation strategies
│   └── path_smoother.py       # Douglas-Peucker simplification + spline fitting
│
├── nav/                        # MODIFIED: Extend existing navigation
│   ├── svg_map_parser.py      # EXTEND: Parse <circle class="poi"> elements
│   ├── map_config.py          # EXTEND: Add poi_positions to MapDefinition
│   ├── navigation.py          # EXTEND: sample_route() delegates to planner
│   ├── obstacle.py            # Reuse: obstacle representations
│   ├── global_route.py        # Reuse: route container
│   └── occupancy_grid.py      # Reference: Shapely usage patterns
│
├── common/
│   └── types.py               # Reuse: Vec2D type alias
│
└── gym_env/
    └── unified_config.py      # EXTEND: Add use_planner flag to configs

tests/
├── test_planner/              # NEW: Planner-specific tests
│   ├── __init__.py
│   ├── test_global_planner.py        # Unit: path generation, clearance
│   ├── test_visibility_graph.py      # Unit: graph construction
│   ├── test_path_smoother.py         # Unit: smoothing algorithms
│   ├── test_poi_sampler.py           # Unit: POI generation
│   └── test_map_integration.py       # Integration: all example maps
│
├── test_nav/                  # EXISTING: Navigation tests
│   └── test_navigation.py     # MODIFY: Add planner integration tests
│
└── fixtures/
    └── test_maps/             # NEW: Sample maps for testing
        ├── simple_corridor.svg
        ├── narrow_passage.svg
        ├── no_path.svg
        └── complex_warehouse.svg

maps/svg_maps/                 # EXISTING: Real maps
└── *.svg                      # MODIFY: Add POI annotations to select maps

examples/
└── advanced/                  # NEW: Planner demonstration
    └── 20_global_planner_demo.py

scripts/
├── validation/
│   └── verify_planner.sh      # NEW: Smoke test for planner
└── benchmark_planner.py       # NEW: Performance profiling script
```

**Structure Decision**: Single library extension within existing robot_sf package. Follows principle of minimal surface expansion - adds one new subpackage (robot_sf/planner/) and extends existing components (svg_map_parser, MapDefinition, navigation.py) to integrate planning capability. No new top-level directories. Test structure mirrors source layout under tests/test_planner/.

## Complexity Tracking

> No violations detected. All Constitution principles pass.

---

## Phase Completion Status

### Phase 0: Outline & Research ✅ COMPLETE

**Status**: Complete (2025-01-10)

**Key Decisions**:
- Primary algorithm: Visibility graph via pyvisgraph library
- Obstacle handling: Shapely buffering (robot_radius + clearance_margin)
- Path smoothing: Douglas-Peucker simplification
- Graph caching: Per-map-hash in-memory cache
- SVG extension: `<circle class="poi">` format for waypoints
- Integration approach: Opt-in via RobotSimulationConfig.use_planner flag
- Error handling: PlanningFailedError with diagnostic information
- Performance targets: <100ms path queries (warm cache), <500ms (cold) @ 50 obstacles

**Artifacts**:
- ✅ `research.md` - All technical decisions documented with rationale
- ✅ All unknowns from Technical Context resolved

---

### Phase 1: Design & Contracts ✅ COMPLETE

**Status**: Complete (2025-01-10)

**Core Entities Defined**:
- `GlobalPlanner` - Main entry point with plan() and plan_multi_goal() methods
- `PlannerConfig` - Configuration dataclass (robot_radius, clearance, smoothing, caching)
- `VisibilityGraph` - Internal wrapper around pyvisgraph (protocol interface)
- `PlanningFailedError` - Typed exception with diagnostic attributes
- `POISampler` - Utility for random/nearest/farthest POI selection
- `MapDefinition` extensions - Added poi_positions and poi_labels fields

**API Contracts**:
- Performance guarantees: First call ≤500ms, cached ≤100ms @ 50 obstacles
- Exception safety: Strong guarantee (no state mutation on failure)
- Backward compatibility: Existing code without POIs continues working
- Thread safety: Read-safe after construction, cache invalidation not concurrent-safe
- Validation: All config constraints enforced in __post_init__

**Artifacts**:
- ✅ `data-model.md` - Complete entity definitions with validation rules (12 sections)
- ✅ `contracts/global_planner_api.md` - Full API specification with examples (12 sections)
- ✅ `quickstart.md` - 10-minute onboarding guide with code samples (10 sections)

---

### Phase 2: Task Breakdown ⏳ PENDING

**Status**: Not Started

**Next Action**: Run `/speckit.tasks` command to generate implementation tasks

**Expected Output**: `tasks.md` file with breakdown of:
1. Module scaffolding (robot_sf/planner/ structure)
2. pyvisgraph integration (visibility graph wrapper)
3. SVG parser extension (POI parsing logic)
4. Path smoothing implementation (Douglas-Peucker)
5. Configuration extension (unified_config.py changes)
6. Integration with navigation (sample_route delegation)
7. Unit test suite (90%+ coverage target)
8. Integration tests (all example maps)
9. Performance benchmarks (validation script)
10. Documentation updates (API docs, examples)

**Estimated Effort**: 3-4 weeks (1 FTE developer)

---

### Phase 3: Implementation ⏳ PENDING

**Status**: Not Started (awaits Phase 2 task breakdown)

**Approach**: Iterative development following Constitution Principle I (thin vertical slice first)

**Recommended Order**:
1. Week 1: Core planner + visibility graph (basic path generation working)
2. Week 2: SVG integration + POI support (factory integration complete)
3. Week 3: Path smoothing + caching (performance targets met)
4. Week 4: Testing + documentation (quality gates pass)

---

## Next Steps

**Immediate**: Run `/speckit.tasks` to generate detailed task breakdown in `tasks.md`

**After Task Generation**: Begin Phase 3 implementation following the generated task sequence
