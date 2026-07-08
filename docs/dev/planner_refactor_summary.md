# Global Planner Refactor Summary

## Overview

Successfully refactored the global planner architecture to separate two distinct planning approaches:

1. **VisibilityPlanner** - Visibility graph-based planning (formerly `GlobalPlanner`)
2. **ClassicGlobalPlanner** - Grid-based planning using python_motion_planning algorithms

## Changes Made

### New Module Structure

```
robot_sf/planner/
├── __init__.py                    # Exports both planners
├── visibility_planner.py          # Renamed from global_planner.py
├── classic_global_planner.py      # NEW - Grid-based planner
├── visualization.py               # Updated type hints
└── ... (other files unchanged)
```

### Key Files Created/Modified

1. **`robot_sf/planner/classic_global_planner.py`** (NEW)
   - Created `ClassicGlobalPlanner` class
   - Wraps python_motion_planning algorithms (ThetaStar, A*, etc.)
   - Integrates with `robot_sf.nav.motion_planning_adapter` for grid conversion
   - Provides coordinate conversion between world space and grid indices
   - Configuration via `ClassicPlannerConfig` dataclass

2. **`robot_sf/planner/visibility_planner.py`** (RENAMED from global_planner.py)
   - Renamed `GlobalPlanner` → `VisibilityPlanner`
   - Updated module docstring
   - No functional changes

3. **`robot_sf/planner/__init__.py`** (UPDATED)
   - Exports both `VisibilityPlanner` and `ClassicGlobalPlanner`
   - Maintains backwards compatibility: `GlobalPlanner = VisibilityPlanner`
   - Enhanced module docstring explaining both approaches

4. **`examples/advanced/27_motion_planning_adapter_test.py`** (REFACTORED)
   - Now uses `ClassicGlobalPlanner` instead of direct ThetaStar usage
   - Demonstrates proper planner API usage
   - Simplified: map conversion happens inside planner

5. **`examples/advanced/28_planner_comparison.py`** (NEW)
   - Side-by-side comparison of both planners
   - Shows performance and path characteristics
   - Educational demo for choosing the right planner

### Updated Imports

The following files were updated to use the new module structure:
- `robot_sf/planner/visualization.py` - Type hints updated to `VisibilityPlanner`
- `tests/test_planner/test_map_integration.py` - Import from package level
- `examples/advanced/24_planner_bottleneck_test.py` - Import from package level
- `examples/advanced/25_planner_diagnostic.py` - Import from package level

## Backwards Compatibility

**All existing code continues to work without changes!**

```python
# Old code (still works)
from robot_sf.planner import GlobalPlanner, PlannerConfig
planner = GlobalPlanner(map_def)

# New code (recommended)
from robot_sf.planner import VisibilityPlanner, PlannerConfig
planner = VisibilityPlanner(map_def)
```

The alias `GlobalPlanner = VisibilityPlanner` ensures no breaking changes.

## Planner Comparison

### VisibilityPlanner
- **Approach**: Visibility graph construction from obstacle corners
- **Representation**: Continuous vector-based coordinates
- **Strengths**: 
  - Fast for sparse environments
  - Optimal paths with few waypoints
  - Good for clear line-of-sight scenarios
- **Limitations**: 
  - May struggle with narrow passages
  - Requires well-defined obstacle corners

### ClassicGlobalPlanner
- **Approach**: Grid-based search with ThetaStar/A* algorithms
- **Representation**: Rasterized grid
- **Strengths**:
  - Better for dense/complex environments
  - Handles narrow passages well
  - Configurable resolution and obstacle inflation
  - Any-angle paths (ThetaStar)
- **Limitations**:
  - Higher memory usage
  - Path may have more waypoints
  - Resolution-dependent performance

## Usage Examples

### VisibilityPlanner (for open spaces)

```python
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import VisibilityPlanner, PlannerConfig

map_def = convert_map("maps/svg_maps/example.svg")
planner = VisibilityPlanner(
    map_def,
    PlannerConfig(
        robot_radius=0.4,
        min_safe_clearance=0.3,
        enable_smoothing=True,
    )
)
path = planner.plan(start=(5.0, 5.0), goal=(45.0, 25.0))
```

### ClassicGlobalPlanner (for complex environments)

```python
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig

map_def = convert_map("maps/svg_maps/example.svg")
planner = ClassicGlobalPlanner(
    map_def,
    ClassicPlannerConfig(
        cells_per_meter=2.0,
        inflate_radius_cells=2,
        algorithm="theta_star",
    )
)
path = planner.plan(start=(5.0, 5.0), goal=(45.0, 25.0))
```

## Testing

All tests pass successfully:
- ✅ 1264 tests passed (full suite)
- ✅ 37 planner-specific tests passed
- ✅ Backwards compatibility verified
- ✅ Both planners work on example maps

## Migration Path

No migration needed! The refactor is fully backwards compatible.

If you want to use the new explicit names:
1. Replace `GlobalPlanner` → `VisibilityPlanner` (optional, for clarity)
2. Consider using `ClassicGlobalPlanner` for grid-based planning needs

## Future Work

Potential enhancements:
- Add more algorithms to ClassicGlobalPlanner (A*, RRT*, etc.)
- Hybrid planner that combines both approaches
- Performance benchmarking suite
- Visual comparison tool for path quality metrics

## Files Summary

**Created:**
- `robot_sf/planner/classic_global_planner.py`
- `examples/advanced/28_planner_comparison.py`

**Renamed:**
- `robot_sf/planner/global_planner.py` → `robot_sf/planner/visibility_planner.py`

**Modified:**
- `robot_sf/planner/__init__.py`
- `robot_sf/planner/visualization.py`
- `examples/advanced/27_motion_planning_adapter_test.py`
- `examples/advanced/24_planner_bottleneck_test.py`
- `examples/advanced/25_planner_diagnostic.py`
- `tests/test_planner/test_map_integration.py`

All changes maintain backwards compatibility and pass all existing tests.
