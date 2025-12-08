# Circle Rasterization Fix: Handle Circles Outside Grid Bounds

## Problem Statement

The `rasterize_circle` function in `robot_sf/nav/occupancy_grid_rasterization.py` had a logic error where circles with centers outside the grid bounds were incorrectly skipped, even when the circles geometrically overlapped the grid.

### Original Bug
```python
# Lines 178-181 (before fix)
if not is_within_grid(center[0], center[1], config, grid_origin_x, grid_origin_y):
    logger.warning(f"Circle center {center} outside grid bounds, skipping")
    return
```

This manual bounds check would skip circles like:
- Circle at (10.5, 5.0) with radius 1.0 (center outside but overlaps grid [0, 10] × [0, 10])
- Circle at (-0.5, 5.0) with radius 1.0 (center left of grid but overlaps)
- Circle at (10.7, 10.7) with radius 1.0 (corner case: overlaps grid corner)

## Solution Overview

The fix involved two key changes:

### 1. Remove Manual Bounds Check
Removed the manual center-in-grid check from `rasterize_circle` and delegated overlap detection to `get_affected_cells`.

### 2. Fix `get_affected_cells` Implementation
Updated `get_affected_cells` in `robot_sf/nav/occupancy_grid_utils.py` to:

1. **Clamp center for iteration**: When the circle center is outside grid bounds, clamp it to valid grid coordinates to find a starting cell for iteration
2. **Use proper circle-rectangle intersection**: Instead of checking distance from circle center to cell center (discrete disk), check if circle overlaps the cell's square area using the closest point algorithm

```python
# Proper circle-rectangle intersection
closest_x = np.clip(world_x, cell_min_x, cell_max_x)
closest_y = np.clip(world_y, cell_min_y, cell_max_y)
dist = np.sqrt((closest_x - world_x) ** 2 + (closest_y - world_y) ** 2)
if dist <= radius:
    affected.append((row, col))
```

## Implementation Details

### Files Modified
1. **robot_sf/nav/occupancy_grid_rasterization.py**
   - Removed lines 178-181 (manual bounds check)
   - Now relies entirely on `get_affected_cells` for overlap detection

2. **robot_sf/nav/occupancy_grid_utils.py**
   - `get_affected_cells()`: Added center clamping logic
   - `get_affected_cells()`: Replaced center-to-center distance with circle-rectangle intersection

### Testing Strategy
Created comprehensive test suite in `tests/test_occupancy_circle_overlap.py` with 20 test cases:

**Test Categories:**
1. **Outside center, overlapping** (4 tests): Edge cases, corners
2. **Outside center, no overlap** (2 tests): Far from grid
3. **Inside center** (1 test): Normal rasterization
4. **Parametrized boundary detection** (13 tests):
   - Edge overlaps (4 tests)
   - Corner overlaps (2 tests)
   - Far outside (5 tests)
   - Barely touching (1 test)
   - Corner overlap (1 test)

**Key Test Cases:**
- Circle at (10.7, 10.7) radius 1.0 → Should overlap corner cell at (99, 99)
- Circle at (10.95, 5.0) radius 1.0 → Should barely touch right edge
- Circle at (15.0, 5.0) radius 1.0 → Should NOT overlap (too far)

## Impact Analysis

### What Changed
- **Behavior**: Circles with centers outside grid now correctly rasterize if they overlap grid area
- **Performance**: Negligible impact (same iteration complexity, slightly more computation per cell)
- **API**: No breaking changes (internal implementation only)

### Who Benefits
- **Global planner**: Can now properly handle obstacles near grid boundaries
- **Path planning**: More accurate occupancy representation at boundaries
- **Simulation**: Pedestrians/obstacles near boundaries now correctly affect grid

## Validation

All occupancy tests pass:
```bash
uv run pytest tests/test_occupancy*.py
# Result: 114 passed, 1 skipped
```

New tests specifically validate the fix:
```bash
uv run pytest tests/test_occupancy_circle_overlap.py
# Result: 20 passed
```

## Related Issues

This fix was discovered during investigation of the global SVG planner implementation. Proper boundary handling is critical for path planning in environments where obstacles may extend beyond the local grid.

## Future Considerations

- The circle-rectangle intersection is now geometrically correct
- Consider similar fixes for `rasterize_line` if boundary issues arise
- Performance optimization opportunity: precompute cell bounds for hot paths

## References

- Test file: `tests/test_occupancy_circle_overlap.py`
- Modified files:
  - `robot_sf/nav/occupancy_grid_rasterization.py`
  - `robot_sf/nav/occupancy_grid_utils.py`
