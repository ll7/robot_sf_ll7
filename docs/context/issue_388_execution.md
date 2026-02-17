# Issue 388 Execution Notes

Issue: [#388](https://github.com/ll7/robot_sf_ll7/issues/388)

## Problem
- Some obstacle paths in the OSM-derived obstacle SVGs are self-intersecting (`path1948`, `path1951`).
- The parser logged invalid polygons and downstream occupancy/grid behavior could drop or mis-handle them.

## Root Cause
- Compound SVG obstacle paths can produce invalid polygons after flattening into one vertex ring.
- Existing parser behavior only logged invalidity and kept raw vertices unchanged.

## Implemented Fix
- Added invalid obstacle repair in `SvgMapConverter._process_obstacle_path(...)`:
  - Use `shapely.validation.make_valid(...)`.
  - Extract polygon members from `Polygon`, `MultiPolygon`, or `GeometryCollection`.
  - Select the largest non-empty polygon as deterministic fallback geometry.
  - Keep raw geometry only when no polygon can be recovered.

## Regression Coverage
- Added `tests/test_svg_obstacle_self_intersection.py`.
- New test asserts that known problematic obstacles (`path1948`, `path1951`) are repaired into valid polygons with positive area.

## Validation Commands
- `uv run ruff check robot_sf/nav/svg_map_parser.py tests/test_svg_obstacle_self_intersection.py`
- `uv run pytest tests/test_svg_obstacle_self_intersection.py tests/test_svg_path_parsing_commands.py`
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
